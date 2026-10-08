"""Unit tests for systemid_core — pure SysID transform logic.

Must run without booting Omniverse. Run:
  cd scripts/simvla && $PY -m pytest test_systemid.py -v
"""
import json
import numpy as np
import pytest

import systemid_core as sc


def _write(tmp_path, name, obj):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(obj))
    return tmp_path


BIMANUAL_MOBILE_EEF = {
    "_comment": "top-level comment",
    "arms": 2, "mobile": True, "threshold_gripper": -0.8,
    "init_joint_pos": {"_comment": "x", "j1": 0.1},
    "real_to_sim_frame_offset_xyz": [0.0, 0.0, 0.0],
    "eef_to_grasp_offset_local_xyz": [0.0, 0.0, 0.0],
    "layouts": {
        "eef": {
            "input": {
                "l_eef_xyz": [0, 1, 2], "l_rot6d": [3, 4, 5, 6, 7, 8], "l_gripper": [9],
                "r_eef_xyz": [10, 11, 12], "r_rot6d": [13, 14, 15, 16, 17, 18], "r_gripper": [19],
                "base_delta": [20, 21, 22],
            },
            "output": ["dLxyz", "dLrot", "dRxyz", "dRrot", "Lgrip", "Rgrip", "base"],
        }
    },
}


def test_load_strips_comment_keys_and_returns_dict(tmp_path):
    d = _write(tmp_path, "bot", BIMANUAL_MOBILE_EEF)
    cfg = sc.load_robot_cfg("bot", d)
    assert cfg["arms"] == 2 and cfg["mobile"] is True
    # top-level _comment keys are stripped; nested init_joint_pos _comment stays raw
    assert "_comment" not in cfg


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sc.load_robot_cfg("nope", tmp_path)


def test_validate_accepts_bimanual_mobile_eef():
    sc.validate_cfg(BIMANUAL_MOBILE_EEF)  # no raise


def test_validate_rejects_arms_mismatch():
    bad = json.loads(json.dumps(BIMANUAL_MOBILE_EEF))
    bad["arms"] = 1  # but layout still has r_* fields
    with pytest.raises(ValueError, match="arms"):
        sc.validate_cfg(bad)


def test_validate_rejects_output_field_without_input():
    # static+eef so the mobile-vs-base_delta check does NOT preempt; offset keys present
    # so the offset check doesn't preempt either; this genuinely reaches _output_backed('base').
    bad = {
        "arms": 1, "mobile": False, "threshold_gripper": -0.8,
        "real_to_sim_frame_offset_xyz": [0.0, 0.0, 0.0],
        "eef_to_grasp_offset_local_xyz": [0.0, 0.0, 0.0],
        "layouts": {"eef": {
            "input": {"l_eef_xyz": [0, 1, 2], "l_rot6d": [3, 4, 5, 6, 7, 8], "l_gripper": [9]},
            "output": ["dLxyz", "base"],
        }},
    }
    with pytest.raises(ValueError, match="base"):
        sc.validate_cfg(bad)


def test_validate_rejects_arm_with_incomplete_input():
    # l_eef_xyz present but l_rot6d missing -> build_eef_fields would KeyError at runtime;
    # validate_cfg now catches it loudly (offset keys present so they don't preempt).
    bad = {
        "arms": 1, "mobile": False, "threshold_gripper": -0.8,
        "real_to_sim_frame_offset_xyz": [0.0, 0.0, 0.0],
        "eef_to_grasp_offset_local_xyz": [0.0, 0.0, 0.0],
        "layouts": {"eef": {
            "input": {"l_eef_xyz": [0, 1, 2], "l_gripper": [9]},  # no l_rot6d
            "output": ["dLxyz", "dLrot", "Lgrip"],
        }},
    }
    with pytest.raises(ValueError, match="l_rot6d"):
        sc.validate_cfg(bad)


def test_validate_rejects_missing_eef_offset_keys():
    bad = json.loads(json.dumps(BIMANUAL_MOBILE_EEF))
    del bad["real_to_sim_frame_offset_xyz"]
    with pytest.raises(ValueError, match="real_to_sim_frame_offset_xyz"):
        sc.validate_cfg(bad)


def test_validate_rejects_output_for_absent_arm():
    # output names a right-arm field but no r_* input exists (arms=1) -> reaches _output_backed.
    bad = {
        "arms": 1, "mobile": False, "threshold_gripper": -0.8,
        "real_to_sim_frame_offset_xyz": [0.0, 0.0, 0.0],
        "eef_to_grasp_offset_local_xyz": [0.0, 0.0, 0.0],
        "layouts": {"eef": {
            "input": {"l_eef_xyz": [0, 1, 2], "l_rot6d": [3, 4, 5, 6, 7, 8], "l_gripper": [9]},
            "output": ["dLxyz", "dRxyz"],
        }},
    }
    with pytest.raises(ValueError, match="dRxyz"):
        sc.validate_cfg(bad)


def test_validate_rejects_wrong_rot6d_width():
    bad = json.loads(json.dumps(BIMANUAL_MOBILE_EEF))
    bad["layouts"]["eef"]["input"]["l_rot6d"] = [3, 4, 5, 6, 7]  # width 5
    with pytest.raises(ValueError, match="l_rot6d"):
        sc.validate_cfg(bad)


def test_select_inputs_slices_named_columns():
    arr = np.arange(3 * 23).reshape(3, 23).astype(np.float64)  # T=3, D=23
    layout = BIMANUAL_MOBILE_EEF["layouts"]["eef"]["input"]
    out = sc.select_inputs(arr, layout)
    assert out["l_eef_xyz"].shape == (3, 3)
    assert np.array_equal(out["l_eef_xyz"][0], [0, 1, 2])
    assert np.array_equal(out["base_delta"][0], [20, 21, 22])


def test_select_inputs_preserves_reorder():
    arr = np.arange(2 * 20).reshape(2, 20).astype(np.float64)
    out = sc.select_inputs(arr, {"joints": [17, 18, 0, 1]})
    assert np.array_equal(out["joints"][0], [17, 18, 0, 1])


def _identity_rot6d(T):
    # 6D of identity rotation: first two columns of I3 -> [1,0,0, 0,1,0]
    r = np.zeros((T, 6)); r[:, 0] = 1.0; r[:, 4] = 1.0
    return r


def test_calibration_adds_frame_offset():
    T = 2
    inputs = {
        "l_eef_xyz": np.zeros((T, 3)), "l_rot6d": _identity_rot6d(T),
        "r_eef_xyz": np.zeros((T, 3)), "r_rot6d": _identity_rot6d(T),
    }
    cfg = {"real_to_sim_frame_offset_xyz": [1.0, 2.0, 3.0],
           "eef_to_grasp_offset_local_xyz": [0.0, 0.0, 0.0]}
    sc.apply_eef_calibration(inputs, cfg)
    assert np.allclose(inputs["l_eef_xyz"][0], [1.0, 2.0, 3.0])
    assert np.allclose(inputs["r_eef_xyz"][0], [1.0, 2.0, 3.0])


def test_calibration_grasp_offset_under_identity_is_local():
    T = 1
    inputs = {
        "l_eef_xyz": np.zeros((T, 3)), "l_rot6d": _identity_rot6d(T),
        "r_eef_xyz": np.zeros((T, 3)), "r_rot6d": _identity_rot6d(T),
    }
    cfg = {"real_to_sim_frame_offset_xyz": [0.0, 0.0, 0.0],
           "eef_to_grasp_offset_local_xyz": [0.0, 0.0, 0.5]}
    sc.apply_eef_calibration(inputs, cfg)
    # identity rotation -> world offset == local offset
    assert np.allclose(inputs["l_eef_xyz"][0], [0.0, 0.0, 0.5])


def test_calibration_grasp_offset_rotated_by_nonidentity_R():
    # 90deg about z: R = [[0,-1,0],[1,0,0],[0,0,1]]. rot6d = first two columns
    # of R flattened = [0,1,0, -1,0,0]. R @ [0.5,0,0] = [0, 0.5, 0].
    rot6d = np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    inputs = {
        "l_eef_xyz": np.zeros((1, 3)), "l_rot6d": rot6d,
        "r_eef_xyz": np.zeros((1, 3)), "r_rot6d": rot6d,
    }
    cfg = {"real_to_sim_frame_offset_xyz": [0.0, 0.0, 0.0],
           "eef_to_grasp_offset_local_xyz": [0.5, 0.0, 0.0]}
    sc.apply_eef_calibration(inputs, cfg)
    assert np.allclose(inputs["l_eef_xyz"][0], [0.0, 0.5, 0.0], atol=1e-6)
    assert np.allclose(inputs["r_eef_xyz"][0], [0.0, 0.5, 0.0], atol=1e-6)


import torch


def test_binarize_gripper_threshold():
    g = torch.tensor([[-1.0], [0.0], [-0.9]])
    out = sc.binarize_gripper(g, -0.8)
    assert torch.allclose(out, torch.tensor([[1.0], [-1.0], [1.0]]))


def test_base_local_to_world_zero_yaw_is_identity():
    base = torch.tensor([[1.0, 2.0, 0.3]])
    yaw = torch.tensor([0.0])
    out = sc.base_local_to_world(base, yaw)
    assert torch.allclose(out, base)


def test_base_local_to_world_rotates_by_yaw():
    base = torch.tensor([[1.0, 0.0, 0.5]])           # vx=1, vy=0, omega=0.5
    yaw = torch.tensor([torch.pi / 2])
    out = sc.base_local_to_world(base, yaw)
    # 90deg: vx_world ~ 0, vy_world ~ 1, omega passes through
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.5]), atol=1e-6)


def _eef_curr(N, device="cpu"):
    return {
        "l_eef_xyz": torch.zeros((N, 3)), "l_rot6d": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(N, 1),
        "l_gripper": torch.full((N, 1), -1.0),
        "r_eef_xyz": torch.zeros((N, 3)), "r_rot6d": torch.tensor([[1.0, 0, 0, 0, 1, 0]]).repeat(N, 1),
        "r_gripper": torch.full((N, 1), 0.0),
        "base_delta": torch.zeros((N, 3)),
    }


def test_build_eef_fields_first_step_deltas_zero():
    cfg = BIMANUAL_MOBILE_EEF
    curr = _eef_curr(2)
    out = sc.build_eef_fields(curr, None, cfg, yaw=torch.zeros(2),
                              reset_mask=torch.ones(2, dtype=torch.bool), device="cpu")
    for k in ("dLxyz", "dLrot", "dRxyz", "dRrot"):
        assert torch.allclose(out[k], torch.zeros(2, 3))
    assert set(out.keys()) >= {"dLxyz", "dLrot", "Lgrip", "dRxyz", "dRrot", "Rgrip", "base"}


def test_build_eef_fields_gripper_and_translation_delta():
    cfg = BIMANUAL_MOBILE_EEF
    prev = _eef_curr(1)
    curr = _eef_curr(1)
    curr["l_eef_xyz"] = torch.tensor([[0.1, 0.0, 0.0]])
    out = sc.build_eef_fields(curr, prev, cfg, yaw=torch.zeros(1),
                              reset_mask=torch.zeros(1, dtype=torch.bool), device="cpu")
    assert torch.allclose(out["dLxyz"], torch.tensor([[0.1, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(out["Lgrip"], torch.tensor([[1.0]]))    # -1.0 < -0.8 -> close
    assert torch.allclose(out["Rgrip"], torch.tensor([[-1.0]]))   #  0.0 !< -0.8 -> open


def test_build_eef_fields_reset_mask_zeros_only_flagged_rows():
    cfg = BIMANUAL_MOBILE_EEF
    prev = _eef_curr(2)
    curr = _eef_curr(2)
    curr["l_eef_xyz"] = torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    mask = torch.tensor([True, False])
    out = sc.build_eef_fields(curr, prev, cfg, yaw=torch.zeros(2), reset_mask=mask, device="cpu")
    assert torch.allclose(out["dLxyz"][0], torch.zeros(3))                 # reset row
    assert torch.allclose(out["dLxyz"][1], torch.tensor([0.2, 0.0, 0.0]))  # kept row


def test_build_eef_fields_single_arm_static_has_no_r_or_base():
    cfg = {"arms": 1, "mobile": False, "threshold_gripper": -0.8,
           "layouts": {"eef": {"input": {"l_eef_xyz": [0, 1, 2], "l_rot6d": [3, 4, 5, 6, 7, 8],
                                          "l_gripper": [9]},
                               "output": ["dLxyz", "dLrot", "Lgrip"]}}}
    curr = {"l_eef_xyz": torch.zeros((1, 3)),
            "l_rot6d": torch.tensor([[1.0, 0, 0, 0, 1, 0]]), "l_gripper": torch.full((1, 1), -1.0)}
    out = sc.build_eef_fields(curr, None, cfg, yaw=None,
                             reset_mask=torch.ones(1, dtype=torch.bool), device="cpu")
    assert set(out.keys()) == {"dLxyz", "dLrot", "Lgrip"}


def test_pack_output_concatenates_in_order():
    fields = {"dLxyz": torch.ones((2, 3)), "dLrot": torch.zeros((2, 3)),
              "Lgrip": torch.full((2, 1), -1.0)}
    out = sc.pack_output(fields, ["dLxyz", "dLrot", "Lgrip"])
    assert out.shape == (2, 7)
    assert torch.allclose(out[:, :3], torch.ones((2, 3)))
    assert torch.allclose(out[:, 6:7], torch.full((2, 1), -1.0))


def test_pack_output_missing_field_raises():
    with pytest.raises(KeyError):
        sc.pack_output({"dLxyz": torch.ones((1, 3))}, ["dLxyz", "base"])


def test_pack_output_follows_layout_order_not_dict_order():
    # fields inserted in build_eef_fields' dict order (l: dxyz,drot,grip; then r...; then base),
    # but the requested layout is anubis.json's order (all dxyz/drot, then grips, then base).
    # These diverge, so a cat that followed dict order instead of the layout would fail here.
    fields = {
        "dLxyz": torch.full((1, 3), 1.0),
        "dLrot": torch.full((1, 3), 2.0),
        "Lgrip": torch.full((1, 1), 3.0),
        "dRxyz": torch.full((1, 3), 4.0),
        "dRrot": torch.full((1, 3), 5.0),
        "Rgrip": torch.full((1, 1), 6.0),
        "base":  torch.full((1, 3), 7.0),
    }
    layout = ["dLxyz", "dLrot", "dRxyz", "dRrot", "Lgrip", "Rgrip", "base"]
    out = sc.pack_output(fields, layout)
    assert out.shape == (1, 17)  # 3+3+3+3+1+1+3
    expected = torch.tensor(
        [[1, 1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 5, 3, 6, 7, 7, 7]], dtype=out.dtype
    )
    assert torch.allclose(out, expected)


def test_episode_frame_range_returns_half_open_contiguous_range():
    ep = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert sc.episode_frame_range(ep, 0) == (0, 3)
    assert sc.episode_frame_range(ep, 1) == (3, 5)
    assert sc.episode_frame_range(ep, 2) == (5, 9)


def test_episode_frame_range_accepts_plain_list():
    assert sc.episode_frame_range([0, 0, 1, 1], 1) == (2, 4)


def test_episode_frame_range_missing_episode_raises():
    with pytest.raises(ValueError, match="7"):
        sc.episode_frame_range(np.array([0, 0, 1]), 7)


def test_episode_frame_range_non_contiguous_raises():
    ep = np.array([0, 1, 0])  # episode 0 split by episode 1
    with pytest.raises(ValueError, match="contiguous"):
        sc.episode_frame_range(ep, 0)


QPOS_INIT = {
    "source": "qpos",
    "joints": {"arm2_base_link_joint": 0, "link21_joint": 1, "arm1_base_link_joint": 7},
    "grippers": {
        "gripper2_joint": {"col": 6, "offset": 0.1, "scale": 1.7, "range": 0.04},
    },
}


def test_init_joints_from_qpos_copies_arm_columns():
    row = np.arange(17, dtype=np.float64)  # col i has value i
    out = sc.init_joints_from_qpos(row, QPOS_INIT)
    assert out["arm2_base_link_joint"] == 0.0
    assert out["link21_joint"] == 1.0
    assert out["arm1_base_link_joint"] == 7.0


def test_init_joints_from_qpos_inverts_gripper_transform():
    row = np.zeros(17, dtype=np.float64)
    row[6] = 0.1  # qpos == offset -> sim joint 0
    out = sc.init_joints_from_qpos(row, QPOS_INIT)
    assert abs(out["gripper2_joint"] - 0.0) < 1e-12
    # round-trip: sim2ruin computes qpos = offset - scale*joint/range
    row[6] = 0.1 - 1.7 * 0.02 / 0.04
    out = sc.init_joints_from_qpos(row, QPOS_INIT)
    assert abs(out["gripper2_joint"] - 0.02) < 1e-12


def test_init_joints_from_qpos_out_of_range_column_raises():
    row = np.zeros(5, dtype=np.float64)  # too short for col 7
    with pytest.raises(ValueError, match="column"):
        sc.init_joints_from_qpos(row, QPOS_INIT)


def _cfg_with_qpos_init(qpos_init):
    cfg = json.loads(json.dumps(BIMANUAL_MOBILE_EEF))
    cfg["qpos_init"] = qpos_init
    return cfg


def test_validate_accepts_valid_qpos_init():
    sc.validate_cfg(_cfg_with_qpos_init({
        "source": "qpos",
        "joints": {"link21_joint": 1},
        "grippers": {"gripper2_joint": {"col": 6, "offset": 0.1, "scale": 1.7, "range": 0.04}},
    }))  # no raise


def test_validate_rejects_qpos_init_without_source():
    with pytest.raises(ValueError, match="source"):
        sc.validate_cfg(_cfg_with_qpos_init({"joints": {"link21_joint": 1}}))


def test_validate_rejects_qpos_init_negative_column():
    with pytest.raises(ValueError, match="link21_joint"):
        sc.validate_cfg(_cfg_with_qpos_init({"source": "qpos", "joints": {"link21_joint": -1}}))


def test_validate_rejects_gripper_entry_missing_key():
    with pytest.raises(ValueError, match="gripper2_joint"):
        sc.validate_cfg(_cfg_with_qpos_init({
            "source": "qpos", "joints": {},
            "grippers": {"gripper2_joint": {"col": 6, "offset": 0.1}},  # no scale/range
        }))


# ---------------------------------------------------------------------------
# RB-Y1 raw npz loader
# ---------------------------------------------------------------------------

def _rby1_npz(tmp_path, T=5, dt=0.05):
    """A tiny RB-Y1 log in the real recorder's layout (24 joints, 14 leader, 2 grippers)."""
    t = np.arange(T) * dt
    names = (["right_wheel", "left_wheel"] + [f"torso_{i}" for i in range(6)]
             + [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)]
             + ["head_0", "head_1"])
    pos = np.zeros((T, 24))
    pos[:, 2:8] = [0.0, 0.785, -1.571, 0.785, 0.0, 0.0]
    pos[:, 8:15] = np.arange(7) + 100.0 + t[:, None]      # right follower: 100+k+t
    pos[:, 15:22] = np.arange(7) + 200.0 + t[:, None]     # left follower:  200+k+t
    leader = np.zeros((T, 14))
    leader[:, :7] = np.arange(7) + 10.0 + t[:, None]      # right leader: 10+k+t
    leader[:, 7:] = np.arange(7) + 20.0 + t[:, None]      # left leader:  20+k+t
    grip = np.zeros((T, 2)); grip[:, 0] = [0, 0, 1, 1, 1][:T]; grip[:, 1] = 0.3
    p = tmp_path / "ep.npz"
    np.savez(p, joint_names=np.array(names), time_s=t, position=pos,
             leader_position=leader, gripper_command=grip, base_command=np.zeros((T, 3)))
    return p


def test_load_rby1_npz_shapes_and_column_layout(tmp_path):
    p = _rby1_npz(tmp_path)
    d = sc.load_rby1_npz(p, fps=20, sync_tol=None)
    assert d["action"].shape == (5, 22)
    assert d["observation.state"].shape == (5, 20)
    assert d["episode_index"].shape == (5,) and (d["episode_index"] == 0).all()
    a0, s0 = d["action"][0], d["observation.state"][0]
    np.testing.assert_allclose(a0[0:7], np.arange(7) + 10.0)       # right leader
    np.testing.assert_allclose(a0[7:14], np.arange(7) + 20.0)      # left leader
    np.testing.assert_allclose(a0[14:20], [0.0, 0.785, -1.571, 0.785, 0.0, 0.0])  # torso held
    np.testing.assert_allclose(s0[0:7], np.arange(7) + 100.0)      # right follower
    np.testing.assert_allclose(s0[7:14], np.arange(7) + 200.0)     # left follower
    np.testing.assert_allclose(s0[14:20], [0.0, 0.785, -1.571, 0.785, 0.0, 0.0])


def test_load_rby1_npz_gripper_is_binary_close_negative(tmp_path):
    d = sc.load_rby1_npz(_rby1_npz(tmp_path), fps=20, sync_tol=None)
    # right command 0,0,1,1,1 -> +1 (open) then -1 (close); left 0.3 -> open throughout
    np.testing.assert_array_equal(d["action"][:, 20], [1, 1, -1, -1, -1])
    np.testing.assert_array_equal(d["action"][:, 21], [1, 1, 1, 1, 1])


def test_load_rby1_npz_resamples_onto_uniform_fps_grid(tmp_path):
    p = _rby1_npz(tmp_path, T=5, dt=0.05)          # 0.2 s of data at 20 Hz
    d = sc.load_rby1_npz(p, fps=40, sync_tol=None)                 # ask for 40 Hz -> 9 samples 0..0.2
    assert d["action"].shape[0] == 9
    # the signals are 10+k+t, so at t=0.025 the right leader joint 0 reads 10.025
    np.testing.assert_allclose(d["action"][1, 0], 10.025)
    np.testing.assert_allclose(d["observation.state"][1, 0], 100.025)


def test_load_rby1_npz_rejects_unexpected_joint_order(tmp_path):
    p = _rby1_npz(tmp_path)
    z = dict(np.load(p))
    names = list(z["joint_names"]); names[8], names[15] = names[15], names[8]
    z["joint_names"] = np.array(names)
    np.savez(p, **z)
    with pytest.raises(ValueError, match="right_arm_0"):
        sc.load_rby1_npz(p, fps=20)


def test_load_rby1_npz_trims_to_first_synced_frame(tmp_path):
    # leader is 10+k+t but the follower starts 0.5 rad away and catches up from frame 2 on.
    p = _rby1_npz(tmp_path, T=5, dt=0.05)
    z = dict(np.load(p))
    z["position"][:2, 8:15] = z["leader_position"][:2, :7] + 0.5      # frames 0,1 unsynced
    z["position"][2:, 8:15] = z["leader_position"][2:, :7]            # synced from frame 2
    z["position"][:, 15:22] = z["leader_position"][:, 7:]
    np.savez(p, **z)
    d = sc.load_rby1_npz(p, fps=20, sync_tol=0.2)
    assert d["action"].shape[0] == 3                                   # frames 2,3,4 (t=0.10..0.20)
    np.testing.assert_allclose(d["action"][0, 0], 10.10)               # first kept frame is t=0.10
    np.testing.assert_allclose(d["observation.state"][0, 0], 10.10)
    d_all = sc.load_rby1_npz(p, fps=20, sync_tol=None)
    assert d_all["action"].shape[0] == 5


def test_load_rby1_npz_never_synced_raises(tmp_path):
    p = _rby1_npz(tmp_path)                                            # follower 90 rad off leader
    with pytest.raises(ValueError, match="never"):
        sc.load_rby1_npz(p, fps=20, sync_tol=0.2)
