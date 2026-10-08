"""Grasp geometry choices must not leak AI Worker mug calibration to other tasks."""

import pytest
import torch
from types import SimpleNamespace

from collector_profile import (
    augmentation_seeds,
    cartesian_fallback_allowed,
    anchor_relative_ik_target,
    arm_grasp_geometries,
    apply_ik_joint_deadzone,
    choose_pregrasp_back_sign,
    grasp_geometry,
    gripper_open_target_reached,
    left_postgrasp_lift_enabled,
    jaw_gap_shows_object_contact,
    mug_grasp_position_tolerance,
    mug_grasp_target_height,
    mug_jaw_contact_ready,
    mug_pad_contacts_loaded,
    mug_pre_lift_grasp_ready,
    post_lift_object_retained,
    postrelease_retreat_target,
    postrelease_final_home_step,
    advance_pregrasp_stage,
    pinch_axis_xy_correction,
    pinch_center_z_correction,
    planar_hold_command,
    recorded_episode_env_ids,
)


def test_strict_planner_profile_disables_cartesian_fallback():
    assert cartesian_fallback_allowed("0")
    assert not cartesian_fallback_allowed("1")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        cartesian_fallback_allowed("yes")


def test_final_home_only_matches_second_reset_after_opening():
    skills = [1, 2, 3, 4, 1, 5, 3, 4, 4]
    payloads = [[0.0] for _ in skills]
    assert postrelease_final_home_step(
        8, skills, payloads, reset_id=4, gripper_id=3)
    assert not postrelease_final_home_step(
        3, skills, payloads, reset_id=4, gripper_id=3)
    assert not postrelease_final_home_step(
        7, skills, payloads, reset_id=4, gripper_id=3)
    payloads[6][0] = 1.0
    assert not postrelease_final_home_step(
        8, skills, payloads, reset_id=4, gripper_id=3)


def test_planar_hold_opposes_world_drift_and_uses_right_positive_action():
    assert planar_hold_command((0, 0, 0), (0.01, -0.01, 0)) == pytest.approx(
        (-0.04, -0.04, 0))
    assert planar_hold_command((0, 0, 0), (0, 0, 0)) == (0, 0, 0)


def test_planar_hold_rotates_to_local_frame_and_wraps_yaw():
    import math

    forward, right, yaw = planar_hold_command(
        (0.01, 0, math.pi / 2), (0, 0, math.pi / 2))
    assert (forward, right, yaw) == pytest.approx((0, 0.04, 0))
    assert planar_hold_command((0, 0, -math.pi + 0.01), (0, 0, math.pi - 0.01))[2] == pytest.approx(0.08)
    assert planar_hold_command((100, -100, 1), (0, 0, 0)) == pytest.approx((0.05, 0.05, 0.15))


def test_augmentation_seeds_are_repeatable_and_separate():
    first = augmentation_seeds(123)
    assert first == augmentation_seeds(123)
    assert first[0] != first[1]
    assert first != augmentation_seeds(124)


@pytest.mark.parametrize(
    ("robot", "object_name", "expected"),
    [
        ("anubis", "bowl0", "authored"),
        ("rby1", "mug0", "authored"),
        ("aiworker", "bowl0", "authored"),
        ("aiworker", "mug0", "cylindrical_mug"),
    ],
)
def test_default_grasp_geometry(robot, object_name, expected):
    assert grasp_geometry(robot, object_name) == expected


def test_left_only_mug_task_selects_left_geometry_independently():
    assert arm_grasp_geometries("aiworker", "none", "mug0") == (
        "authored", "cylindrical_mug")


@pytest.mark.parametrize(("gap", "expected"), [
    (0.0003, False), (0.0078, False), (0.015, True), (0.0709, True),
    (float("nan"), False), (float("inf"), False),
])
def test_subgrasp_requires_positive_loaded_jaw_gap(gap, expected):
    assert jaw_gap_shows_object_contact(gap) is expected


@pytest.mark.parametrize(("left", "right", "expected"), [
    (18.7, 2.4, True), (18.7, 0.0, False), (0.0, 0.0, False),
    (1.0, 1.0, True), (float("nan"), 2.0, False),
])
def test_instrumented_mug_grasp_requires_load_on_both_pads(left, right, expected):
    assert mug_pad_contacts_loaded(left, right) is expected


@pytest.mark.parametrize(("pad_forces", "expected"), [
    (None, True), (None, False), ((0.0, 0.0), False),
    ((18.46, 2.40), True), ((18.46, 0.0), False),
    ((float("nan"), 2.0), False),
])
def test_pre_lift_mug_gate_uses_geometry_without_sensors_and_force_with_sensors(pad_forces, expected):
    # The second case is the low-effort trace: proximity and a wide open gap looked valid,
    # but neither jaw contacted the mug. The two None cases distinguish generic tasks (pass)
    # from a geometric failure (fail) when sensors are absent.
    if pad_forces is None and expected is False:
        assert not mug_pre_lift_grasp_ready(0.08, 0.067, 0.026, 0.069)
    else:
        assert mug_pre_lift_grasp_ready(0.08, 0.068, 0.011, 0.055,
                                        pad_forces_n=pad_forces) is expected


def test_bilateral_pad_contact_supersedes_uncalibrated_root_relative_geometry():
    # RB-Y1/AI Worker body origins are not the pad contact frame. Instrumented bilateral load
    # should authorize real contact even when the wrist/body-root proxy is outside its window.
    assert mug_pre_lift_grasp_ready(
        0.10, 0.068, 0.080, 0.20, pad_forces_n=(5.0, 4.0)
    )
    assert not mug_pre_lift_grasp_ready(
        0.10, 0.068, 0.080, 0.20, pad_forces_n=(5.0, 0.0)
    )
    assert mug_pre_lift_grasp_ready(
        0.10, 0.068, float("nan"), float("nan"), pad_forces_n=(5.0, 4.0)
    )


@pytest.mark.parametrize(
    ("lifted_by", "jaw_gap", "expected"),
    [
        (0.1629, 0.0163, True),
        (0.0, 0.054, False),
        (-0.0205, 0.0042, False),
        (float("nan"), 0.02, False),
    ],
)
def test_pose_bank_grasp_requires_object_retention_after_lift(
    lifted_by, jaw_gap, expected,
):
    assert post_lift_object_retained(lifted_by, jaw_gap) is expected


def test_post_lift_grasp_requires_bilateral_pad_load_when_available():
    assert post_lift_object_retained(0.08, 0.04, pad_forces_n=(5.0, 4.0))
    assert not post_lift_object_retained(0.08, 0.04, pad_forces_n=(5.0, 0.0))


def test_pregrasp_side_uses_measured_wrist_not_base_for_elevated_grasp():
    # Tool +Z points down. The base is below the object, but this wrist is above it;
    # selecting by base position would put the planned waypoint below the object.
    sign = choose_pregrasp_back_sign(
        goal_position=(0.0, 0.0, 0.9),
        tool_axis=(0.0, 0.0, -1.0),
        reference_position=(0.0, 0.0, 1.4),
        standoff_m=0.18,
    )
    assert sign == -1


def test_pregrasp_side_selects_the_nearer_lateral_approach():
    assert choose_pregrasp_back_sign(
        goal_position=(0.0, 0.0, 0.0),
        tool_axis=(2.0, 0.0, 0.0),
        reference_position=(0.4, 0.0, 0.0),
        standoff_m=0.2,
    ) == 1


def test_aiworker_mug_grasp_centers_the_jaw_gap_on_the_object():
    assert pinch_center_z_correction(
        (1.0, 2.0, 0.8), (0.0, 0.0, 0.06), (1.0, 2.0, 0.9)
    ) == pytest.approx(0.04)
    assert pinch_center_z_correction(
        (1.0, 2.0, 0.8), (0.0, 0.0, 0.06), (1.0, 2.0, 0.9), 0.04
    ) == pytest.approx(0.08)


def test_corrective_ik_jog_rebases_the_accumulated_target_at_current_pose():
    controller = SimpleNamespace(
        ee_pos_des=torch.zeros((2, 3)), ee_quat_des=torch.zeros((2, 4)))
    term = SimpleNamespace(_ik_controller=controller)
    position = torch.tensor([0.4, -0.2, 0.8])
    quaternion = torch.tensor([0.7, 0.1, 0.2, 0.3])

    anchor_relative_ik_target(term, 1, position, quaternion)

    assert torch.equal(controller.ee_pos_des[1], position)
    assert torch.equal(controller.ee_quat_des[1], quaternion)
    assert torch.equal(controller.ee_pos_des[0], torch.zeros(3))


def test_pregrasp_stages_far_position_orientation_standoff_and_approach_one_way():
    stage = 0
    stage = advance_pregrasp_stage(stage, 0.029, 40.0, 0.200)
    assert stage == 1
    # Position noise cannot undo reaching the distant rotation waypoint.
    stage = advance_pregrasp_stage(stage, 0.200, 40.0, 0.200)
    assert stage == 1
    # The longer AI Worker approach can plateau at 17.3 degrees at the far waypoint.
    # Transition there, then require the full orientation near the mug.
    assert advance_pregrasp_stage(1, 0.200, 19.0, 0.200) == 2
    assert advance_pregrasp_stage(1, 0.200, 20.1, 0.200) == 1
    stage = advance_pregrasp_stage(stage, 0.200, 4.9, 0.200)
    assert stage == 2
    # Near standoff is where a bounded far-waypoint residual can be corrected, but the final
    # axial approach stays blocked until strict grasp orientation is met.
    stage = advance_pregrasp_stage(stage, 0.200, 5.1, 0.029)
    assert stage == 2
    stage = advance_pregrasp_stage(stage, 0.200, 4.9, 0.029)
    assert stage == 3
    # Neither error can move the approach backwards.
    assert advance_pregrasp_stage(stage, 1.0, 180.0, 1.0) == 3


@pytest.mark.parametrize("stage,position,rotation", [(-1, 0.0, 0.0), (4, 0.0, 0.0),
                                                       (0, float("nan"), 0.0),
                                                       (0, 0.0, -1.0)])
def test_pregrasp_transition_rejects_invalid_state(stage, position, rotation):
    with pytest.raises(ValueError):
        advance_pregrasp_stage(stage, position, rotation, 0.0)


def test_mug_reach_tolerance_is_bounded_and_non_mugs_keep_their_gate():
    assert mug_grasp_position_tolerance("mug0", 0.012, True) == 0.040
    assert mug_grasp_position_tolerance("mug0", 0.05, True) == 0.05
    assert mug_grasp_position_tolerance("bowl0", 0.012, True) == 0.012
    assert mug_grasp_position_tolerance("mug0", 0.012, False) == 0.012


@pytest.mark.parametrize(
    ("lateral", "height", "expected"),
    [(0.019, 0.055, True), (0.020, 0.055, True), (0.0201, 0.055, False),
     # The public mug is 85.7 mm tall: 79.4 mm is at its rim, not a secure side grip.
     (0.0144, 0.0794, False), (0.0144, 0.074, True), (0.0144, 0.055, True),
     (0.01, 0.04, False), (0.01, 0.15, False), (float("nan"), 0.08, False)],
)
def test_mug_jaw_contact_gate_uses_bounded_physical_window(lateral, height, expected):
    assert mug_jaw_contact_ready(lateral, height) is expected


def test_mug_grasp_target_is_strictly_inside_contact_window():
    target = mug_grasp_target_height()
    assert target == pytest.approx(0.0575)
    assert mug_jaw_contact_ready(0.0, target)
    assert mug_grasp_target_height(requested_m=0.065) == pytest.approx(0.065)
    for invalid in (0.04, 0.075, 0.08, float("nan")):
        with pytest.raises(ValueError, match="SIMVLA_MUG_GRASP_HEIGHT"):
            mug_grasp_target_height(requested_m=invalid)


def test_open_gripper_gate_waits_for_both_joints_to_reach_full_open():
    positions = torch.tensor([[0.0399, 0.0399], [0.0386, 0.0386]])
    targets = torch.tensor([0.04, 0.04])
    assert gripper_open_target_reached(positions, targets, 0.0002).tolist() == [True, False]
    with pytest.raises(ValueError, match="SIMVLA_GRIPPER_OPEN_TARGET_TOL_M"):
        gripper_open_target_reached(positions, targets, float("nan"))


def test_postrelease_retreat_moves_toward_base_without_an_upward_hook():
    target = postrelease_retreat_target(
        (1.10, -3.33, 0.91), (1.10, -3.33, 0.85), (1.12, -2.90, 0.0),
        0.12, 0.05)
    assert target[0] == pytest.approx(1.1056, abs=1e-4)
    assert target[1] == pytest.approx(-3.2101, abs=1e-4)
    assert target[2] == pytest.approx(0.96)
    with pytest.raises(ValueError, match="coincident"):
        postrelease_retreat_target((1, 2, 3), (0, 0, 0), (0, 0, 0), 0.12, 0.05)


def test_override_is_explicit_and_validated():
    assert grasp_geometry("anubis", "bowl0", "cylindrical_mug") == "cylindrical_mug"
    with pytest.raises(ValueError, match="SIMVLA_GRASP_GEOMETRY"):
        grasp_geometry("aiworker", "mug0", "other")


def test_pinch_axis_correction_centres_the_jaw_gap_without_changing_height():
    goal = (1.0, -0.4, 0.95)
    jaw_offset = (0.0, 0.0, 0.01)
    obj = (1.05, -0.35, 0.90)
    correction = pinch_axis_xy_correction(goal, jaw_offset, obj)
    corrected = (goal[0] + correction[0], goal[1] + correction[1], goal[2])
    jaw_mid = (corrected[0] + jaw_offset[0], corrected[1] + jaw_offset[1])
    assert jaw_mid == pytest.approx(obj[:2])
    assert corrected[2] == goal[2]


def test_pinch_axis_correction_rejects_short_positions():
    with pytest.raises(ValueError, match="at least 2/3 coordinates"):
        pinch_axis_xy_correction((1,), (0, 0, 0), (1, 2, 3))


@pytest.mark.parametrize("value", [None, "", "0"])
def test_left_lift_is_off_by_default(value):
    assert not left_postgrasp_lift_enabled(value)


def test_left_lift_requires_explicit_one():
    assert left_postgrasp_lift_enabled("1")
    with pytest.raises(ValueError, match="SIMVLA_POSTGRASP_LIFT_LEFT"):
        left_postgrasp_lift_enabled("true")


def test_missing_episode_env_id_uses_reset_or_success_order():
    assert recorded_episode_env_ids(2, [1, 3], [3]) == [1, 3]
    assert recorded_episode_env_ids(1, [1, 3], [3]) == [3]
    assert recorded_episode_env_ids(2, [1], [1]) == [None, None]


class _FakeEnv:
    def __init__(self):
        self.terms = {}
        for name in ("armL_action", "armR_action"):
            controller = type("Controller", (), {})()
            controller.cfg = type("Cfg", (), {"delta_joint_deadzone": 0.01})()
            self.terms[name] = type("Term", (), {"_ik_controller": controller})()
        self.action_manager = type("ActionManager", (), {"get_term": lambda _, name: self.terms[name]})()


def test_ik_joint_deadzone_defaults_to_zero_for_collection():
    env = _FakeEnv()
    assert apply_ik_joint_deadzone(env) == 0.0
    assert all(term._ik_controller.cfg.delta_joint_deadzone == 0.0 for term in env.terms.values())


def test_ik_joint_deadzone_can_be_explicitly_set():
    env = _FakeEnv()
    assert apply_ik_joint_deadzone(env, "0.0005") == 0.0005
    assert all(term._ik_controller.cfg.delta_joint_deadzone == 0.0005 for term in env.terms.values())


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "abc"])
def test_ik_joint_deadzone_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        apply_ik_joint_deadzone(_FakeEnv(), value)
def test_aiworker_pad_offsets_use_surface_centres_not_linkage_origins():
    from collector_profile import jaw_pad_offset
    right = jaw_pad_offset("gripper_l_rh_p12_rn_r2")
    left = jaw_pad_offset("gripper_l_rh_p12_rn_l2")
    assert right[2] == left[2] == (0.04 - 0.00071787) / 2
    assert right[1] == -left[1]
    assert jaw_pad_offset("gripper2R") == (0, 0, .1088)
    assert jaw_pad_offset("unconfigured_finger") == (0, 0, 0)
    assert jaw_pad_offset("ee_finger_r1") == (-.003, 0, -.0305)
    assert jaw_pad_offset("ee_finger_l2") == (-.003, 0, -.0305)


def test_measured_locks_refresh_torso_but_preserve_synthetic_jaws():
    from collector_profile import measured_locked_joints
    original = {"torso_0": 0., "torso_1": 0., "synthetic_jaw": .04}
    result = measured_locked_joints(original, ["torso_0", "torso_1", "arm_0"], [.012, .00001, .9])
    assert result == {"torso_0": .012, "torso_1": 0., "synthetic_jaw": .04}
    assert original["torso_0"] == 0
    with pytest.raises(ValueError, match="non-finite"):
        measured_locked_joints(original, ["torso_0"], [float("nan")])


@pytest.mark.parametrize("batched", [False, True])
def test_pad_midpoint_rotates_offsets_and_does_not_mutate_body_data(batched):
    from collector_profile import jaw_contact_midpoint
    positions = torch.tensor([[[0., .06, 1.], [0., -.06, 1.]]])
    before = positions.clone()
    # Half turn about X flips the local contact-surface Z displacement.
    robot = SimpleNamespace(body_names=["gripper_l_rh_p12_rn_r2", "gripper_l_rh_p12_rn_l2"],
        data=SimpleNamespace(body_pos_w=positions,
            body_quat_w=torch.tensor([[[0., 1., 0., 0.], [0., 1., 0., 0.]]])))
    midpoint = jaw_contact_midpoint(robot, [0] if batched else 0, 0, 1)
    expected = torch.tensor([0., 0., 1. - .019641065])
    torch.testing.assert_close(midpoint, expected[None] if batched else expected)
    torch.testing.assert_close(positions, before)


def test_rby1_pad_centres_respect_mirrored_finger_frames():
    from collector_profile import jaw_contact_midpoint
    robot = SimpleNamespace(body_names=["ee_finger_r1", "ee_finger_r2"],
        data=SimpleNamespace(body_pos_w=torch.tensor([[[.053, 0., .04], [-.053, 0., .04]]]),
            body_quat_w=torch.tensor([[[1., 0., 0., 0.], [0., 0., 0., 1.]]])))
    torch.testing.assert_close(jaw_contact_midpoint(robot, 0, 0, 1), torch.tensor([0., 0., .0095]))
