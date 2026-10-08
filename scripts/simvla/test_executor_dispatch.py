"""The port must not change which branch the executor picks.

v1 decided by sniffing payload floats — torch.isclose against magic sentinels, `>= 900` against
flags, `>= 1000` against pot/range markers. v2 decides by integer skill id. For every skill that was
already correct, both must select the SAME branch; otherwise this refactor silently changed robot
behaviour, and a wrong arm goal is not something a test suite notices for you.

The differential test below is the load-bearing one. Its v1 side is NOT this file's opinion of what
v1 did: `_v1_branch` is transcribed from goal_format's CHANNEL_SENTINELS — a mutation-tested model
of the executor's real per-channel dispatch, built in Task 2 and cross-checked against the corpus by
a standalone oracle script. Its v2 side is the real executor_dispatch, driven from the real
skills.py registry.

Two deliberate differences, and they are the point of the whole spec:
  * the refrigerator: v1 recognises NOTHING for `N [-0.4]*3`, so the payload falls through and
    becomes the env-frame corner (-0.4, -0.4). 73 files. v2 says nav.open_articulation with
    back_off_m=0.4 and resolves 0.4 m behind the live base pose.
  * the 109 bowl-place fall-throughs: `A_r [-0.12]*3` writes the BOWL_PLACE sentinel into the
    POSITION slots, but the executor matches arm sentinels in the QUATERNION slots [3:7], so it
    reads a zero quaternion and IKs to the literal point (-0.12, -0.12, -0.12).

Run: pytest scripts/simvla/test_executor_dispatch.py -v
No Omniverse. (skills.py needs torch + isaaclab.utils.math, which the executor has too.)
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import executor_dispatch
import goal_format
import skills  # noqa: F401  — populates the REGISTRY that SKILL_ID() is derived from
from executor_dispatch import (
    ACTION_WIDTH,
    PADDING_SKILL_ID,
    PAYLOAD_SLOTS,
    UnexecutableStep,
    build_script,
    dispatch_ids,
    encode_payload,
    load_script,
)
from goal_format import Step
from skill_contract import REGISTRY, SKILL_ID


# ==================================================================================================
# The v1 dispatch, as the executor really performed it. goal_format.CHANNEL_SENTINELS IS the table.
# ==================================================================================================

def _v1_branch(action, payload):
    """Which branch would v1's executor have picked for this (action, payload)?

    Mirrors simvla_gen's pre-port logic exactly, including the two things that made it wrong:
    payloads are zero-padded to 14 before any check, and each sentinel is matched in exactly ONE
    channel and ONE slot range. Returns None when nothing matches — that is the fall-through, and
    the fall-through is the bug.
    """
    if action in ("G_r", "G_l"):
        return "gripper"
    if not isinstance(payload, list) or not payload:
        return None
    if isinstance(payload[0], list):
        return "authored"                                   # candidate grasp rows

    t = goal_format._pad(payload)

    if action == "N":
        for value, skill in goal_format.CHANNEL_SENTINELS["N"]:
            if goal_format._isclose_all(t[0:3], value):
                return {"nav.open_articulation": "pull", "nav.close_articulation": "push"}[skill]
        return None                                          # -> a literal (x, y, yaw). THE BUG.

    if action == "N_s":
        if t[2] <= -goal_format.POT_RANGE_MARKER:
            return "to_range"
        if t[2] >= goal_format.POT_RANGE_MARKER:
            return "to_pot"
        return "authored"

    if action in ("A_r", "A_l"):
        for value, skill in goal_format.CHANNEL_SENTINELS[action]:
            if goal_format._isclose_all(t[3:7], value):      # QUATERNION slots, not position
                return skill
        if t[0] >= goal_format.FLAG_THRESHOLD:
            return "reset"
        if t[3] >= goal_format.FLAG_THRESHOLD:
            return "place"
        # A stray sentinel in [3:7] this channel does not dispatch, or a zero quaternion left by a
        # short payload, is a fall-through: the executor IKs to a meaningless pose. Not "authored".
        if goal_format._matched_sentinel(t[3:7]) is not None:
            return None
        if all(v == 0.0 for v in t[3:7]):
            return None
        return "authored"

    if action == "A_b":
        if t[0] >= goal_format.FLAG_THRESHOLD and t[7] >= goal_format.FLAG_THRESHOLD:
            return "reset"
        if t[2] >= goal_format.POT_RANGE_MARKER:
            return "pot"
        if t[3] >= goal_format.FLAG_THRESHOLD and t[10] >= goal_format.FLAG_THRESHOLD:
            return "place"
        if t[0] >= goal_format.FLAG_THRESHOLD or t[7] >= goal_format.FLAG_THRESHOLD \
                or t[3] >= goal_format.FLAG_THRESHOLD or t[10] >= goal_format.FLAG_THRESHOLD:
            return None                                      # halves disagree: no single branch
        return "authored"

    return None


#: The v2 branch each skill id selects, per action block. This is read off the ported simvla_gen:
#: a skill id that is not in a block's mask list falls into that block's "authored pose" path.
_V2_BRANCH = {
    "nav.open_articulation": "pull",
    "nav.close_articulation": "push",
    "nav.to_pot": "to_pot",
    "nav.to_range": "to_range",
    "nav.to_prim": "authored",
    # Authored [x, y, yaw] on N_s, exactly like nav.to_prim: the base pose is computed while
    # authoring, so it lands in the nav block's authored-pose path and needs no branch of its
    # own. That is what makes the push-chair chain runnable on the unmodified executor.
    "nav.push_prim": "authored",
    # Authored [x, y, yaw] on N_s, same as nav.to_prim and nav.push_prim: the park pose is computed
    # while authoring (the door is shut at t=0, so where to stand is a fact about the scene), so it
    # lands in the nav block's authored-pose path and needs no branch of its own.
    "nav.to_door_handle": "authored",
    # Both hands to an authored 14-float pose, so the A_b block's normal_mask branch plans it to
    # those absolute poses exactly as arm.pose does -- no branch of its own. What makes this table
    # have to name it at all is the assertion below: it must equal the dispatch set exactly.
    "arm.push_pose": "authored",
    "arm.reset": "reset",
    "arm.place": "place",
    "arm.pot": "pot",
    "arm.bowl_place": "arm.bowl_place",
    "arm.bottle_to_position": "arm.move_left",      # A_r, v1's MOVE_LEFT
    "arm.mug_to_position": "arm.move_right",        # A_l, v1's MOVE_RIGHT
    "arm.bottle_pour": "arm.bottle_tilt",           # v1's BOTTLE_TILT
    "arm.move_front": "arm.move_front",
    "arm.grasp_target": "arm.grasp_target",
    "arm.pause": "arm.pause",
    "arm.pose": "authored",
    "arm.grasp": "authored",
    "arm.handle_pregrasp": "authored",
    "arm.handle_grasp": "authored",
    "arm.fridge_handle_grasp": "authored",
    # Authored 7-float poses on an arm channel, exactly like the other handle planners: bbox centre
    # plus a per-direction quaternion, so they land in the authored-arm path and need no branch.
    "arm.bar_handle_pregrasp": "authored",
    "arm.bar_handle_grasp": "authored",
    # A 7-vector absolute pose like the two above, so it rides the same generic path -- the only
    # difference is WHERE the pose comes from (the grasp pose rotated about a horizontal hinge).
    "arm.door_arc_pull": "authored",
    "arm.fridge_handle_pregrasp": "authored",
    # A wide-object bimanual grasp: plan = _authored(...), so it lands in the same authored-arm
    # branch arm.grasp does. Added here because this table must equal the registry exactly and
    # arm.squeeze reached the registry without it -- which is what had this assertion red.
    "arm.squeeze": "authored",
    # The base arcs about a door's hinge. Its own branch in simvla_gen's nav block, beside pull and
    # push, because it resolves three params rather than one.
    "nav.open_door_arc": "arc",
    "gripper.set": "gripper",
}


def _v2_branch(action, payload, tmp_path):
    """Load this v1 (action, payload) the way the ported executor does, and report its branch."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"goals": [[[action, payload]]]}))
    script, _meta, _ids = load_script(path)
    return _V2_BRANCH[script[0].skill]


# --- fixtures. Slot-distinguishable BY CONSTRUCTION: no two floats in a payload are equal, so a
# --- wrong-slot read cannot pass by luck. (An all-identical payload is exactly how a wrong-slot
# --- bug slipped through on this branch once already.)
_QUAT = [0.6322, -0.587, 0.3344, -0.3793]          # a real, non-uniform rotation
_POS = [0.68, -0.25, 0.97]                          # three distinct coordinates
_POS2 = [1.11, -0.42, 0.83]

V1_CASES = [
    # (action, payload, expected branch) — every skill that was already correct in v1.
    ("N",   [-0.25, -0.25, -0.25],                          "pull"),
    ("N",   [0.25, 0.25, 0.25],                             "push"),
    ("N_s", [0.77, -1.65, 0.31],                            "authored"),
    ("N_s", [0.77, -1.65, 1000.0],                          "to_pot"),
    ("N_s", [0.77, -1.65, -1000.0],                         "to_range"),
    ("A_r", [0.3] * 7,                                      "arm.pause"),
    ("A_r", [-0.12] * 7,                                    "arm.bowl_place"),
    ("A_r", [*_POS, 3.2, 3.2, 3.2, 3.2],                    "arm.move_left"),
    ("A_r", [*_POS, 3.3, 3.3, 3.3, 3.3],                    "arm.bottle_tilt"),
    ("A_r", [999.0, 0.0, 1.102, *_QUAT],                    "reset"),
    ("A_r", [*_POS, 999.0, 0.0, 0.0, 0.0],                  "place"),
    ("A_r", [*_POS, *_QUAT],                                "authored"),
    ("A_r", [[*_POS, *_QUAT], [*_POS2, *_QUAT]],            "authored"),   # grasp candidates
    ("A_l", [*_POS, 3.1, 3.1, 3.1, 3.1],                    "arm.move_right"),
    ("A_l", [*_POS, 3.5, 3.5, 3.5, 3.5],                    "arm.move_front"),
    ("A_l", [*_POS, 3.4, 3.4, 3.4, 3.4],                    "arm.grasp_target"),
    ("A_l", [999.0, 0.0, 1.102, *_QUAT],                    "reset"),
    ("A_l", [*_POS, 999.0, 0.0, 0.0, 0.0],                  "place"),
    ("A_l", [*_POS, *_QUAT],                                "authored"),
    ("A_b", [999.0, 0.0, 1.1, *_QUAT] + [999.0, 0.0, 1.2, *_QUAT],        "reset"),
    ("A_b", [*_POS, 999.0, 0.0, 0.0, 0.0] + [*_POS2, 999.0, 0.0, 0.0, 0.0], "place"),
    ("A_b", [*_POS, *_QUAT] + [*_POS2, *_QUAT],             "authored"),
    ("A_b", [0.0, 0.0, 1000.0, *_QUAT] + [*_POS2, *_QUAT],  "pot"),
    ("G_r", True,                                           "gripper"),
    ("G_l", False,                                          "gripper"),
]


@pytest.mark.parametrize("action,payload,expected", V1_CASES,
                         ids=[f"{a}-{e}" for a, _, e in V1_CASES])
def test_v1_and_v2_select_the_same_branch(action, payload, expected, tmp_path):
    """THE test. Every skill that already worked must dispatch identically after the port."""
    assert _v1_branch(action, payload) == expected, "the v1 model itself is wrong — fix it first"
    assert _v2_branch(action, payload, tmp_path) == expected, \
        f"v2 changed the branch for {action} {payload!r}"


def test_the_v1_model_is_not_a_rubber_stamp():
    """_v1_branch must be able to disagree, or the differential test proves nothing."""
    assert _v1_branch("N", [-0.4, -0.4, -0.4]) is None
    assert _v1_branch("A_r", [-0.12, -0.12, -0.12]) is None
    # A sentinel this channel does not dispatch is NOT that skill.
    assert _v1_branch("A_l", [0.0, 0.0, 0.0, 3.2, 3.2, 3.2, 3.2]) is None   # MOVE_LEFT is A_r's
    assert _v1_branch("A_r", [0.0, 0.0, 0.0, 3.1, 3.1, 3.1, 3.1]) is None   # MOVE_RIGHT is A_l's


# ==================================================================================================
# The two deliberate differences.
# ==================================================================================================

def test_the_refrigerator_is_a_deliberate_difference(tmp_path):
    """v1 selects NOTHING for [-0.4]*3, so the step becomes a literal env-frame coordinate and the
    robot drives to the same fixed corner in every kitchen. 73 of the 7,905 goal files on disk."""
    assert _v1_branch("N", [-0.4, -0.4, -0.4]) is None, "v1 recognises nothing — that is the bug"

    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"goals": [[["N", [-0.4, -0.4, -0.4]]]]}))
    with pytest.raises(goal_format.UnrecognisedStep, match="refrigerator"):
        load_script(path)          # loud, rather than reproducing the drive-to-a-corner


def test_the_109_bowl_place_fallthroughs_are_a_deliberate_difference(tmp_path):
    """A_r [-0.12]*3 puts BOWL_PLACE in the POSITION slots; the executor matches arm sentinels in
    the QUATERNION slots, reads (0,0,0,0), and IKs to the literal point. 109 steps on disk."""
    assert _v1_branch("A_r", [-0.12, -0.12, -0.12]) is None

    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"goals": [[["A_r", [-0.12, -0.12, -0.12]]]]}))
    with pytest.raises(goal_format.UnrecognisedStep, match="quaternion"):
        load_script(path)


def test_the_v2_refrigerator_resolves_behind_the_live_base_not_to_a_corner(tmp_path):
    """The whole point, end to end: author the fridge in v2, load it the way simvla_gen does, feed
    the payload to the resolver the way simvla_gen does, and land 0.4 m BEHIND the base."""
    goal_format.write_v2(
        tmp_path / "v2.json",
        [Step(skill="nav.open_articulation", action="N", params={"back_off_m": 0.4}, goal=None)],
        meta={"kitchen_type": "island"},
    )
    script, _meta, _ids = load_script(tmp_path / "v2.json")

    # simvla_gen writes the spec into payloads_tensor and reads back_off_m out of slot 0.
    payload = torch.zeros(14)
    payload[: len(script[0].spec)] = torch.tensor(script[0].spec)
    assert payload[0].item() == pytest.approx(0.4)
    assert payload[1].item() == 0.0 and payload[2].item() == 0.0, \
        "a runtime step's payload is zero apart from its argument — no sentinel came back"

    # Two envs, at different yaws and different origins: a wrong env or a wrong axis cannot pass.
    ctx = _fake_ctx(base_xy=[(2.0, 5.0), (-1.0, 3.0)], yaws=[0.0, math.pi / 2],
                    origins=[(10.0, 20.0, 0.0), (30.0, 40.0, 0.0)])
    envs = torch.tensor([0, 1])
    back_off = payload[0].expand(2)

    src, yaw = REGISTRY["nav.open_articulation"].cls().resolve(
        ctx, envs, {"back_off_m": back_off}, None)

    # env 0: yaw 0 -> facing +x -> back off along -x.  env 1: yaw pi/2 -> facing +y -> back off -y.
    assert src[0].tolist() == pytest.approx([2.0 - 10.0 - 0.4, 5.0 - 20.0], abs=1e-5)
    assert src[1].tolist() == pytest.approx([-1.0 - 30.0, 3.0 - 40.0 - 0.4], abs=1e-5)
    assert yaw.tolist() == pytest.approx([0.0, math.pi / 2], abs=1e-5)

    # And emphatically NOT the fixed env-frame corner v1 drove to.
    for row in src.tolist():
        assert row != pytest.approx([-0.4, -0.4], abs=1e-3)


def test_back_off_is_per_env_not_a_shared_scalar():
    """Two envs on different steps of the script resolve with different back-offs. Collapsing them
    to one scalar would give the fridge a drawer's 0.25 m and nothing would say so."""
    ctx = _fake_ctx(base_xy=[(2.0, 5.0), (2.0, 5.0)], yaws=[0.0, 0.0],
                    origins=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    src, _yaw = REGISTRY["nav.open_articulation"].cls().resolve(
        ctx, torch.tensor([0, 1]), {"back_off_m": torch.tensor([0.25, 0.4])}, None)
    assert src[0, 0].item() == pytest.approx(2.0 - 0.25)
    assert src[1, 0].item() == pytest.approx(2.0 - 0.40)


class _Data:
    def __init__(self, pos, quat):
        self.body_pos_w = pos
        self.body_quat_w = quat


class _Robot:
    def __init__(self, data):
        self.data = data


class _Ctx:
    def __init__(self, robot, env_origins, base_link_idx):
        self.robot = robot
        self.env_origins = env_origins
        self.base_link_idx = base_link_idx
        self.r_eef_idx = 1
        self.l_eef_idx = 2
        self.has_ramen = False
        self.has_sweet_potato = False
        self.rigid_objects = None


def _fake_ctx(base_xy, yaws, origins):
    """A robot whose base_link (body 0) sits at base_xy with yaw `yaws`, per env."""
    n = len(base_xy)
    pos = torch.zeros(n, 3, 3)
    quat = torch.zeros(n, 3, 4)
    for i, ((x, y), yaw) in enumerate(zip(base_xy, yaws)):
        pos[i, 0] = torch.tensor([x, y, 0.0])
        quat[i, 0] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return _Ctx(_Robot(_Data(pos, quat)), torch.tensor(origins), base_link_idx=0)


# ==================================================================================================
# skill_ids_tensor: built right, including the padding rows.
# ==================================================================================================

def _skill_ids_tensor(scripts, max_len):
    """Exactly what simvla_gen's two tensor builders do, minus the CUDA."""
    rows = []
    for script in scripts:
        ids = [st.skill_id for st in script]
        ids += [PADDING_SKILL_ID] * (max_len - len(ids))
        rows.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(rows)


def test_skill_ids_tensor_is_built_from_a_v2_file_including_padding(tmp_path):
    steps = [
        Step(skill="nav.to_prim", action="N_s", params={"prim_path": "/W/drawer"}, goal=[0.7, -1.6, 0.0]),
        Step(skill="arm.handle_grasp", action="A_r", params={"prim_path": "/W/h"},
             goal=[*_POS, *_QUAT]),
        Step(skill="gripper.set", action="G_r", params={"grasp": True}, goal=True),
        Step(skill="nav.open_articulation", action="N", params={"back_off_m": 0.4}, goal=None),
        Step(skill="arm.reset", action="A_r", params={}, goal=None),
    ]
    goal_format.write_v2(tmp_path / "v2.json", steps, meta={"kitchen_type": "l_shaped"})
    script, meta, ids = load_script(tmp_path / "v2.json")

    assert meta["kitchen_type"] == "l_shaped"
    assert [st.skill for st in script] == [s.skill for s in steps]

    # 2 envs, padded to 7 (the executor pads every env's script to the longest one).
    tensor = _skill_ids_tensor([script, script], max_len=7)
    assert tensor.shape == (2, 7)
    assert tensor.dtype == torch.long

    expected = [ids[s.skill] for s in steps]
    assert tensor[0, :5].tolist() == expected
    assert tensor[1, :5].tolist() == expected
    assert tensor[0, 5:].tolist() == [PADDING_SKILL_ID, PADDING_SKILL_ID]

    # The five ids must be five DIFFERENT ints, or the tensor cannot distinguish the steps.
    assert len(set(expected)) == 5


def test_the_padding_id_cannot_be_mistaken_for_a_skill():
    """-1 is not a skill id, so every `skill_ids_tensor == SKILL_ID[x]` mask is False on padding."""
    ids = dispatch_ids()
    assert PADDING_SKILL_ID == -1
    assert PADDING_SKILL_ID not in ids.values()
    assert min(ids.values()) == 0 and sorted(ids.values()) == list(range(len(ids))), \
        "ids are 0..n-1, which is why -1 is safe as padding"


def test_dispatch_ids_are_unique_ints_and_the_registry_keeps_its_SKILL_ID(tmp_path):
    ids = dispatch_ids()
    assert all(isinstance(v, int) for v in ids.values())
    assert len(set(ids.values())) == len(ids)
    # A registry skill's dispatch id IS its SKILL_ID(), so the two tables cannot drift.
    for name, i in SKILL_ID().items():
        assert ids[name] == i
    # Every registry skill AND every legacy-only skill has one.
    assert set(ids) == set(REGISTRY) | set(executor_dispatch.LEGACY_ONLY_SKILLS)


def test_every_skill_the_executor_can_meet_has_a_branch():
    """The dispatch table and the executor's branch table are the same set — no skill can arrive
    that simvla_gen has never heard of. (This is the G_b bug's shape, one level up.)"""
    assert set(_V2_BRANCH) == set(dispatch_ids()), \
        "a skill exists that this file does not know which branch it takes"


# ==================================================================================================
# The one-shot latch. This is the property a per-step differential test CANNOT see, because it is
# temporal, and it is the trap this port sets for itself.
#
# v1's sentinel dispatch was one-shot for free: resolving a step overwrote the very slots the
# sentinel lived in, so the isclose stopped matching and the goal froze. Dispatching on a skill id
# has no such side effect — the id is still there on the next tick. nav.open_articulation resolves
# to "0.25 m behind wherever the base is NOW", so a re-resolving executor would move the goal with
# the robot and reverse forever.
#
# simvla_gen boots Omniverse and needs a GPU, so this cannot be run. It can only be checked
# statically: every runtime resolver is guarded by ~resolved, and every one sets the latch.
# ==================================================================================================

_GEN = Path(__file__).parent / "simvla_gen.py"


def test_every_runtime_resolver_is_one_shot():
    import re
    code = _GEN.read_text()

    call_sites = re.findall(r'resolve_skill\(\s*"|skills\.resolve_\w+\(_ctx', code)
    latch_sets = re.findall(r"resolved_tensor\[[^\]]+\] = True", code)
    # Include multiline calls, such as the left-arm pause resolver. The pairing below
    # is the real assertion; this count also makes a new unlatched resolver fail loudly.
    assert len(call_sites) == 12, f"expected the 12 runtime resolvers, found {len(call_sites)}"
    assert len(latch_sets) == len(call_sites), (
        f"{len(call_sites)} resolver call sites but {len(latch_sets)} set the resolved latch. An "
        f"unlatched resolver re-fires every tick: nav.open_articulation would back off 0.25 m from "
        f"the base's NEW pose each time and the robot would reverse until it timed out."
    )

    # The three blocks that dispatch sentinel-replacing skills each read the latch...
    assert len(re.findall(r"~resolved_tensor\[", code)) == 3, "N / A_r / A_l each need the guard"
    # ...and rebuilding an env's payload clears it, exactly where v1's sentinels came back.
    assert "resolved_tensor[env_i] = False" in code


def test_the_flag_branches_are_deliberately_not_latched():
    """reset / place / pot must NOT be latched, and this records why.

    v1 read them out of `arm_r_goals = payloads_tensor[rows, cols]` — advanced indexing, which
    returns a COPY. The 999.0 flag was therefore never destroyed in payloads_tensor, and the branch
    re-fired on every replan. Latching them would freeze a goal v1 recomputed, which is the same
    class of error as failing to latch the resolvers, in the other direction.
    """
    import re
    code = _GEN.read_text()
    for mask in ("reset_mask = ", "place_mask = ", "reset_mask_l = ", "place_mask_l = ",
                 "reset_mask_l_b = ", "place_mask_l_b = ", "pot_mask_b = "):
        line = next(ln for ln in code.splitlines() if ln.strip().startswith(mask))
        assert "resolved" not in line, f"{mask.strip()} must not be latched: {line.strip()}"
    assert len(re.findall(r"payloads_tensor\[needs_plan_indices", code)) >= 1, (
        "the reset/place branches must keep reading a COPY of the payload, or the reasoning above "
        "stops holding"
    )


# ==================================================================================================
# The payload encoding.
# ==================================================================================================

def test_a_v2_place_goal_keeps_its_authored_xyz_and_carries_no_flag():
    """arm.place is plan(), not resolve(). Modelling it as runtime-resolved throws away 4,178
    authored place positions. The executor supplies only the quaternion, from the live eef."""
    step = Step(skill="arm.place", action="A_r", params={}, goal=list(_POS))
    spec = encode_payload(step, "arm.place")
    assert spec[:3] == pytest.approx(_POS)
    assert spec[3:] == [0.0, 0.0, 0.0, 0.0], "no 999.0 flag: the skill id says it is a place"
    assert len(spec) == 7


def test_a_runtime_step_is_all_zeros_apart_from_its_argument():
    for skill, action in [("arm.reset", "A_r"), ("arm.pause", "A_l")]:
        spec = encode_payload(Step(skill=skill, action=action, goal=None), skill)
        assert spec == [0.0] * 7, f"{skill} must carry no floats at all"

    # arm.bowl_place carries FOUR ARGUMENTS, not none: forward_m, down_m, min_eef_z, roll_deg.
    # Same shape as nav.open_articulation's back_off_m in slot 0, just wider -- arguments to a
    # named skill, not sentinels, read by simvla_gen before the resolver overwrites the row.
    # They exist because passing NO params made resolve() fall back to its own constants, which
    # discarded every authored place height silently.
    spec = encode_payload(Step(skill="arm.bowl_place", action="A_r", goal=None), "arm.bowl_place")
    assert spec[4:] == [0.0] * 3, "arm.bowl_place must not carry floats past slot 3"
    assert spec[0] == 0.23 and spec[1] == 0.05 and spec[3] == -20.0

    spec = encode_payload(
        Step(skill="nav.close_articulation", action="N", params={"back_off_m": 0.3}, goal=None),
        "nav.close_articulation")
    # FIVE slots: ACTION_WIDTH["N"] widened so nav.open_door_arc can carry its radial back-off
    # and its retreat direction. The articulation skills still use slot 0 alone; the rest stay
    # zero.
    assert spec == [0.3, 0.0, 0.0, 0.0, 0.0]


def test_a_v1_articulation_step_gets_v1s_hardcoded_back_off(tmp_path):
    """v1's back-off was not a number, it was the lookup key PULL, and the resolver hardcoded
    0.25 m. The v1 file records no back_off_m, so the skill's declared default has to be 0.25 or
    every existing pull/push step silently changes distance."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"goals": [[["N", [-0.25, -0.25, -0.25]]]]}))
    script, _meta, _ids = load_script(path)
    assert script[0].skill == "nav.open_articulation"
    assert script[0].spec == [0.25, 0.0, 0.0, 0.0, 0.0]   # N is 5 slots; see ACTION_WIDTH


def test_no_marker_survives_into_any_payload(tmp_path):
    """Every value v1 used AS A MARKER is gone from the floats: no 999.0 flag, no ±1000 pot/range
    marker, and no sentinel left in the quaternion slots [3:7] that the arm blocks used to sniff.

    Note what this does NOT forbid: a nav step's slot 0 holds 0.25, which is numerically equal to
    v1's PUSH sentinel. That coincidence — a lookup key that reads exactly like the distance it
    produces — is what invited the refrigerator bug. It is harmless now precisely because nothing
    sniffs the payload any more: 0.25 there is a back-off in metres, and the skill id, not the
    float, says so.
    """
    markers = {999.0, 1000.0, -1000.0}
    cases = [(a, p) for a, p, _ in V1_CASES if not isinstance(p, bool)]

    for action, payload in cases:
        path = tmp_path / "v1.json"
        path.write_text(json.dumps({"goals": [[[action, payload]]]}))
        spec = load_script(path)[0][0].spec
        rows = spec if spec and isinstance(spec[0], list) else [spec]
        for row in rows:
            assert not (markers & set(row)), f"{action} {payload!r} still carries a flag/marker"
            if action in ("A_r", "A_l", "A_b"):
                assert goal_format._matched_sentinel(row[3:7]) is None, \
                    f"{action} {payload!r} still has a sentinel in the slots v1 dispatched on"


def test_a_v1_pot_step_keeps_its_branch_but_loses_its_marker(tmp_path):
    """nav.to_pot / arm.pot were dispatched on a ±1000 parked in a coordinate slot. The branch is
    now the skill id and the slot is a plain zero — the branch overwrites it from the live pot."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"goals": [[["N_s", [0.77, -1.65, 1000.0]]]]}))
    script, _meta, _ids = load_script(path)
    assert script[0].skill == "nav.to_pot"
    assert script[0].spec == [0.0, 0.0, 0.0]


def test_an_A_b_step_whose_halves_disagree_is_refused(tmp_path):
    """A_b drives both arms with ONE skill. No corpus step disagrees; if one appears, say so."""
    with pytest.raises(UnexecutableStep, match="halves disagree"):
        build_script([Step(skill="arm.both", action="A_b",
                           params={"left": "reset", "right": "place"}, goal=None)])


def test_an_unknown_skill_is_refused_at_load(tmp_path):
    with pytest.raises(UnexecutableStep, match="no @skill declares"):
        build_script([Step(skill="arm.teleport", action="A_r", goal=[0.0] * 7)])


# ==================================================================================================
# A whole file, both versions, through the same door.
# ==================================================================================================

_V1_SCRIPT = [
    ["N_s", [0.77, -1.65, 0.0]],
    ["A_r", [[*_POS, *_QUAT], [*_POS2, *_QUAT]]],
    ["G_r", True],
    ["N", [-0.25, -0.25, -0.25]],
    ["A_r", [999.0, 0.0, 1.102, *_QUAT]],
    ["A_r", [*_POS2, 999.0, 0.0, 0.0, 0.0]],
    ["G_r", False],
]

_V2_SCRIPT = [
    Step(skill="nav.to_prim", action="N_s", params={"prim_path": "/W/d"}, goal=[0.77, -1.65, 0.0]),
    Step(skill="arm.grasp", action="A_r", params={"prim_path": "/W/d"},
         goal=[[*_POS, *_QUAT], [*_POS2, *_QUAT]]),
    Step(skill="gripper.set", action="G_r", params={"grasp": True}, goal=True),
    Step(skill="nav.open_articulation", action="N", params={"back_off_m": 0.25}, goal=None),
    Step(skill="arm.reset", action="A_r", params={}, goal=None),
    Step(skill="arm.place", action="A_r", params={"prim_path": "/W/d"}, goal=list(_POS2)),
    Step(skill="gripper.set", action="G_r", params={"grasp": False}, goal=False),
]


def test_a_v1_file_and_the_equivalent_v2_file_produce_the_same_script(tmp_path):
    """Both versions run, and for the skills v1 got right they produce the SAME executor script:
    same actions, same skill ids, same payload floats. That is the port's whole promise."""
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({"kitchen_type": "island", "goals": [_V1_SCRIPT]}))
    goal_format.write_v2(tmp_path / "v2.json", _V2_SCRIPT, meta={"kitchen_type": "island"})

    s1, m1, _ = load_script(v1)
    s2, m2, _ = load_script(tmp_path / "v2.json")

    assert m1["kitchen_type"] == m2["kitchen_type"] == "island"
    assert [st.action for st in s1] == [st.action for st in s2]
    assert [st.skill for st in s1] == [st.skill for st in s2]
    assert [st.skill_id for st in s1] == [st.skill_id for st in s2]
    assert [st.spec for st in s1] == [st.spec for st in s2]

    # ...and the script is not degenerate: it exercises six different skills, including the three
    # kinds that could each be got wrong on their own — a runtime nav step, a full-overwrite reset,
    # and a place whose authored xyz must survive.
    assert len({st.skill for st in s1}) == 6
    assert s1[5].skill == "arm.place" and s1[5].spec[:3] == pytest.approx(_POS2)


# ==================================================================================================
# nav.open_door_arc: three params into three slots, and the string-to-sign mapping
# ==================================================================================================

def test_the_arc_skill_encodes_its_three_params_into_the_n_row():
    """ACTION_WIDTH["N"] is 3, and that is the ENTIRE budget: radius, sweep, and the hinge sign.

    The declared param is the STRING "Right"/"Left"; the payload carries +1/-1, because
    payloads_tensor is float32 and a resolver cannot read a string out of it. That mapping lives
    here and nowhere else.
    """
    step = Step(
        skill="nav.open_door_arc", action="N",
        params={"radius_m": 0.717, "hinge_side": "Right", "sweep_deg": 30.0},
        goal=None,
    )
    row = encode_payload(step, "nav.open_door_arc")
    assert row == pytest.approx([0.717, 30.0, 1.0, 0.05, 45.0])


def test_the_arc_skill_encodes_a_left_hinge_as_minus_one():
    step = Step(
        skill="nav.open_door_arc", action="N",
        params={"radius_m": 0.60, "hinge_side": "Left", "sweep_deg": 25.0},
        goal=None,
    )
    assert encode_payload(step, "nav.open_door_arc") == pytest.approx([0.60, 25.0, -1.0, 0.05, 45.0])


def test_the_arc_skill_falls_back_to_its_declared_defaults():
    """A step authored without params still has to execute as SOMETHING defined, and the defaults
    come from the @skill declaration rather than a second copy in this module that could drift."""
    step = Step(skill="nav.open_door_arc", action="N", params={}, goal=None)
    assert encode_payload(step, "nav.open_door_arc") == pytest.approx([0.72, 30.0, 1.0, 0.05, 45.0])


def test_an_unknown_hinge_side_is_refused_rather_than_silently_signed():
    """A third value has no defined direction. Falling through to 0.0 would multiply the sweep by
    zero and spin the base in place while the door never moved -- a full run recorded, nothing
    said."""
    step = Step(
        skill="nav.open_door_arc", action="N",
        params={"radius_m": 0.7, "hinge_side": "Middle", "sweep_deg": 30.0},
        goal=None,
    )
    with pytest.raises(UnexecutableStep, match="hinge_side"):
        encode_payload(step, "nav.open_door_arc")


def test_diag_deg_rides_in_slot_four():
    """The retreat DIRECTION is a parameter, not an environment variable.

    Slots 0..2 are overwritten by the resolved (x, y, yaw), so the direction has to live above
    them — beside back_off_m in slot 3, which takes the identical route for the identical reason.
    """
    step = Step(
        skill="nav.open_door_arc", action="N",
        params={
            "radius_m": 0.6119823, "hinge_side": "Right", "sweep_deg": 0.0,
            "back_off_m": 0.15, "diag_deg": 30.0,
        },
        goal=None,
    )
    row = encode_payload(step, "nav.open_door_arc")
    assert len(row) == 5
    assert row[3] == pytest.approx(0.15)
    assert row[4] == pytest.approx(30.0)


def test_diag_deg_falls_back_to_its_declared_default():
    """A goal file authored before this param existed must still run, at 45 degrees — the
    geometric optimum for one fixed direction over a 0-to-90 swing."""
    step = Step(
        skill="nav.open_door_arc", action="N",
        params={
            "radius_m": 0.6119823, "hinge_side": "Right", "sweep_deg": 0.0, "back_off_m": 0.15,
        },
        goal=None,
    )
    row = encode_payload(step, "nav.open_door_arc")
    assert row[4] == pytest.approx(45.0)


def test_the_nav_channel_is_five_slots_wide():
    """Widening costs nothing — simvla_gen pads every row to PAYLOAD_SLOTS (14) and every "N"
    consumer reads slots explicitly rather than by length — but it must be stated, because
    encode_payload sizes `row` from it."""
    assert ACTION_WIDTH["N"] == 5
    assert ACTION_WIDTH["N"] <= PAYLOAD_SLOTS


# ---------------------------------------------------------------------------------------------
# arm.bowl_place's PARAMS MUST RIDE THE PAYLOAD, the same way nav.open_articulation's back_off_m
# does. Without this the executor calls resolve() with no params at all and the skill silently
# falls back to its own defaults -- which is exactly what happened: three place variants were
# A/B'd across ~280 episodes and scored identically, because none of them ever reached the arm.
# ---------------------------------------------------------------------------------------------

def test_bowl_place_params_ride_the_payload_slots():
    import goal_format
    step = goal_format.Step(
        skill="arm.bowl_place", action="A_r", goal=None,
        params={"forward_m": 0.23, "down_m": 0.40, "min_eef_z": 0.8266, "roll_deg": 0.0},
        language="Set the mug down on the table")

    row = encode_payload(step, "arm.bowl_place")

    assert row[:4] == [0.23, 0.40, 0.8266, 0.0]

    left_step = goal_format.Step(
        skill="arm.bowl_place", action="A_l", goal=None,
        params={"forward_m": 0.12, "down_m": 0.20, "min_eef_z": 0.88, "roll_deg": 5.0, "lateral_m": -.25},
        language="Set the mug down with the left arm")
    assert encode_payload(left_step, "arm.bowl_place")[:4] == [0.12, 0.20, 0.88, 5.0]
    assert encode_payload(left_step, "arm.bowl_place")[4] == -.25


def test_bowl_place_with_no_params_still_encodes_the_skills_own_defaults():
    """An old goal file records no params. The row must carry the constants the skill was born
    with, not zeros -- a zero min_eef_z is a floor at the world origin, and a zero forward_m
    would place the object on top of itself."""
    import goal_format
    step = goal_format.Step(skill="arm.bowl_place", action="A_r", goal=None,
                            params={}, language="Move the object over the table")

    row = encode_payload(step, "arm.bowl_place")

    assert row[0] == 0.23
    assert row[1] == 0.05
    assert row[3] == -20.0
    assert row[4] == 0.0
    assert row[2] < -1e30, "an unset floor must be -inf, not 0"
