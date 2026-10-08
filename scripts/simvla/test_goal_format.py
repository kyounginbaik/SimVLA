"""Tests for the goal file format, and an audit of the v1 files on disk.

The v1 reader's only job is to agree with simvla_gen.py. So the centrepiece here is
`test_reader_agrees_with_the_executor_on_every_payload_the_generator_emits`, a differential test
against an oracle that runs the executor's *actual* arithmetic — real torch.isclose on a real
float32 tensor — transcribed from the dispatch blocks by line number.

Fixtures are SLOT-DISTINGUISHABLE on purpose. A payload of [0.3] * 7 cannot tell you whether the
reader matched slots [0:4], [3:7] or [4:8]; the earlier version of this file used exactly that and
nine mutations survived. Every arm fixture below carries a real position in [:3] and the sentinel
only in [3:7], so a wrong-slot decode fails.

Run with: pytest scripts/simvla/test_goal_format.py -v
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

import goal_format as gf
from goal_format import (
    Step,
    UnrecognisedStep,
    audit_v1,
    decode_v1,
    read,
    read_v1_legacy,
    write_v2,
)

# v1's magic floats. They used to be skill_runtime.SkillSentinel / SkillFlag, imported here; those
# classes are gone (nothing dispatches on a float any more) and goal_format's legacy reader is now
# their only home, so this file reads them from there — the module under test — and
# test_the_v1_sentinel_values_are_frozen pins them to literals so "the module under test" cannot
# quietly become "whatever the module says today".
class SkillSentinel:                                                       # noqa: N801  (was a class)
    PAUSE = gf.PAUSE
    BOWL_PLACE = gf.BOWL_PLACE
    MOVE_LEFT = gf.MOVE_LEFT
    MOVE_RIGHT = gf.MOVE_RIGHT
    BOTTLE_TILT = gf.BOTTLE_TILT
    GRASP_TARGET = gf.GRASP_TARGET
    MOVE_FRONT = gf.MOVE_FRONT
    PULL = gf.PULL
    PUSH = gf.PUSH


class SkillFlag:                                                           # noqa: N801  (was a class)
    THRESHOLD = gf.FLAG_THRESHOLD
    RESET = gf.FLAG_VALUE
    PLACE = gf.FLAG_VALUE


# A real right-arm pose, so no two slots are interchangeable.
POS = [0.6785, -0.2503, 0.9660]
QUAT = [0.6322, -0.5870, 0.3344, -0.3793]

# Exactly what simvla_data_generator.py used to emit.
RESET_PAYLOAD = [SkillFlag.RESET, 0.0, 1.102, 0.6322, -0.587, 0.3344, -0.3793]
PLACE_PAYLOAD = [*POS, SkillFlag.PLACE, 0.0, 0.0, 0.0]

# skill -> (sentinel value, the ONE action whose block matches it). From simvla_gen.py.
ARM_SENTINEL_OWNER = {
    "arm.bowl_place": (SkillSentinel.BOWL_PLACE, "A_r"),
    "arm.move_left": (SkillSentinel.MOVE_LEFT, "A_r"),
    "arm.bottle_tilt": (SkillSentinel.BOTTLE_TILT, "A_r"),
    "arm.pause": (SkillSentinel.PAUSE, "A_r"),
    "arm.move_right": (SkillSentinel.MOVE_RIGHT, "A_l"),
    "arm.move_front": (SkillSentinel.MOVE_FRONT, "A_l"),
    "arm.grasp_target": (SkillSentinel.GRASP_TARGET, "A_l"),
}
NAV_SENTINEL_OWNER = {
    "nav.open_articulation": (SkillSentinel.PULL, "N"),
    "nav.close_articulation": (SkillSentinel.PUSH, "N"),
}


def v1_file(tmp_path, steps, **meta):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({**meta, "goals": [steps]}))
    return path


# =================================================================================================
# The oracle: simvla_gen.py's dispatch, run with real torch on a real float32 payload tensor.
# =================================================================================================

def executor_dispatch(action, payload):
    """What simvla_gen.py ACTUALLY does. Returns (label, goal_slots_left_intact).

    `goal_slots_left_intact` is the set of payload indices the executor does NOT overwrite — i.e.
    the part of the goal that was authored and must survive a migration. Transcribed from:
      N    1863-1865   A_r  2095-2099, 2151-2202   A_l  2311-2314, 2356-2378
      A_b  2488-2596   N_s  1932-1979              G_b  1538 (TASK_IDS)
    """
    if action == "G_b":
        return "KeyError: 'G_b'", None
    if action in ("G_r", "G_l"):
        return "gripper.set", "all"
    if not isinstance(payload, list) or not payload:
        return "malformed", None

    if isinstance(payload[0], list):
        # The executor picks one row at random and drops it into the payload.
        verdicts = {executor_dispatch(action, row)[0] for row in payload}
        if verdicts == {"LITERAL"}:
            return "LITERAL", "all"
        if len(payload) == 1:
            return executor_dispatch(action, payload[0])
        return "NONDETERMINISTIC", None

    # payload = torch.zeros(14); payload[:n] = data   (simvla_gen.py:1563-1570)
    t = torch.zeros(14, dtype=torch.float32)
    t[: len(payload)] = torch.tensor(payload, dtype=torch.float32)

    def isclose_all(x, target):
        return bool(torch.isclose(x, torch.full_like(x, target)).all())

    if action == "N":
        for name, (val, _) in NAV_SENTINEL_OWNER.items():
            if isclose_all(t[:3], val):
                return name, "none"
        # Only three skills author an "N" goal: pull, push, and the refrigerator
        # (simvla_data_generator.py:1479, 1485, 1599). A fall-through here IS the bug.
        return "FALL-THROUGH", None

    if action == "N_s":
        if t[2] <= -1000:
            return "nav.to_range", "none"
        if t[2] >= 1000:
            return "nav.to_pot", "none"
        return "LITERAL", "all"

    if action in ("A_r", "A_l"):
        for name, (val, owner) in ARM_SENTINEL_OWNER.items():
            if owner == action and isclose_all(t[3:7], val):
                return name, "none"
        if t[0] >= SkillFlag.THRESHOLD:
            return "arm.reset", "none"                   # 2181/2371: [:7] all replaced
        if t[3] >= SkillFlag.THRESHOLD:
            return "arm.place", "0:3"                    # 2188/2376: ONLY [3:7] replaced
        if _fell_through(t[3:7]):
            return "FALL-THROUGH", None
        return "LITERAL", "all"

    if action == "A_b":
        reset_l = bool(t[0] >= SkillFlag.THRESHOLD)
        if not reset_l and t[2] >= 1000:
            return "arm.pot", "none"                     # 2561: both halves fully overwritten
        modes = []
        for lo in (0, 7):
            if t[lo] >= SkillFlag.THRESHOLD:
                modes.append("reset")
            elif t[lo + 3] >= SkillFlag.THRESHOLD:
                modes.append("place")                    # 2592/2606: ONLY that half's [3:7]
            elif _fell_through(t[lo + 3:lo + 7]):
                return "FALL-THROUGH", None
            else:
                modes.append("pose")
        if modes == ["reset", "reset"]:
            return "arm.reset", "none"
        return f"A_b[{modes[0]}+{modes[1]}]", "halves"

    return "unknown action", None


def _fell_through(quat_slots):
    """The executor is about to IK to these quaternion slots. Is what it found a real rotation?

    Two ways it is not, and both are the refrigerator bug's family — the executor does not crash,
    it just drives somewhere meaningless:
      - a SENTINEL, in the sentinel slots of a channel whose block does not match it. The isclose
        never runs, so 3.1 becomes a literal quaternion of (3.1, 3.1, 3.1, 3.1).
      - ZEROS, which is what zero-padding a short payload leaves behind. Not a rotation.
    """
    for value in gf.ALL_SENTINELS:
        x = torch.full_like(quat_slots, value)
        if bool(torch.isclose(quat_slots, x).all()):
            return True
    return bool(torch.all(quat_slots == 0.0))


def generator_corpus():
    """Every (action, payload) simvla_data_generator.py can emit, plus the known-broken ones."""
    cases = []
    for action in ("A_r", "A_l", "A_b"):
        # Every sentinel skill is @register_skill'd for A_r, A_l AND A_b (lines 1490-1531).
        for skill, (val, _owner) in ARM_SENTINEL_OWNER.items():
            cases.append((f"{skill} @{action}", action, [val] * 7))
        cases.append((f"arm.reset @{action}", action, list(RESET_PAYLOAD)))
        cases.append((f"arm.place @{action}", action, list(PLACE_PAYLOAD)))
    # skill_arm_grasp emits 7-float rows for A_r/A_l and 14-float rows for A_b (line 1257).
    for action in ("A_r", "A_l"):
        cases.append((f"arm.grasp @{action}", action, [[*POS, *QUAT], [0.5, 0.1, 1.0, *QUAT]]))
        cases.append((f"arm.grasp 1 candidate @{action}", action, [[*POS, *QUAT]]))
        cases.append((f"arm.reset wrapped in 1 row @{action}", action, [list(RESET_PAYLOAD)]))
    cases.append(("arm.grasp @A_b", "A_b", [[*POS, *QUAT, 0.5, 0.1, 1.0, *QUAT]]))
    cases += [
        ("nav.open_articulation @N", "N", [SkillSentinel.PULL] * 3),
        ("nav.close_articulation @N", "N", [SkillSentinel.PUSH] * 3),
        ("refrigerator (the bug) @N", "N", [-0.4, -0.4, -0.4]),
        ("nav.to_prim @N_s", "N_s", [0.77, -1.65, 0.0]),
        ("nav.to_range @N_s", "N_s", [0.0, 0.0, -1000.0]),
        ("nav.to_pot @N_s", "N_s", [0.0, 0.0, 1000.0]),
        ("gripper @G_r", "G_r", True),
        ("gripper @G_l", "G_l", False),
        ("gripper @G_b (KeyErrors)", "G_b", True),
        # A_b's true 14-float layout: [:7] left, [7:14] right.
        ("A_b both pose", "A_b", [*POS, *QUAT, 0.5, 0.1, 1.0, *QUAT]),
        ("A_b both reset", "A_b", [*RESET_PAYLOAD, *RESET_PAYLOAD]),
        ("A_b both place", "A_b", [*PLACE_PAYLOAD, *PLACE_PAYLOAD]),
        ("A_b left reset, right pose", "A_b", [*RESET_PAYLOAD, *POS, *QUAT]),
        ("A_b left pose, right place", "A_b", [*POS, *QUAT, *PLACE_PAYLOAD]),
        ("A_b pot mode", "A_b", [0.0, 0.0, 1000.0, *QUAT, 0.5, 0.1, 1.0, *QUAT]),
        # Wrong-channel and short payloads that really are on disk.
        ("bowl_place as 3 floats @A_r", "A_r", [SkillSentinel.BOWL_PLACE] * 3),
    ]
    return cases


@pytest.mark.parametrize("label,action,payload", generator_corpus(),
                         ids=[c[0] for c in generator_corpus()])
def test_reader_agrees_with_the_executor_on_every_payload_the_generator_emits(
        label, action, payload):
    """The differential test. The reader is a mirror of the executor or it is nothing.

    Two things must agree: WHICH skill (or 'this is not dispatched at all'), and WHICH SLOTS of the
    payload survive into the goal. Getting the skill right while dropping the goal is exactly the
    arm.place bug — it decoded to the right name and destroyed the position.
    """
    truth, intact = executor_dispatch(action, payload)

    # `intact is None` means the executor's behaviour is not a goal anybody authored. Those, and
    # ONLY those, may the reader reject — rejecting a real goal would be just as destructive.
    rejectable = intact is None

    try:
        step = decode_v1(action, payload)
    except UnrecognisedStep:
        assert rejectable, (
            f"reader rejected {label}, but the executor dispatches it as {truth} and the goal "
            f"({intact}) is real — rejecting it destroys authored data"
        )
        return

    assert not rejectable, (
        f"reader accepted {label} as {step.skill!r}, but the executor's verdict is {truth!r} — "
        f"that is not a goal anyone authored, and the reader must say so"
    )

    if truth == "LITERAL":
        # The executor uses the payload verbatim. So must the reader — and it must not claim a
        # runtime skill. (Only ordinary authored poses may land here.)
        assert step.skill in ("arm.pose", "arm.grasp", "nav.to_prim", "gripper.set"), (
            f"reader called {label} {step.skill!r}, but the executor treats it as a literal goal"
        )
        assert step.goal == payload
        return

    if truth.startswith("A_b["):
        modes = truth[len("A_b["):-1].split("+")
        assert step.skill in ("arm.both", "arm.grasp")
        if step.skill == "arm.both":
            assert [step.params["left"], step.params["right"]] == modes
        return

    assert step.skill == truth, f"{label}: reader says {step.skill!r}, executor says {truth!r}"

    if intact == "none":
        assert step.goal is None, (
            f"{label}: the executor overwrites the whole payload, so there is nothing to author"
        )
    elif intact == "0:3":
        assert step.goal == list(payload[:3]), (
            f"{label}: the executor keeps payload[:3] as the IK target and overwrites only the "
            f"quaternion — the reader must preserve it, not drop it"
        )


# =================================================================================================
# Slot correctness. Every sentinel, on every channel, in the right slots and the wrong ones.
# =================================================================================================

@pytest.mark.parametrize("skill,val,owner",
                         [(s, v, o) for s, (v, o) in ARM_SENTINEL_OWNER.items()])
def test_every_arm_sentinel_is_matched_in_the_quaternion_slots_only(skill, val, owner):
    """The sentinel sits in [3:7]. A real position sits in [:3]. Decoding [0:4] or [4:8] fails."""
    step = decode_v1(owner, [*POS, val, val, val, val])

    assert step.skill == skill
    assert step.goal is None


@pytest.mark.parametrize("skill,val,owner",
                         [(s, v, o) for s, (v, o) in ARM_SENTINEL_OWNER.items()])
def test_an_arm_sentinel_in_the_position_slots_is_not_dispatched(skill, val, owner):
    """[val]*3 is not a sentinel — the executor reads slots [3:7] as padding zeros and drives to
    the literal position. 109 `A_r` steps on disk are exactly this ([-0.12] * 3)."""
    with pytest.raises(UnrecognisedStep):
        decode_v1(owner, [val, val, val])


@pytest.mark.parametrize("skill,val,owner",
                         [(s, v, o) for s, (v, o) in ARM_SENTINEL_OWNER.items()])
def test_an_arm_sentinel_on_the_wrong_channel_is_not_dispatched(skill, val, owner):
    """C3. Each sentinel is matched in exactly ONE action block. MOVE_RIGHT on `A_r` is not
    'move right' — the A_r block never runs that isclose, so 3.1 becomes a literal quaternion."""
    other = "A_l" if owner == "A_r" else "A_r"

    with pytest.raises(UnrecognisedStep, match="does not match that sentinel"):
        decode_v1(other, [*POS, val, val, val, val])


@pytest.mark.parametrize("val", [SkillSentinel.PULL, SkillSentinel.PUSH])
def test_a_nav_sentinel_on_an_arm_channel_is_not_dispatched(val):
    with pytest.raises(UnrecognisedStep):
        decode_v1("A_r", [*POS, val, val, val, val])


@pytest.mark.parametrize("skill,val",
                         [(s, v) for s, (v, _) in NAV_SENTINEL_OWNER.items()])
def test_nav_sentinels_are_matched_in_the_position_slots(skill, val):
    step = decode_v1("N", [val, val, val])

    assert step.skill == skill
    assert step.goal is None


@pytest.mark.parametrize("skill,val",
                         [(s, v) for s, (v, _) in ARM_SENTINEL_OWNER.items()])
def test_an_arm_sentinel_on_the_nav_channel_is_not_dispatched(skill, val):
    """The N block matches PULL and PUSH and nothing else."""
    with pytest.raises(UnrecognisedStep, match="refrigerator"):
        decode_v1("N", [val, val, val])


def test_no_sentinel_is_dispatched_on_A_b():
    """C3. The A_b block runs no isclose at all — a sentinel skill wired to `A_b` is broken."""
    for _skill, (val, _owner) in ARM_SENTINEL_OWNER.items():
        with pytest.raises(UnrecognisedStep):
            decode_v1("A_b", [val] * 7)


# =================================================================================================
# The flags. Reset drops its goal; place KEEPS its position. (C1)
# =================================================================================================

@pytest.mark.parametrize("action", ["A_r", "A_l"])
def test_reset_is_fully_runtime_resolved(action):
    """simvla_gen.py:2181/2371 overwrites arm_goals[:7] — every slot. Nothing survives."""
    step = decode_v1(action, list(RESET_PAYLOAD))

    assert step.skill == "arm.reset"
    assert step.goal is None
    assert step.is_runtime_resolved()


@pytest.mark.parametrize("action", ["A_r", "A_l"])
def test_place_keeps_its_authored_position(action):
    """C1. simvla_gen.py:2188/2376 overwrites ONLY [3:7] (the quat, from the live eef pose). The
    position in [:3] is skill_arm_place's bbox-derived target and it IS the IK goal. A reader that
    returns goal=None here silently destroys it — 4,178 steps' worth, corpus-wide."""
    step = decode_v1(action, list(PLACE_PAYLOAD))

    assert step.skill == "arm.place"
    assert step.goal == POS
    assert not step.is_runtime_resolved()


def test_the_reset_flag_is_read_at_slot_0_and_the_place_flag_at_slot_3():
    """Both flags in one payload would be ambiguous; keep them apart and prove the slots."""
    assert decode_v1("A_r", [999.0, *POS[1:], *QUAT]).skill == "arm.reset"
    assert decode_v1("A_r", [*POS, 999.0, 0.0, 0.0, 0.0]).skill == "arm.place"

    # A 999 anywhere else is not a flag: slots 1, 2, 4, 5, 6 are never checked.
    assert decode_v1("A_r", [POS[0], 999.0, POS[2], *QUAT]).skill == "arm.pose"
    assert decode_v1("A_r", [*POS, QUAT[0], 999.0, QUAT[2], QUAT[3]]).skill == "arm.pose"


def test_a_sentinel_beats_a_flag_in_the_same_payload():
    """The executor runs its isclose block (2096) before it reads the flags (2151), and the
    resolver has already overwritten the payload by then."""
    step = decode_v1("A_r", [999.0, 0.0, 0.0, *[SkillSentinel.PAUSE] * 4])

    assert step.skill == "arm.pause"


# =================================================================================================
# A_b's 14-float layout. (C2)
# =================================================================================================

def test_A_b_models_both_halves():
    """C2. [:7] is the LEFT arm, [7:14] the RIGHT. Flags at absolute slots 0, 3 and 7, 10. A reader
    that checks only slots 0 and 3 sees the left half's flag and throws away the right arm's goal."""
    right = [0.5, 0.1, 1.0, *QUAT]
    step = decode_v1("A_b", [*RESET_PAYLOAD, *right])

    assert step.params == {"left": "reset", "right": "pose"}
    assert step.goal["left"] is None                 # reset: fully overwritten
    assert step.goal["right"] == right               # NOT destroyed by the left half's flag


def test_A_b_place_keeps_both_positions():
    step = decode_v1("A_b", [*PLACE_PAYLOAD, *PLACE_PAYLOAD])

    assert step.params == {"left": "place", "right": "place"}
    assert step.goal == {"left": POS, "right": POS}


def test_A_b_right_half_flags_are_read_at_slots_7_and_10():
    left = [*POS, *QUAT]

    assert decode_v1("A_b", [*left, *RESET_PAYLOAD]).params["right"] == "reset"
    assert decode_v1("A_b", [*left, *PLACE_PAYLOAD]).params["right"] == "place"
    # slot 8, 9, 11, 12, 13 are never checked.
    assert decode_v1("A_b", [*left, 0.5, 999.0, 1.0, *QUAT]).params["right"] == "pose"


def test_A_b_both_reset_is_the_reset_skill():
    step = decode_v1("A_b", [*RESET_PAYLOAD, *RESET_PAYLOAD])

    assert step.skill == "arm.reset"
    assert step.goal is None


def test_A_b_pot_mode_is_runtime_resolved():
    """simvla_gen.py:2561 — left slot 2 >= 1000 overwrites the position and quaternion of BOTH."""
    step = decode_v1("A_b", [0.0, 0.0, 1000.0, *QUAT, 0.5, 0.1, 1.0, *QUAT])

    assert step.skill == "arm.pot"
    assert step.goal is None


def test_A_b_rejects_a_short_payload():
    """A 7-float A_b payload gives the right arm a zero position and a zero quaternion."""
    with pytest.raises(UnrecognisedStep, match="14"):
        decode_v1("A_b", list(RESET_PAYLOAD))


# =================================================================================================
# N_s.
# =================================================================================================

def test_N_s_is_a_literal_goal():
    step = decode_v1("N_s", [0.77, -1.65, 0.0])

    assert step.skill == "nav.to_prim"
    assert step.goal == [0.77, -1.65, 0.0]


def test_N_s_range_and_pot_markers_are_runtime_resolved():
    """simvla_gen.py:1949/1967 — a yaw of +/-1000 is a marker, not a yaw. Both overwrite the
    position AND the yaw from live scene state. Migrating them as literals would encode 1000 rad."""
    assert decode_v1("N_s", [0.0, 0.0, -1000.0]).skill == "nav.to_range"
    assert decode_v1("N_s", [0.0, 0.0, -1000.0]).goal is None
    assert decode_v1("N_s", [0.0, 0.0, 1000.0]).skill == "nav.to_pot"
    assert decode_v1("N_s", [0.0, 0.0, 1000.0]).goal is None


# =================================================================================================
# Grippers.
# =================================================================================================

@pytest.mark.parametrize("action", ["G_r", "G_l"])
@pytest.mark.parametrize("value", [True, False])
def test_gripper_carries_its_value(action, value):
    step = decode_v1(action, value)

    assert step.skill == "gripper.set"
    assert step.goal is value


def test_G_b_cannot_run():
    """simvla_gen.py:1538 TASK_IDS has no 'G_b'; the file KeyErrors while it is being loaded."""
    with pytest.raises(UnrecognisedStep, match="G_b"):
        decode_v1("G_b", True)


# =================================================================================================
# Tolerance. (I1)
# =================================================================================================

def test_the_tolerance_is_torch_isclose_not_an_absolute_epsilon():
    """torch.isclose is atol + rtol*|target| = 1e-8 + 1e-5*|target|, so MOVE_FRONT (3.5) gets
    3.5e-5 of slack — 35x what an absolute 1e-6 would give. A reader using 1e-6 rejects a payload
    the executor accepts, and calls a runtime skill a literal coordinate."""
    near = SkillSentinel.MOVE_FRONT + 2e-5           # inside isclose, outside an absolute 1e-6

    step = decode_v1("A_l", [*POS, near, near, near, near])
    assert step.skill == "arm.move_front" and step.goal is None

    far = SkillSentinel.MOVE_FRONT + 1e-3            # outside both -> an ordinary literal pose
    assert decode_v1("A_l", [*POS, far, far, far, far]).skill == "arm.pose"


@pytest.mark.parametrize("value", sorted(gf.ALL_SENTINELS))
def test_isclose_matches_torch_exactly_across_the_whole_boundary(value):
    """Sweep the tolerance band, densely, either side of the edge, and demand the same answer torch
    gives. This is what pins the float32 rounding: the payload lives in a float32 tensor, so the
    executor compares f32(payload) to f32(sentinel). Doing the same arithmetic in float64 flips the
    verdict for values within ~2e-7 of the edge — e.g. torch says 0.29999700204 is NOT the PAUSE
    sentinel, and a float64 reader says it is."""
    band = gf.ISCLOSE_ATOL + gf.ISCLOSE_RTOL * abs(value)

    for k in range(-3000, 3001):
        delta = band * (1 + k * 1e-6)
        for probe in (value + delta, value - delta):
            x = torch.tensor([probe] * 4, dtype=torch.float32)
            expected = bool(torch.isclose(x, torch.full_like(x, value)).all())

            assert gf._isclose_all([probe] * 4, value) is expected, (value, probe)


def test_padding_is_zeros_exactly_as_the_executor_pads():
    """simvla_gen.py:1563 — `payload = torch.zeros(14); payload[:n] = data`. The padding value is
    load-bearing: it is WHY a 3-float `A_r` step reads as a zero quaternion rather than as the
    bowl_place sentinel it was meant to be."""
    expected = torch.zeros(14, dtype=torch.float32)
    expected[:3] = torch.tensor([-0.12, -0.12, -0.12], dtype=torch.float32)

    assert gf._pad([-0.12, -0.12, -0.12]) == expected.tolist()
    assert gf._pad([*POS, *QUAT])[7:] == [0.0] * 7


def test_all_slots_must_match_not_just_one():
    """torch.isclose(...).all(dim=1) — one slot off and the sentinel does not fire; the payload is
    then an ordinary quaternion and the arm IKs to it."""
    val = SkillSentinel.PAUSE
    step = decode_v1("A_r", [*POS, val, val, val, 0.9])

    assert step.skill == "arm.pose"
    assert step.goal == [*POS, val, val, val, 0.9]


# =================================================================================================
# is_runtime is one predicate, not two. (I2)
# =================================================================================================

def test_a_near_miss_cannot_be_both_a_runtime_skill_and_a_literal_goal():
    """I2. The old reader decided the skill id with isclose and `is_runtime` with `in` — so a value
    within tolerance but not exactly equal got a runtime skill name AND kept its payload as a
    literal goal. There is now one predicate: `goal is None`."""
    near = SkillSentinel.PULL + 1e-7                 # isclose says yes; `x in dict` says no

    step = decode_v1("N", [near, near, near])

    assert step.skill == "nav.open_articulation"
    assert step.goal is None and step.is_runtime_resolved()


# =================================================================================================
# The audit. (C4)
# =================================================================================================

def test_audit_flags_the_refrigerator(tmp_path):
    path = v1_file(tmp_path, [
        ["N", [-0.25, -0.25, -0.25]],                # PULL — recognised
        ["N", [-0.4, -0.4, -0.4]],                   # recognised by nothing
    ])

    problems = audit_v1(path)

    assert [p["payload"] for p in problems] == [[-0.4, -0.4, -0.4]]
    assert problems[0]["action"] == "N" and problems[0]["index"] == 1


def test_audit_flags_a_sentinel_on_the_wrong_channel(tmp_path):
    """C4. The false negative that mattered: 3.1 IS a sentinel, so the old audit looked it up in
    one global table, found `arm.move_right`, and reported the step clean. But the A_r block never
    matches MOVE_RIGHT, so the executor drives the arm to a quaternion of (3.1, 3.1, 3.1, 3.1)."""
    path = v1_file(tmp_path, [["A_r", [*POS, *[SkillSentinel.MOVE_RIGHT] * 4]]])

    assert len(audit_v1(path)) == 1


def test_audit_flags_a_three_float_arm_sentinel(tmp_path):
    """The other real one: [-0.12] * 3 on A_r. It is BOWL_PLACE's value, in the wrong slots."""
    path = v1_file(tmp_path, [["A_r", [-0.12, -0.12, -0.12]]])

    assert len(audit_v1(path)) == 1


def test_audit_flags_G_b(tmp_path):
    """C4. A G_b step does not merely misbehave — it KeyErrors as the file is loaded."""
    path = v1_file(tmp_path, [["G_b", True]])

    assert len(audit_v1(path)) == 1


def test_audit_is_clean_on_a_well_formed_file(tmp_path):
    path = v1_file(tmp_path, [
        ["N", [-0.25, -0.25, -0.25]],
        ["N_s", [0.77, -1.65, 0.0]],
        ["A_r", [*POS, *[SkillSentinel.PAUSE] * 4]],
        ["A_r", list(PLACE_PAYLOAD)],
        ["A_b", [*RESET_PAYLOAD, *RESET_PAYLOAD]],
        ["G_r", True],
    ])

    assert audit_v1(path) == []


def test_legacy_gripper_boolean_keeps_its_close_open_meaning():
    assert decode_v1("G_r", True).params == {"grasp": True}
    assert decode_v1("G_l", False).params == {"grasp": False}


# =================================================================================================
# Reading and writing.
# =================================================================================================

def test_v2_round_trips(tmp_path):
    steps = [
        Step(skill="nav.to_prim", action="N_s", params={"prim_path": "/world/mug0"},
             goal=[0.77, -1.65, 0.0]),
        Step(skill="arm.pause", action="A_r", params={}, goal=None),
    ]
    path = tmp_path / "task.json"
    write_v2(path, steps, meta={"kitchen_num": 1, "kitchen_type": "l_shaped"})

    loaded, meta = read(path)

    assert loaded == steps
    assert meta["kitchen_num"] == 1
    assert json.loads(path.read_text())["version"] == 2


def test_a_runtime_resolved_step_says_so_instead_of_hiding_a_float(tmp_path):
    """The whole point. v1 encoded 'resolve me at runtime' as 0.3 repeated across the quaternion
    slots. A real quaternion could not equal that, but nothing said so, and a value that matched
    nothing (the refrigerator's -0.4) silently became a coordinate."""
    path = tmp_path / "task.json"
    write_v2(path, [Step(skill="arm.pause", action="A_r", params={}, goal=None)], meta={})

    raw = json.loads(path.read_text())
    step = raw["goals"][0][0]

    assert step["goal"] is None
    assert step["skill"] == "arm.pause"
    assert "0.3" not in json.dumps(raw), "no magic float should appear in a v2 file"


def test_v2_writer_rejects_non_finite_numbers(tmp_path):
    path = tmp_path / "task.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        write_v2(path, [Step(skill="arm.pause", action="A_r", params={"bad": float("-inf")},
                             goal=None)], meta={})


def test_read_dispatches_on_the_version_key(tmp_path):
    path = v1_file(tmp_path, [["N_s", [0.1, 0.2, 0.3]], ["G_r", True]], kitchen_type="l_shaped")

    steps, meta = read(path)

    assert [s.action for s in steps] == ["N_s", "G_r"]
    assert meta["kitchen_type"] == "l_shaped"


def test_read_v1_names_the_step_and_the_file_when_it_cannot_decode(tmp_path):
    path = v1_file(tmp_path, [["N", [-0.25] * 3], ["N", [-0.4] * 3]])

    with pytest.raises(UnrecognisedStep, match="step 1"):
        read_v1_legacy(path)


def test_an_unknown_skill_id_is_a_hard_error_at_load(tmp_path):
    """I3. Opt-in: pass the registry's skill ids and a goal file naming anything else fails loudly,
    naming the skill and the file. Without it the module stays registry-free (and stdlib-pure)."""
    path = tmp_path / "task.json"
    write_v2(path, [Step(skill="arm.levitate", action="A_r", goal=None)], meta={})

    with pytest.raises(UnrecognisedStep, match="arm.levitate"):
        read(path, known_skills={"arm.pause", "arm.reset"})

    assert read(path)[0][0].skill == "arm.levitate"          # no registry -> no check


def test_known_skills_also_guards_a_v1_file(tmp_path):
    path = v1_file(tmp_path, [["A_r", list(PLACE_PAYLOAD)]])

    with pytest.raises(UnrecognisedStep, match="arm.place"):
        read(path, known_skills={"arm.pause"})

    assert read(path, known_skills={"arm.place"})[0][0].goal == POS


# =================================================================================================
# The constants are frozen: they are what 7,723 files on disk MEAN.
# =================================================================================================

def test_the_v1_sentinel_values_are_frozen():
    """These nine floats + the 900 threshold are the v1 goal format's whole vocabulary.

    They used to be duplicated — skill_runtime.SkillSentinel had one copy, goal_format the other,
    and this test compared them. skill_runtime's copy is gone: nothing dispatches on a float any
    more, so a *live* sentinel constant is exactly the thing this refactor removed. goal_format's
    legacy reader is the last owner, and what it owns is not a tunable — it is the decoder ring for
    7,723 files that nobody is going to rewrite. So the values are pinned to literals here rather
    than to another module: changing one silently re-interprets the corpus, and this test is the
    only thing standing in front of that.
    """
    assert (gf.PAUSE, gf.BOWL_PLACE, gf.MOVE_LEFT, gf.MOVE_RIGHT, gf.BOTTLE_TILT,
            gf.GRASP_TARGET, gf.MOVE_FRONT, gf.PULL, gf.PUSH) == (
        0.3, -0.12, 3.2, 3.1, 3.3, 3.4, 3.5, -0.25, 0.25
    )
    assert gf.FLAG_THRESHOLD == 900.0
    assert gf.FLAG_VALUE == 999.0
    assert gf.POT_RANGE_MARKER == 1000.0
    assert set(gf.ALL_SENTINELS) == {0.3, -0.12, 3.2, 3.1, 3.3, 3.4, 3.5, -0.25, 0.25}


def test_the_sentinels_have_no_home_outside_the_legacy_reader():
    """SkillSentinel / SkillFlag are deleted, and skill_runtime must not grow them back.

    The point of the refactor is that "which float is this?" is no longer a question anyone asks at
    run time. If the classes reappear in skill_runtime — the module BOTH processes import — then a
    dispatch site can reach one again, and it will, because it is easier than plumbing a skill id.
    """
    import skill_runtime

    for gone in ("SkillSentinel", "SkillFlag", "RUNTIME_RESOLVERS", "register_runtime_resolver"):
        assert not hasattr(skill_runtime, gone), (
            f"skill_runtime.{gone} is back. v1's floats live in goal_format's legacy reader, where "
            f"they decode old files; anywhere else they are a dispatch mechanism again."
        )


def test_every_sentinel_is_dispatched_by_exactly_one_channel():
    owners = {}
    for action, sentinels in gf.CHANNEL_SENTINELS.items():
        for value, _skill in sentinels:
            owners.setdefault(value, []).append(action)

    assert all(len(a) == 1 for a in owners.values()), owners
    assert set(owners) == set(gf.ALL_SENTINELS), "a sentinel no channel dispatches is dead"


# =================================================================================================
# Purity.
# =================================================================================================

def test_module_is_stdlib_only():
    """goal_format.py must be importable with no GPU and no Omniverse, same constraint as
    skill_contract.py — simvla_data_generator.py and simvla_gen.py both boot Omniverse at import
    (~60s). Parse the AST rather than just importing, so a lazily-imported (function-local)
    third-party dependency cannot slip past a naive check. skill_contract is the one first-party
    module goal_format.py is allowed to depend on; everything else must be stdlib."""
    stdlib_names = set(sys.stdlib_module_names)

    module_path = Path(__file__).parent / "goal_format.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])

    allowed = stdlib_names | {"skill_contract"}
    non_stdlib = imported_roots - allowed
    assert not non_stdlib, f"goal_format.py imports non-stdlib module(s): {non_stdlib}"

    # Belt and suspenders: actually importing it must not pull in any Omniverse/torch/etc module.
    # Diff sys.modules across the import rather than asserting on the absolute set — an absolute
    # check would blame this module for torch, which a sibling test in the same process
    # (test_kitchen_*.py, test_timeout_logic.py) already imported. That fails only under
    # `pytest scripts/simvla/`, i.e. exactly how CI runs it.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("goal_format_purity_check", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    banned_prefixes = ("omni", "isaaclab", "pxr", "torch", "scene_synthesizer", "numpy")
    pulled_in = set(sys.modules) - before
    banned = sorted(m for m in pulled_in if m.split(".")[0] in banned_prefixes)
    assert not banned, f"importing goal_format.py pulled in {banned}"
