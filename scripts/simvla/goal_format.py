"""SimVLA: reading and writing goal files.

v2 says which skill produced each step and whether its goal is resolved at run time. v1 encoded
that as a magic float repeated across the payload slots, recovered with torch.isclose — which is
how the refrigerator skill shipped a value (-0.4) that matched nothing, fell through, and became a
literal coordinate.

The v1 reader is a quarantined legacy path. It exists so the existing files keep executing; it is
not a migration, and nothing writes v1.

The reader is a MIRROR OF THE EXECUTOR, not of the generator. simvla_gen.py is the only thing that
decides what a v1 payload means, and it dispatches PER CHANNEL: each sentinel is matched in exactly
one action block, in exactly one slot range. A sentinel emitted on the wrong channel is not
"the skill it names" — the executor never looks for it there, so it silently becomes a literal
coordinate. That is the refrigerator bug's whole shape, and the corpus has three instances of it.

The dispatch, transcribed from simvla_gen.py (line numbers as of 42e0a173):

    N     1863-1865  slots [:3]   isclose vs PULL, PUSH                          -- these 2 only
    A_r   2095-2099  slots [3:7]  isclose vs BOWL_PLACE, MOVE_LEFT,
                                  BOTTLE_TILT, PAUSE                             -- these 4 only
          2151/2154  slot 0 >= 900 -> reset (whole 7-vector overwritten)
                     slot 3 >= 900 -> place (ONLY [3:7] overwritten; [:3] is the
                                      authored, bbox-derived IK target -- keep it)
    A_l   2311-2314  slots [3:7]  isclose vs MOVE_RIGHT, MOVE_FRONT,
                                  GRASP_TARGET                                   -- these 3 only
          2356/2359  same reset/place flags as A_r
    A_b   2488-2596  NO isclose at all. 14 floats: [:7] = LEFT, [7:14] = RIGHT.
                     reset/place flags per half -> absolute slots 0, 3 (left)
                     and 7, 10 (right). Plus a pot mode on left slot 2 >= 1000.
    N_s   1932-1979  literal goal, EXCEPT slot 2 (yaw) <= -1000 -> go-to-range and
                     >= 1000 -> go-to-pot, both fully runtime-resolved.
    G_r/G_l          payload[0] = +1/-1; literal.
    G_b              TASK_IDS (simvla_gen.py:1538) has no "G_b" -> KeyError at load.
                     A G_b step does not run. It is not a valid v1 step.

Payloads are zero-padded to 14 float32 slots before dispatch (simvla_gen.py:1560-1570), so a
short payload does NOT skip a slot check -- it feeds zeros into it. An `A_r` step carrying the
3-float `[-0.12, -0.12, -0.12]` is not a bowl_place: the executor reads its quaternion slots as
padding zeros, matches nothing, and drives the arm to the literal position (-0.12, -0.12, -0.12)
with a zero quaternion. 109 steps in the corpus do exactly this.

Stdlib only.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------
# v1's magic floats. THIS IS THE ONLY PLACE THEY EXIST.
#
# They used to be skill_runtime.SkillSentinel / SkillFlag, and they used to be a live dispatch
# mechanism: the goal file said "pause" by putting 0.3 in four quaternion slots and simvla_gen
# recovered the verb with torch.isclose. That is gone — a step names its skill, and the executor
# dispatches on an integer skill id.
#
# But the 7,723 v1 files on disk did not change when the code did, and what those files MEAN is
# defined by these floats. So the values survive here, in the quarantined legacy reader, where they
# are archaeology rather than dispatch: read once, at the edge, turned into a skill name, and never
# seen again. Nothing outside this module imports them, and no executor branch compares against one
# (test_skill_dispatch.py::test_sentinel_dispatch_is_extinct enforces both, on the AST).
#
# The reader's job is to agree with what the OLD executor did, exactly, bug for bug. So do not
# "fix" anything below — if it looks wrong, it is faithfully reproducing something that was wrong,
# which is the only way to know what an old file was asking for.
# ---------------------------------------------------------------------------------------------

PAUSE = 0.3
BOWL_PLACE = -0.12
MOVE_LEFT = 3.2
MOVE_RIGHT = 3.1
BOTTLE_TILT = 3.3
GRASP_TARGET = 3.4
MOVE_FRONT = 3.5
PULL = -0.25
PUSH = 0.25

#: The category flag (was SkillFlag): v1 wrote FLAG_VALUE into one slot of an otherwise-meaningful
#: goal vector — position[0] for a reset, quaternion[0] for a place — and the executor recovered it
#: with a range check, not an isclose. So the READER only ever needs the threshold; the value is
#: what the old writer emitted, kept so the format is described in one place.
FLAG_THRESHOLD = 900.0
FLAG_VALUE = 999.0

#: N_s / A_b range+pot markers: an out-of-range value in the yaw (N_s) or left-z (A_b) slot.
POT_RANGE_MARKER = 1000.0

#: torch.isclose defaults. |a - target| <= atol + rtol * |target|, elementwise, then .all().
#: NOT an absolute 1e-6 — the tolerance scales with the target, and 3.5 gets 3.5e-5 of slack.
ISCLOSE_ATOL = 1e-8
ISCLOSE_RTOL = 1e-5

#: simvla_gen.py:1563 — `payload = torch.zeros(max_payload_size)`, max_payload_size = 14.
PAYLOAD_SLOTS = 14

#: Which sentinels each action block actually matches, and in which slots. This IS the dispatch.
#: A sentinel absent from an action's list is invisible to that action's block — it falls through.
ARM_QUAT_SLOTS = slice(3, 7)
NAV_POS_SLOTS = slice(0, 3)

CHANNEL_SENTINELS: dict[str, list[tuple[float, str]]] = {
    "N": [(PULL, "nav.open_articulation"), (PUSH, "nav.close_articulation")],
    "A_r": [(BOWL_PLACE, "arm.bowl_place"), (MOVE_LEFT, "arm.move_left"),
            (BOTTLE_TILT, "arm.bottle_tilt"), (PAUSE, "arm.pause")],
    "A_l": [(MOVE_RIGHT, "arm.move_right"), (MOVE_FRONT, "arm.move_front"),
            (GRASP_TARGET, "arm.grasp_target")],
    "A_b": [],  # the A_b block does no isclose whatsoever
}

ALL_SENTINELS = (PAUSE, BOWL_PLACE, MOVE_LEFT, MOVE_RIGHT, BOTTLE_TILT,
                 GRASP_TARGET, MOVE_FRONT, PULL, PUSH)


@dataclass
class Step:
    skill: str
    action: str
    params: dict = field(default_factory=dict)
    goal: Any | None = None          # None => resolved at run time; nothing to author

    #: The author's sentence for this step ("Move Right arm to place in drawer"). It lived in the
    #: <task>.reloadable.json twin, which v2 replaces — so it has to live on the step, or collapsing
    #: the two files would silently throw it away. stream_finalize_lerobot.py reads
    #: data["goals"][0][0]["language"] to label the dataset, and the GUI shows it when reloading a
    #: goal. A v1 file (the *sim* half of the pair) never carried it: there it is "".
    language: str = ""

    def is_runtime_resolved(self) -> bool:
        """The single predicate. `goal is None` and 'the executor overwrites it' are one thing."""
        return self.goal is None


class UnrecognisedStep(ValueError):
    """A v1 payload the executor's dispatch would not recognise.

    Not a parse error: v1 does not fail on these, it silently reinterprets them as literal
    coordinates. That is the bug. Surfacing it is the point.
    """


# ---------------------------------------------------------------------------------------------
# The executor's arithmetic, in stdlib.
# ---------------------------------------------------------------------------------------------

def _f32(x: float) -> float:
    """Round to float32. The executor's payloads live in a float32 tensor; the JSON is float64."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _isclose(value: float, target: float) -> bool:
    """torch.isclose(value, target) with torch's defaults, elementwise."""
    return abs(_f32(value) - _f32(target)) <= ISCLOSE_ATOL + ISCLOSE_RTOL * abs(_f32(target))


def _isclose_all(slots: list[float], target: float) -> bool:
    """torch.isclose(x, full_like(x, target)).all(dim=1) — every slot in the range must match."""
    return bool(slots) and all(_isclose(v, target) for v in slots)


def _pad(payload: list[float]) -> list[float]:
    """simvla_gen.py: `payload = zeros(14); payload[:n] = data`. Short payloads read as zeros."""
    vals = [_f32(float(v)) for v in payload[:PAYLOAD_SLOTS]]
    return vals + [0.0] * (PAYLOAD_SLOTS - len(vals))


def _matched_sentinel(slots: list[float]) -> float | None:
    """Any sentinel these slots are all-close to, regardless of which channel owns it."""
    for value in ALL_SENTINELS:
        if _isclose_all(slots, value):
            return value
    return None


# ---------------------------------------------------------------------------------------------
# The dispatch.
# ---------------------------------------------------------------------------------------------

def _decode_arm(action: str, payload: list[float]) -> Step:
    """A_r / A_l. Sentinels in [3:7] (this channel's only), then the reset/place flags."""
    if len(payload) < 7:
        # Padded, so [3:7] is zeros: the executor reads a zero quaternion and drives to the
        # literal position. Never what was meant. The 109 [-0.12]*3 `A_r` steps land here.
        raise UnrecognisedStep(
            f"a {action!r} step carries only {len(payload)} floats; the executor zero-pads it and "
            f"reads slots [3:7] as a quaternion of (0,0,0,0), which is not a rotation. If this was "
            f"meant as a sentinel it is in the wrong slots — arm sentinels are matched in [3:7], "
            f"not [:3]."
        )

    t = _pad(payload)
    quat = t[ARM_QUAT_SLOTS]

    for value, skill in CHANNEL_SENTINELS[action]:
        if _isclose_all(quat, value):
            return Step(skill=skill, action=action, goal=None)

    # A sentinel this channel does not dispatch. The executor never looks for it here; it becomes
    # a literal quaternion of e.g. (3.1, 3.1, 3.1, 3.1). Refrigerator bug, different channel.
    stray = _matched_sentinel(quat)
    if stray is not None:
        owners = [a for a, sl in CHANNEL_SENTINELS.items() if any(v == stray for v, _ in sl)]
        raise UnrecognisedStep(
            f"a {action!r} step carries sentinel {stray} in its quaternion slots [3:7], but the "
            f"{action!r} block does not match that sentinel — only "
            f"{[s for _, s in CHANNEL_SENTINELS[action]]}. It is dispatched on "
            f"{owners or ['no channel']}. Here the executor reads it as a literal quaternion."
        )

    if t[0] >= FLAG_THRESHOLD:
        # simvla_gen.py:2181 / 2371 — `arm_goals[reset, :7] = reset_pose`. All 7 slots replaced.
        return Step(skill="arm.reset", action=action, goal=None)

    if t[3] >= FLAG_THRESHOLD:
        # simvla_gen.py:2188 / 2376 — ONLY [3:7] is replaced (with the live eef quat). [:3] is the
        # authored place position from skill_arm_place, derived from the target's bbox, and it is
        # the IK target. arm.place is a plan() skill with a runtime-filled quaternion, NOT a
        # resolve() skill. Dropping [:3] here would destroy the goal.
        return Step(skill="arm.place", action=action, goal=list(payload[:3]))

    return Step(skill="arm.pose", action=action, goal=list(payload))


def _decode_arm_both(payload: list[float]) -> Step:
    """A_b. No sentinels. 14 floats: [:7] left, [7:14] right; flags at 0, 3 and 7, 10."""
    if len(payload) < PAYLOAD_SLOTS:
        raise UnrecognisedStep(
            f"an 'A_b' step carries only {len(payload)} floats, not {PAYLOAD_SLOTS}. The executor "
            f"reads [:7] as the LEFT arm goal and [7:14] as the RIGHT arm goal; a short payload "
            f"gives the right arm a zero position and a zero quaternion. (A sentinel-emitting "
            f"skill wired to 'A_b' lands here: the A_b block runs no isclose at all, so a "
            f"7-float sentinel payload is silently a literal goal.)"
        )

    t = _pad(payload)

    for half, lo in (("left", 0), ("right", 7)):
        stray = _matched_sentinel(t[lo + 3:lo + 7])
        if stray is not None:
            raise UnrecognisedStep(
                f"an 'A_b' step carries sentinel {stray} in the {half} arm's quaternion slots "
                f"[{lo + 3}:{lo + 7}]. The A_b block runs no isclose at all — the executor reads "
                f"it as a literal quaternion."
            )

    reset_l = t[0] >= FLAG_THRESHOLD
    reset_r = t[7] >= FLAG_THRESHOLD

    # simvla_gen.py:2561 — `pot_mask_b = (arm_l_goals_b[:, 2] >= 1000)`, read AFTER the reset
    # overwrite, so a reset left half can never be in pot mode. Both halves' position AND
    # quaternion are then overwritten from pot0's live pose.
    if not reset_l and t[2] >= POT_RANGE_MARKER:
        return Step(skill="arm.pot", action="A_b", goal=None)

    def _half(lo: int) -> Any:
        if t[lo] >= FLAG_THRESHOLD:                    # reset: whole 7-vector overwritten
            return "reset", None
        if t[lo + 3] >= FLAG_THRESHOLD:                # place: only the quaternion is overwritten
            return "place", list(payload[lo:lo + 3])
        return "pose", list(payload[lo:lo + 7])

    mode_l, goal_l = _half(0)
    mode_r, goal_r = _half(7)

    if mode_l == "reset" and mode_r == "reset":
        return Step(skill="arm.reset", action="A_b", params={"left": mode_l, "right": mode_r},
                    goal=None)

    return Step(skill="arm.both", action="A_b", params={"left": mode_l, "right": mode_r},
                goal={"left": goal_l, "right": goal_r})


def _decode_nav(payload: list[float]) -> Step:
    """N. Sentinels in [:3]. Nothing else authors an N goal — a fall-through is the bug."""
    t = _pad(payload)
    pos = t[NAV_POS_SLOTS]

    for value, skill in CHANNEL_SENTINELS["N"]:
        if _isclose_all(pos, value):
            return Step(skill=skill, action="N", goal=None)

    raise UnrecognisedStep(
        f"an 'N' step carries payload {list(payload)!r}. The 'N' block matches only PULL "
        f"({PULL}) and PUSH ({PUSH}) in slots [:3]; everything else falls through and is driven "
        f"to as a literal (x, y, yaw). This is the refrigerator bug: skill_pull_articulation_pull "
        f"(simvla_data_generator.py:1599) returns [-0.4, -0.4, -0.4], which matches nothing."
    )


def _decode_nav_prim(payload: list[float]) -> Step:
    """N_s. Literal, unless the yaw slot carries the range / pot marker."""
    if len(payload) < 3:
        raise UnrecognisedStep(
            f"an 'N_s' step carries only {len(payload)} floats; the executor reads slots "
            f"[:2] as (x, y) and slot 2 as yaw."
        )

    t = _pad(payload)

    # simvla_gen.py:1949 / 1967 — both overwrite goal_pos_e AND goal_yaw from live scene state.
    if t[2] <= -POT_RANGE_MARKER:
        return Step(skill="nav.to_range", action="N_s", goal=None)
    if t[2] >= POT_RANGE_MARKER:
        return Step(skill="nav.to_pot", action="N_s", goal=None)

    return Step(skill="nav.to_prim", action="N_s", goal=list(payload))


def _is_plain_pose(step: Step) -> bool:
    """Did this decode to an ordinary authored arm pose — no flag, no sentinel, no runtime fill?

    Only such rows may be offered as interchangeable grasp candidates.
    """
    if step.skill == "arm.pose":
        return True
    if step.skill == "arm.both":
        return step.params.get("left") == "pose" and step.params.get("right") == "pose"
    return False


def decode_v1(action: str, payload: Any) -> Step:
    """What would simvla_gen.py do with this (action, payload)? Raises UnrecognisedStep if the
    honest answer is 'silently reinterpret it as a literal coordinate' or 'crash at load'."""
    if action in ("G_r", "G_l"):
        return Step(skill="gripper.set", action=action,
                    params={"grasp": bool(payload)}, goal=payload)

    if action == "G_b":
        # simvla_gen.py:1538 TASK_IDS = {"N", "A_l", "A_r", "G_l", "G_r", "N_s", "A_b"}. No "G_b".
        raise UnrecognisedStep(
            "a 'G_b' step cannot run: simvla_gen.py's TASK_IDS (line 1538) has no 'G_b', so "
            "TASK_IDS[task] raises KeyError while the goal file is being loaded (line 1456/1559). "
            "simvla_data_generator.py registers skill_gripper_toggle for 'G_b' anyway."
        )

    if not isinstance(payload, list) or not payload:
        raise UnrecognisedStep(f"a {action!r} step carries payload {payload!r}, which is not a "
                               f"non-empty list of floats.")

    # A list of candidate poses (skill_arm_grasp). The executor picks ONE row at random and drops
    # it into the payload — so every row goes through the same dispatch as a flat payload would.
    if isinstance(payload[0], list):
        rows = payload
        if len(rows) == 1:
            # Deterministic: this row IS the payload. Decode it as one. Eight `A_r` steps in the
            # corpus are a reset pose wrapped in a 1-row list — the executor resets, and so do we.
            step = decode_v1(action, rows[0])
            if not _is_plain_pose(step):
                return step                              # a wrapped reset / place / sentinel
        else:
            for i, row in enumerate(rows):
                step = decode_v1(action, row)
                if not _is_plain_pose(step):
                    raise UnrecognisedStep(
                        f"a {action!r} step offers {len(rows)} candidate poses, but row {i} "
                        f"decodes as {step.skill!r}. The executor picks a row at random, so the "
                        f"step's meaning would depend on the draw."
                    )
        return Step(skill="arm.grasp", action=action, goal=list(rows))

    if action == "N":
        return _decode_nav(payload)
    if action == "N_s":
        return _decode_nav_prim(payload)
    if action == "A_b":
        return _decode_arm_both(payload)
    if action in ("A_r", "A_l"):
        return _decode_arm(action, payload)

    raise UnrecognisedStep(
        f"{action!r} is not a v1 action. simvla_gen.py's TASK_IDS knows only "
        f"N, A_l, A_r, G_l, G_r, N_s, A_b."
    )


# ---------------------------------------------------------------------------------------------
# File I/O.
# ---------------------------------------------------------------------------------------------

def expand_route(step: Step) -> list[Step]:
    """A step whose goal carries via-points (nav_clearance.NavRoute) becomes that many nav steps
    ahead of the park step; any other step comes back alone. Both the emitter (task_emit.plan_steps)
    and the GUI's save write goal files through this, so a detour is the same chain in either."""
    via = list(getattr(step.goal, "via", ()) or ())
    if not hasattr(step.goal, "via"):
        return [step]
    out = [Step(skill=step.skill, action=step.action, params=step.params, goal=[float(v) for v in p],
                language=f"{step.language} (via-point {k + 1} of {len(via)})")
           for k, p in enumerate(via)]
    out.append(Step(skill=step.skill, action=step.action, params=step.params,
                    goal=[float(v) for v in step.goal], language=step.language))
    return out


def write_v2(path, steps: list[Step], meta: dict) -> None:
    payload = dict(meta)
    payload["version"] = 2
    payload["goals"] = [[
        {"skill": s.skill, "action": s.action, "params": s.params, "goal": s.goal,
         "language": s.language}
        for s in steps
    ]]
    Path(path).write_text(json.dumps(payload, indent=4, allow_nan=False))


def read(path, known_skills: set[str] | None = None) -> tuple[list[Step], dict]:
    """Read a goal file. Dispatches on the version key; no version means v1.

    `known_skills`, when given, is the set of skill ids the caller can execute — a goal file naming
    anything else is a hard error at load, rather than a KeyError somewhere deep in the run. It is
    opt-in so this module stays importable with no registry (and so the stdlib-purity test can
    exercise it standalone); pass `set(REGISTRY)` from the skill contract.
    """
    data = json.loads(Path(path).read_text())
    if data.get("version") == 2:
        steps = [
            Step(skill=s["skill"], action=s["action"], params=s.get("params", {}), goal=s["goal"],
                 language=s.get("language", ""))
            for s in data["goals"][0]
        ]
        meta = {k: v for k, v in data.items() if k not in ("goals", "version")}
    else:
        steps, meta = read_v1_legacy(path)

    if known_skills is not None:
        for i, step in enumerate(steps):
            if step.skill not in known_skills:
                raise UnrecognisedStep(
                    f"{path}: step {i} names skill {step.skill!r}, which is not in the skill "
                    f"registry. Known skills: {sorted(known_skills)}."
                )
    return steps, meta


def read_v1_legacy(path) -> tuple[list[Step], dict]:
    """Decode the old [action, payload] pairs. Quarantined — nothing writes this format."""
    data = json.loads(Path(path).read_text())
    steps: list[Step] = []

    for i, (action, payload) in enumerate(data["goals"][0]):
        try:
            steps.append(decode_v1(action, payload))
        except UnrecognisedStep as exc:
            raise UnrecognisedStep(
                f"{path}: step {i}: {exc} Regenerate this goal with the current generator."
            ) from None

    meta = {k: v for k, v in data.items() if k != "goals"}
    return steps, meta


def audit_v1(path) -> list[dict]:
    """Every step the executor's dispatch would NOT recognise. Empty means clean.

    "Recognised" means the executor's own per-channel dispatch matches it. A step whose payload
    happens to equal some OTHER channel's sentinel is not recognised — that channel's block never
    runs the isclose, and the value becomes a literal coordinate. Reporting such a step as clean is
    the exact false negative that let the refrigerator ship.
    """
    data = json.loads(Path(path).read_text())
    problems = []
    for i, (action, payload) in enumerate(data["goals"][0]):
        try:
            decode_v1(action, payload)
        except UnrecognisedStep as exc:
            problems.append({"index": i, "action": action, "payload": payload,
                             "reason": str(exc)})
    return problems
