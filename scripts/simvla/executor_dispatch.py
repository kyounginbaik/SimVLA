"""SimVLA: the executor's script — which skill produced each step, and its payload.

simvla_gen.py used to answer "what should I do with this step?" by sniffing the payload floats:
torch.isclose against magic sentinels in 9 places, `>= 900.0` against flags in 8 more, and a
`>= 1000` marker in 3 more. That question can be wrong, and it was — the refrigerator is the
clearest example.

It now asks which skill produced the step. This module is where the answer is computed, and it is
deliberately pure: stdlib + skill_contract + goal_format, no torch, no Omniverse. main() is 2,300
lines and boots Isaac at import, so anything left in there cannot be tested at all.

`load_script(path)` returns, per step:

    action    the effector channel ("N", "A_r", ...)      -> TASK_IDS  -> task_ids_tensor
    skill_id  a small int naming the skill                              -> skill_ids_tensor
    spec      the floats (or bool, or list of candidate rows)           -> payloads_tensor

THE INTS ARE IN-MEMORY, PER-RUN. They index the dispatch and nothing else. Goal files persist the
skill NAME; the int is derived at load, because sorted-order ids shift the moment a skill is added
and a persisted int would silently reinterpret every file ever written.

v1 files still load, through goal_format's quarantined legacy reader, and are re-encoded into the
same payload shape a v2 file produces. That re-encoding is lossless for the executor: every slot
v1 used as a *marker* (the sentinels, the 999 flags, the ±1000 pot/range markers) is a slot no
branch reads any more, and every slot v1 used as a *value* is carried through. See
`encode_payload`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import goal_format
from skill_contract import REGISTRY, SKILL_ID

#: The skill id of a padding row. Real ids are assigned 0..n-1, so -1 cannot collide with one, and
#: `skill_ids_tensor == SKILL_ID[anything]` is False on every padded step. It mirrors
#: simvla_gen's PADDING_TASK_ID, which is -1 for the same reason: a padded row must be a step the
#: executor cannot mistake for work.
PADDING_SKILL_ID = -1

#: goal_format's legacy reader names three skills after the v1 sentinel they were encoded with.
#: They ARE registry skills — same class, same geometry, older name. (skills.py registers
#: SkillSentinel.MOVE_LEFT -> arm.bottle_to_position, MOVE_RIGHT -> arm.mug_to_position,
#: BOTTLE_TILT -> arm.bottle_pour; this table is the same fact, on the reading side.)
LEGACY_SKILL_ALIASES = {
    "arm.move_left": "arm.bottle_to_position",
    "arm.move_right": "arm.mug_to_position",
    "arm.bottle_tilt": "arm.bottle_pour",
}

#: Unregistered executor branches retained for v1 compatibility. Explicit v2
#: diagnostic goals may also use arm.pose; it is validated as a raw pose, not
#: exposed as a registry/composer skill. These branches still need dispatch ids.
#:
#:   arm.pose        an ordinary authored arm pose. Not a skill: it is the fall-through branch that
#:                   every plan()-skill's goal also lands in (arm.grasp, arm.handle_grasp, ...).
#:   arm.pot         A_b, both arms to the pot rim. 2 steps on disk.
#:   nav.to_pot      N_s with the +1000 yaw marker. 4 steps.
#:   nav.to_range    N_s with the -1000 yaw marker. 1 step.
#:   arm.move_front  A_l, SkillSentinel.MOVE_FRONT. 1 step. skills.py has the resolver; no skill
#:   arm.grasp_target  A_l, SkillSentinel.GRASP_TARGET. 1 step.   emits either sentinel.
LEGACY_ONLY_SKILLS = (
    "arm.grasp_target",
    "arm.move_front",
    "arm.pose",
    "arm.pot",
    "nav.to_pot",
    "nav.to_range",
)

#: How many float slots the executor reads for each action channel. A_b is 14 because it is two
#: 7-vectors: [:7] left, [7:14] right (simvla_gen.py:2515).
#: "N" is FIVE, not three. It carries (x, y, yaw) when driven, but nav.open_door_arc needs two
#: further runtime arguments -- its radial back-off (slot 3) and its retreat DIRECTION (slot 4) --
#: and a runtime step's params can only reach the executor through these slots: envs in one batch
#: can be on different steps, so a per-step scalar read would hand them all the first env's value.
#: Widening costs nothing: simvla_gen pads every payload row to PAYLOAD_SLOTS (14) regardless, and
#: every "N" consumer reads slots explicitly rather than by length.
ACTION_WIDTH = {"N": 5, "N_s": 3, "A_r": 7, "A_l": 7, "A_b": 14}

#: simvla_gen.py: `payload = torch.zeros(14)`.
PAYLOAD_SLOTS = 14

#: v1's A_b halves each carry their own mode. A_b drives both arms with ONE skill, so a step whose
#: halves disagree has no v2 spelling. There are none in the corpus (25 A_b steps, all agreeing).
_BOTH_HALF_TO_SKILL = {"place": "arm.place", "pose": "arm.pose", "reset": "arm.reset"}


class UnexecutableStep(ValueError):
    """A goal step the executor has no branch for. Loud at load, not silent at run time."""


@dataclass
class ScriptStep:
    action: str
    skill: str
    skill_id: int
    spec: Any                       # bool | list[float] | list[list[float]] (grasp candidates)
    language: str = ""
    params: dict = field(default_factory=dict)


def dispatch_ids() -> dict[str, int]:
    """skill name -> the int the executor compares against. Registry first, in SKILL_ID() order.

    Legacy-only names are appended after the registry's, so a registry skill's id is exactly its
    SKILL_ID() value and the two tables cannot drift. Nothing persists these.
    """
    ids = dict(SKILL_ID())
    next_id = len(ids)
    for name in sorted(LEGACY_ONLY_SKILLS):
        if name in ids:                                   # a legacy name got registered for real
            continue
        ids[name] = next_id
        next_id += 1
    return ids


def normalise_skill(step: goal_format.Step) -> str:
    """The skill this step really is, in the dispatch's vocabulary.

    Two rewrites, both lossless:
      * the three sentinel-named aliases become their registry ids;
      * `arm.both` — v1's A_b step with a mode per half — becomes the one skill that drives both
        arms, which is what A_b means. Halves that disagree are rejected rather than guessed at.
    """
    skill = LEGACY_SKILL_ALIASES.get(step.skill, step.skill)

    if skill == "arm.both":
        left, right = step.params.get("left"), step.params.get("right")
        if left != right:
            raise UnexecutableStep(
                f"an 'A_b' step whose halves disagree (left={left!r}, right={right!r}). A_b drives "
                f"both arms with ONE skill, so there is no skill id for this step. No such step "
                f"exists in the corpus; if one has appeared, split it into an A_l and an A_r step."
            )
        if left not in _BOTH_HALF_TO_SKILL:
            raise UnexecutableStep(f"an 'A_b' step with unknown half-mode {left!r}")
        skill = _BOTH_HALF_TO_SKILL[left]

    return skill


def _param_default(skill: str, name: str):
    spec = REGISTRY.get(skill)
    if spec is None:
        return None
    for param in spec.params:
        if param.name == name:
            return getattr(param, "default", None)
    return None


def _pad(values, width: int) -> list[float]:
    vals = [float(v) for v in list(values)[:width]]
    return vals + [0.0] * (width - len(vals))


def encode_payload(step: goal_format.Step, skill: str) -> Any:
    """The floats the executor puts in payloads_tensor for this step.

    A runtime-resolved step (`goal is None`) gets an all-zero row, because the executor's branch
    overwrites it in full before anything reads it. Slot 0 then carries the skill's runtime
    argument, if it has one — `back_off_m` for the articulation skills. That is NOT a sentinel
    coming back: the skill id already says what the step is, so slot 0 is a plain argument to a
    known skill and nothing sniffs it. (v1 could not express it at all: its "back-off" was the
    lookup key PULL/PUSH and the 0.25 m was hardcoded in the resolver, which is how a fridge
    author asking for 0.4 m got a drive to the env-frame corner (-0.4, -0.4).)
    """
    action = step.action

    if action in ("G_r", "G_l"):
        return bool(step.goal)

    width = ACTION_WIDTH.get(action)
    if width is None:
        raise UnexecutableStep(
            f"{action!r} is not an action the executor has a channel for "
            f"(TASK_IDS: N, A_l, A_r, G_l, G_r, N_s, A_b)."
        )

    goal = step.goal

    if goal is None:
        row = [0.0] * width
        if skill in ("nav.open_articulation", "nav.close_articulation"):
            back_off = step.params.get("back_off_m")
            if back_off is None:
                # A v1 file records no back-off; v1's resolver hardcoded 0.25 m, which is also the
                # skill's declared default. Same number, arrived at two ways — assert it.
                back_off = _param_default(skill, "back_off_m")
            if back_off is None:
                raise UnexecutableStep(f"{skill!r} has no back_off_m and no declared default")
            row[0] = float(back_off)
        elif skill == "arm.bowl_place":
            # Five params into five slots of an A_r or A_l row (both action widths are 7), read
            # back in simvla_gen before the resolver overwrites them with the answer -- the same
            # mechanism nav.open_articulation's back_off_m uses.
            #
            # WHY THIS EXISTS. Without it the executor called resolve() with NO params, so the
            # skill fell back to its own constants and every authored value was silently discarded.
            # Three place variants (blind 0.05, floor 0.811, floor 0.88) were then A/B'd over ~280
            # episodes and scored identically -- a clean-looking negative result for a knob that
            # had never been connected.
            #
            # Defaults come from the SKILL, not from zeros: a zero min_eef_z is a floor at the
            # world origin and a zero forward_m would place the object on top of itself.
            for _slot, _name in enumerate(("forward_m", "down_m", "min_eef_z", "roll_deg", "lateral_m")):
                _v = step.params.get(_name)
                if _v is None:
                    _v = _param_default(skill, _name)
                if _v is None:
                    raise UnexecutableStep(f"{skill!r} has no {_name} and no declared default")
                row[_slot] = float(_v)
        elif skill == "nav.open_door_arc":
            # FIVE params into five slots. The executor's arc branch then overwrites slots 0..2
            # with the resolved (x, y, yaw) -- safe only because simvla_gen clones every one of
            # them BEFORE resolving, the ordering back_off_m already relies on.
            #
            # WHY THE PARAMS RIDE IN TENSOR SLOTS AT ALL, rather than being read from
            # ScriptStep.params: envs in one batch can be on DIFFERENT steps of the script, so
            # their radius/sweep/side can differ. A per-step scalar read would hand every env the
            # first one's values -- the hazard _back_off's docstring records.
            radius = step.params.get("radius_m")
            if radius is None:
                radius = _param_default(skill, "radius_m")
            sweep = step.params.get("sweep_deg")
            if sweep is None:
                sweep = _param_default(skill, "sweep_deg")
            side = step.params.get("hinge_side")
            if side is None:
                side = _param_default(skill, "hinge_side")
            if radius is None or sweep is None:
                raise UnexecutableStep(
                    f"{skill!r} is missing radius_m or sweep_deg and has no declared default"
                )
            if side not in ("Right", "Left"):
                # A third value has no defined direction, and falling through to 0.0 would
                # multiply the sweep by zero: the base would spin in place, the door would never
                # move, and a whole run would record without anything saying so.
                raise UnexecutableStep(
                    f"{skill!r} has hinge_side {side!r}; it must be 'Right' or 'Left'. The sign it "
                    f"maps to is what decides which way the base arcs."
                )
            back = step.params.get("back_off_m")
            if back is None:
                back = _param_default(skill, "back_off_m") or 0.0
            row[0] = float(radius)
            row[1] = float(sweep)
            row[2] = 1.0 if side == "Right" else -1.0
            row[3] = float(back)
            diag = step.params.get("diag_deg")
            if diag is None:
                diag = _param_default(skill, "diag_deg")
            if diag is None:
                raise UnexecutableStep(
                    f"{skill!r} has no diag_deg and no declared default. The retreat DIRECTION has "
                    f"no safe fallback: 0 would resolve to a straight-back pull, which is the "
                    f"behaviour this parameter exists to replace."
                )
            row[4] = float(diag)
        return row

    if isinstance(goal, dict):                  # v1 A_b: {"left": [...], "right": [...]}
        left = goal.get("left") or []
        right = goal.get("right") or []
        return _pad(left, 7) + _pad(right, 7)

    if isinstance(goal, list) and goal and isinstance(goal[0], list):
        return [_pad(row, width) for row in goal]      # arm.grasp: interchangeable candidates

    if isinstance(goal, list):
        return _pad(goal, width)

    raise UnexecutableStep(f"step {skill!r} on {action!r} has goal {goal!r}, which is not a payload")


def build_script(steps: list[goal_format.Step]) -> tuple[list[ScriptStep], dict[str, int]]:
    """Decoded goal steps -> what the executor executes. Pure; this is the whole port, testably."""
    ids = dispatch_ids()
    script: list[ScriptStep] = []

    for i, step in enumerate(steps):
        try:
            skill = normalise_skill(step)
            if skill not in ids:
                raise UnexecutableStep(
                    f"names skill {step.skill!r}, which no @skill declares and which is not one of "
                    f"the legacy-only ids {sorted(LEGACY_ONLY_SKILLS)}. The executor has no branch "
                    f"for it."
                )
            spec = encode_payload(step, skill)
        except UnexecutableStep as exc:
            raise UnexecutableStep(f"step {i} ({step.action!r}): {exc}") from None

        script.append(
            ScriptStep(action=step.action, skill=skill, skill_id=ids[skill], spec=spec,
                       language=step.language, params=dict(step.params))
        )

    return script, ids


def load_script(path) -> tuple[list[ScriptStep], dict, dict[str, int]]:
    """Read a goal file (v2 native, v1 through the legacy reader) -> (script, metadata, ids)."""
    steps, meta = goal_format.read(path)
    try:
        script, ids = build_script(steps)
    except UnexecutableStep as exc:
        raise UnexecutableStep(f"{path}: {exc}") from None
    return script, meta, ids
