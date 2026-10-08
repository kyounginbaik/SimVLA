"""SimVLA: refuse to emit a task sequence that cannot work.

Stdlib only (plus skill_contract and task_template).

A symbolic execution of the script: it tracks, per arm, whether the gripper is holding something,
and whether each articulation has been opened. The point is to move failure from "a GPU run that
dies with a KeyError forty minutes in, or spins silently until its time limit" to "the composer
will not let you save this, and it says why."
"""

from __future__ import annotations

from .skill_contract import REGISTRY, PrimPath
from .task_template import TaskTemplate, TemplateStep

#: The skills that bring an arm TO an object so the gripper can close on it. A gripper.set(grasp=True)
#: that follows none of these on the same arm is closing on air.
#: Bimanual heuristic squeeze skills. They ARE grasp skills for the sequence check (they put the hand
#: on the object, so a following gripper.set close is legitimate), but the run config treats them
#: apart: a squeeze binds BOTH arms to ONE object on purpose and pins it, so it is exempt from the
#: two-arm manipulation refusal that a pair of ordinary grasps would trip. See derive_run_config.
SQUEEZE_SKILLS = frozenset({"arm.squeeze"})

#: arm.bar_handle_grasp is here for the same reason the other two handle planners are: it brings the
#: hand onto a handle, so a gripper.set(grasp) after it is closing on something. It is a THIRD
#: handle planner because neither existing one fits a horizontal bar whose prim translation is not
#: on the bar -- see its docstring in skills.py.
GRASP_SKILLS = frozenset({
    "arm.grasp", "arm.handle_grasp", "arm.fridge_handle_grasp", "arm.bar_handle_grasp",
}) | SQUEEZE_SKILLS

#: The skills that put down what the gripper is holding.
RELEASE_SKILLS = frozenset({"arm.place", "arm.bowl_place"})


class SequenceError(Exception):
    """The composed sequence is incoherent. Raised before anything is written."""


def arms_of(action: str) -> tuple[str, ...]:
    """Which arms an action drives. A_b drives both.

    Public, and imported by task_runconfig, because the copy that lived there was blind to `_b`:
    every arm skill declares actions=("A_r", "A_l", "A_b"), so `arm.reset(A_b)` — reset both arms,
    the natural way to author it — read as "drives no arm at all", and the derivation silently
    dropped the right arm's grasp check. One reading of an action code, in one place.
    """
    if action.endswith("_r"):
        return ("right",)
    if action.endswith("_l"):
        return ("left",)
    if action.endswith("_b"):
        return ("left", "right")
    return ()


def gripper_closes(step: TemplateStep) -> bool:
    """Does this gripper.set step CLOSE the gripper?

    `grasp` is optional and its default is the CONTRACT's, not this module's: skills.py declares
    Bool("grasp", default=True), and the goal writer fills an omitted param from that same
    declaration. So a step authored as `{}` is a close. Read as `bool(params.get("grasp"))` it came
    out a release — falsy by omission — and a release is what clears "this arm is holding
    something" on both sides of the codebase: here a later arm.place is reported as placing nothing,
    and in task_runconfig the grasp-check scan walks past the reset that would have verified the
    grasp. The default is read from the registry so the two cannot drift from the declaration.
    """
    if "grasp" in step.params:
        return bool(step.params["grasp"])
    spec = REGISTRY.get("gripper.set")
    if spec is None:
        raise SequenceError(
            "gripper.set is not in the skill registry: import skills before reading a sequence"
        )
    for p in spec.params:
        if p.name == "grasp":
            return bool(p.default)
    raise SequenceError("gripper.set declares no 'grasp' param, so a step that omits it has no meaning")


def validate_sequence(t: TaskTemplate) -> None:
    """Symbolically execute the script.

    TWO states per arm, not one. `at_object` is set by a grasp skill (the arm has moved to the
    thing); `holding` is set only when the gripper actually CLOSES. Collapsing them into a single
    flag would make `arm.grasp` alone look like a completed pickup, and a script that reaches for
    the bowl, never closes the gripper, and then places it would validate.
    """
    problems: list[str] = []
    at_object = {"left": False, "right": False}
    holding = {"left": False, "right": False}
    open_articulations = 0

    for i, step in enumerate(t.steps):
        spec = REGISTRY.get(step.skill)
        if spec is None:
            continue                       # task_template.validate_template already reports this

        for p in spec.params:
            if isinstance(p, PrimPath):
                value = step.params.get(p.name, "")
                # A PrimPath must be a non-empty string. `if not value` alone is falsy-based: it
                # catches "" but not a truthy non-string like 0.25, which would then reach the
                # executor as a prim_path.
                if not isinstance(value, str) or not value:
                    problems.append(
                        f"step {i} ({step.skill}): {p.name} is unbound — it points at nothing"
                    )

        arms = arms_of(step.action)

        if step.skill in GRASP_SKILLS:
            for arm in arms:
                at_object[arm] = True      # the arm is now AT the object, ready to close

        elif step.skill == "gripper.set":
            closing = gripper_closes(step)
            for arm in arms:
                if closing:
                    if not at_object[arm]:
                        problems.append(
                            f"step {i}: the {arm} gripper closes on nothing — no grasp skill on "
                            f"the {arm} arm precedes it"
                        )
                    else:
                        holding[arm] = True
                else:
                    holding[arm] = False
                    at_object[arm] = False

        elif step.skill in RELEASE_SKILLS:
            for arm in arms:
                if not holding[arm]:
                    problems.append(
                        f"step {i} ({step.skill}): the {arm} arm is not holding anything to place"
                    )

        elif step.skill == "nav.open_articulation":
            open_articulations += 1

        elif step.skill == "nav.close_articulation":
            if open_articulations == 0:
                problems.append(
                    f"step {i}: closes an articulation that was never opened"
                )
            else:
                open_articulations -= 1

    if problems:
        raise SequenceError(
            f"{len(problems)} problem(s) in the sequence for {t.name!r}:\n  "
            + "\n  ".join(problems)
        )
