"""The sequence validator. A validator that rejects nothing is the bug to expect here, so every
rule below is asserted to FAIL on a sequence that breaks it — and the real, known-good
Isaac-Kitchen-v813-00 script is asserted to pass.

Run: pytest scripts/simvla/test_task_validate.py -v
"""

import ast
import sys
from pathlib import Path

import pytest

import skills  # noqa: F401 — populates the registry
from task_template import TemplateStep
from task_validate import SequenceError, validate_sequence
from scripts.simvla.test_task_template import bowl_to_drawer


def test_the_reference_task_passes():
    """Isaac-Kitchen-v813-00 ran to a recorded demonstration. If this fails, the validator is
    wrong — do not weaken the task to fit it."""
    validate_sequence(bowl_to_drawer())


def test_closing_a_gripper_on_nothing_is_rejected():
    """G_r closes the right gripper. If no right-arm grasp skill preceded it, it closes on air."""
    t = bowl_to_drawer()
    t.steps.insert(0, TemplateStep("gripper.set", "G_r", {"grasp": True}))
    with pytest.raises(SequenceError, match="closes on nothing"):
        validate_sequence(t)


def test_placing_what_was_never_picked_up_is_rejected():
    t = bowl_to_drawer()
    del t.steps[2]                      # remove the G_r that closes on the bowl
    t.subtask_groups = []               # indices shift; not what this test is about
    with pytest.raises(SequenceError, match="is not holding"):
        validate_sequence(t)


def test_closing_an_articulation_that_was_never_opened_is_rejected():
    t = bowl_to_drawer()
    t.steps[7] = TemplateStep("arm.pause", "A_r", {})     # was nav.open_articulation
    with pytest.raises(SequenceError, match="never opened"):
        validate_sequence(t)


def test_an_unbound_prim_path_is_rejected():
    """A PrimPath param that is empty points at nothing."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": ""})
    with pytest.raises(SequenceError, match="unbound"):
        validate_sequence(t)


def test_a_non_string_prim_path_is_rejected():
    """A PrimPath param that is a truthy non-string (e.g. a float) is not caught by a falsy-only
    check — `if not value` lets 0.25 straight through. It must still be rejected as unbound."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": 0.25})
    with pytest.raises(SequenceError, match="unbound"):
        validate_sequence(t)


def test_a_missing_prim_path_param_is_rejected():
    """arm.grasp declares PrimPath(prim_path); omitting it entirely is the same defect."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {})
    with pytest.raises(SequenceError, match="unbound"):
        validate_sequence(t)


def test_a_gripper_set_that_omits_grasp_is_a_close():
    """gripper.set declares Bool("grasp", default=True) — an omitted `grasp` is a CLOSE, and the
    goal writer fills it in from that same declaration. Read here as bool(params.get("grasp")) it
    was a RELEASE: `holding` was cleared at step 2, and the arm.bowl_place at step 8 was then
    reported as 'the right arm is not holding anything to place'. The validator would have refused a
    script the contract calls valid. The default comes from the registry now, not from a second
    copy."""
    t = bowl_to_drawer()
    t.steps[2] = TemplateStep("gripper.set", "G_r", {}, "Grasp the bowl — grasp defaults to True")
    validate_sequence(t)                 # the bowl IS held when step 8 places it


def test_an_arm_action_that_drives_both_arms_is_tracked_on_both():
    """A_b drives left and right. `arm.grasp(A_b)` puts BOTH arms at the object, so both grippers
    may close on it — and neither close is 'closing on nothing'."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_b", {"prim_path": "@target"})
    t.steps.insert(2, TemplateStep("gripper.set", "G_l", {"grasp": True}))
    t.subtask_groups = []                # indices shift; not what this test is about
    validate_sequence(t)


def test_the_two_arms_are_tracked_independently():
    """By step 2 the right arm has already grasped and closed on the bowl -- holding[right] is
    True. A validator with one global 'holding' flag would see 'something is held' and let the
    left arm place right there, before the left arm has ever grasped anything. Insert exactly
    that: prove the arms do not leak into each other."""
    t = bowl_to_drawer()
    t.steps.insert(3, TemplateStep("arm.place", "A_l", {"prim_path": "@container"}))
    t.subtask_groups = []               # indices shift; not what this test is about
    with pytest.raises(SequenceError, match="left"):
        validate_sequence(t)


def test_every_violation_is_reported_at_once():
    t = bowl_to_drawer()
    t.steps.insert(0, TemplateStep("gripper.set", "G_l", {"grasp": True}))
    t.steps[2] = TemplateStep("arm.grasp", "A_r", {"prim_path": ""})
    t.subtask_groups = []
    with pytest.raises(SequenceError) as exc:
        validate_sequence(t)
    message = str(exc.value)
    assert "closes on nothing" in message and "unbound" in message


def test_module_is_stdlib_only():
    module_path = Path(__file__).parent / "task_validate.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"skill_contract", "task_template"}
    assert not (roots - allowed), f"task_validate.py imports {roots - allowed}"
