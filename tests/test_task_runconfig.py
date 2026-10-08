"""Deriving the run config, so nobody transcribes it again.

The expected values here are not opinions. This exact configuration was run on a GPU node and
produced a 960-frame LeRobot dataset with EXIT=0:

    --obj_name bowl0  --target_idx 1  --sub_grasp_idx_r 4  --sub_grasp_idx_l 999
    --sub_good_goal_count_l 0  --sub_good_goal_count_r 7  --export_groups "4,5,6,7;8,9,10;11"

If the derivation disagrees, the derivation is wrong.

The reference task pins ONE check index (4) and ONE target_idx (1), which a constant satisfies — and
v813's target_idx is doubly treacherous, because its grasp is at STEP 1 and the bowl is also the
kitchen's OBJECT 1. That coincidence hid a wrong rule (target_idx = the bound prim's object_index)
through a whole review: --target_idx is a step index, and simvla_gen.py:1522 compares it against
`step_i` enumerating the goal script. So the suite carries templates whose right answers are
DIFFERENT numbers — a check at 5 (the reset-before-grasp ordering), a left check at 7, a target_idx
of 2 on an object whose object_index is 3, a target_idx of 5, and a target_idx belonging to the LEFT
arm's grasp. Together they are what makes `return 4`, `target_idx = 1`, `target_idx = object_index`
and `target_idx = the right arm's grasp` fail rather than pass.

v813 hides one more thing: its manipulation object is the RIGHT arm's, so a derivation that always
reads target_idx off the right arm gets v813 right and gets the mirror of it silently wrong. Which
hand carries the bowl is an authoring choice, not a property of the task.

Run: pytest scripts/simvla/test_task_runconfig.py -v
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from simvla import skills  # noqa: F401
from simvla.task_bind import BindingError, BoundPrim
from simvla.task_runconfig import GRASP_CHECK_DISABLED, NO_TARGET_IDX, derive_run_config
from simvla.task_template import Role, TaskTemplate, TemplateStep, validate_template
from simvla.task_validate import SequenceError, validate_sequence
from test_task_template import bowl_to_drawer


def _minimal_success(role: str = "target") -> dict:
    """The smallest success condition validate_template will accept: one physical predicate on a
    declared role. None of this module is about success conditions — it is about derive_run_config
    — but `success` is now a required field on every template, and validate_template is what several
    fixtures below call to prove they are legal scripts before deriving a run config from them. Every
    template that needs this declares a role named "target", so the default is enough everywhere it
    is used. derive_run_config itself never reads t.success."""
    return {"all": [{"obj_z": {"role": f"@{role}", "lo": 0.9}}]}


def v813_bindings() -> dict[str, BoundPrim]:
    """What binding Isaac-Kitchen-v813-00 produces. bowl0 is a rigid body; a drawer handle is not
    — it is part of the base_cabinet articulation, which is exactly why its grasp check must be
    disabled."""
    return {
        "target": BoundPrim(prim_path="/world/bowl0", is_rigid_body=True, object_index=1),
        "container": BoundPrim(prim_path="/world/base_cabinet/drawer_0_0", is_rigid_body=False, object_index=-1),
        "container_handle": BoundPrim(
            prim_path="/world/base_cabinet/drawer_0_0/door_handle", is_rigid_body=False, object_index=-1
        ),
    }


def test_it_derives_exactly_the_config_that_recorded_a_demonstration():
    cfg = derive_run_config(bowl_to_drawer(), v813_bindings())
    assert cfg.obj_name == "bowl0"
    assert cfg.obj_name_l == "none"
    assert cfg.target_idx == 1, "the step of the right arm's grasp, which is also step 1"
    assert cfg.sub_grasp_idx_r == 4
    assert cfg.sub_grasp_idx_l == GRASP_CHECK_DISABLED
    assert cfg.sub_good_goal_count_l == 0
    assert cfg.sub_good_goal_count_r == 7, "the right check is live, so it may be asked for successes"
    assert cfg.export_groups == "4,5,6,7;8,9,10;11"
    assert cfg.task_language == "Put bowl inside drawer."
    assert cfg.task_type == "NavManipulation"


def test_the_coupling_that_spun_a_run_for_two_hours_cannot_be_stated_wrongly():
    """--sub_grasp_idx_l 999 disables the left grasp check; --sub_good_goal_count_l 7 then asks for
    seven left-arm successes the disabled check is the only thing that could produce. The run
    reaches stage=collect_left and spins to its time limit printing nothing. Here the two come from
    the same bound prims, so they cannot disagree."""
    cfg = derive_run_config(bowl_to_drawer(), v813_bindings(), sub_good_goal_count=7)
    assert cfg.sub_grasp_idx_l == GRASP_CHECK_DISABLED
    assert cfg.sub_good_goal_count_l == 0, "a disabled check must never be asked for successes"


# ---------------------------------------------------------------------------------------------
# WHAT each arm grasps: every grasp skill, not just arm.grasp.
# ---------------------------------------------------------------------------------------------


def test_the_left_arm_of_the_reference_task_binds_to_the_handle_it_grasps():
    """v813's left arm reaches the handle with arm.handle_grasp, never with arm.grasp. Recognising
    only arm.grasp made the left target look ABSENT — 999 then came out of "no left grasp step
    found", not out of the rigid-body guard the module documents. Same numbers, wrong reason, and
    the guard below had never once executed. Prove the binding is real: the left arm's target IS
    the handle, and it is the handle's non-rigidity that switches the check off."""
    bindings = v813_bindings()
    handle = bindings["container_handle"]
    assert handle.name == "door_handle" and not handle.is_rigid_body

    cfg = derive_run_config(bowl_to_drawer(), bindings, sub_good_goal_count=7)
    # If the handle were a rigid body this would be "door_handle"; it is not, so the guard fires.
    assert cfg.obj_name_l == "none"
    assert cfg.sub_grasp_idx_l == GRASP_CHECK_DISABLED


def left_arm_carries_a_mug(grasp_skill: str) -> tuple[TaskTemplate, dict[str, BoundPrim]]:
    """v813 with the objects swapped between the arms' ROLES: the RIGHT arm works the drawer handle
    (non-rigid: no manipulation object), the LEFT arm picks up a mug with `grasp_skill`.

      0 N_s nav  1 A_r handle_grasp(@container_handle)  2 G_r close  3 A_r reset
      4 A_l <grasp_skill>(@second = mug0)  5 G_l close  6 A_l reset  -> the left check is at 7

    The right arm must have NO rigid object of its own here: two rigid manipulation objects is now a
    refusal (only one asset is randomized, and nothing in the template says which), so a fixture that
    left the bowl in the right hand would be testing the refusal, not the left arm's guard.
    """
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.handle_grasp", "A_r", {"prim_path": "@container_handle"},
                              "Right arm to the handle — a means, not an object")
    t.steps[4] = TemplateStep(grasp_skill, "A_l", {"prim_path": "@second"}, "Left arm to the mug")
    t.steps[5] = TemplateStep("gripper.set", "G_l", {"grasp": True}, "Grasp the mug")
    t.steps[6] = TemplateStep("arm.reset", "A_l", {}, "Home, mug in hand")
    bindings = v813_bindings()
    bindings["second"] = BoundPrim(prim_path="/world/mug0", is_rigid_body=True, object_index=0)
    return t, bindings


def test_a_left_arm_that_handle_grasps_a_rigid_body_keeps_its_check_enabled():
    """The guard is on the TARGET, not on the skill name. An arm.handle_grasp whose prim happens to
    be a rigid object (a lid, a detached handle) is checkable, and its check must stay on. This is
    the case that was silently dead: matching only arm.grasp, the left arm bound to nothing here and
    the check was disabled for a rigid body the run could perfectly well have verified."""
    t, bindings = left_arm_carries_a_mug("arm.handle_grasp")

    cfg = derive_run_config(t, bindings, sub_good_goal_count=7)
    assert cfg.obj_name_l == "mug0"
    assert cfg.sub_grasp_idx_l == 7, "the step after the reset that follows the close"
    assert cfg.sub_good_goal_count_l == 7


def test_a_left_grasp_on_a_rigid_body_keeps_its_check_enabled():
    """The disabling is a consequence of the handle not being a rigid body, not a hardcoded 999.
    Bind the left arm to a real object and the check must come back on.

    The template says grasp -> close -> reset, in that order. An arm cannot return home holding
    something it has not closed on yet, and a fixture that reset the left arm BEFORE its
    gripper.set would be asking the derivation to bless a physically incoherent script."""
    t, bindings = left_arm_carries_a_mug("arm.grasp")

    cfg = derive_run_config(t, bindings, sub_good_goal_count=7)
    assert cfg.obj_name_l == "mug0"
    assert cfg.sub_grasp_idx_l == 7, "the step after the reset that follows the close"
    assert cfg.sub_good_goal_count_l == 7


def test_a_non_rigid_left_target_is_disabled_even_when_a_reset_would_give_an_index():
    """The two disablements are not the same, and only this template tells them apart. v813's left
    arm never resets, so a broken rigid-body guard STILL yields 999 there — by accident. Give the
    left arm a reset after its close and the index scan would happily return 8. The check must
    still be off: env.scene.rigid_objects["door_handle"] is a KeyError, and a check that cannot run
    is worse than no check."""
    t = bowl_to_drawer()
    t.steps.insert(7, TemplateStep("arm.reset", "A_l", {}, "Left arm home, handle in hand"))

    cfg = derive_run_config(t, v813_bindings(), sub_good_goal_count=7)
    assert cfg.sub_grasp_idx_l == GRASP_CHECK_DISABLED, "a non-rigid target has no checkable grasp"
    assert cfg.obj_name_l == "none"
    assert cfg.sub_good_goal_count_l == 0


def fridge_bindings() -> dict[str, BoundPrim]:
    """A kitchen with a fridge. bowl0 is a rigid body; the fridge and its handle are not."""
    return {
        "target": BoundPrim(prim_path="/world/bowl0", is_rigid_body=True, object_index=1),
        "fridge": BoundPrim(prim_path="/world/fridge", is_rigid_body=False, object_index=-1),
        "fridge_handle": BoundPrim(
            prim_path="/world/fridge/door_handle", is_rigid_body=False, object_index=-1
        ),
    }


def fridge_then_bowl() -> TaskTemplate:
    """One arm, two grasps: open the fridge, THEN pick up the bowl inside it.

      0 N_s  nav.to_prim(@fridge)
      1 A_r  arm.fridge_handle_grasp(@fridge_handle)   — a grasp, but of a means
      2 G_r  gripper.set(close)                        — closed on the door handle
      3 N    nav.open_articulation(0.4)                — the fridge is open
      4 G_r  gripper.set(open)                         — let the handle go
      5 A_r  arm.grasp(@target)                        — the bowl: THIS is the manipulation object
      6 G_r  gripper.set(close)
      7 A_r  arm.reset                                 — home, bowl in hand
      8 N    nav.close_articulation                    — the check fires here

    arm.fridge_handle_grasp is a registered skill with a live planner, and this sequence validates.
    """
    return TaskTemplate(
        name="fridge_then_bowl",
        language="Take the bowl out of the fridge.",
        roles=[
            Role("target", object_type="bowl"),
            Role("fridge", articulation_with="fridge"),
            Role("fridge_handle", handle_of="fridge"),
        ],
        steps=[
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@fridge"}, "Move to the fridge"),
            TemplateStep("arm.fridge_handle_grasp", "A_r", {"prim_path": "@fridge_handle"}, "Grasp the fridge handle"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Close on the handle"),
            TemplateStep("nav.open_articulation", "N", {"back_off_m": 0.4}, "Open the fridge"),
            TemplateStep("gripper.set", "G_r", {"grasp": False}, "Release the handle"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Right arm to the bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the bowl"),
            TemplateStep("arm.reset", "A_r", {}, "Home, bowl in hand"),
            TemplateStep("nav.close_articulation", "N", {}, "Close the fridge"),
        ],
        subtask_groups=[[1, 2, 3], [5, 6, 7]],
        success=_minimal_success(),
    )


def test_the_manipulation_object_is_the_first_RIGID_grasp_not_the_first_grasp():
    """The arm opens the fridge with arm.fridge_handle_grasp and then picks up the bowl. Taking the
    arm's FIRST grasp step bound it to the fridge handle, which is part of an articulation; the
    rigid-body guard then blanked the config to obj_name="none", no target_idx, and no grasp check —
    for a task whose object is right there in the second grasp. `--obj_name none` is
    env.scene.rigid_objects["none"]: the bare KeyError this module exists to prevent.

    A handle is a means. The manipulation object is the first grasp the arm makes on a rigid body."""
    t = fridge_then_bowl()
    validate_template(t)
    validate_sequence(t)                    # the registry already supports this task

    cfg = derive_run_config(t, fridge_bindings())
    assert cfg.obj_name == "bowl0", "the fridge handle is a means, not the object of the task"
    assert cfg.target_idx == 5, "the bowl's grasp is step 5 — the handle's grasp at 1 is not the task"
    assert cfg.sub_grasp_idx_r == 8, "the step after the reset that follows the close ON THE BOWL"
    assert cfg.obj_name_l == "none"         # the left arm does nothing here
    assert cfg.sub_grasp_idx_l == GRASP_CHECK_DISABLED


def test_a_reset_while_holding_the_handle_is_not_the_bowls_grasp_check():
    """The same arm, resetting home while it still holds the fridge door — legal, and authored:

      0 N_s nav  1 A_r fridge_handle_grasp  2 G_r close  3 N open  4 A_r reset (holding the DOOR)
      5 G_r open  6 A_r arm.grasp(@target)  7 G_r close  8 A_r reset (holding the bowl)  9 N close

    A check-index scan that starts at the top of the script latches the close at 2 and returns 5:
    the bowl is still on a fridge shelf, the hand is holding a handle, every candidate fails the
    0.15 m distance check and the run spins to its time limit printing nothing. The scan starts at
    the grasp that bound the manipulation object, so it can only ever return 9."""
    t = fridge_then_bowl()
    t.steps = [
        t.steps[0],
        t.steps[1],
        t.steps[2],
        t.steps[3],
        TemplateStep("arm.reset", "A_r", {}, "Home, still holding the fridge door"),
        t.steps[4],
        t.steps[5],
        t.steps[6],
        t.steps[7],
        t.steps[8],
    ]
    t.subtask_groups = []
    validate_sequence(t)

    cfg = derive_run_config(t, fridge_bindings())
    assert cfg.obj_name == "bowl0"
    assert cfg.sub_grasp_idx_r == 9, "the reset at 4 holds the door, not the bowl"


# ---------------------------------------------------------------------------------------------
# WHICH arms an action drives: A_b drives both, and every arm skill declares it.
# ---------------------------------------------------------------------------------------------


def grasp_with_both_arms() -> TaskTemplate:
    """Both arms to the same object — a two-handed pickup. Every arm skill declares A_b; only
    gripper.set does not (it declares G_r and G_l, because a G_b in a v1 goal file died with a
    KeyError), so each gripper still closes on its own line.

      0 N_s nav  1 A_b arm.grasp(@target)  2 G_r close  3 G_l close  4 A_b arm.reset  5 N_s nav
    """
    return TaskTemplate(
        name="grasp_with_both_arms",
        language="Pick up the bowl with both hands.",
        roles=[Role("target", object_type="bowl")],
        steps=[
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target"}, "Move to bowl"),
            TemplateStep("arm.grasp", "A_b", {"prim_path": "@target"}, "Both arms to the bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Right gripper closes"),
            TemplateStep("gripper.set", "G_l", {"grasp": True}, "Left gripper closes"),
            TemplateStep("arm.reset", "A_b", {}, "Both arms home, bowl in hand"),
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target"}, "Carry it away"),
        ],
        subtask_groups=[[1, 2, 3, 4]],
        success=_minimal_success(),
    )


def test_a_two_armed_manipulation_grasp_is_refused_because_the_executor_moves_one_arm():
    """`arm.grasp(A_b, @target)` reads as both arms grasping the bowl, and both arms DO bind — that
    part is right, and it is why this used to derive target_idx=1 with a live check on each arm.

    The derivation is still a lie, and the lie is the executor's. An A_b payload is 14 floats,
    [:7] LEFT and [7:14] RIGHT (goal_format.py:28), and the reset event's ±5 cm object sample is
    applied at simvla_gen.py:1523 as `pose_exec[:6] += noise` — the LEFT half only. (Six lines above,
    the authoring noise patches both halves explicitly, `pose_save[0:3]` and `pose_save[7:10]`: the
    executor's omission is an oversight, not a convention.) So on this template the bowl moves at
    reset, the left arm follows it, and the RIGHT arm — the one sub_grasp_idx_r checks — goes on
    reaching for where the bowl no longer is. The miss is under 5 cm and the check's threshold is
    0.15 m, so the check PASSES: a whole dataset of near-misses, recorded in silence.

    Widening the executor's patch is a runtime change that needs its own GPU verification. This
    module's job is to refuse what it cannot express correctly, and to say why."""
    t = grasp_with_both_arms()
    validate_template(t)
    validate_sequence(t)                    # it is a legal script; it is not a derivable run config

    with pytest.raises(SequenceError, match="A_b"):
        derive_run_config(t, v813_bindings(), sub_good_goal_count=7)


def test_a_two_armed_grasp_of_something_that_is_not_the_object_is_not_refused():
    """The refusal is on the step --target_idx would NAME, not on A_b anywhere in the script. Two
    arms hauling a drawer open together grasp no rigid body, so no sample ever lands on that step and
    there is nothing to correct: the A_b grasp is fine, and the right arm's bowl is still the task."""
    t = bowl_to_drawer()
    t.steps[5] = TemplateStep("arm.handle_grasp", "A_b", {"prim_path": "@container_handle"},
                              "Both arms haul the drawer open")

    cfg = derive_run_config(t, v813_bindings(), sub_good_goal_count=7)
    assert cfg.obj_name == "bowl0"
    assert cfg.target_idx == 1, "the right arm's A_r grasp of the bowl — the A_b step grasps a handle"
    assert cfg.sub_grasp_idx_r == 4


# ---------------------------------------------------------------------------------------------
# WHEN a grasp can be checked: after the reset that FOLLOWS the close.
# ---------------------------------------------------------------------------------------------


def reset_first_then_grasp() -> TaskTemplate:
    """Legal authoring: put the arm home, THEN go get the thing. The reset that verifies the grasp
    is the second one, at step 4, so the check is at step 5. An arm that has not closed its gripper
    yet is holding nothing, and step 1 is a nav step."""
    return TaskTemplate(
        name="reset_first",
        language="Pick up the bowl.",
        roles=[Role("target", object_type="bowl")],
        steps=[
            TemplateStep("arm.reset", "A_r", {}, "Arm home before we set off"),
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target"}, "Move to bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Right arm to bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the bowl"),
            TemplateStep("arm.reset", "A_r", {}, "Home, bowl in hand"),
            TemplateStep("nav.close_articulation", "N", {}, "Close the drawer"),
        ],
        subtask_groups=[[2, 3, 4]],
    )


def test_the_check_follows_the_reset_that_comes_after_the_close_not_the_first_reset():
    """The arm's FIRST reset is step 0 — before it has even navigated to the bowl. A scan gated only
    on "a close exists somewhere in the episode" returns 1: a nav step, gripper empty. Every
    candidate then fails the grasp check and the run spins to its time limit printing nothing."""
    cfg = derive_run_config(reset_first_then_grasp(), v813_bindings())
    assert cfg.sub_grasp_idx_r == 5, "the step after the SECOND reset, the one that follows the close"


def test_a_reset_of_both_arms_still_verifies_the_right_arms_grasp():
    """`arm.grasp(A_r), gripper.set(G_r, close), arm.reset(A_b)` — reset BOTH arms at once, the
    natural way to author it. arm.reset declares A_b (every arm skill does) and the validator
    accepts it, but a reading blind to `_b` saw a step that drives no arm at all, never found the
    right arm's reset, and silently returned 999: the right grasp check off, on a template that
    validates cleanly."""
    t = reset_first_then_grasp()
    t.steps[4] = TemplateStep("arm.reset", "A_b", {}, "Both arms home, bowl in hand")

    cfg = derive_run_config(t, v813_bindings())
    assert cfg.sub_grasp_idx_r == 5, "A_b resets the right arm too"


def test_a_gripper_set_that_omits_grasp_is_a_close_because_the_contract_says_so():
    """skills.py declares Bool("grasp", default=True), and the goal writer fills an omitted param
    from that declaration — so a step authored as `{}` is a CLOSE. Read as bool(params.get("grasp"))
    it came out a RELEASE: the closed latch was cleared, the reset at 4 no longer counted as holding
    anything, and the check was disabled on a perfectly good pickup. The default now comes from the
    registry, so this module cannot drift from the contract."""
    t = reset_first_then_grasp()
    t.steps[3] = TemplateStep("gripper.set", "G_r", {}, "Grasp the bowl — grasp defaults to True")

    cfg = derive_run_config(t, v813_bindings())
    assert cfg.sub_grasp_idx_r == 5, "an omitted `grasp` is the contract's default: True, a close"


def test_an_arm_that_never_closes_its_gripper_has_no_grasp_to_check():
    t = reset_first_then_grasp()
    t.steps[3] = TemplateStep("gripper.set", "G_r", {"grasp": False}, "Never closes")
    cfg = derive_run_config(t, v813_bindings())
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED


def test_a_close_with_no_reset_after_it_has_nowhere_to_check():
    t = reset_first_then_grasp()
    t.steps = t.steps[:4]                       # reset, nav, grasp, close — and then nothing
    t.subtask_groups = []
    cfg = derive_run_config(t, v813_bindings())
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED


def test_a_reset_that_is_the_last_step_has_no_following_step_to_check_at():
    t = reset_first_then_grasp()
    t.steps = t.steps[:5]                       # ..., close, reset — the reset is now last
    t.subtask_groups = []
    cfg = derive_run_config(t, v813_bindings())
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED, "there is no step 5 to check at"


def close_place_release_reset() -> TaskTemplate:
    """Legal authoring, the release-before-reset mirror of `reset_first_then_grasp`: close on the
    object, place it, OPEN the gripper, and only THEN reset home. `N_s(0) A_r-arm.grasp(1)
    G_r-close(2) A_r-arm.place(3) G_r-open(4) A_r-arm.reset(5) N(6)`. By step 5 the object is
    already on the counter and the arm is empty-handed — there is no step at which "is the object
    still within 0.15 m of the end-effector" could ever pass."""
    return TaskTemplate(
        name="close_place_release_reset",
        language="Move the bowl to the counter.",
        roles=[Role("target", object_type="bowl")],
        steps=[
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target"}, "Move to bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Right arm to bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the bowl"),
            TemplateStep("arm.place", "A_r", {"prim_path": "@target"}, "Place the bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": False}, "Release the bowl"),
            TemplateStep("arm.reset", "A_r", {}, "Home, empty-handed"),
            TemplateStep("nav.close_articulation", "N", {}, "Move on"),
        ],
        subtask_groups=[[1, 2, 3, 4, 5]],
    )


def test_a_reset_after_release_has_no_valid_check_point_even_though_a_close_happened():
    """Without clearing `closed` on release, this template would return 6 — the step after the
    reset at 5 — even though the arm let go of the bowl at step 4. Every candidate would then fail
    the 0.15 m distance check against an object already sitting on the counter, and the run would
    spin to its time limit printing nothing: the same failure class as C1's reset-before-grasp
    ordering, on the other side of the sequence. The arm never resets WHILE holding the object, so
    there is no valid check point at all."""
    cfg = derive_run_config(close_place_release_reset(), v813_bindings())
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED


# ---------------------------------------------------------------------------------------------
# WHAT the hand is holding when it closes. Knowing THAT a gripper closed is not knowing what on.
# ---------------------------------------------------------------------------------------------


def one_armed_v813() -> TaskTemplate:
    """v813 done by ONE arm — the natural way to author it, and it validates:

      0 N_s nav.to_prim(@target)
      1 A_r arm.grasp(@target)              — the bowl
      2 G_r gripper.set(close)              — holding the bowl
      3 A_r arm.place(@container)           — bowl into the drawer
      4 G_r gripper.set(open)               — let the bowl go; it is in the drawer now
      5 A_r arm.handle_grasp(@container_handle)
      6 G_r gripper.set(close)              — holding the DRAWER HANDLE
      7 A_r arm.reset                       — home, handle in hand
      8 A_r arm.pause

    The arm closes its gripper TWICE, on two different things. A scan that latches any close after
    the bowl's grasp returns 8, and at step 8 the bowl is lying in the drawer while the hand holds a
    door: the 0.15 m rigid-object check fails on every env, every env resets, the run never leaves
    stage=collect_right, and it spins to its time limit printing nothing.
    """
    return TaskTemplate(
        name="one_armed_v813",
        language="Put bowl inside drawer, one arm.",
        roles=[
            Role("target", object_type="bowl"),
            Role("container", articulation_with="drawer"),
            Role("container_handle", handle_of="container"),
        ],
        steps=[
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target"}, "Move to bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Right arm to bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the bowl"),
            TemplateStep("arm.place", "A_r", {"prim_path": "@container"}, "Place bowl in drawer"),
            TemplateStep("gripper.set", "G_r", {"grasp": False}, "Release the bowl"),
            TemplateStep("arm.handle_grasp", "A_r", {"prim_path": "@container_handle"}, "Grasp the handle"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Close on the handle"),
            TemplateStep("arm.reset", "A_r", {}, "Home, handle in hand"),
            TemplateStep("arm.pause", "A_r", {}, "Hold"),
        ],
        subtask_groups=[[1, 2, 3], [5, 6, 7]],
        success=_minimal_success(),
    )


def test_a_second_close_on_a_drawer_handle_is_not_the_bowls_grasp_check():
    """The failure this whole section exists for. It passes validate_template AND validate_sequence,
    and it derived sub_grasp_idx_r=8 — a check on a hand holding a door, against a bowl already in
    the drawer. The arm never returns home while holding the bowl, so there is no check point."""
    t = one_armed_v813()
    validate_template(t)
    validate_sequence(t)

    cfg = derive_run_config(t, v813_bindings(), sub_good_goal_count=7)
    assert cfg.obj_name == "bowl0", "the bowl is still the manipulation object"
    assert cfg.target_idx == 1, "and its grasp is still step 1"
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED, "at step 8 the hand holds the drawer handle"
    assert cfg.sub_good_goal_count_r == 0, "and a disabled check must not be asked for successes"


def test_a_hand_redirected_to_a_handle_before_it_ever_closes_has_no_check():
    """`arm.grasp(@target)` with no close, then `arm.handle_grasp(@handle)`, close, reset. The only
    close in the script happens with the hand on the HANDLE. The bowl was reached for and never
    picked up; a scan that only asks "did this arm close after the bowl's grasp step?" says yes."""
    t = TaskTemplate(
        name="reached_then_redirected",
        language="Open the drawer.",
        roles=[
            Role("target", object_type="bowl"),
            Role("container", articulation_with="drawer"),
            Role("container_handle", handle_of="container"),
        ],
        steps=[
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Reach for the bowl"),
            TemplateStep("arm.handle_grasp", "A_r", {"prim_path": "@container_handle"}, "Go to the handle instead"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Close on the handle"),
            TemplateStep("arm.reset", "A_r", {}, "Home, handle in hand"),
            TemplateStep("arm.pause", "A_r", {}, "Hold"),
        ],
        subtask_groups=[],
        success=_minimal_success(),
    )
    validate_template(t)
    validate_sequence(t)

    cfg = derive_run_config(t, v813_bindings(), sub_good_goal_count=7)
    assert cfg.obj_name == "bowl0"
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED, "the gripper closed on the handle, not the bowl"
    assert cfg.sub_good_goal_count_r == 0


def test_a_hand_redirected_to_another_rigid_object_before_it_closes_has_no_check():
    """The same redirect, but to a RIGID body — so the rigid-body guard cannot catch it, and only
    tracking WHAT the hand is on can. obj_name is bowl0 (the first rigid grasp), but the hand closes
    on the mug. Checking bowl0 against a hand full of mug fails every env, every reset."""
    t = TaskTemplate(
        name="bowl_then_mug",
        language="Pick up the mug.",
        roles=[Role("target", object_type="bowl"), Role("second", object_type="mug")],
        steps=[
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Reach for the bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@second"}, "Take the mug instead"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Close on the mug"),
            TemplateStep("arm.reset", "A_r", {}, "Home, mug in hand"),
            TemplateStep("arm.pause", "A_r", {}, "Hold"),
        ],
        subtask_groups=[],
        success=_minimal_success(),
    )
    validate_template(t)
    validate_sequence(t)

    bindings = v813_bindings()
    bindings["second"] = BoundPrim(prim_path="/world/mug0", is_rigid_body=True, object_index=0)

    cfg = derive_run_config(t, bindings, sub_good_goal_count=7)
    assert cfg.obj_name == "bowl0"
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED, "the gripper closed on the mug, not the bowl"
    assert cfg.sub_good_goal_count_r == 0


def bowl_then_mug_both_picked_up() -> TaskTemplate:
    """Two rigid grasps by one arm, both completed: put the bowl down, then pick up the mug.

      0 N_s nav  1 A_r grasp(@target)  2 G_r close  3 A_r reset  -> the bowl's check is at 4
      4 A_r place(@target)  5 G_r open  6 A_r grasp(@second)  7 G_r close  8 A_r reset  9 A_r pause

    The manipulation object is the FIRST rigid grasp. Taking the LAST would call this task's object
    the mug, randomize step 6, and check at 9 — every number different, and every one wrong.
    """
    return TaskTemplate(
        name="bowl_then_mug_both_picked_up",
        language="Put the bowl down and pick up the mug.",
        roles=[Role("target", object_type="bowl"), Role("second", object_type="mug")],
        steps=[
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target"}, "Move to the bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Right arm to the bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the bowl"),
            TemplateStep("arm.reset", "A_r", {}, "Home, bowl in hand"),
            TemplateStep("arm.place", "A_r", {"prim_path": "@target"}, "Put the bowl down"),
            TemplateStep("gripper.set", "G_r", {"grasp": False}, "Release the bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@second"}, "Right arm to the mug"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the mug"),
            TemplateStep("arm.reset", "A_r", {}, "Home, mug in hand"),
            TemplateStep("arm.pause", "A_r", {}, "Hold"),
        ],
        subtask_groups=[],
        success=_minimal_success(),
    )


def test_the_manipulation_object_is_the_FIRST_rigid_grasp_not_the_last():
    """Every other template in this suite has at most ONE rigid grasp per arm, so "first" and "last"
    agree everywhere and the whole suite stayed green when _grasp_target was mutated to take the
    last. Here they disagree on all three derived values at once."""
    t = bowl_then_mug_both_picked_up()
    validate_template(t)
    validate_sequence(t)

    bindings = v813_bindings()
    bindings["second"] = BoundPrim(prim_path="/world/mug0", is_rigid_body=True, object_index=0)

    cfg = derive_run_config(t, bindings)
    assert cfg.obj_name == "bowl0", "the LAST rigid grasp is the mug; the task is about the bowl"
    assert cfg.target_idx == 1, "the bowl's grasp is step 1; the mug's is step 6"
    assert cfg.sub_grasp_idx_r == 4, "the reset holding the BOWL is step 3; the mug's is step 8"


# ---------------------------------------------------------------------------------------------
# target_idx: what the run randomizes.
# ---------------------------------------------------------------------------------------------


def test_target_idx_is_the_step_of_the_grasp_not_the_objects_place_in_the_kitchen():
    """--target_idx is a STEP index. events.py:903 stashes the ±5 cm pose sample the reset applies to
    the manipulation object; simvla_gen.py:1522 adds it to the goal step where `step_i ==
    target_idx`, and `step_i` enumerates the GOAL SCRIPT. So target_idx names the step whose authored
    pose has to follow the object when the object moves — the manipulation grasp — and NOT the
    object's place in the kitchen's object list.

    v813 cannot tell the two rules apart: its grasp is step 1 and its bowl is object 1. This template
    can. `reset_first_then_grasp` grasps at STEP 2, and the bowl bound here is OBJECT 3. Under the
    object-index rule the noise lands on step 3 — the gripper close — leaving the grasp itself
    reaching for where the bowl no longer is."""
    bindings = v813_bindings()
    bindings["target"] = BoundPrim(prim_path="/world/bowl3", is_rigid_body=True, object_index=3)

    cfg = derive_run_config(reset_first_then_grasp(), bindings)
    assert cfg.obj_name == "bowl3"
    assert cfg.target_idx == 2, "the grasp is step 2; 3 is the bowl's slot in the object list"


def mirrored_v813() -> TaskTemplate:
    """v813 with the arms swapped: the RIGHT arm opens the drawer, the LEFT arm carries the bowl.

      0  N_s nav.to_prim(@target)
      1  A_l arm.grasp(@target)                    — the LEFT arm picks up the bowl
      2  G_l gripper.set(close)
      3  A_l arm.reset                             — home, bowl in hand; the left check is at 4
      4  A_r arm.handle_pregrasp(@container_handle)
      5  A_r arm.handle_grasp(@container_handle)   — the RIGHT arm takes the drawer handle
      6  G_r gripper.set(close)
      7  N   nav.open_articulation
      8  A_l arm.place(@container)
      9  G_l gripper.set(open)
     10  A_l arm.reset
     11  N   nav.close_articulation

    Nothing about v813 requires the bowl to be the RIGHT arm's. Which hand carries it is an
    authoring choice, and this one validates exactly as the original does."""
    t = bowl_to_drawer()
    t.name = "mirrored_v813"
    swap = {"A_r": "A_l", "A_l": "A_r", "G_r": "G_l", "G_l": "G_r"}
    t.steps = [
        TemplateStep(s.skill, swap.get(s.action, s.action), dict(s.params), s.language)
        for s in t.steps
    ]
    return t


def test_a_right_arm_opening_a_drawer_gets_the_same_guard_as_the_left():
    """The right arm reaching a drawer handle is legal — the mirror of v813. Ungated, it yielded
    obj_name="door_handle", and env.scene.rigid_objects has no such key: a bare KeyError the moment
    the (live) check dereferences it.

    It is also where sub_good_goal_count_r earns its existence. With the guard on, the right check is
    disabled — and --sub_grasp_idx_r 999 with the CLI's default --sub_good_goal_count_r 7 asks for
    seven right-arm successes that only the disabled check could have produced. The run reaches
    stage=collect_right and spins to its time limit printing nothing: the exact failure
    sub_good_goal_count_l exists to prevent, on the arm that had no guard. A coupling enforced on one
    arm and not the other is not enforced."""
    t = mirrored_v813()
    validate_template(t)
    validate_sequence(t)

    cfg = derive_run_config(t, v813_bindings(), sub_good_goal_count=7)
    assert cfg.obj_name == "none", "a drawer handle is not in env.scene.rigid_objects"
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED
    assert cfg.sub_good_goal_count_r == 0, "a disabled RIGHT check must never be asked for successes"
    assert cfg.obj_name_l == "bowl0", "the bowl is the LEFT arm's here, and it is still the object"
    assert cfg.sub_grasp_idx_l == 4


def test_the_randomized_step_is_the_manipulation_grasps_even_when_that_arm_is_the_left():
    """The reset event randomizes exactly ONE object (events.py:903 — the asset whose x-range is
    (-0.05, 0.05)), and target_idx names the goal step whose authored pose must follow it. That step
    belongs to whichever arm GRASPS the manipulation object, which is not always the right one.

    Here it is the left arm's, at step 1. Deriving target_idx from the right arm unconditionally
    emits the no-target sentinel instead: the bowl is then displaced by up to 5 cm at reset while the
    left arm's authored grasp pose stays where the bowl no longer is. The grasp misses by up to 5 cm,
    the 0.15 m check may well still pass, and the run records a whole dataset of near-misses without
    ever saying so. That is the silent one, and the silent ones are why this module exists."""
    t = mirrored_v813()
    cfg = derive_run_config(t, v813_bindings(), sub_good_goal_count=7)

    assert cfg.obj_name == "none", "the right arm has no manipulation object"
    assert cfg.target_idx == 1, "the LEFT arm's grasp of the bowl is step 1, and the bowl is what moves"
    assert t.steps[cfg.target_idx].skill == "arm.grasp"


def test_two_rigid_manipulation_objects_are_refused_rather_than_tie_broken():
    """Both arms carry a rigid body: the right arm the bowl (step 1), the left arm a mug (step 4).
    The reset randomizes exactly ONE asset, and target_idx names ONE step — but WHICH asset is
    randomized is a fact of the env config (events.py: the asset whose x-range is (-0.05, 0.05)) and
    appears nowhere in the TaskTemplate or in a BoundPrim. The information is genuinely absent, so no
    derivation can infer it.

    This module used to pin target_idx to the right arm. That guess is wrong about half the time, and
    when it loses, the right hand collects a displacement its object never underwent while the left
    arm's object moves with no step tracking it. Both errors are under 5 cm — inside the 0.15 m grasp
    check — so nothing prints and the dataset looks fine. A refusal names both candidates instead."""
    t = bowl_to_drawer()
    t.roles.append(Role("second", object_type="mug"))
    t.steps[4] = TemplateStep("arm.grasp", "A_l", {"prim_path": "@second"})
    t.steps[5] = TemplateStep("gripper.set", "G_l", {"grasp": True})
    t.steps[6] = TemplateStep("arm.reset", "A_l", {})
    bindings = v813_bindings()
    bindings["second"] = BoundPrim(prim_path="/world/mug0", is_rigid_body=True, object_index=0)
    validate_template(t)
    validate_sequence(t)                    # a legal script; still not a derivable run config

    with pytest.raises(SequenceError) as exc:
        derive_run_config(t, bindings, sub_good_goal_count=7)
    assert "/world/bowl0" in str(exc.value) and "/world/mug0" in str(exc.value), (
        "a refusal that does not name both candidates leaves the author guessing too"
    )


def test_no_manipulation_object_on_either_arm_gets_a_target_idx_that_names_no_step():
    """Both arms on the drawer handle: nothing rigid is grasped, so the task randomizes NO
    manipulation object and there is no step whose authored pose has to follow anything.

    The sentinel must therefore match NO step. 0 does not: simvla_gen.py:1522 adds the ±5 cm sample
    wherever `step_i == target_idx`, and step 0 is a real step — it would collect the displacement of
    an object that was never displaced. Being a step index, the sentinel is only ever COMPARED
    against `step_i` (never used to subscript the goal), so a negative value can equal no step."""
    t = bowl_to_drawer()
    t.steps = [s for s in t.steps if s.action not in ("A_r", "G_r")]
    t.subtask_groups = []

    cfg = derive_run_config(t, v813_bindings())
    assert cfg.obj_name == "none" and cfg.obj_name_l == "none"
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED
    assert cfg.sub_grasp_idx_l == GRASP_CHECK_DISABLED
    assert cfg.target_idx == NO_TARGET_IDX
    assert all(step_i != cfg.target_idx for step_i in range(len(t.steps))), (
        "a no-target sentinel that equals a step index randomizes that step's authored pose"
    )


# ---------------------------------------------------------------------------------------------
# A grasp whose prim does not resolve. "Unbound" is not "not a rigid body".
# ---------------------------------------------------------------------------------------------


def test_a_grasp_on_an_unbound_role_is_refused_rather_than_blanking_the_whole_config():
    """The silent one. An unresolvable prim used to read exactly like "this arm grasps nothing
    rigid": obj_name="none", target_idx=-1, sub_grasp_idx_r=999, sub_good_goal_count_r=0 — and an
    EMPTY bindings dict produced byte-identical output. Nothing crashed and nothing spun; the run
    recorded a full dataset with the grasp check switched off and the bowl never randomized.

    v813's own template, one role short of a binding, must not derive that config."""
    t = bowl_to_drawer()
    bindings = v813_bindings()
    del bindings["target"]

    with pytest.raises(BindingError) as exc:
        derive_run_config(t, bindings)
    assert "step 1" in str(exc.value) and "@target" in str(exc.value), "name the step and the prim"

    with pytest.raises(BindingError):
        derive_run_config(bowl_to_drawer(), {})


def test_a_grasp_on_a_literal_prim_path_is_refused_even_though_a_template_may_write_one():
    """`arm.grasp(A_r, "/world/bowl0")` — validate_template explicitly permits a literal prim path
    (task_template.py:117) and validate_sequence accepts it, so this reaches the derivation. A raw
    string is not a BoundPrim: it says nothing about is_rigid_body, which is the whole basis of the
    grasp-check guard. Read as "unresolvable", it silently disabled the check and un-randomized the
    object for a task whose object is named right there in the step."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": "/world/bowl0"}, "Right arm to bowl")
    validate_template(t)
    validate_sequence(t)

    with pytest.raises(BindingError) as exc:
        derive_run_config(t, v813_bindings())
    assert "step 1" in str(exc.value) and "/world/bowl0" in str(exc.value)


# ---------------------------------------------------------------------------------------------
# The sentinel is a claim about scripts, and it has a length.
# ---------------------------------------------------------------------------------------------


def no_rigid_grasp_padded_to(n_steps: int) -> TaskTemplate:
    """v813 with the right arm's steps removed — neither arm grasps a rigid body, so obj_name is
    "none" and both checks are 999 — padded out to `n_steps` with arm.pause."""
    t = bowl_to_drawer()
    t.steps = [s for s in t.steps if s.action not in ("A_r", "G_r")]
    t.steps += [TemplateStep("arm.pause", "A_r", {}, "Hold") for _ in range(n_steps - len(t.steps))]
    t.subtask_groups = []
    assert len(t.steps) == n_steps
    return t


def test_a_script_long_enough_to_contain_step_999_is_refused():
    """"No step ever has this index" is an assertion about scripts, and it holds only below 1000 of
    them. This template grasps nothing rigid, so it derives obj_name="none" and sub_grasp_idx_r=999 —
    and step 999 EXISTS here, so the mask fires at a real step and the check dereferences
    env.scene.rigid_objects["none"]: the bare KeyError this module exists to prevent, arriving by way
    of the sentinel that was supposed to prevent it."""
    t = no_rigid_grasp_padded_to(1000)

    with pytest.raises(SequenceError, match="999"):
        derive_run_config(t, v813_bindings())


def test_the_longest_script_the_sentinel_survives_still_derives():
    """The boundary, from the other side: 999 steps are indices 0..998, so 999 still names no step
    and the config is honest. A guard written as `>=` would refuse this one for nothing."""
    t = no_rigid_grasp_padded_to(GRASP_CHECK_DISABLED)

    cfg = derive_run_config(t, v813_bindings())
    assert cfg.sub_grasp_idx_r == GRASP_CHECK_DISABLED
    assert all(step_i != cfg.sub_grasp_idx_r for step_i in range(len(t.steps)))


# ---------------------------------------------------------------------------------------------


def test_task_type_follows_the_actions_used():
    t = bowl_to_drawer()
    assert derive_run_config(t, v813_bindings()).task_type == "NavManipulation"

    t.steps = [s for s in t.steps if s.action in ("N", "N_s")]
    t.subtask_groups = []
    assert derive_run_config(t, v813_bindings()).task_type == "Navigation"


def test_a_script_with_no_steps_is_refused_rather_than_called_manipulation():
    """The action scan fell through to "Manipulation" for an empty script — nothing is manipulated,
    and every other field would be a sentinel. There is no honest run config for no task, so it says
    so instead of naming one."""
    t = bowl_to_drawer()
    t.steps = []
    t.subtask_groups = []

    try:
        derive_run_config(t, v813_bindings())
    except ValueError as exc:
        assert "no steps" in str(exc)
    else:
        raise AssertionError("an empty script must not derive a run config")


def test_to_meta_is_what_the_goal_file_carries():
    meta = derive_run_config(bowl_to_drawer(), v813_bindings()).to_meta()
    assert meta["obj_name"] == "bowl0"
    assert meta["sub_good_goal_count_l"] == 0
    assert meta["sub_good_goal_count_r"] == 7, "simvla_gen reads this one too, and its default is 7"
    assert meta["export_groups"] == "4,5,6,7;8,9,10;11"


def test_module_is_stdlib_only():
    """It must be unit-testable with no GPU and no Omniverse. Parse the AST, so a function-local
    third-party import cannot slip past."""
    module_path = Path(importlib.util.find_spec("simvla.task_runconfig").origin)
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    allowed = set(sys.stdlib_module_names) | {"skill_contract", "task_template", "task_bind", "task_validate"}
    assert not (roots - allowed), f"task_runconfig.py imports {roots - allowed}"

    # Diff sys.modules across the import — an absolute check would blame this module for torch,
    # which a sibling test (skills, imported above) already pulled into the shared process.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("simvla.task_runconfig_purity", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    banned = ("omni", "isaaclab", "pxr", "torch", "trimesh", "numpy", "tkinter", "PIL", "scipy")
    pulled = {m for m in set(sys.modules) - before if m.split(".")[0] in banned}
    assert not pulled, f"importing task_runconfig.py pulled in {sorted(pulled)}"
