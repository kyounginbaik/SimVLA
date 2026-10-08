"""Tests for the task template — a task as a composition of skills over roles.

Run: pytest scripts/simvla/test_task_template.py -v
Stdlib only: no Omniverse, no torch.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

import skills  # noqa: F401  — populates skill_contract.REGISTRY
from scene_spec import Extent, SceneObject
from task_template import (
    Role,
    TaskTemplate,
    TaskTemplateError,
    TemplateStep,
    from_json,
    language_problems,
    role_ref,
    to_json,
    validate_template,
)


def bowl_to_drawer() -> TaskTemplate:
    """The reference task: Isaac-Kitchen-v813-00's script, as a template.

    Not "a task like v813's" — v813's, step for step. Two of these steps were transcribed from the
    plan rather than from the file, and both were wrong in the silent direction. The file on disk
    settles it, and test_task_emit.py holds the template to it:

      step 0  which_arm="Right". nav.to_prim reads which_arm as a ±5 cm arm_bias on the base pose
              (simvla_data_generator.py:1090) and plan_nav_to_prim reads it with a bare .get(), so
              an omitted one is not "unset" — it is "Both", and the robot parks 5 cm off.

      step 8  arm.bowl_place, not arm.place. Two different skills: `arm.place` ("Move arm to place")
              plans a bbox-derived xyz above the target and is AUTHORED; `arm.bowl_place` ("Place
              object") is RUNTIME-resolved — +23 cm forward of wherever the base ended up, −5 cm
              down, −20° wrist roll. The demonstrated run used the second: its step 8 payload is
              [-0.12] * 7 (goal_format.BOWL_PLACE, matched in the quaternion slots) and its
              .reloadable.json twin records `usage: "Place object"`. Authoring the first here does
              not merely rename the step — it hands the arm a different goal, from a different
              frame, computed at a different time, on a task nobody has run.

    The `language` strings are the DEMONSTRATED file's, verbatim — capitalisation ("Left Gripper
    Grasp"), phrasing ("Move back to open drawer", not "Open the drawer"), terseness ("Release")
    and all. They are not decoration and they are not this template's to improve: they are the
    dataset's labels. stream_finalize_lerobot.infer_task_language reads goals[0][0]["language"] as
    the task string every frame is trained against, and simvla_gen.py:811 feeds the per-step ones to
    `sub_task_l`. A tidied-up wording here is a different label on the data. Seven of these twelve
    were tidied up once; test_task_emit's reproduction test now compares them, and the file on disk
    wins.
    """
    return TaskTemplate(
        name="bowl_to_drawer",
        language="Put bowl inside drawer.",
        roles=[
            Role("target", object_type="bowl"),
            Role("container", articulation_with="drawer"),
            Role("container_handle", handle_of="container"),
        ],
        steps=[
            TemplateStep("nav.to_prim", "N_s", {"prim_path": "@target", "which_arm": "Right"}, "Move to bowl"),
            TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"}, "Right arm to bowl"),
            TemplateStep("gripper.set", "G_r", {"grasp": True}, "Grasp the bowl"),
            TemplateStep("arm.reset", "A_r", {}, "Reset to original position"),
            TemplateStep("arm.handle_pregrasp", "A_l", {"prim_path": "@container_handle"}, "Left Arm to pre-grasp pull handle"),
            TemplateStep("arm.handle_grasp", "A_l", {"prim_path": "@container_handle"}, "Left Arm to grasp pull handle"),
            TemplateStep("gripper.set", "G_l", {"grasp": True}, "Left Gripper Grasp"),
            TemplateStep("nav.open_articulation", "N", {"back_off_m": 0.25}, "Move back to open drawer"),
            TemplateStep("arm.bowl_place", "A_r", {}, "Move Right arm to place in drawer"),
            TemplateStep("gripper.set", "G_r", {"grasp": False}, "Release"),
            TemplateStep("arm.reset", "A_r", {}, "Reset to original position"),
            TemplateStep("nav.close_articulation", "N", {}, "Move front to close drawer"),
        ],
        subtask_groups=[[4, 5, 6, 7], [8, 9, 10], [11]],
        scene=[
            SceneObject("bowl0", "bowl", Extent(1.0, 0.05), "dishwasher", lift=Extent(0.775, 0.025)),
            SceneObject("mug0", "mug", Extent(1.0, 0.05), "island"),
        ],
        success={"all": [
            {"obj_z": {"role": "@target", "lo": 0.6, "hi": 0.8}},
            {"eef_home": {"arm": "right", "radius": 0.18}},
            {"joint_pos": {"role": "@container", "joint": "corpus_to_drawer_0_0", "hi": 0.03}},
            {"last_subtask": {}},
        ]},
    )


def test_the_reference_task_validates():
    """Isaac-Kitchen-v813-00 has been run to a recorded demonstration. If the template that
    expresses it does not validate, the validator is wrong, not the task."""
    validate_template(bowl_to_drawer())


def test_role_ref_distinguishes_a_role_from_a_literal_path():
    assert role_ref("@target") == "target"
    assert role_ref("/world/bowl0") is None


def test_a_step_naming_an_unregistered_skill_is_rejected():
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.telekinesis", "A_r", {"prim_path": "@target"})
    with pytest.raises(TaskTemplateError, match="arm.telekinesis"):
        validate_template(t)


def test_a_step_using_an_action_the_skill_does_not_declare_is_rejected():
    """gripper.set declares ('G_r','G_l'). N_s is not one of them."""
    t = bowl_to_drawer()
    t.steps[2] = TemplateStep("gripper.set", "N_s", {"grasp": True})
    with pytest.raises(TaskTemplateError, match="N_s"):
        validate_template(t)


def test_a_param_the_skill_does_not_declare_is_rejected():
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": "@target", "velocity": 3})
    with pytest.raises(TaskTemplateError, match="velocity"):
        validate_template(t)


def test_a_role_reference_with_no_matching_role_is_rejected():
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": "@nonesuch"})
    with pytest.raises(TaskTemplateError, match="nonesuch"):
        validate_template(t)


def test_a_non_string_primpath_value_is_rejected():
    """0.25 is neither a role reference nor a literal prim path. role_ref() returns None for it
    (its isinstance(value, str) guard fails open), and the old validator treated that None as
    "must be a literal path" and accepted it — a float would reach the executor as a prim_path."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": 0.25})
    with pytest.raises(TaskTemplateError, match=r"step 1.*prim_path.*0\.25"):
        validate_template(t)


def test_a_none_primpath_value_is_rejected():
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": None})
    with pytest.raises(TaskTemplateError, match=r"step 1.*prim_path"):
        validate_template(t)


def test_an_empty_string_primpath_value_is_rejected():
    """"" is a string, so it survives an isinstance(value, str) check — but it is falsy, and a
    validator that only checks non-string-ness (or only checks truthiness) can let one of these
    two cases through. Both must be rejected."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {"prim_path": ""})
    with pytest.raises(TaskTemplateError, match=r"step 1.*prim_path"):
        validate_template(t)


def test_a_role_must_declare_exactly_one_matcher():
    with pytest.raises(TaskTemplateError, match="exactly one"):
        Role("target", object_type="bowl", articulation_with="drawer")
    with pytest.raises(TaskTemplateError, match="exactly one"):
        Role("target")


def test_handle_of_must_name_a_declared_role():
    t = bowl_to_drawer()
    t.roles[2] = Role("container_handle", handle_of="cupboard")
    with pytest.raises(TaskTemplateError, match="cupboard"):
        validate_template(t)


def test_subtask_groups_must_index_real_steps():
    t = bowl_to_drawer()
    t.subtask_groups = [[4, 5, 99]]
    with pytest.raises(TaskTemplateError, match="99"):
        validate_template(t)


def test_every_problem_is_reported_at_once():
    """Fixing a twelve-step template one error per run is whack-a-mole."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.telekinesis", "A_r", {})
    t.steps[2] = TemplateStep("gripper.set", "N_s", {"grasp": True})
    with pytest.raises(TaskTemplateError) as exc:
        validate_template(t)
    message = str(exc.value)
    assert "arm.telekinesis" in message and "N_s" in message


def test_json_round_trip_is_lossless():
    t = bowl_to_drawer()
    back = from_json(to_json(t))
    assert back == t


def test_a_template_carries_a_scene():
    from scene_spec import SceneObject, Extent
    from task_template import TaskTemplate, Role, TemplateStep
    t = TaskTemplate(
        name="x", language="y",
        roles=[Role("target", object_type="bowl")],
        steps=[TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"})],
        scene=[SceneObject("bowl0", "bowl", Extent(1.0, 0.05), "dishwasher", lift=Extent(0.775, 0.025))],
    )
    assert t.scene[0].object_type == "bowl"


def test_scene_round_trips_through_json():
    from task_template import to_json, from_json
    t = bowl_to_drawer()
    assert from_json(to_json(t)) == t          # includes the scene


def test_a_file_without_a_scene_key_loads_as_an_empty_scene():
    from task_template import from_json
    import json
    raw = {"name": "n", "language": "l",
           "roles": [{"name": "target", "object_type": "bowl",
                      "articulation_with": None, "handle_of": None}],
           "steps": [], "subtask_groups": []}
    t = from_json(json.dumps(raw))
    assert t.scene == []                        # the no-scene fallback boundary


def test_validate_requires_a_scene_object_for_a_manipulation_role():
    from task_template import TaskTemplate, Role, TemplateStep, validate_template, TaskTemplateError
    from scene_spec import SceneObject, Extent
    t = TaskTemplate(
        name="x", language="y",
        roles=[Role("target", object_type="bowl")],
        steps=[TemplateStep("arm.grasp", "A_r", {"prim_path": "@target"})],
        scene=[SceneObject("mug0", "mug", Extent(1.0), "island")],   # no bowl!
    )
    with pytest.raises(TaskTemplateError, match="bowl"):
        validate_template(t)


def test_an_empty_scene_still_validates_as_the_fallback_case():
    from task_template import validate_template
    t = bowl_to_drawer()
    t.scene = []
    validate_template(t)                        # empty scene defers to the generator, no raise


# ---------------------------------------------------------------------------------------------
# The language is a DIRECTORY NAME. simvla_gen.py:3217 —
#     f"{os.environ['SIMVLA_LEROBOT_ROOT']}/{args_cli.task_language}/{args_cli.task}"
# — and derive_run_config puts t.language into the goal file as task_language, verbatim. Until now a
# human typed it at a shell and would have noticed; the composer lets a GUI user author it.
# ---------------------------------------------------------------------------------------------


def test_the_demonstrated_language_is_legal_and_keeps_its_trailing_period():
    """The one task known to have recorded a dataset end to end is labelled 'Put bowl inside
    drawer.' — the dataset's own label, period and all. A check that refuses a sentence is a check
    that refuses the reference task."""
    assert bowl_to_drawer().language == "Put bowl inside drawer."
    assert language_problems("Put bowl inside drawer.") == []
    validate_template(bowl_to_drawer())


@pytest.mark.parametrize("language", [
    "Put the bowl in the drawer / fridge",   # a sentence a GUI user would type. A NESTED directory.
    "Put bowl inside drawer\\fridge",        # the same, the other separator
    "../../etc/tasks",                       # climbs out of SIMVLA_LEROBOT_ROOT
    "Put bowl inside drawer..",              # '..' need not be alone to be a path component
    "/Put bowl inside drawer.",              # leading '/': the path is ABSOLUTE; the root is dropped
    "Put bowl inside drawer. ",              # a directory nothing typed at a shell will ever match
    " Put bowl inside drawer.",
    "Put bowl\x00inside drawer.",            # a NUL
    "",                                      # an unnamed path component, and an unlabelled dataset
])
def test_a_language_that_is_a_path_is_refused(language):
    """None of these FAILS at run time. Every one of them records a dataset — somewhere else, or
    under a name split in two. That is the shape of failure this whole contract exists to remove, so
    it is refused where the string is authored, not where the path is built."""
    assert language_problems(language), f"{language!r} must not reach a goal file"

    t = bowl_to_drawer()
    t.language = language
    with pytest.raises(TaskTemplateError, match="language"):
        validate_template(t)


def test_a_language_is_still_allowed_to_be_a_sentence():
    """The refusal is about path components, not about English. Punctuation, capitals, commas and a
    final period are what the dataset's labels actually look like."""
    for language in ("Put bowl inside drawer.", "Open the drawer, then place the mug.",
                     "Take the bowl out of the fridge!", "Pick up the bowl with both hands"):
        assert language_problems(language) == []


def test_validate_template_reports_a_bad_language_alongside_everything_else():
    """One pass, every problem — a language error must not preempt the rest of the report."""
    t = bowl_to_drawer()
    t.language = "a/b"
    t.steps[1] = TemplateStep("arm.telekinesis", "A_r", {})
    with pytest.raises(TaskTemplateError) as exc:
        validate_template(t)
    assert "language" in str(exc.value) and "arm.telekinesis" in str(exc.value)


def test_module_is_stdlib_only():
    """It must be unit-testable with no GPU and no Omniverse. Parse the AST, so a function-local
    third-party import cannot slip past."""
    module_path = Path(__file__).parent / "task_template.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    allowed = set(sys.stdlib_module_names) | {"skill_contract", "scene_spec", "predicate_contract"}
    assert not (roots - allowed), f"task_template.py imports {roots - allowed}"

    # Diff sys.modules across the import — an absolute check would blame this module for torch,
    # which a sibling test already imported into the shared process.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("task_template_purity", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    banned = ("omni", "isaaclab", "pxr", "torch", "trimesh", "numpy", "tkinter", "PIL", "scipy")
    pulled = {m for m in set(sys.modules) - before if m.split(".")[0] in banned}
    assert not pulled, f"importing task_template.py pulled in {sorted(pulled)}"


# ---- composed success / retry conditions -------------------------------------------------------

GOOD_SUCCESS = {"all": [
    {"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.2}},
    {"obj_z": {"role": "@target", "lo": 0.90}},
    {"last_subtask": {}},
]}


def _grasp_template(**over):
    """A minimal one-object grasp task, valid except for whatever the caller overrides."""
    kw = dict(
        name="grasp_thing",
        language="Grasp the bottle.",
        roles=[Role(name="target", object_type="bottle")],
        steps=[
            TemplateStep(skill="arm.grasp", action="A_r", params={"prim_path": "@target"}),
            TemplateStep(skill="gripper.set", action="G_r", params={"grasp": True}),
        ],
        scene=[SceneObject(name="bottle0", object_type="bottle",
                           size=Extent(center=1.0, spread=0.02), placement="island", lift=None)],
        success=GOOD_SUCCESS,
    )
    kw.update(over)
    return TaskTemplate(**kw)


def test_a_template_with_no_success_condition_is_refused():
    """A task without a success condition emits mdp.task2 — a put-in-drawer check — whatever the
    task is, and demo export is gated on that term firing."""
    with pytest.raises(TaskTemplateError, match="success"):
        validate_template(_grasp_template(success=None))


def test_a_valid_success_condition_passes():
    validate_template(_grasp_template())


def test_a_success_condition_naming_an_undeclared_role_is_refused():
    bad = {"all": [{"obj_z": {"role": "@nosuchrole", "lo": 0.9}}]}
    with pytest.raises(TaskTemplateError, match="nosuchrole"):
        validate_template(_grasp_template(success=bad))


def test_a_bad_retry_condition_is_refused():
    bad = {"any": [{"obj_z": {"role": "@target"}}]}       # neither bound given
    with pytest.raises(TaskTemplateError, match="obj_z"):
        validate_template(_grasp_template(retry=bad))


def test_an_absent_retry_is_allowed_and_emit_supplies_the_default():
    t = _grasp_template(retry=None)
    validate_template(t)
    assert t.retry is None


def test_the_conditions_survive_the_json_round_trip():
    retry = {"any": [{"obj_z": {"role": "@target", "hi": 0.3}}]}
    t = _grasp_template(retry=retry)
    back = from_json(to_json(t))
    assert back.success == GOOD_SUCCESS
    assert back.retry == retry
