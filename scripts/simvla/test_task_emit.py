"""The reproduction test: the composer must be able to rebuild a task that is known to work.

Isaac-Kitchen-v813-00 was run on a GPU node and produced a 960-frame LeRobot dataset with EXIT=0.
If the composer path cannot reproduce its goal script, the composer is wrong.

THE ORACLE IS THE FILE ON DISK, and it comes in two halves that are checked against each other:

    goals/Isaac-Kitchen-v813-00.json             the v1 payloads the executor ran
    goals/Isaac-Kitchen-v813-00.reloadable.json  the twin, which records the AUTHORING inputs

Neither half alone is enough. v1 cannot name a skill: an ordinary authored arm pose decodes to
`arm.pose`, the fall-through branch that every plan() skill's goal lands in, so the payload cannot
say whether step 4 was a handle_pregrasp or a plain pose. The twin can — its `usage` string IS the
skill's label, and skills._legacy_skill_for is the same lookup the GUI's reload path uses. And the
twin cannot say what the executor DID with a step; the payload can. Together they pin all four
things the emitter must get right: the skill, the action, the params (which_arm included), and
whether the goal was authored or left null for the run to resolve.

What the oracle CANNOT supply is the authored poses themselves — plan() reads the USD stage, and
this test has no GPU. So plan_fn REPLAYS them out of the demonstrated file, keyed by CONTENT
(skill, action, resolved params), never by step index: an emitter that planned the wrong step, or
that handed plan() an unresolved '@target', misses the key and the test fails rather than quietly
agreeing with itself.

Run: pytest scripts/simvla/test_task_emit.py -v
"""

import ast
import dataclasses
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import executor_dispatch
import goal_format
import skills  # noqa: F401  — populates skill_contract.REGISTRY
from scene_spec import Extent, SceneObject
from skill_contract import REGISTRY
from skills import _legacy_skill_for
from predicate_contract import SpecError
from task_bind import BoundPrim, bind_many
from task_emit import (
    UnsafeOutputDir,
    build_arg_parser,
    default_retry,
    emit,
    guard_out_dir,
    kitchen_subs,
    kitchen_task_name,
    load_template,
    parse_kitchens,
    parse_subs,
    resolved_condition,
    resolved_params,
    steps_for_kitchen,
    terminations_block,
)
from task_template import from_json, to_json
from task_runconfig import GRASP_CHECK_DISABLED
from task_template import Role, TaskTemplate, TaskTemplateError, TemplateStep
from scripts.simvla.test_task_bind import kitchen
from scripts.simvla.test_task_runconfig import v813_bindings
from scripts.simvla.test_task_template import bowl_to_drawer

HERE = Path(__file__).parent

#: Read-only bundled fixtures, independent of a caller's collection environment.
GOALS = HERE.parents[1] / "examples" / "goals"
V813 = "Isaac-Kitchen-v813-00"


@pytest.fixture(autouse=True)
def isolate_goal_corpus(monkeypatch):
    # Exercise the same output guard without reading/writing the user's external
    # SIMVLA_GOALS_DIR (which may contain a different robot or not be mounted).
    monkeypatch.setattr("task_emit.GOALS_DIR", GOALS.resolve())


def stub_plan(skill_id, action, params):
    """Stand in for the real plan(), which needs the USD stage. Returns a distinguishable pose per
    skill so a step that picks up the wrong skill's goal is visible."""
    return [float(len(skill_id)), 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------------------------
# The oracle.
# ---------------------------------------------------------------------------------------------


def demonstrated_steps() -> list[goal_format.Step]:
    """v813's twelve steps, as the executor read them: skill named by the twin's `usage`, payload
    (and therefore authored-vs-runtime) taken from the file that actually ran."""
    payloads, _ = goal_format.read_v1_legacy(GOALS / f"{V813}.json")
    twin = json.loads((GOALS / f"{V813}.reloadable.json").read_text())["goals"][0]
    assert len(payloads) == len(twin) == 12

    steps = []
    for payload, authored in zip(payloads, twin):
        spec, params = _legacy_skill_for(authored["action"], authored["parameters"])
        steps.append(
            goal_format.Step(
                skill=spec.id,
                action=authored["action"],
                # Only the params the skill DECLARES: the twin is a GUI dump and carries the empty
                # prim_path box of a skill that has no prim. An omitted one is the contract's
                # default — which is exactly what the emitter must write.
                params={p.name: _declared(p, params) for p in spec.params},
                goal=payload.goal,
                language=authored["language"],
            )
        )
    return steps


def _declared(param, authored: dict):
    if param.name in authored:
        return authored[param.name]
    default = getattr(param, "default", None)
    assert default is not None, f"the twin omits {param.name!r}, which has no declared default"
    return default


def demonstrated_meta() -> dict:
    """The goal file minus its steps — the kitchen's randomization ranges. simvla_gen.py:826 reads
    initial_pos_ranges off it, so a goal file without them does not load."""
    raw = json.loads((GOALS / f"{V813}.json").read_text())
    return {k: v for k, v in raw.items() if k not in {"goals", "run_config"}}


def test_the_two_halves_of_the_oracle_agree_about_what_is_authored():
    """A guard on the oracle itself, before anything is asserted against it: for every step, the
    skill the AUTHOR picked is runtime-resolved exactly when the payload the EXECUTOR ran was a
    sentinel it overwrites. If these two ever disagreed, the file would not mean what either half
    says, and every assertion below would be built on it."""
    for i, step in enumerate(demonstrated_steps()):
        assert REGISTRY[step.skill].is_runtime == (step.goal is None), (
            f"step {i} ({step.skill}): the twin says is_runtime="
            f"{REGISTRY[step.skill].is_runtime}, the payload says goal is None = {step.goal is None}"
        )


# ---------------------------------------------------------------------------------------------
# THE REPRODUCTION TEST.
# ---------------------------------------------------------------------------------------------


def replay_planner(demonstrated: list[goal_format.Step]):
    """plan(), replayed out of the demonstrated file — CONTENT-ADDRESSED, not step-indexed.

    The six authoring skills read the USD stage (BBoxCache, raycasts, a Tk thumbnail chooser), so
    their poses cannot be recomputed here. They can be looked up: a step's plan() is a pure function
    of (skill, action, resolved params), and the demonstrated file records what each of them
    returned. Keying on content rather than on `i` is what keeps the test honest — an emitter that
    called plan() for the wrong step, left a role unresolved, or dropped which_arm gets a KeyError,
    not a pass.
    """
    table = {}
    for step in demonstrated:
        if step.goal is not None:
            table[_key(step.skill, step.action, step.params)] = step.goal

    def plan_fn(skill_id, action, params):
        key = _key(skill_id, action, params)
        if key not in table:
            raise KeyError(
                f"the emitter asked plan() for {key!r}, which the demonstrated file never planned. "
                f"It planned: {sorted(table)}"
            )
        return table[key]

    return plan_fn


def _key(skill: str, action: str, params: dict):
    return (skill, action, tuple(sorted((k, repr(v)) for k, v in params.items())))


def test_it_reproduces_the_goal_file_of_a_task_that_recorded_a_demonstration(tmp_path):
    """The whole chain, against the only thing that can settle it: template -> validate -> bind ->
    plan -> run config -> v2 file, checked step for step against the file a GPU node ran.

    This is the test the composer exists to pass. It is not a shape check: every step's skill,
    action, params, goal AND language must equal the demonstrated one's — all five of a Step's
    fields, because a field nobody compares is a field the template may quietly change — and the run
    config must equal the nine flags that produced the dataset."""
    demonstrated = demonstrated_steps()

    written, failed = emit(
        bowl_to_drawer(),
        {V813: v813_bindings()},
        replay_planner(demonstrated),
        out_dir=tmp_path,
        kitchen_meta={V813: demonstrated_meta()},
    )
    assert failed == [], failed
    assert written == [str(tmp_path / f"{V813}.json")]

    # `version` is the FILE's, and goal_format.read strips it out of the meta it returns (a v1 file
    # has no version key at all, and the executor's `data` must not sprout one). So read the file.
    assert json.loads(Path(written[0]).read_text())["version"] == 2
    steps, meta = goal_format.read(written[0], known_skills=set(REGISTRY))

    assert [s.skill for s in steps] == [s.skill for s in demonstrated], (
        "the composed script names a different skill somewhere than the one that was demonstrated"
    )
    assert [s.action for s in steps] == [s.action for s in demonstrated]
    expected_params = [
        {k: v for k, v in s.params.items()
         if not (isinstance(v, float) and not math.isfinite(v))}
        for s in demonstrated
    ]
    assert [s.params for s in steps] == expected_params, (
        "roles must resolve to the prims the demonstration used, and every finite declared param "
        "must be stated; non-finite sentinels cannot be represented in strict JSON"
    )
    assert [s.goal for s in steps] == [s.goal for s in demonstrated], (
        "the authored poses land on the steps that planned them, and every runtime step is null"
    )
    assert [s.language for s in steps] == [s.language for s in demonstrated], (
        "the fifth field. `language` is the DATA's label, not a comment: "
        "stream_finalize_lerobot.infer_task_language reads goals[0][0]['language'] as the string "
        "every frame of the LeRobot dataset is trained against, and simvla_gen.py:811 feeds the "
        "per-step ones to sub_task_l. The demonstrated file says 'Left Gripper Grasp' and 'Move "
        "back to open drawer'; a template that tidies those up is labelling the data differently "
        "from the run that produced it. The file wins — verbatim, capitals included."
    )

    assert meta["run_config"] == {
        "obj_name": "bowl0",
        "obj_name_l": "none",
        "target_idx": 1,
        "sub_grasp_idx_r": 2,
        "sub_grasp_idx_l": GRASP_CHECK_DISABLED,
        "sub_good_goal_count_r": 7,
        "sub_good_goal_count_l": 0,
        "export_groups": "4,5,6,7;8,9,10;11",
        "task_language": "Put bowl inside drawer.",
        "task_type": "NavManipulation",
    }, "the nine flags a human used to transcribe by counting steps"

    # The kitchen's randomization ranges are not the template's to invent, and simvla_gen.py:826
    # reads them straight off the goal file: a v2 file without them raises KeyError at load.
    for key, value in demonstrated_meta().items():
        if key == "task_name":
            continue                       # decorative; the executor never reads it
        assert meta[key] == value, f"the emitted file lost the kitchen's {key!r}"


def test_the_reproduction_test_compares_every_field_a_step_has():
    """A guard on the test above, because its claim is total: the composed file IS the demonstrated
    file. It can only mean that if every field of a Step is checked — and for a while `language` was
    not, so seven of the twelve steps' labels differed from the run's and the suite said nothing. A
    sixth field added to Step later would be the same silence again. Fail here instead."""
    compared = {"skill", "action", "params", "goal", "language"}
    assert {f.name for f in dataclasses.fields(goal_format.Step)} == compared, (
        "goal_format.Step has a field the reproduction test does not compare. Compare it there, then "
        "name it here — an authored field nobody checks is one the template may quietly change."
    )


def test_the_executor_cannot_tell_the_composed_file_from_the_demonstrated_one(tmp_path):
    """One level lower, through the executor's own loader. `load_script` is what simvla_gen calls;
    it decides which branch each step takes (skill_id) and what floats go into payloads_tensor
    (spec). Run both files through it and the rows must be identical.

    The skill NAMES are compared through the twin above; here what is compared is the executable
    content — because a v1 file cannot name an authored pose, so the demonstrated file's steps 4 and
    5 load as `arm.pose`, the legacy fall-through. That is a limit of the v1 format, not a
    difference in what runs: the action, the payload floats and the authored/runtime split are what
    the executor acts on, and those must match exactly."""
    demonstrated = demonstrated_steps()
    written, failed = emit(
        bowl_to_drawer(), {V813: v813_bindings()}, replay_planner(demonstrated),
        out_dir=tmp_path, kitchen_meta={V813: demonstrated_meta()},
    )
    assert failed == []

    ours, _, _ = executor_dispatch.load_script(written[0])
    theirs, _, _ = executor_dispatch.load_script(GOALS / f"{V813}.json")

    assert [s.action for s in ours] == [s.action for s in theirs]
    assert [s.spec for s in ours] == [s.spec for s in theirs], (
        "the floats the executor would put in payloads_tensor differ"
    )
    for i, (mine, theirs_i) in enumerate(zip(ours, theirs)):
        if theirs_i.skill == "arm.pose":
            # v1's fall-through: an authored pose with no skill name. Ours must be a real authored
            # arm skill on the same channel — which is more than the old file could say.
            assert not REGISTRY[mine.skill].is_runtime and mine.action.startswith("A_")
        else:
            assert mine.skill == theirs_i.skill, f"step {i}"


# ---------------------------------------------------------------------------------------------
# Composition, with a stub planner. No file, no oracle — just the emitter's own logic.
# ---------------------------------------------------------------------------------------------


def test_it_reproduces_the_script_of_a_task_that_recorded_a_demonstration():
    steps = steps_for_kitchen(bowl_to_drawer(), v813_bindings(), stub_plan)

    assert [s.action for s in steps] == [
        "N_s", "A_r", "G_r", "A_r", "A_l", "A_l", "G_l", "N", "A_r", "G_r", "A_r", "N"
    ], "this is Isaac-Kitchen-v813-00's script"
    assert [s.skill for s in steps] == [
        "nav.to_prim", "arm.grasp", "gripper.set", "arm.reset",
        "arm.handle_pregrasp", "arm.handle_grasp", "gripper.set", "nav.open_articulation",
        "arm.bowl_place", "gripper.set", "arm.reset", "nav.close_articulation",
    ], "step 8 is 'Place object' (arm.bowl_place) in the demonstrated file, not 'Move arm to place'"


# ---- multi-rotation fan-out + island double-nav: the dataset is 12 rotations, not one ----

def test_kitchen_task_name_carries_the_rotation():
    """A kitchen is not one task but twelve — one per rotation. The task id carries the sub-number, and
    defaults to -00 so callers naming 'the kitchen' get its first rotation."""
    assert kitchen_task_name(813) == "Isaac-Kitchen-v813-00"       # default rotation
    assert kitchen_task_name(813, 0) == "Isaac-Kitchen-v813-00"
    assert kitchen_task_name(813, 11) == "Isaac-Kitchen-v813-11"
    assert kitchen_task_name(5, 3) == "Isaac-Kitchen-v05-03"       # zero-padded, both fields


def test_kitchen_subs_finds_exactly_the_rotation_usds_on_disk(tmp_path):
    """kitchen_subs reads which rotations physically exist, so an emit fans out over the real USDs and
    neither invents a missing rotation nor stops at -00. Noise (other kitchens, non-USD) is ignored."""
    for name in ("kitchen_05_00.usd", "kitchen_05_01.usd", "kitchen_05_02.usd",
                 "kitchen_05_00.py", "kitchen_06_00.usd", "notes.txt"):
        (tmp_path / name).write_text("x")
    assert kitchen_subs(5, kitchen_dir=tmp_path) == [0, 1, 2]
    assert kitchen_subs(6, kitchen_dir=tmp_path) == [0]
    assert kitchen_subs(99, kitchen_dir=tmp_path) == []            # a kitchen with no USD -> []


def test_island_kitchens_are_not_double_navved_because_the_corpus_is_not():
    """The editor's save_goal_file doubles a nav step for kitchen_type == 'island', but an audit of the
    released corpus found only 3 of 1690 island goal files carry a doubled nav (and those are
    malformed). Reproducing the dataset means NOT doubling — so steps_for_kitchen emits nav once,
    island or not. This pins that decision so a future 'fix' to re-add doubling fails here."""
    steps = steps_for_kitchen(bowl_to_drawer(), v813_bindings(), stub_plan)
    assert [s.action for s in steps] == [
        "N_s", "A_r", "G_r", "A_r", "A_l", "A_l", "G_l", "N", "A_r", "G_r", "A_r", "N"
    ]
    assert len(steps) == 12                                        # no nav step doubled


def test_roles_are_resolved_to_real_prims():
    steps = steps_for_kitchen(bowl_to_drawer(), v813_bindings(), stub_plan)
    assert steps[1].params["prim_path"] == "/world/bowl0", "@target became the bound prim"
    assert steps[4].params["prim_path"].endswith("/door_handle")
    assert not any(
        str(v).startswith("@") for s in steps for v in s.params.values()
    ), "no role reference may survive into a goal file"


def test_a_step_referencing_a_role_that_did_not_bind_is_refused():
    """Silently leaving '@container_handle' in a prim_path hands the executor a path no stage has."""
    bindings = v813_bindings()
    del bindings["container_handle"]
    with pytest.raises(KeyError, match="container_handle"):
        steps_for_kitchen(bowl_to_drawer(), bindings, stub_plan)


def test_every_finite_declared_param_is_stated_in_the_file_even_when_the_template_omits_it():
    """A param the template leaves out is not absent — it has the contract's default, and that
    default is load-bearing. nav.to_prim reads which_arm as a ±5 cm arm_bias
    (simvla_data_generator.py:1090) and executor_dispatch reads back_off_m off the step's params at
    load. Leaving the slot empty puts the value in nobody's file and everybody's memory, which is
    the disease. The writer fills it from the same declaration task_validate.gripper_closes reads."""
    steps = steps_for_kitchen(bowl_to_drawer(), v813_bindings(), stub_plan)
    assert steps[11].skill == "nav.close_articulation"
    assert steps[11].params == {"back_off_m": 0.25, "which_arm": "Both"}, (
        "the template writes neither; the contract declares both"
    )
    for step in steps:
        finite_params = {
            p.name for p in REGISTRY[step.skill].params
            if not hasattr(p, "default")
            or not (isinstance(p.default, float) and not math.isfinite(p.default))
        }
        assert set(step.params) == finite_params


def test_runtime_skills_carry_no_goal_and_authored_skills_do():
    steps = steps_for_kitchen(bowl_to_drawer(), v813_bindings(), stub_plan)
    for s in steps:
        if REGISTRY[s.skill].is_runtime:
            assert s.goal is None, f"{s.skill} is runtime-resolved; its goal must be null"
        else:
            assert s.goal is not None, f"{s.skill} is authored; it must carry a goal"


def test_a_planned_skill_is_never_asked_to_resolve_itself_at_run_time():
    """plan_fn must not be called for a runtime skill: there is no plan() to call. arm.reset's would
    raise NotImplementedError, and nav.open_articulation's goal depends on where the base ends up."""
    called = []

    def recording_plan(skill_id, action, params):
        called.append(skill_id)
        return stub_plan(skill_id, action, params)

    steps_for_kitchen(bowl_to_drawer(), v813_bindings(), recording_plan)
    assert not [s for s in called if REGISTRY[s].is_runtime]
    assert called == [
        "nav.to_prim", "arm.grasp", "gripper.set", "arm.handle_pregrasp", "arm.handle_grasp",
        "gripper.set", "gripper.set",
    ]


def test_an_authored_skill_whose_plan_returns_nothing_is_refused_rather_than_written_as_null():
    """`goal: null` in a v2 file means ONE thing: the executor resolves this step at run time. An
    authored skill has no resolver, so a null goal for one is not "unspecified", it is a step the
    executor drives to (0, 0, 0) with a zero quaternion — the refrigerator bug's exact shape, in the
    new encoding. A plan() that returns None is a bug in the planner and must not reach a file."""
    def blank_plan(skill_id, action, params):
        return None if skill_id == "arm.place" else stub_plan(skill_id, action, params)

    t = bowl_to_drawer()
    t.steps[8] = TemplateStep("arm.place", "A_r", {"prim_path": "@container"}, "Place bowl")

    with pytest.raises(ValueError, match="arm.place"):
        steps_for_kitchen(t, v813_bindings(), blank_plan)


def test_arm_place_is_authored_and_keeps_the_position_its_planner_computed():
    """arm.place — a DIFFERENT skill from the arm.bowl_place v813 uses — computes a real
    bbox-derived xyz which IS the IK target; only its quaternion is refined from the live eef.
    Modelling it as runtime-resolved discarded 4,178 place positions once already. Nothing in the
    reference task exercises it, so it is exercised here."""
    t = bowl_to_drawer()
    t.steps[8] = TemplateStep("arm.place", "A_r", {"prim_path": "@container"}, "Place bowl in drawer")

    steps = steps_for_kitchen(t, v813_bindings(), stub_plan)
    assert steps[8].skill == "arm.place"
    assert not REGISTRY["arm.place"].is_runtime
    assert steps[8].goal == stub_plan("arm.place", "A_r", {})
    assert steps[8].params["prim_path"] == "/world/base_cabinet/drawer_0_0"


def test_a_step_that_omits_the_prim_it_reaches_for_is_refused():
    """A PrimPath has no default — there is no prim a skill reaches for 'by default' — so an omitted
    one cannot be filled in the way which_arm and back_off_m are. The old code left the key ABSENT
    and said nothing, and the emitted step told the executor to reach for nothing. validate_template
    refuses this before emit() ever gets there, but steps_for_kitchen is public and a direct caller
    got the silent version, while this module's own test asserts every declared param is stated."""
    t = bowl_to_drawer()
    t.steps[1] = TemplateStep("arm.grasp", "A_r", {}, "Right arm to bowl")   # no prim_path

    with pytest.raises(TaskTemplateError, match="prim_path"):
        steps_for_kitchen(t, v813_bindings(), stub_plan)


def test_resolved_params_leaves_a_literal_prim_path_alone():
    """A template may write a literal path (task_template.py:117). It is not a role; it passes
    through untouched rather than being looked up and lost."""
    spec = REGISTRY["arm.grasp"]
    step = TemplateStep("arm.grasp", "A_r", {"prim_path": "/world/bowl0"})
    assert resolved_params(step, {}, spec) == {"prim_path": "/world/bowl0"}


def test_resolved_params_omits_non_finite_identity_defaults():
    """The no-floor sentinel is an internal arithmetic identity, not valid JSON data."""
    spec = REGISTRY["arm.bowl_place"]
    step = TemplateStep("arm.bowl_place", "A_r", {})

    params = resolved_params(step, {}, spec)

    assert "min_eef_z" not in params
    assert params == {"forward_m": 0.23, "down_m": 0.05, "roll_deg": -20.0, "lateral_m": 0.0}


# ---------------------------------------------------------------------------------------------
# emit(): the batch.
# ---------------------------------------------------------------------------------------------


def test_the_written_file_is_v2_and_reads_back(tmp_path):
    written, failed = emit(
        bowl_to_drawer(), {V813: v813_bindings()}, stub_plan, out_dir=tmp_path
    )
    assert failed == []
    assert json.loads(Path(written[0]).read_text())["version"] == 2
    steps, meta = goal_format.read(written[0], known_skills=set(REGISTRY))
    assert meta["run_config"]["obj_name"] == "bowl0"
    assert meta["run_config"]["sub_good_goal_count_l"] == 0
    assert len(steps) == 12


def test_a_kitchen_whose_plan_fails_is_skipped_and_reported(tmp_path):
    """A bad kitchen must not take the batch down, and must never be dropped quietly."""
    def exploding_plan(skill_id, action, params):
        if skill_id == "arm.grasp":
            raise RuntimeError("no reachable grasp candidate")
        return stub_plan(skill_id, action, params)

    written, failed = emit(
        bowl_to_drawer(),
        {V813: v813_bindings()},
        exploding_plan,
        out_dir=tmp_path,
    )
    assert written == []
    assert len(failed) == 1 and "no reachable grasp candidate" in failed[0]


def test_a_bug_in_the_planner_takes_the_batch_down_instead_of_reading_as_twelve_bad_kitchens(
    tmp_path,
):
    """The per-kitchen catch is for a SCENE that cannot satisfy the template. It is not for a bug.

    A plan_fn that returns something json cannot serialise (a numpy array — the real case; here, an
    object) is wrong for every kitchen alike. Under `except Exception` the batch reported twelve
    identical "failures" and exited normally: one systematic bug in the caller's code, dressed up as
    twelve bad kitchens, in a list a human skims. It must stop the batch."""
    def unserialisable_plan(skill_id, action, params):
        return [object()] * 7               # stands in for the numpy array json.dumps chokes on

    kitchens = {V813: v813_bindings(), "Isaac-Kitchen-v813-01": v813_bindings()}
    with pytest.raises(TypeError):
        emit(bowl_to_drawer(), kitchens, unserialisable_plan, out_dir=tmp_path)


def test_the_real_goals_directory_is_never_written_into(tmp_path):
    """Protect the shipped goal examples against direct and symlinked output paths."""
    args = (bowl_to_drawer(), {V813: v813_bindings()}, stub_plan)
    before = {p.name: p.stat().st_mtime for p in GOALS.iterdir()}
    assert f"{V813}.json" in before, "the bundled example must exist for this guard"

    with pytest.raises(UnsafeOutputDir, match="goals"):
        emit(*args, out_dir=GOALS)
    with pytest.raises(UnsafeOutputDir, match="goals"):
        emit(*args, out_dir=GOALS / "v2")                  # nor a subdirectory of it

    # And not by the back door. `goals` is itself a symlink, so a check on the literal path is
    # bypassed by either end of it: pass the directory it points AT, or a fresh symlink of your own.
    with pytest.raises(UnsafeOutputDir, match="goals"):
        emit(*args, out_dir=GOALS.resolve())

    sneaky = tmp_path / "somewhere_harmless"
    sneaky.symlink_to(GOALS.resolve(), target_is_directory=True)
    with pytest.raises(UnsafeOutputDir, match="goals"):
        emit(*args, out_dir=sneaky)

    assert {p.name: p.stat().st_mtime for p in GOALS.iterdir()} == before, (
        "a goal file was created, overwritten or touched — which is the thing being prevented"
    )


def test_one_bad_kitchen_does_not_take_the_batch_down(tmp_path):
    """The whole reason emit() reports instead of raising: a fan-out over twelve rotations of a
    kitchen must not lose the eleven that plan cleanly. And the one it loses must be NAMED — a
    batch that silently writes 11 of 12 files reads as 'it covered everything'."""
    def fails_on_one(skill_id, action, params):
        if params.get("prim_path") == "/world/bowl9":
            raise RuntimeError("bowl9 is inside a wall")
        return stub_plan(skill_id, action, params)

    bad = dict(v813_bindings())
    bad["target"] = BoundPrim(prim_path="/world/bowl9", is_rigid_body=True, object_index=9)

    written, failed = emit(
        bowl_to_drawer(),
        {V813: v813_bindings(), "Isaac-Kitchen-v813-01": bad},
        fails_on_one,
        out_dir=tmp_path,
    )
    assert [Path(p).name for p in written] == [f"{V813}.json"]
    assert len(failed) == 1 and "Isaac-Kitchen-v813-01" in failed[0] and "wall" in failed[0]


def test_the_batch_takes_the_bindings_bind_many_produces(tmp_path):
    """End to end from a KitchenScene, not from a hand-written bindings dict: bind_many is what the
    CLI will call, and its output is what emit consumes."""
    scenes = {V813: kitchen([("mug", "/world/mug0"), ("bowl", "/world/bowl0")])}
    bound, skipped = bind_many(bowl_to_drawer(), scenes)
    assert skipped == []

    written, failed = emit(bowl_to_drawer(), bound, stub_plan, out_dir=tmp_path)
    assert failed == []
    steps, meta = goal_format.read(written[0], known_skills=set(REGISTRY))
    assert steps[1].params["prim_path"] == "/world/bowl0"
    assert steps[4].params["prim_path"] == "/world/base_cabinet/drawer_0_0/door_handle"
    assert meta["run_config"]["obj_name"] == "bowl0"


def test_a_sequence_that_cannot_run_is_never_written(tmp_path):
    """validate_sequence guards the batch, not each kitchen: an incoherent script is a bug in the
    composition, and writing eleven copies of it before saying so helps nobody."""
    from task_validate import SequenceError

    t = bowl_to_drawer()
    del t.steps[2]                              # the close that picks the bowl up
    t.subtask_groups = []

    with pytest.raises(SequenceError):
        emit(t, {V813: v813_bindings()}, stub_plan, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [], "nothing may be written before the script is validated"


def test_a_language_that_is_a_path_never_reaches_a_goal_file(tmp_path):
    """The exposure the composer creates. `task_language` is pasted into the LeRobot output path
    (simvla_gen.py:3217, f"{SIMVLA_LEROBOT_ROOT}/{task_language}/{task}"), and a v2 goal file now
    CARRIES it — derive_run_config copies t.language straight in. A GUI user writing "Put the bowl in
    the drawer / fridge" would emit twelve goal files that each record their dataset into a nested
    directory, and nothing would say so. emit() validates the template before it opens a file."""
    t = bowl_to_drawer()
    t.language = "Put the bowl in the drawer / fridge"

    with pytest.raises(TaskTemplateError, match="language"):
        emit(t, {V813: v813_bindings()}, stub_plan, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [], "not one file may be written with a path for a label"


def test_the_language_the_goal_file_carries_is_the_one_the_dataset_was_labelled_with(tmp_path):
    """And the legal one still goes all the way through, period intact."""
    written, failed = emit(bowl_to_drawer(), {V813: v813_bindings()}, stub_plan, out_dir=tmp_path)
    assert failed == []
    _, meta = goal_format.read(written[0], known_skills=set(REGISTRY))
    assert meta["run_config"]["task_language"] == "Put bowl inside drawer."


def test_a_disabled_grasp_check_is_never_asked_for_successes(tmp_path):
    """The coupling that spun a run for two hours, carried into the file the run actually reads."""
    written, _ = emit(
        bowl_to_drawer(), {V813: v813_bindings()}, stub_plan, out_dir=tmp_path,
        sub_good_goal_count=7,
    )
    _, meta = goal_format.read(written[0], known_skills=set(REGISTRY))
    cfg = meta["run_config"]
    assert cfg["sub_grasp_idx_l"] == GRASP_CHECK_DISABLED and cfg["sub_good_goal_count_l"] == 0
    assert cfg["sub_grasp_idx_r"] == 2 and cfg["sub_good_goal_count_r"] == 7


# ---------------------------------------------------------------------------------------------
# The checked-in template, and the CLI's Omniverse-free helpers. main() itself is never run here —
# it boots Omniverse — so what is tested is everything that decides whether a run can even start.
# ---------------------------------------------------------------------------------------------

TEMPLATES = HERE / "templates"


def test_aiworker_left_mug_template_keeps_loaded_arm_at_its_lifted_pose():
    template = load_template(
        HERE.parents[1] / "examples" / "experimental"
        / "mug_counter_to_table_aiworker_left.json")
    assert [(step.skill, step.action) for step in template.steps[1:4]] == [
        ("arm.grasp", "A_l"), ("gripper.set", "G_l"), ("arm.pause", "A_l")]
    assert [(step.skill, step.action) for step in template.steps[-2:]] == [
        ("arm.reset", "A_l"), ("arm.reset", "A_l")]


def test_the_checked_in_template_round_trips_to_the_fixture():
    """templates/bowl_to_drawer.json must BE to_json(bowl_to_drawer()) — not a hand-written lookalike.
    A real round-trip means the file on disk can never drift from the fixture the reproduction test
    holds the whole composer to: change the fixture without regenerating the file and this fails."""
    on_disk = from_json((TEMPLATES / "bowl_to_drawer.json").read_text())
    assert on_disk == bowl_to_drawer()
    # And it is the writer's own output, byte for byte (modulo a trailing newline).
    assert (TEMPLATES / "bowl_to_drawer.json").read_text().rstrip("\n") == to_json(bowl_to_drawer())


def test_parse_kitchens_reads_a_comma_list():
    assert parse_kitchens("813,422") == [813, 422]
    assert parse_kitchens("813") == [813]
    assert parse_kitchens(" 813 , 422 ") == [813, 422], "whitespace is tolerated"
    assert parse_kitchens("813,,422") == [813, 422], "an empty entry is skipped, not zero"


@pytest.mark.parametrize("spec", ["", " ", ",", "813,x", "8.1", "v813"])
def test_parse_kitchens_refuses_junk_rather_than_dropping_it(spec):
    """A kitchen number that is not an integer is refused loudly. Silently dropping it would emit
    fewer files than asked and read as 'those kitchens had no matching object'."""
    with pytest.raises(ValueError):
        parse_kitchens(spec)


def test_parse_subs_selects_explicit_rotations_and_preserves_full_fanout_by_default():
    assert parse_subs(None) is None
    assert parse_subs("0") == [0]
    assert parse_subs(" 0, 11 ") == [0, 11]


@pytest.mark.parametrize("spec", ["", "0,", "0,x", "-1", "12", "0,0"])
def test_parse_subs_rejects_invalid_or_ambiguous_rotations(spec):
    with pytest.raises(ValueError):
        parse_subs(spec)


def test_kitchen_task_name_maps_a_number_to_its_first_rotation():
    assert kitchen_task_name(813) == "Isaac-Kitchen-v813-00"
    assert kitchen_task_name(42) == "Isaac-Kitchen-v42-00"


def test_load_template_validates_and_returns_the_fixture():
    """The checked-in template loads through the CLI's own loader and passes both validators."""
    assert load_template(TEMPLATES / "bowl_to_drawer.json") == bowl_to_drawer()


def test_load_template_refuses_an_incoherent_script_before_omniverse_boots(tmp_path):
    """A template with a path for a language is refused here, in milliseconds — not after a
    three-minute boot. It is the same TaskTemplateError emit() would raise, brought forward."""
    t = bowl_to_drawer()
    t.language = "Put the bowl in the drawer / fridge"
    bad = tmp_path / "bad.json"
    bad.write_text(to_json(t))
    with pytest.raises(TaskTemplateError, match="language"):
        load_template(bad)


def test_load_template_refuses_a_sequence_that_cannot_run(tmp_path):
    from task_validate import SequenceError

    t = bowl_to_drawer()
    del t.steps[2]                       # the close that picks the bowl up
    t.subtask_groups = []
    bad = tmp_path / "bad.json"
    bad.write_text(to_json(t))
    with pytest.raises(SequenceError):
        load_template(bad)


def test_guard_out_dir_refuses_the_goals_corpus_up_front():
    """The same refusal emit() makes, run before Omniverse boots so `--out .../goals` fails fast."""
    with pytest.raises(UnsafeOutputDir, match="goals"):
        guard_out_dir(GOALS)
    with pytest.raises(UnsafeOutputDir, match="goals"):
        guard_out_dir(GOALS.resolve())


def test_guard_out_dir_returns_a_safe_path_untouched(tmp_path):
    out = tmp_path / "composed"
    assert guard_out_dir(out) == out
    assert not out.exists(), "the guard does not create anything; emit() does that"


def test_arg_parser_requires_all_three_flags():
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--template", "t.json", "--kitchens", "813,422", "--out", "/tmp/x"]
    )
    assert args.template == "t.json"
    assert args.kitchens == "813,422"
    assert args.out == "/tmp/x"
    assert args.subs is None
    assert parser.parse_args(
        ["--template", "t.json", "--kitchens", "813", "--subs", "0,2", "--out", "/tmp/x"]
    ).subs == [0, 2]

    for missing in (
        ["--kitchens", "813", "--out", "/tmp/x"],
        ["--template", "t.json", "--out", "/tmp/x"],
        ["--template", "t.json", "--kitchens", "813"],
    ):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(missing)


# ---------------------------------------------------------------------------------------------
# The import that must not happen.
# ---------------------------------------------------------------------------------------------


def test_module_is_stdlib_only():
    """It must be unit-testable with no GPU and no Omniverse. Parse the AST, so a function-local
    third-party import cannot slip past."""
    module_path = HERE / "task_emit.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    allowed = set(sys.stdlib_module_names) | {
        "goal_format", "predicate_contract", "skill_contract", "task_bind", "task_runconfig",
        "task_template", "task_validate",
    }
    assert not (roots - allowed), f"task_emit.py imports {roots - allowed}"


def test_importing_it_boots_nothing():
    """In a clean interpreter — not this one, where `import skills` has already pulled torch in.
    The emitter runs in the AUTHORING process, which has Omniverse; the point is that it does not
    NEED it, because everything that does is behind plan_fn. An import that dragged pxr in would
    make the composition logic untestable again, which is the whole reason plan_fn is injected."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import task_emit                                   # and nothing else

            banned = ("omni", "isaaclab", "pxr", "torch", "trimesh", "numpy", "tkinter", "PIL",
                      "scipy", "shapely")
            pulled = sorted(m for m in sys.modules if m.split(".")[0] in banned)
            assert not pulled, f"importing task_emit pulled in {pulled}"
            print("clean")
        """)],
        cwd=str(HERE), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.stdout.split() == ["clean"]


def test_main_populates_the_registry_before_it_validates_the_template():
    """main() validates the template (load_template) BEFORE it boots Omniverse. validate_template
    checks each step's skill against skill_contract.REGISTRY, which the @skill decorators fill only on
    `import skills`. If main() doesn't load them first, it refuses EVERY real template with 'no skill
    <id> in the registry' before Omniverse is reached — the CLI never ran a template end-to-end.

    A subprocess, because THIS module imports skills at the top (so an in-process check sees a
    registry that is already populated, exactly the blind spot that let the bug ship). Prove: a bare
    import leaves the real template unvalidatable, and _load_skill_registry() (what main() calls) fixes
    it. The reference template ships in the repo, so this needs no fixture."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import task_emit                                   # and NOT skills
            tmpl = "templates/bowl_to_drawer.json"

            try:
                task_emit.load_template(tmpl)
                raise SystemExit("expected an empty-registry failure before _load_skill_registry()")
            except Exception as exc:
                assert "in the registry" in str(exc), f"unexpected error: {exc!r}"

            task_emit._load_skill_registry()                  # what main() does before load_template
            t = task_emit.load_template(tmpl)                 # now the skills resolve
            assert [s.skill for s in t.steps], "template validated but has no steps"
            print("validated", len(t.steps))
        """)],
        cwd=str(HERE), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.stdout.split()[0] == "validated" and int(proc.stdout.split()[1]) > 0


# ---------------------------------------------------------------------------------------------
# composed conditions reach the emitted env config.
# ---------------------------------------------------------------------------------------------

GOOD_SUCCESS = {"all": [
    {"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.2}},
    {"obj_z": {"role": "@target", "lo": 0.90}},
    {"last_subtask": {}},
]}


def _grasp_template(**over):
    """A minimal one-object grasp task. Repeated here rather than imported from
    test_task_template.py: test modules do not import each other, and a reader of this file should
    not have to open another one."""
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


def test_terminations_block_emits_both_terms_as_data():
    text = terminations_block(
        {"all": [{"obj_z": {"role": "bottle0", "lo": 0.9}}]},
        {"any": [{"robot_fell": {"z": -0.1}}]},
    )
    assert "func=mdp.composed" in text
    assert '"role": "bottle0"' in text
    assert "success" in text and "retry" in text
    assert "mdp.task2" not in text, "the drawer check must not survive"


def test_terminations_block_is_valid_python():
    """It is pasted into a module that must import."""
    import ast
    src = "class T:\n" + "\n".join(
        "    " + line for line in terminations_block(
            {"all": [{"obj_z": {"role": "bottle0", "lo": 0.9}}]},
            {"any": [{"robot_fell": {"z": -0.1}}]},
        ).splitlines()
    )
    ast.parse(src)


def test_terminations_block_renders_bools_as_python_not_json_literals():
    """json.dumps writes JSON's true/false, which parses fine as Python SYNTAX (a bare NAME) but is
    undefined at runtime — the generated env config would NameError at import, not at emit, the
    first time a composed spec carries an explicit Bool. obj_near_prim.xy_only
    (predicate_contract.py) is the one Bool param a composed spec can carry today; nothing in the
    checked-in template exercises it, so it is exercised here rather than discovered later on a
    GPU node."""
    spec = {"all": [{"obj_near_prim": {
        "role": "bottle0", "target_role": "base_cabinet", "xy_only": True,
    }}]}
    text = terminations_block(spec, {"any": [{"robot_fell": {"z": -0.1}}]})
    assert "true" not in text, "JSON's lowercase true is not a Python name"
    assert '"xy_only": True' in text

    class _FakeDoneTerm:                    # a stand-in that just remembers its kwargs
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ns: dict = {"mdp": type("mdp", (), {"composed": None}), "DoneTerm": _FakeDoneTerm}
    exec("class T:\n" + "\n".join("    " + l for l in text.splitlines()), ns)
    resolved = ns["T"].success.kwargs["params"]["spec"]["all"][0]["obj_near_prim"]["xy_only"]
    assert resolved is True, "a Python bool, not the string 'true' or an undefined name"


_RETRY = {"any": [{"robot_fell": {"z": -0.1}}]}


def _exec_terminations(text):
    """The block's two specs, as the generated env config's import would build them. exec, not a
    text assertion: what matters is the VALUE the config ends up with."""
    class _FakeDoneTerm:                    # a stand-in that just remembers its kwargs
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ns: dict = {"mdp": type("mdp", (), {"composed": None}), "DoneTerm": _FakeDoneTerm}
    exec("class T:\n" + "\n".join("    " + l for l in text.splitlines()), ns)
    return (ns["T"].success.kwargs["params"]["spec"], ns["T"].retry.kwargs["params"]["spec"])


@pytest.mark.parametrize("body", [
    'x": true',       # THE HOLE: json.dumps escapes the quote, so this dumps as '"x\\": true"' and
    'x": false',      # a textual '": true' -> '": True' rewrite fires INSIDE the string. The value
    'x": null',       # round-trips as `x": True` — valid Python, silently wrong, nothing raised.
    "true", "false", "null",              # the substring cases the textual rewrite already handled
    "drawer_true_0", "nullify", "false_front",
    'quote"inside', "back\\slash", "line\nbreak", "tab\there",
    "dräwer",                             # json.dumps escapes it as \u00e4
])
def test_a_string_param_reaches_the_config_with_exactly_its_own_value(body):
    """The spec is DATA; a body/joint name is matched against the articulation's own names, so a
    single character changed in transit means the leaf silently matches nothing (or the wrong
    thing). Compared as VALUES after exec, which is the only form that catches a rewrite that
    happens to still parse."""
    spec = {"all": [{"obj_near_prim": {
        "role": "bowl0", "target_role": "base_cabinet", "body": body, "radius": 0.1,
    }}]}
    success, retry = _exec_terminations(terminations_block(spec, _RETRY))
    assert success == spec
    assert retry == _RETRY


def test_the_rendered_literal_is_still_the_json_text_where_json_is_valid_python():
    """The rewrite went structural; the OUTPUT must not drift. Every spec with no bool/None renders
    byte-for-byte as json.dumps did, so regenerating an existing env config stays a no-op."""
    spec = {"all": [{"obj_z": {"role": "bowl0", "lo": 0.6, "hi": 0.8}},
                    {"joint_pos": {"role": "base_cabinet", "joint": "corpus_to_drawer_0_0",
                                   "hi": 0.03}},
                    {"last_subtask": {}}]}
    assert json.dumps(spec) in terminations_block(spec, _RETRY)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_number_never_reaches_the_generated_config(value):
    """json.dumps writes Infinity/-Infinity/NaN and json.loads reads them back, so a hand-written
    template CAN carry one (it is a real float, so validate_spec's type check passed until it was
    taught otherwise). In a .py file those are undefined NAMEs — the config would NameError at
    import. validate_spec refuses them at authoring; this is the writer's own backstop, for a spec
    that reaches it unvalidated."""
    with pytest.raises(SpecError, match="non-finite"):
        terminations_block({"all": [{"obj_z": {"role": "bowl0", "lo": value}}]}, _RETRY)


def test_the_generated_config_text_is_inserted_verbatim_not_as_a_regex_template():
    """simvla_data_generator._write_env_config_file splices the generated blocks in with re.sub,
    whose `repl` is a TEMPLATE: backslashes in it are re-read. '\\u' — which json.dumps writes for
    every non-ASCII character — raises `re.error: bad escape \\u`, and a literal backslash silently
    collapses. Worse, _generate_env_config wraps the call in a blanket `except Exception:
    self.log(...)`, so the raise is swallowed and the kitchen is recorded as written while its
    config is stale or absent. A callable `repl` is inserted verbatim.

    Structural (ast): the module imports pxr/omni/tkinter at its top and cannot be imported here.
    """
    import re as _re
    src = (HERE / "simvla_data_generator.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_write_env_config_file"), None)
    assert fn, "simvla_data_generator no longer defines _write_env_config_file"
    subs = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "sub"]
    assert subs, "the splice is no longer done with re.sub — re-check this test"
    for call in subs:
        assert isinstance(call.args[1], ast.Lambda), (
            "re.sub's replacement must be a callable, not the generated text itself: "
            + ast.dump(call.args[1])[:120]
        )

    # And the hazard is reachable through the writer, not hypothetical:
    text = terminations_block(
        {"all": [{"obj_near_prim": {"role": "bowl0", "target_role": "base_cabinet",
                                    "body": "dräwer", "radius": 0.1}}]}, _RETRY)
    assert "\\u" in text, "json.dumps escapes non-ASCII, so the replacement text carries a backslash"
    with pytest.raises(_re.error):
        _re.sub("X", text, "X")          # what passing it as a template would have done


def test_generated_env_config_can_be_routed_outside_the_source_checkout():
    """Public goal regeneration must not overwrite tracked/generated task configs in-place."""
    src = (HERE / "simvla_data_generator.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_write_env_config_file"), None)
    assert fn, "simvla_data_generator no longer defines _write_env_config_file"
    writer = ast.get_source_segment(src, fn)
    assert 'SIMVLA_ENV_CFG_OUTPUT_DIR' in writer
    assert 'os.environ.get' in writer


PROXIMITY_TRAP = {"all": [{"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.2}}]}


def test_load_template_surfaces_the_proximity_trap(tmp_path, capsys):
    """spec_warnings reached exactly one caller — the composer's POST /template — so the identical
    condition in a HAND-WRITTEN template was emitted in silence. The trap is the one that once
    scored a non-grasping policy 4/20: an object on the ~0.82 m counter is within 0.2 m of a
    passing gripper. Warned, never refused: this cannot know the counter height."""
    t = dataclasses.replace(bowl_to_drawer(), success=PROXIMITY_TRAP)
    path = tmp_path / "trap.json"
    path.write_text(to_json(t))
    loaded = load_template(path)                     # NOT refused
    assert loaded.success == PROXIMITY_TRAP
    assert "4/20" in capsys.readouterr().err


def test_emit_surfaces_the_proximity_trap(tmp_path, capsys):
    """The other door into the writer: a caller that built the template in memory and never went
    through load_template."""
    t = dataclasses.replace(bowl_to_drawer(), success=PROXIMITY_TRAP)
    written, failed = emit(t, {V813: v813_bindings()}, stub_plan, out_dir=tmp_path)
    assert written and not failed
    assert "4/20" in capsys.readouterr().err


def test_a_condition_with_a_lift_check_emits_silently(tmp_path, capsys):
    """The warning must be about the trap, not about composed conditions in general."""
    emit(bowl_to_drawer(), {V813: v813_bindings()}, stub_plan, out_dir=tmp_path)
    assert "4/20" not in capsys.readouterr().err


def test_default_retry_uses_the_first_rigid_body_role():
    """Not a role that happens to be named 'target' — the same prim _target_prim already picks."""
    t = _grasp_template()          # from the template tests; roles = [target: bottle]
    bindings = {"target": BoundPrim(prim_path="/world/bottle0", is_rigid_body=True, object_index=0)}
    spec = default_retry(t, bindings)
    assert spec == {"any": [
        {"obj_z": {"role": "bottle0", "hi": 0.3}},
        {"robot_fell": {"z": -0.1}},
    ]}


def test_default_retry_without_a_rigid_role_is_robot_fell_alone():
    t = _grasp_template(roles=[Role(name="container", articulation_with="drawer")], success={
        "all": [{"joint_pos": {"role": "@container", "joint": 0, "lo": 0.1}}]})
    bindings = {"container": BoundPrim(prim_path="/world/base_cabinet",
                                       is_rigid_body=False, object_index=-1)}
    assert default_retry(t, bindings) == {"any": [{"robot_fell": {"z": -0.1}}]}


def test_an_articulation_role_resolves_to_the_articulation_the_scene_actually_has():
    """@container binds to /world/base_cabinet/drawer_0_0, and `drawer_0_0` is NOT a scene entity.

    mdp.composed.build_context looks a spec's role up in env.scene.rigid_objects /
    env.scene.articulations, and _generate_env_config enumerates FIRST-LEVEL prims only
    (len(path.split("/")) == 3) — so the generated config declares `base_cabinet =
    ArticulationCfg(...)` (whose joints include corpus_to_drawer_0_0) and nothing called
    `drawer_0_0`. Resolving @container to the last path segment emitted a spec that raised
    "the composed condition references 'drawer_0_0'" on the FIRST termination evaluation of every
    kitchen generated from the checked-in template — in the generator and in eval both, so zero
    demos. The checked-in template is the one that shipped it, so it is the one held here."""
    success, _retry = resolved_condition(bowl_to_drawer(), v813_bindings())
    leaves = success["all"]
    assert {"obj_z": {"role": "bowl0", "lo": 0.6, "hi": 0.8}} in leaves, "a rigid role is its prim"
    assert {"joint_pos": {"role": "base_cabinet", "joint": "corpus_to_drawer_0_0", "hi": 0.03}} \
        in leaves, "the articulation, not the drawer segment inside it"
    assert not any("drawer_0_0" == p.get("role") for leaf in leaves for p in leaf.values()
                   if isinstance(p, dict)), "no leaf may name a prim the scene has no entity for"


def test_the_resolved_condition_names_only_entities_the_generated_config_declares():
    """The rule stated as the generator states it: an entity is a FIRST-LEVEL prim. Every role
    operand the resolution produces must be one — for the handle role too, whose prim is two levels
    below its articulation (/world/base_cabinet/drawer_0_0/door_handle -> base_cabinet)."""
    bindings = v813_bindings()
    first_level = {b.prim_path.split("/")[2] for b in bindings.values()}
    for role, b in bindings.items():
        assert b.scene_entity in first_level, role
    assert bindings["container_handle"].scene_entity == "base_cabinet"
    assert bindings["container"].scene_entity == "base_cabinet"
    assert bindings["target"].scene_entity == "bowl0" == bindings["target"].name

    t = bowl_to_drawer()
    success, retry = resolved_condition(t, bindings)
    for spec in (success, retry):
        for _pid, params in _leaf_params(spec):
            for key in ("role", "target_role"):
                if key in params:
                    assert params[key] in first_level, f"{params[key]!r} is no scene entity"


def _leaf_params(spec):
    from predicate_contract import leaves
    return leaves(spec)


# ---------------------------------------------------------------------------------------------
# Via-points: a routed nav is a chain of nav steps
# ---------------------------------------------------------------------------------------------

def routed_plan(skill_id, action, params):
    """A nav planner that had to route around furniture: the park pose plus two via-points."""
    import nav_clearance as nc
    if skill_id == "nav.to_prim":
        return nc.NavRoute([1.0, 2.0, 0.0], via=[(0.5, 0.0, 1.57), (0.5, 2.0, 0.0)])
    return stub_plan(skill_id, action, params)


def test_a_routed_nav_is_written_as_via_point_steps_before_the_park():
    """The run's nav is turn-drive-turn per step and the runtime advances to the next step when
    one finishes, so a detour is a chain of nav steps: the via-points first, the park last, each
    the same skill/action/params as the step the template wrote."""
    t = bowl_to_drawer()
    steps = steps_for_kitchen(t, v813_bindings(), routed_plan)
    assert len(steps) == len(t.steps) + 2
    assert [s.skill for s in steps[:3]] == ["nav.to_prim"] * 3
    assert steps[0].goal == [0.5, 0.0, 1.57]
    assert steps[1].goal == [0.5, 2.0, 0.0]
    assert steps[2].goal == [1.0, 2.0, 0.0]
    assert type(steps[2].goal) is list                      # a plain list in the file
    assert steps[0].action == steps[2].action == t.steps[0].action
    assert steps[0].params == steps[1].params == steps[2].params
    assert "via" in steps[0].language.lower() and "via" in steps[1].language.lower()
    assert steps[2].language == t.steps[0].language
    assert [s.skill for s in steps[3:]] == [s.skill for s in t.steps[1:]]


def test_a_nav_without_a_route_is_written_exactly_as_before():
    import nav_clearance as nc

    def plain(skill_id, action, params):
        if skill_id == "nav.to_prim":
            return nc.NavRoute([1.0, 2.0, 0.0])
        return stub_plan(skill_id, action, params)

    t = bowl_to_drawer()
    steps = steps_for_kitchen(t, v813_bindings(), plain)
    assert len(steps) == len(t.steps)
    assert steps[0].goal == [1.0, 2.0, 0.0] and type(steps[0].goal) is list


def test_the_expanded_template_shifts_step_indices_and_subtask_groups():
    """Everything the run config derives -- the grasp step, the grasp-check step, the export
    groups -- is a STEP INDEX. Via-point steps go before their park, so every later index moves
    by the number inserted, and the via-points join their park's subtask group."""
    from task_emit import plan_steps
    t = bowl_to_drawer()
    n = len(t.steps)
    t.subtask_groups = [[0, 1, 2], list(range(3, 7)), list(range(7, n))]
    steps, t2 = plan_steps(t, v813_bindings(), routed_plan)
    assert len(t2.steps) == n + 2 == len(steps)
    assert [s.skill for s in t2.steps] == [s.skill for s in steps]
    assert t2.subtask_groups == [[0, 1, 2, 3, 4], list(range(5, 9)), list(range(9, n + 2))]
    assert t2.name == t.name and t2.roles == t.roles and t2.success == t.success


def test_the_run_config_indices_follow_the_via_point_steps():
    from task_emit import plan_steps
    from task_runconfig import derive_run_config
    t = bowl_to_drawer()
    plain_cfg = derive_run_config(t, v813_bindings())
    _, t2 = plan_steps(t, v813_bindings(), routed_plan)
    cfg = derive_run_config(t2, v813_bindings())
    assert cfg.target_idx == plain_cfg.target_idx + 2
    if plain_cfg.sub_grasp_idx_r != GRASP_CHECK_DISABLED:
        assert cfg.sub_grasp_idx_r == plain_cfg.sub_grasp_idx_r + 2
    assert t2.steps[cfg.target_idx].skill == t.steps[plain_cfg.target_idx].skill


def test_the_written_file_carries_the_via_steps_and_matching_indices(tmp_path):
    from task_emit import emit_one
    t = bowl_to_drawer()
    path = emit_one(t, "Isaac-Kitchen-v813-00", v813_bindings(), routed_plan, tmp_path,
                    kitchen_meta=demonstrated_meta())
    raw = json.loads(Path(path).read_text())
    goals = raw["goals"][0]
    assert [g["skill"] for g in goals[:3]] == ["nav.to_prim"] * 3
    rc = raw["run_config"]
    grasp_steps = [i for i, g in enumerate(goals) if g["skill"] == "arm.grasp"]
    assert rc["target_idx"] == grasp_steps[0]



def test_expand_route_is_what_both_the_emitter_and_the_gui_save_write():
    """One function turns a step whose goal is a NavRoute into via steps plus the park step, so
    the GUI's save (which plans per step and writes v2 directly) writes the same chain the
    emitter does."""
    import nav_clearance as nc
    step = goal_format.Step(skill="nav.to_prim", action="N_s", params={"prim_path": "/world/mug0"},
                            goal=nc.NavRoute([1.0, 2.0, 0.0], via=[(0.5, 0.0, 1.57)]), language="Drive")
    out = goal_format.expand_route(step)
    assert [s.goal for s in out] == [[0.5, 0.0, 1.57], [1.0, 2.0, 0.0]]
    assert all(type(s.goal) is list for s in out)
    assert out[0].language == "Drive (via-point 1 of 1)" and out[1].language == "Drive"
    plain = goal_format.Step(skill="arm.grasp", action="A_r", params={}, goal=[[1, 2, 3]], language="g")
    assert goal_format.expand_route(plain) == [plain]
