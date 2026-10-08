"""The goal file carries its own run config; the CLI overrides it.

The expected values in test_the_real_v813_run_config_merges_to_the_nine_known_good_values are not
opinions. That exact configuration was run on a GPU node and produced a 960-frame LeRobot dataset
with EXIT=0. If the merge disagrees, the merge is wrong.

Run: pytest scripts/simvla/test_run_config_merge.py -v
"""

import ast
import subprocess
import sys
import textwrap
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import executor_dispatch
import skills  # noqa: F401  — populates skill_contract.REGISTRY
from run_config_merge import (
    FIELD_TYPES,
    RUN_CONFIG_FIELDS,
    RunConfigError,
    merge_run_config,
    parse_export_groups,
    validate_grasp_check_steps,
)
from task_runconfig import derive_run_config
from scripts.simvla.test_task_runconfig import v813_bindings
from scripts.simvla.test_task_template import bowl_to_drawer

HERE = Path(__file__).parent

META = {
    "run_config": {
        "obj_name": "bowl0",
        "obj_name_l": "none",
        "target_idx": 1,
        "sub_grasp_idx_r": 4,
        "sub_grasp_idx_l": 999,
        "sub_good_goal_count_l": 0,
        "export_groups": "4,5,6,7;8,9,10;11",
        "task_language": "Put bowl inside drawer.",
        "task_type": "NavManipulation",
    }
}


def args_with_defaults() -> Namespace:
    """A COMPLETE namespace, as argparse produces: merge_run_config reads `vars(args)` as argparse's
    own vocabulary — it is what tells `--task` from an abbreviation of `--task_language` — and it
    says so, and refuses a partial one (test_a_partial_namespace_is_refused)."""
    return Namespace(
        task="Isaac-Kitchen-v813-00",
        obj_name="mug0", obj_name_l="none", target_idx=1, sub_grasp_idx_r=4,
        sub_grasp_idx_l=9, sub_good_goal_count_r=7, sub_good_goal_count_l=7, export_groups=None,
        task_language="Put bottle to the sink.", task_type="LocoManipulation",
    )


def test_the_goal_file_supplies_what_the_cli_did_not():
    args = args_with_defaults()
    overrode = merge_run_config(args, META, argv=["simvla_gen.py", "--task", "Isaac-Kitchen-v813-00"])

    assert args.obj_name == "bowl0", "the default mug0 must not survive; the goal file knows better"
    assert args.sub_grasp_idx_l == 999
    assert args.sub_good_goal_count_l == 0
    assert args.export_groups == [[4, 5, 6, 7], [8, 9, 10], [11]]   # the flag's own type=, applied
    assert args.task_type == "NavManipulation"
    assert "obj_name" in overrode


def test_an_explicit_flag_wins_over_the_goal_file():
    args = args_with_defaults()
    args.obj_name = "mug0"
    merge_run_config(args, META, argv=["simvla_gen.py", "--obj_name", "mug0"])
    assert args.obj_name == "mug0", "the user said so on the command line"


def test_an_explicit_flag_equal_to_the_default_still_wins():
    """Detecting 'explicitly passed' by comparing to the argparse default would silently ignore a
    user who typed the default value on purpose."""
    args = args_with_defaults()
    merge_run_config(args, META, argv=["simvla_gen.py", "--target_idx", "1"])
    assert args.target_idx == 1
    # And a *different* field is still taken from the goal file.
    assert args.obj_name == "bowl0"


def test_a_v1_goal_file_changes_nothing():
    """The v1 files have no run_config. They must keep working exactly as before."""
    args = args_with_defaults()
    overrode = merge_run_config(args, {"version": 1}, argv=["simvla_gen.py"])
    assert overrode == []
    assert args.obj_name == "mug0"


@pytest.mark.parametrize("empty", [{}, [], "", 0, None])
def test_a_run_config_block_that_is_present_and_empty_is_not_a_v1_file(empty):
    """ABSENT and EMPTY are different facts, and `if not config` could not tell them apart.

    No block at all is v1 — every goal file that has ever been run, including the one that recorded
    960 frames — and it merges as a no-op. A block that is THERE and says nothing is a v2 file whose
    writer produced nothing: to_meta() writes all ten fields, so nothing here can emit one, which is
    the whole reason to refuse it before something can. Silently treating it as v1 would run the task
    on argparse's defaults — mug0 on a bowl, export_groups=None, no LeRobot export at all — and the
    dataset would be recorded before anyone found out."""
    args = args_with_defaults()
    with pytest.raises(RunConfigError, match="run_config"):
        merge_run_config(args, {"run_config": empty}, argv=["simvla_gen.py"])
    assert args.obj_name == "mug0", "and it must not have merged half of one first"


def test_a_partial_namespace_is_refused():
    """`vars(args)` is argparse's vocabulary, and reading it is what stops `--task` being expanded as
    an abbreviation of task_language/task_type. A Namespace that is not the one argparse parsed makes
    that reading a lie, silently: the merge is only correct for the FULL namespace, so it says so."""
    partial = Namespace(obj_name="mug0", task_language="Put bottle to the sink.")
    with pytest.raises(RunConfigError, match="sub_grasp_idx_r"):
        merge_run_config(partial, META, argv=["simvla_gen.py", "--task", "x"])


def test_a_v1_file_is_a_no_op_even_for_a_caller_that_could_not_be_merged_into():
    """v1 reaches no setattr and asks the namespace nothing. It is the path every recorded run took,
    and no check added for v2's sake may fire on it."""
    partial = Namespace(obj_name="mug0")
    assert merge_run_config(partial, {"version": 1}, argv=["simvla_gen.py"]) == []
    assert vars(partial) == {"obj_name": "mug0"}


@pytest.mark.parametrize(
    ("filename", "object_name", "object_name_l", "grasp_side"),
    [
        ("Isaac-Kitchen-v813-00.json", "bowl0", "none", "r"),
        ("Isaac-Kitchen-v813-00.grasp12.json", "bowl0", "none", "r"),
        ("Isaac-Kitchen-v813-00.banked.json", "bowl0", "none", "r"),
        ("Isaac-Kitchen-v813r-00.json", "mug0", "none", "r"),
        ("Isaac-Kitchen-v813a-00.json", "none", "mug0", "l"),
        ("Isaac-Kitchen-v813a-00.grasp0.json", "none", "mug0", "l"),
    ],
)
def test_the_public_v813_goal_files_load_and_merge_their_run_config(
    filename, object_name, object_name_l, grasp_side
):
    """Every shipped robot goal has a valid grasp gate and self-contained run config."""
    script, goal_meta, _ = executor_dispatch.load_script(
        HERE.parents[1] / "examples" / "goals" / filename
    )
    assert goal_meta["run_config"][f"sub_grasp_idx_{grasp_side}"] == 2

    args = simvla_gen_defaults()
    overrode = merge_run_config(
        args, goal_meta,
        argv=["simvla_gen.py", "--task", Path(filename).stem],
    )
    assert f"sub_grasp_idx_{grasp_side}" in overrode
    assert args.obj_name == object_name
    assert args.obj_name_l == object_name_l
    assert getattr(args, f"sub_grasp_idx_{grasp_side}") == 2
    validate_grasp_check_steps(args, script)


def test_anubis_single_grasp_fixture_is_the_banked_authored_candidate():
    original, _, _ = executor_dispatch.load_script(
        HERE.parents[1] / "examples" / "goals" / "Isaac-Kitchen-v813-00.json"
    )
    selected, _, _ = executor_dispatch.load_script(
        HERE.parents[1] / "examples" / "goals" / "Isaac-Kitchen-v813-00.grasp12.json"
    )
    assert len(original[1].spec) == 18
    assert selected[1].spec == [original[1].spec[12]]


def test_aiworker_single_grasp_fixture_is_the_stable_approach_candidate():
    original, _, _ = executor_dispatch.load_script(
        HERE.parents[1] / "examples" / "goals" / "Isaac-Kitchen-v813a-00.json"
    )
    selected, _, _ = executor_dispatch.load_script(
        HERE.parents[1] / "examples" / "goals" / "Isaac-Kitchen-v813a-00.grasp0.json"
    )
    assert len(original[1].spec) == 10
    assert selected[1].spec == [original[1].spec[0]]


def test_active_grasp_check_must_point_at_its_closing_gripper_step():
    args = args_with_defaults()
    args.sub_good_goal_count_r = 0
    args.sub_grasp_idx_l = 4  # stale AI Worker goal: index 4 is nav.to_prim, close is index 2
    args.sub_good_goal_count_l = 1
    script = [
        SimpleNamespace(action="N_s", skill="nav.to_prim", params={}),
        SimpleNamespace(action="A_l", skill="arm.grasp", params={}),
        SimpleNamespace(action="G_l", skill="gripper.set", params={"grasp": True}),
        SimpleNamespace(action="A_l", skill="arm.reset", params={}),
        SimpleNamespace(action="N_s", skill="nav.to_prim", params={}),
    ]

    with pytest.raises(RunConfigError, match="not a closing G_l step"):
        validate_grasp_check_steps(args, script)

    args.sub_grasp_idx_l = 2
    validate_grasp_check_steps(args, script)


# ---------------------------------------------------------------------------------------------
# The reference task, end to end: what task_emit ACTUALLY writes -> what simvla_gen ACTUALLY runs.
# ---------------------------------------------------------------------------------------------

def simvla_gen_defaults() -> Namespace:
    """simvla_gen.py's argparse defaults, verbatim (simvla_gen.py:135-198). They are the values a
    v1 file leaves in place, and the values a v2 file must be able to overrule — every one of them
    is wrong for v813 except target_idx and sub_grasp_idx_r, which are right by coincidence."""
    return Namespace(
        task="Isaac-Kitchen-v813-00",   # not a run-config field: a fact about THIS run, not the task
        obj_name="mug0",
        obj_name_l="none",
        target_idx=1,
        sub_grasp_idx_r=4,
        sub_grasp_idx_l=9,
        sub_good_goal_count_r=7,
        sub_good_goal_count_l=7,
        export_groups=None,
        task_language="Put bottle to the sink.",
        task_type="LocoManipulation",
    )


def v813_meta() -> dict:
    """The meta block task_emit writes for Isaac-Kitchen-v813-00 — derived, not transcribed."""
    return {"task_name": "bowl_to_drawer",
            "run_config": derive_run_config(bowl_to_drawer(), v813_bindings()).to_meta()}


def test_the_real_v813_run_config_merges_to_the_derived_safe_values():
    """No flags at all. The generated file supplies the safe derived values.

    export_groups is asserted TWICE, and neither assertion is weaker than the one it replaced. The
    authored value is the flag's TEXT, `--export_groups "4,5,6,7;8,9,10;11"`, and that is what the
    goal file states — pinned below, unchanged. What the RUN held in args_cli.export_groups was never
    that string: --export_groups is `type=parse_export_groups`, so argparse had already turned it into
    [[4,5,6,7],[8,9,10],[11]] before a line of simvla_gen ran, and that is what finalize_lerobot
    iterates. The merge now applies the same type= by default, so args ends up holding what the
    runtime receives the typed value. The grasp-check index intentionally differs from the legacy
    demonstrated run: new goals check the settled close before any optional post-grasp lift."""
    args = simvla_gen_defaults()
    merge_run_config(args, v813_meta(), argv=["simvla_gen.py", "--task", "Isaac-Kitchen-v813-00"])

    assert args.obj_name == "bowl0"
    assert args.obj_name_l == "none"
    assert args.target_idx == 1
    assert args.sub_grasp_idx_r == 2
    assert args.sub_grasp_idx_l == 999
    assert args.sub_good_goal_count_r == 7
    assert args.sub_good_goal_count_l == 0
    assert v813_meta()["run_config"]["export_groups"] == "4,5,6,7;8,9,10;11"   # the known-good text
    assert args.export_groups == [[4, 5, 6, 7], [8, 9, 10], [11]]              # ...under its type=
    assert args.task_type == "NavManipulation"


def test_a_disabled_check_is_never_left_asking_for_successes():
    """sub_good_goal_count_r is the tenth field of RunConfig, and to_meta() writes it. Leaving it
    out of the merge is not a smaller feature — it reinstates the exact bug this project exists to
    kill: a mirror of v813 (the RIGHT arm works the drawer handle) derives sub_grasp_idx_r=999 and
    sub_good_goal_count_r=0, and a merge blind to that field leaves argparse's default of 7 in
    place. The run then waits for seven right-arm successes that the disabled check is the only
    thing that could produce: stage=collect_right, spinning to its time limit, printing nothing."""
    assert "sub_good_goal_count_r" in RUN_CONFIG_FIELDS

    args = simvla_gen_defaults()
    meta = {"run_config": {"sub_grasp_idx_r": 999, "sub_good_goal_count_r": 0}}
    merge_run_config(args, meta, argv=["simvla_gen.py"])
    assert args.sub_good_goal_count_r == 0, "a disabled check must never be asked for successes"


def test_every_field_of_a_run_config_can_be_merged():
    """RUN_CONFIG_FIELDS and RunConfig cannot drift: a field task_emit writes and the merge does not
    read is a value that lives in the file and is silently ignored at run time."""
    written = set(derive_run_config(bowl_to_drawer(), v813_bindings()).to_meta())
    assert written == set(RUN_CONFIG_FIELDS)


# ---------------------------------------------------------------------------------------------
# What "explicitly passed" has to survive: the forms argparse itself accepts.
# ---------------------------------------------------------------------------------------------

def test_the_inline_equals_form_is_explicit():
    args = simvla_gen_defaults()
    args.obj_name = "mug0"
    overrode = merge_run_config(args, v813_meta(), argv=["simvla_gen.py", "--obj_name=mug0"])
    assert args.obj_name == "mug0", "--obj_name=mug0 is the same flag as --obj_name mug0"
    assert "obj_name" not in overrode


def a_parser() -> ArgumentParser:
    """simvla_gen's parser, in the two respects that matter here: allow_abbrev is on (the default),
    and --task exists alongside --task_language and --task_type."""
    parser = ArgumentParser()
    for flag in ("--task", *(f"--{f}" for f in RUN_CONFIG_FIELDS)):
        parser.add_argument(flag)
    return parser


def test_an_unambiguous_abbreviation_is_explicit():
    """argparse's allow_abbrev is on by default, so `--target_id 5` really does set target_idx — and
    a merge that only recognised the full spelling would call the flag 'not passed' and overwrite
    the user's 5 with the file's 1. That is the one failure this detection scheme exists to
    prevent."""
    assert a_parser().parse_args(["--target_id", "5"]).target_idx == "5"   # argparse really does this

    args = simvla_gen_defaults()
    args.target_idx = 5
    overrode = merge_run_config(args, v813_meta(), argv=["simvla_gen.py", "--target_id", "5"])
    assert args.target_idx == 5
    assert "target_idx" not in overrode


def test_an_ambiguous_prefix_is_a_command_that_never_runs():
    """`--obj_nam` prefixes both --obj_name and --obj_name_l. argparse REFUSES it, so no run can
    reach the merge with one, and the merge owes it nothing."""
    with pytest.raises(SystemExit):
        a_parser().parse_args(["--obj_nam", "mug0"])


def test_a_flag_of_its_own_is_not_an_abbreviation_of_another():
    """--task is a flag in its own right and prefixes task_language and task_type. Reading it as an
    abbreviation would take BOTH of those away from the goal file on every run that names its task
    — which is every run."""
    assert a_parser().parse_args(["--task", "Isaac-Kitchen-v813-00"]).task_language is None

    args = simvla_gen_defaults()
    overrode = merge_run_config(args, v813_meta(), argv=["simvla_gen.py", "--task", "Isaac-Kitchen-v813-00"])
    assert args.task_language == "Put bowl inside drawer."
    assert args.task_type == "NavManipulation"
    assert "task_language" in overrode and "task_type" in overrode


def test_a_flags_value_is_not_mistaken_for_a_flag():
    """`--task Isaac-Kitchen-v813-00` is one flag and one value. Only the flag counts."""
    args = simvla_gen_defaults()
    merge_run_config(args, v813_meta(), argv=["simvla_gen.py", "--task", "Isaac-Kitchen-v813-00"])
    assert args.obj_name == "bowl0"


# ---------------------------------------------------------------------------------------------
# export_groups: the goal file states a STRING, argparse hands the program a list of lists.
# ---------------------------------------------------------------------------------------------

def test_export_groups_is_coerced_the_way_argparse_would_have_coerced_it():
    """--export_groups has `type=parse_export_groups` (simvla_gen.py:188), so args.export_groups is
    ALWAYS a list of lists by the time the program sees it — it is handed straight to
    finalize_lerobot(export_groups=...), which does `for group in export_groups: [int(x) for x in
    group]` (hdf5_dataset_file_handler.py:639). Setting the goal file's raw '4,5,6,7;8,9,10;11'
    would iterate that string CHARACTER BY CHARACTER and die on int(',') — after the run has already
    recorded its whole dataset. The merge applies the same coercion argparse would have.

    NO `coerce=` KWARG. That is the point: the coercion is a fact about the FIELD, and it is declared
    with the field (run_config_merge.FIELD_TYPES). It used to be the caller's to remember — one call
    site passed it and it was the only one — so dropping the kwarg in a refactor, or adding a second
    caller that never knew about it, brought the raw string back, and it surfaced in finalize_lerobot,
    with the dataset already on disk."""
    args = simvla_gen_defaults()
    merge_run_config(args, v813_meta(), argv=["simvla_gen.py"])
    assert args.export_groups == [[4, 5, 6, 7], [8, 9, 10], [11]]


def test_coercion_is_not_applied_to_a_field_the_user_typed():
    """argparse already coerced what the user typed. Coercing it twice would parse a list."""
    args = simvla_gen_defaults()
    args.export_groups = [[0, 1]]                       # what argparse produced from '0,1'
    merge_run_config(args, v813_meta(), argv=["simvla_gen.py", "--export_groups", "0,1"])
    assert args.export_groups == [[0, 1]]


def test_a_caller_may_still_say_it_wants_no_coercion():
    """The default is a default, not a law: `coerce={}` is an explicit request for the raw values,
    and it is how a caller that is not argparse (a report, a migration) reads the file's own text."""
    args = simvla_gen_defaults()
    merge_run_config(args, v813_meta(), argv=["simvla_gen.py"], coerce={})
    assert args.export_groups == "4,5,6,7;8,9,10;11"


def test_an_int_field_stated_as_text_does_not_stay_text():
    """The int entries in FIELD_TYPES are not symmetry. target_idx is only ever COMPARED against a
    step index (`step_i == args_cli.target_idx`, simvla_gen.py:1522), so a hand-edited file stating
    "1" would leave it the STRING "1": no step ever equals it, the reset event still displaces the
    object by up to 5 cm, and NO step's authored pose follows it. The grasp check's threshold is
    0.15 m, so it passes anyway, and the run records a dataset that nothing announces as wrong."""
    args = simvla_gen_defaults()
    merge_run_config(args, {"run_config": {"target_idx": "3"}}, argv=["simvla_gen.py"])
    assert args.target_idx == 3


def test_the_merge_coerces_with_the_types_the_parser_actually_declares():
    """THE ANTI-DRIFT PIN, and the reason FIELD_TYPES is allowed to be a second statement of the
    parser's `type=`s at all. simvla_gen.py cannot be imported (it boots Isaac), so its parser is
    read as an AST instead — no import, no GPU. A flag whose type= changes without FIELD_TYPES is a
    coercion that has quietly stopped matching what the program reads."""
    tree = ast.parse((HERE / "simvla_gen.py").read_text())

    declared = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("--")):
            continue
        field = str(node.args[0].value)[2:]
        if field not in RUN_CONFIG_FIELDS:
            continue
        kinds = [k.value for k in node.keywords if k.arg == "type"]
        assert kinds, f"--{field} declares no type=; FIELD_TYPES would be asserting a fiction"
        declared[field] = kinds[0].id

    assert set(declared) == set(RUN_CONFIG_FIELDS), "every run-config field is a flag of simvla_gen's"
    assert declared == {f: FIELD_TYPES[f].__name__ for f in RUN_CONFIG_FIELDS}


def test_simvla_gen_does_not_keep_a_second_copy_of_the_export_groups_parser():
    """It IMPORTS parse_export_groups from run_config_merge — the flag's type= and the merge's
    coercion are the same object. Two copies is two copies that drift, and the drift only shows up in
    finalize_lerobot, i.e. after the run has recorded its dataset."""
    tree = ast.parse((HERE / "simvla_gen.py").read_text())

    defined = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "parse_export_groups" not in defined, "simvla_gen defines it again; it must import it"

    imported = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "run_config_merge"
        for alias in n.names
    }
    assert {"merge_run_config", "parse_export_groups"} <= imported


def test_the_export_groups_parser_refuses_what_it_cannot_parse():
    """It is the flag's type=, so it is also the goal file's. From argv it only ever sees a string;
    from a run_config block it could see the already-parsed list a confused writer emitted, and
    `.strip()` on that is an AttributeError that names neither the field nor the file."""
    assert parse_export_groups("4,5,6,7;8,9,10;11") == [[4, 5, 6, 7], [8, 9, 10], [11]]
    assert parse_export_groups(None) is None
    assert parse_export_groups("  ") is None
    with pytest.raises(ValueError, match="ints only"):
        parse_export_groups("4,x")
    with pytest.raises(ValueError, match="export_groups"):
        parse_export_groups([[4, 5]])


# ---------------------------------------------------------------------------------------------

def test_run_config_merge_imports_nothing_heavy():
    """simvla_gen.py cannot be imported without Isaac, so the merge is only testable at all if it
    stays out of that dependency. It is stdlib, and it must remain stdlib."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import run_config_merge                            # and nothing else

            banned = ("omni", "isaaclab", "pxr", "torch", "trimesh", "numpy", "tkinter", "PIL",
                      "scipy", "shapely")
            pulled = sorted(m for m in sys.modules if m.split(".")[0] in banned)
            assert not pulled, f"importing run_config_merge pulled in {pulled}"
            print("clean")
        """)],
        cwd=str(HERE), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.stdout.split() == ["clean"]
