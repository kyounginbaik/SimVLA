"""SimVLA: a task's config comes from its goal file; the CLI overrides it.

Stdlib only.

--obj_name, --sub_grasp_idx_r/_l, --sub_good_goal_count_r/_l, --export_groups and --target_idx are
all facts about the TASK, and the goal file already implies every one of them. Transcribing them by
hand is how the README shipped `--obj_name "bottle0"` for a task whose goal file targets /world/bowl0
(a bare KeyError, forty minutes into a GPU run), and how `--sub_grasp_idx_l 999` came to be paired
with `--sub_good_goal_count_l 7` (a run that spins to its time limit printing nothing).

A v2 goal file carries them, in the `run_config` block task_runconfig.RunConfig.to_meta() writes. A
v1 file does not, and merging is then a NO-OP — the existing v1 tasks (including the only task known
to have recorded a dataset end to end) keep working exactly as they do today, on exactly the flags
they are launched with.

WHAT COUNTS AS "THE USER SAID SO" is read from argv, never from the parsed values. An argparse
default is indistinguishable from a user's value once parsed — `--target_idx 1` and no flag at all
both arrive as `args.target_idx == 1` — so comparing against the default would silently discard a
user who typed the default on purpose, and (far worse, the other way round) leaving the comparison
out entirely would let argparse's default overwrite the file's derived value for every task whose
config happens to differ from it, which is every task.
"""

from __future__ import annotations

#: The fields a v2 goal file may supply — RunConfig's, exactly. Every one is a fact about the task,
#: not about this run, and test_run_config_merge pins this tuple against RunConfig.to_meta()'s keys:
#: a field the writer emits and the merge does not read is a value that sits in the file and is
#: silently ignored at run time. `sub_good_goal_count_r` is in here for that reason and not by
#: symmetry — omit it, and a task whose RIGHT grasp check is disabled (the mirror of v813: the right
#: arm works the drawer handle) keeps argparse's default of 7 and waits for seven successes only the
#: disabled check could have produced.
RUN_CONFIG_FIELDS = (
    "obj_name",
    "obj_name_l",
    "target_idx",
    "sub_grasp_idx_r",
    "sub_grasp_idx_l",
    "sub_good_goal_count_r",
    "sub_good_goal_count_l",
    "export_groups",
    "task_language",
    "task_type",
)


class RunConfigError(ValueError):
    """The goal file's run_config block cannot be merged, or the caller cannot merge into."""


def validate_grasp_check_steps(args, script) -> None:
    """Refuse active grasp checkpoints that do not name a closing gripper step.

    A stale run_config can otherwise wait forever: the collector only evaluates the physical grasp
    gate while processing a gripper command, but the old index is compared on every step. `script`
    is the decoded executor sequence, so this check follows the exact indices the runtime uses.
    """
    for arm, action in (("r", "G_r"), ("l", "G_l")):
        index = getattr(args, f"sub_grasp_idx_{arm}")
        count = getattr(args, f"sub_good_goal_count_{arm}")
        if count <= 0:
            continue
        if index < 0 or index >= len(script):
            raise RunConfigError(
                f"active {arm.upper()} grasp check index {index} is outside the "
                f"{len(script)}-step goal; regenerate the goal with the current task emitter "
                f"or pass a valid --sub_grasp_idx_{arm}."
            )
        step = script[index]
        if (step.action != action or step.skill != "gripper.set"
                or not bool(step.params.get("grasp", False))):
            raise RunConfigError(
                f"active {arm.upper()} grasp check index {index} names "
                f"{step.skill!r}/{step.action!r}, not a closing {action} step; regenerate the "
                f"goal with the current task emitter or pass --sub_grasp_idx_{arm} pointing "
                "to the manipulation-object close."
            )


def parse_export_groups(s):
    """--export_groups' own `type=`. '0,1,2;3,4' -> [[0, 1, 2], [3, 4]]. None/empty -> None.

    IT LIVES HERE, and simvla_gen.py imports it from here, for one reason: it is BOTH the flag's
    `type=` and the coercion the merge has to apply to the same value coming out of a goal file, and
    two copies of it would be two copies that drift. simvla_gen.py cannot be imported without Isaac,
    so the arrow can only point this way — and this module is stdlib, which is what lets it.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        # From argparse this cannot happen (argv is strings). From a goal file's run_config it can:
        # a writer that emitted the already-parsed [[4,5,6,7]] instead of the flag's TEXT would
        # otherwise reach `.strip()` and die as an AttributeError with no mention of the file.
        raise ValueError(
            f"export_groups must be the flag's text, like '0,1,2;3,4' — got {type(s).__name__} "
            f"{s!r}. (A goal file's run_config states the TEXT; the list of lists is what this "
            f"function makes of it.)"
        )

    s = s.strip()
    if not s:
        return None

    groups = []
    for g in s.split(";"):
        g = g.strip()
        if not g:
            continue
        items = [x.strip() for x in g.split(",") if x.strip() != ""]
        if not items:
            continue
        try:
            groups.append([int(x) for x in items])
        except ValueError as e:
            raise ValueError(
                f"Bad --export_groups='{s}'. Use like '0,1,2;3,4' (ints only)."
            ) from e

    return groups if groups else None


#: FIELD -> the callable argparse applies to that flag (its `type=`), for every field a goal file may
#: supply. This is the merge's DEFAULT coercion, and it is here rather than at the call site because
#: a fact about a field belongs with the field.
#:
#: It used to be the caller's to remember — simvla_gen.py passed coerce={"export_groups": ...} and
#: was the only caller that did. Drop that kwarg in a refactor, or add a second caller without it,
#: and the goal file's TEXT ("4,5,6,7;8,9,10;11") is assigned to args.export_groups raw, where the
#: rest of the program expects the list of lists argparse makes of it: finalize_lerobot iterates it
#: (`for group in export_groups: [int(x) for x in group]`, hdf5_dataset_file_handler.py:639), walks
#: the STRING character by character, and dies on int(',') — after the run has recorded its entire
#: dataset. The one thing standing between that and a GPU node should not be a call site's memory.
#:
#: The int entries are not decoration either. A goal file is JSON, and to_meta() writes real ints —
#: but a hand-edited file that states "target_idx": "1" would otherwise leave args.target_idx a
#: STRING, and target_idx is only ever compared (`step_i == args_cli.target_idx`, simvla_gen.py:1522):
#: "1" != 1 for every step, so no step is ever the target, the object is randomized and NOTHING
#: follows it. int("1") is 1 and int(1) is 1 — on a well-formed file every entry here is the
#: identity, which is exactly what a declaration of the flag's type should be.
#:
#: test_run_config_merge pins this map against simvla_gen.py's parser (by AST, without importing it):
#: a flag whose `type=` changes and whose entry here does not is a coercion that silently stops
#: matching what the program reads.
FIELD_TYPES = {
    "obj_name": str,
    "obj_name_l": str,
    "target_idx": int,
    "sub_grasp_idx_r": int,
    "sub_grasp_idx_l": int,
    "sub_good_goal_count_r": int,
    "sub_good_goal_count_l": int,
    "export_groups": parse_export_groups,
    "task_language": str,
    "task_type": str,
}


def _explicit_flags(args, argv: list[str]) -> set[str]:
    """Which run-config fields the user actually typed.

    PRECONDITION: `args` is the COMPLETE Namespace argparse produced — one attribute per dest the
    parser defines, not a hand-built subset. `vars(args)` is read as argparse's own vocabulary, and
    it is what tells `--task` (a flag in its own right) apart from an abbreviation of `task_language`
    /`task_type`; a Namespace missing `task` would see the token as a prefix of two fields, and a
    Namespace missing, say, `num_envs` would expand `--num_envs` against nothing at all. Only the
    run-config half of that vocabulary is checkable from here — those are the only names this module
    knows — and it is checked below. The rest is the caller's word, and in the program that matters
    the caller passes `args_cli` itself (simvla_gen.py:836).

    Read from argv, NOT by comparing against the argparse default: a user may legitimately pass a
    value that equals the default, and treating that as 'not passed' would silently ignore them.

    Three forms have to be recognised, because argparse accepts all three:

      --obj_name bowl0    the flag and its value, as two tokens
      --obj_name=bowl0    the same thing, inline
      --obj_nam bowl0     an UNAMBIGUOUS ABBREVIATION. parser.allow_abbrev is on by default, so this
                          really does set obj_name — and a matcher that only knew the full spelling
                          would call the flag "not passed" and overwrite the user's value with the
                          file's. That is the one thing this function exists to prevent.

    An abbreviation is only honoured when it names exactly ONE field: argparse itself REFUSES an
    ambiguous prefix ("--obj_nam" against --obj_name and --obj_name_l is an error and the program
    never gets here), so an ambiguous token is not a flag we could be shadowing.

    A token that is some OTHER flag's exact name is not an abbreviation of anything — `--task` is a
    flag in its own right, and expanding it as a prefix of `task_language`/`task_type` would take
    those two away from the goal file for every run that names its task. `vars(args)` is argparse's
    own vocabulary (one key per dest it defined), which is how that is known here.
    """
    dests = set(vars(args))

    missing = [f for f in RUN_CONFIG_FIELDS if f not in dests]
    if missing:
        raise RunConfigError(
            f"merge_run_config was handed a Namespace that does not define {missing}. It must be the "
            f"complete namespace argparse parsed: `vars(args)` is read as argparse's vocabulary — it "
            f"is the only thing that tells a flag of its own (--task) from an abbreviation of a field "
            f"(--task_language, --task_type) — and a partial one silently mis-expands the flags it has "
            f"never heard of. (These names are also what the merge is about to setattr; a field the "
            f"parser does not define is a value nothing reads.)"
        )

    flags: set[str] = set()

    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        if not name:                                  # the bare "--" end-of-options separator
            continue
        if name in RUN_CONFIG_FIELDS:                 # exact, and exact always wins in argparse too
            flags.add(name)
            continue
        if name in dests:                             # some other flag, spelled in full
            continue
        prefixed = [f for f in RUN_CONFIG_FIELDS if f.startswith(name)]
        if len(prefixed) == 1:                        # an unambiguous abbreviation IS the flag
            flags.add(prefixed[0])

    return flags


def merge_run_config(args, goal_meta: dict, argv: list[str], coerce: dict | None = None) -> list[str]:
    """Fill args from the goal file's run_config, for every flag the user did not type.

    Returns the field names it set, so the caller can log what came from the file.

    `coerce` maps a field to the callable argparse would have applied to it — i.e. the flag's
    `type=` — and it DEFAULTS to FIELD_TYPES, which declares that for every field. A caller may pass
    its own map (or {} for none); no caller has to REMEMBER to. A goal file states values as JSON,
    but a flag's `type=` may hand the program something else entirely: --export_groups is
    `type=parse_export_groups` (simvla_gen.py:188), so args.export_groups is ALWAYS a list of lists
    by the time anything reads it, while the file states the flag's TEXT, "4,5,6,7;8,9,10;11".
    Assigning that string raw would hand finalize_lerobot a string where it iterates groups (`for
    group in export_groups: [int(x) for x in group]`, hdf5_dataset_file_handler.py:639) — int(',')
    raises, after the run has recorded its entire dataset. Nothing is coerced for a field the user
    typed: argparse already did it.
    """
    meta = goal_meta or {}

    # ABSENT and PRESENT-BUT-EMPTY are different facts, and `if not config` told them apart from
    # nothing.
    #
    # NO run_config block is a v1 goal file — which is every goal file that has ever been run, the
    # 960-frame one included. It merges nothing, reaches no setattr, and leaves the command line
    # exactly as the user typed it. Correct, and the common case.
    #
    # An EMPTY run_config block is not that. It is a v2 file whose writer produced nothing, and
    # nothing in this repo can emit one (to_meta() writes all ten fields) — which is exactly why it
    # is refused now rather than met later. Read as v1, it would run the task on argparse's defaults,
    # which are another task's: obj_name=mug0 on a bowl (bare KeyError), sub_grasp_idx_l=9 against
    # sub_good_goal_count_l=7 (a run that spins to its time limit), export_groups=None (a recorded
    # dataset with no LeRobot export at all). None of those says a word about the goal file.
    if "run_config" not in meta:
        return []                       # a v1 goal file; nothing to merge

    config = meta["run_config"]
    if not isinstance(config, dict) or not config:
        raise RunConfigError(
            f"the goal file carries a `run_config` block, and it is {config!r}. A v2 file states "
            f"{list(RUN_CONFIG_FIELDS)} — RunConfig.to_meta() writes every one of them — and a v1 "
            f"file has no run_config block at all, which merges as a no-op. This is neither, so "
            f"whatever wrote the file is broken. Falling back to the flags would run this task on "
            f"another task's defaults and never mention it."
        )

    explicit = _explicit_flags(args, argv)
    coerce = FIELD_TYPES if coerce is None else coerce
    overrode = []
    for field in RUN_CONFIG_FIELDS:
        if field in config and field not in explicit:
            value = config[field]
            if field in coerce:
                value = coerce[field](value)
            setattr(args, field, value)
            overrode.append(field)
    return overrode
