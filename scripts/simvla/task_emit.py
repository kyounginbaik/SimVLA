"""SimVLA: turn a task template into v2 goal files, one per kitchen.

Stdlib only (plus skill_contract, goal_format, task_template, task_validate, task_bind,
task_runconfig).

The composer authors INTENT; this computes GEOMETRY. The split is forced, and it is the reason
plan() is INJECTED rather than called: six skills' plan() bodies read the USD stage (BBoxCache, a
physx raycast, a Tk thumbnail chooser of grasp candidates) and live in simvla_data_generator.py,
which imports pxr, omni.usd, tkinter and PIL. Nothing that imports that module can be unit-tested —
it boots Omniverse at import. So `plan_fn(skill_id, action, params)` arrives from the caller: in
the authoring process it calls the skill's real plan(); in a test it is a stub, or a replay of a
goal file that already ran. What is left here is the COMPOSITION — which skill, which prim, which
step carries which pose, and which steps carry none — which is where the bugs live, and it is
testable on a laptop.

TWO THINGS A STEP CAN SAY ABOUT ITS GOAL, and only two:

    goal: [..]   authored. plan() computed it while the stage was open; the file IS the answer.
    goal: null   resolved at run time. The executor overwrites it from live robot state.

`SkillSpec.is_runtime` is the single predicate that decides which, and it is a fact of the skill,
not of the shape of the floats. v1 said the same thing by writing a magic float into a coordinate
slot and recovering it with torch.isclose — which is how a skill shipped the value -0.4, matched
nothing, fell through, and became a literal coordinate in 73 goal files. So a null goal here means
exactly one thing, and an authored skill whose plan() returns None is REFUSED (below) rather than
written as a null the executor would drive to as (0, 0, 0).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import goal_format
from predicate_contract import SpecError, resolve_roles
from skill_contract import REGISTRY, PrimPath, SkillSpec
from task_bind import BindingError, BoundPrim, KitchenScene, bind_many, bind_roles
from task_runconfig import derive_run_config
from task_template import (
    TaskTemplate,
    TaskTemplateError,
    TemplateStep,
    from_json,
    role_ref,
    template_warnings,
    validate_template,
)
from task_validate import SequenceError, validate_sequence

#: plan(skill_id, action, params) -> the authored goal. See the module docstring.
PlanFn = Callable[[str, str, dict], Any]

#: The corpus: 15,810 real goal files, reachable from here through a symlink. emit() REFUSES to
#: write into it (below). Resolved, because the symlink is exactly how the check gets bypassed.
_BUNDLED_GOALS = Path(__file__).resolve().parents[2] / "examples/goals"
GOALS_DIR = Path(os.environ.get("SIMVLA_GOALS_DIR", str(_BUNDLED_GOALS if _BUNDLED_GOALS.is_dir()
                else Path(__file__).resolve().parent / "goals"))).expanduser().resolve()

#: Where the kitchen USDs live. Each kitchen has one USD PER ROTATION — kitchen_<n>_00.usd ..
#: kitchen_<n>_11.usd — a physically rotated copy (30*sub degrees). The dataset is that fan-out: the
#: corpus holds ~12 goal files per kitchen (v<n>-00 .. v<n>-11), not one. kitchen_subs() reads which
#: rotations actually exist so an emit reproduces the full set, not just the first.
_DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / "source/isaaclab_assets/data"
KITCHEN_DIR = Path(os.environ.get("SIMVLA_ASSETS_DIR", _DEFAULT_ASSETS)).expanduser().resolve() / "Kitchen"


class RoleUnbound(KeyError):
    """A step names a role that did not bind in THIS kitchen — a fact of the scene, not of the
    template (validate_template already passed). KeyError so the message reads the same as the one
    this used to raise."""


class PlanFailure(ValueError):
    """plan() could not produce this step's goal in THIS kitchen: it raised (no reachable grasp
    candidate, the object is inside a wall), or it returned None. A fact of the scene.

    ValueError so it still reads as one to a caller that only knows the old contract, and a NAMED
    type so emit() can skip the kitchen that has it without also skipping the kitchen whose plan()
    is fine and whose emitter is broken.
    """


#: THE ONLY REASONS A KITCHEN IS SKIPPED. emit() catches these and no others.
#:
#: The per-kitchen catch exists for ONE thing: a scene that cannot satisfy the template — a role
#: with no prim, a grasp on nothing rigid, a plan() that finds no candidate. It used to be a bare
#: `except Exception`, which cannot tell that from a bug in this module: a plan_fn returning a numpy
#: array makes json.dumps raise TypeError, and twelve kitchens then report twelve identical
#: "failures" for one systematic bug in the caller's code — the batch looks like twelve bad scenes
#: and reads as data. Everything not listed here propagates and stops the batch, which is what a bug
#: should do.
SCENE_FAILURES = (BindingError, SequenceError, RoleUnbound, PlanFailure)


class UnsafeOutputDir(ValueError):
    """emit() was pointed at the real goals corpus."""


def surface_warnings(t: TaskTemplate, where: str) -> list[str]:
    """Print the template's non-fatal condition traps to stderr, and return them.

    template_warnings had exactly ONE caller — the composer's POST /template — so a condition
    authored in the browser was warned about and the identical condition in a hand-written template
    was not. The trap it reports (proximity with no height check) is not academic: an object resting
    on the ~0.82 m counter is within 0.2 m of a passing gripper, and that condition once scored a
    non-grasping policy 4/20. A warning nobody sees on the path that WRITES the files is not a
    warning. Not fatal, and deliberately so — this cannot know the counter height, so it must not
    refuse.
    """
    warnings = template_warnings(t)
    for w in warnings:
        print(f"[task_emit] WARNING ({where}): {w}", file=sys.stderr, flush=True)
    return warnings


def resolved_params(
    step: TemplateStep, bindings: dict[str, BoundPrim], spec: SkillSpec
) -> dict:
    """The params as the goal file states them: every @role replaced by the prim it bound to, and
    every param the template omitted filled in from the skill's own declaration.

    NO ROLE REFERENCE MAY REACH A GOAL FILE. '@target' is a name in a template, and the executor has
    no stage with a prim at '/world/@target'.

    NO DECLARED PARAM MAY BE LEFT UNSAID, either — an omitted param is not an absent one, it is the
    contract's default, and the defaults are load-bearing. nav.to_prim reads which_arm as a ±5 cm
    arm_bias on the base pose (simvla_data_generator.py:1090) and plan_nav_to_prim reads it with a
    bare params.get(), so an omitted which_arm silently means "Both". executor_dispatch reads
    back_off_m off a step's params at load. A value that lives in nobody's file and everybody's
    memory is the disease this whole contract exists to cure, so the writer states it — from the
    same declaration task_validate.gripper_closes reads its default from, so the two cannot drift.
    """
    out = dict(step.params)
    for p in spec.params:
        if isinstance(p, PrimPath):
            if p.name not in out:
                # A PrimPath has no default to fall back on — there is no such thing as "the prim
                # this skill reaches for by default" — so unlike every other param there is nothing
                # to state, and the old code left the key ABSENT and said nothing. Through emit()
                # this is unreachable (validate_template refuses it first), but steps_for_kitchen is
                # public: a direct caller got a step with no prim_path, and the executor a step that
                # reaches for nothing. Absent is not a value. Refuse it here too.
                raise TaskTemplateError(
                    f"step {step.skill!r} omits {p.name!r}, which {step.skill!r} declares as a prim "
                    f"path. It has no default — a skill has no prim it reaches for 'by default' — so "
                    f"the goal file would simply not say where to reach. (validate_template reports "
                    f"this too; steps_for_kitchen is reachable without it.)"
                )
            ref = role_ref(out[p.name])
            if ref is None:
                continue                    # a literal prim path: a template may write one
            bound = bindings.get(ref)
            if bound is None:
                raise RoleUnbound(
                    f"step {step.skill!r} references role {ref!r} in {p.name!r}, and it did not "
                    f"bind in this kitchen (bound: {sorted(bindings) or 'nothing'}). A goal file "
                    f"carrying '@{ref}' names a prim no stage has."
                )
            out[p.name] = bound.prim_path
        elif p.name not in out:
            # IEEE infinities are useful runtime identities (for example, "no minimum
            # placement height"), but they are not JSON numbers. Keep that semantic default
            # implicit; the skill's resolver reads the same declared default when the key is
            # absent. Explicit non-finite values are rejected by template validation.
            if isinstance(p.default, float) and not math.isfinite(p.default):
                continue
            out[p.name] = p.default
    return out


def steps_for_kitchen(
    t: TaskTemplate, bindings: dict[str, BoundPrim], plan_fn: PlanFn
) -> list[goal_format.Step]:
    """The template's steps, bound to one kitchen's prims and planned. Pure: no Omniverse, no I/O.
    (plan_steps below is the same call returning the EXPANDED template as well; this one keeps
    the steps-only signature its tests and callers use.)"""
    return plan_steps(t, bindings, plan_fn)[0]


def plan_steps(
    t: TaskTemplate, bindings: dict[str, BoundPrim], plan_fn: PlanFn
) -> tuple[list[goal_format.Step], TaskTemplate]:
    """(the planned steps, the template those steps came from).

    THE TWO CAN DIFFER: a nav planner that had to route around furniture returns a
    nav_clearance.NavRoute -- the park pose plus via-points -- and each via-point is written as
    its own nav step AHEAD of the park (the run's nav is turn-drive-turn per step, and the
    runtime advances to the next step when one finishes, so a detour is a chain of steps). Every
    step index the run config derives from the template -- the grasp step, the grasp-check step,
    the export groups -- moves with the insertion, so the returned template has the via steps in
    it and its subtask groups remapped; derive_run_config must read THAT template.

    NOTE ON ISLAND KITCHENS: save_goal_file in the editor doubles a nav step for kitchen_type ==
    'island'. That code is vestigial — an audit of the released corpus found only 3 of 1690 island
    goal files carry a doubled nav (and those 3 are malformed), so the real dataset does NOT double.
    Reproducing the corpus means NOT doubling; the doubling was deliberately left out here."""
    steps: list[goal_format.Step] = []
    expanded_steps: list[TemplateStep] = []
    inserted_before: list[int] = []

    for i, step in enumerate(t.steps):
        spec = REGISTRY.get(step.skill)
        if spec is None:
            raise KeyError(
                f"step {i} names skill {step.skill!r}, which is not in the registry. "
                f"(validate_template reports this too; nothing may reach a file without a skill.)"
            )

        params = resolved_params(step, bindings, spec)

        if spec.is_runtime:
            # Stated, not encoded as a magic float in a coordinate slot. plan_fn is NOT called:
            # there is no plan() to call (arm.reset's raises NotImplementedError by design), and
            # the goal genuinely does not exist until the robot is where it is.
            goal = None
        else:
            try:
                goal = plan_fn(step.skill, step.action, params)
            except Exception as exc:
                # THE PLAN BOUNDARY, and the only place a caller's exception is translated. plan()
                # reads the USD stage, so what it raises is a fact about THIS kitchen's scene ("no
                # reachable grasp candidate") — the one thing emit() is entitled to skip a kitchen
                # for. Naming it as such is what lets emit() catch it and NOT catch a TypeError out
                # of its own json.dumps. The cause is chained, never swallowed.
                raise PlanFailure(
                    f"step {i} ({step.skill}): plan() raised {type(exc).__name__}: {exc}"
                ) from exc
            if goal is None:
                raise PlanFailure(
                    f"step {i} ({step.skill}) is authored, but plan() returned None. In a v2 file "
                    f"`goal: null` means 'the executor resolves this at run time' — and nothing "
                    f"resolves {step.skill!r}, so the executor would drive to (0, 0, 0) with a zero "
                    f"quaternion. A planner that computes nothing is a bug, not a null goal."
                )

        chain = goal_format.expand_route(goal_format.Step(
            skill=step.skill, action=step.action, params=params, goal=goal, language=step.language))
        inserted_before.append(len(chain) - 1)
        for s in chain[:-1]:
            steps.append(s)
            expanded_steps.append(dataclasses.replace(step, language=s.language))
        steps.append(chain[-1])
        expanded_steps.append(step)

    if not any(inserted_before):
        return steps, t
    # remap: template step i lands at i + (via steps inserted at or before it); its via steps
    # take the indices just before it and join its subtask group
    offsets = []
    total = 0
    for n in inserted_before:
        total += n
        offsets.append(total)
    groups = []
    for group in t.subtask_groups:
        g = []
        for i in group:
            g.extend(range(i + offsets[i] - inserted_before[i], i + offsets[i] + 1))
        groups.append(g)
    expanded = dataclasses.replace(t, steps=expanded_steps, subtask_groups=groups)
    return steps, expanded


def emit(
    t: TaskTemplate,
    bindings_by_kitchen: dict[str, dict[str, BoundPrim]],
    plan_fn: PlanFn,
    out_dir,
    sub_good_goal_count: int = 7,
    kitchen_meta: dict[str, dict] | None = None,
) -> tuple[list[str], list[str]]:
    """Write one v2 goal file per kitchen. Returns (written_paths, failures).

    A kitchen whose SCENE cannot satisfy the template is SKIPPED AND REPORTED (SCENE_FAILURES, and
    nothing else), and the rest still generate: a fan-out over the twelve rotations of a kitchen must
    not lose the eleven that plan cleanly, and the one it loses must be named. A batch that silently
    writes 11 of 12 files reads as "it covered everything". A bug in this module or in plan()'s
    RETURN VALUE is not a bad kitchen, and is not caught: it stops the batch.

    `out_dir` may not be the real goals corpus. See _refuse_the_goals_corpus.

    The TEMPLATE's own errors are not caught. An incoherent script is a bug in the composition, not
    in a kitchen, and writing eleven copies of it before saying so helps nobody — so both validators
    run once, up front, before anything is written. (They are also derive_run_config's stated
    precondition; it re-checks neither.)

    `kitchen_meta` is the goal file's non-step half — initial_pos_ranges, initial_rot_yaw_range,
    island_bound, kitchen_type. It is the KITCHEN's, not the template's: simvla_gen.py:826 reads
    initial_pos_ranges straight off the loaded file, so a v2 file without it does not run.
    """
    validate_template(t)
    validate_sequence(t)                       # never write a sequence that cannot run
    surface_warnings(t, t.name)                # non-fatal traps, said out loud before anything is written

    out = Path(out_dir)
    _refuse_the_goals_corpus(out)              # before mkdir, and before a single file is opened
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    failed: list[str] = []

    for task_name, bindings in bindings_by_kitchen.items():
        try:
            path = emit_one(
                t, task_name, bindings, plan_fn, out,
                sub_good_goal_count=sub_good_goal_count,
                kitchen_meta=(kitchen_meta or {}).get(task_name, {}),
            )
            written.append(path)
        except SCENE_FAILURES as exc:          # a bad KITCHEN must not take the batch down
            failed.append(f"{task_name}: {type(exc).__name__}: {exc}")

    return written, failed


def emit_one(
    t: TaskTemplate,
    task_name: str,
    bindings: dict[str, BoundPrim],
    plan_fn: PlanFn,
    out: Path,
    sub_good_goal_count: int = 7,
    kitchen_meta: dict | None = None,
) -> str:
    """Plan and write ONE kitchen's v2 goal file; return the path. Raises SCENE_FAILURES if the
    scene cannot satisfy the template (a bad kitchen, to be skipped and named by the caller).

    THE CALLER MUST HAVE THIS KITCHEN'S STAGE LIVE. plan_fn's plan() bodies read app.stage /
    app.free_squares / app.kitchen_data, which are per-kitchen mutable state — so this kitchen's
    steps MUST be planned before the next kitchen's _load_kitchen_scene replaces that state. The
    two-pass form (enumerate every scene, THEN plan them all) planned every kitchen against the
    LAST-opened stage and poisoned every pose; _generate drives per-kitchen for exactly this reason.
    The batch emit() above is only correct for a stage-independent plan_fn (the tests' stub) or a
    single kitchen."""
    steps, planned_t = plan_steps(t, bindings, plan_fn)
    # THE EXPANDED template: a routed nav inserted via-point steps, and every index the run
    # config carries (target_idx, sub_grasp_idx_*, export_groups) must name steps in the FILE.
    cfg = derive_run_config(planned_t, bindings, sub_good_goal_count=sub_good_goal_count)

    meta = dict(kitchen_meta or {})
    meta["task_name"] = t.name
    meta["run_config"] = cfg.to_meta()

    path = Path(out) / f"{task_name}.json"
    goal_format.write_v2(path, steps, meta=meta)
    return str(path)


def _refuse_the_goals_corpus(out: Path) -> None:
    """scripts/simvla/goals is 15,810 goal files that runs have already used, and emit() writes
    <task_name>.json — the very names that are already in there. One --out away from overwriting
    them in place, with no backup and no undo, so the one directory this may not write into is named
    and refused.

    RESOLVED, both sides. `goals` is a symlink (-> simvla_others/goals); comparing the link path
    alone is bypassed by passing the target, and comparing the target alone is bypassed by passing
    the link. Path.resolve() collapses both to the same real directory, and catches a symlink of
    one's own making pointed at it.
    """
    target = Path(out).resolve()
    if target == GOALS_DIR or GOALS_DIR in target.parents:
        raise UnsafeOutputDir(
            f"refusing to emit into {out} — it resolves to {target}, inside the real goals corpus "
            f"({GOALS_DIR}). Those files are the record of runs that already happened, and emit() "
            f"writes '<kitchen>.json', which is exactly what they are called: this would overwrite "
            f"them in place. Emit somewhere new and copy in deliberately."
        )


# =============================================================================
# The CLI. main() boots Omniverse; everything above it does not.
# =============================================================================
# THE IMPORT ORDER IN main() IS LOAD-BEARING, and it is why this file's heavy dependencies are
# reached through importlib rather than a top-level `import`:
#
#   * The module-purity tests (test_task_emit.py) parse THIS file's AST and refuse any import whose
#     root is not stdlib-or-a-named-sibling — and ast.walk descends into function bodies, so even a
#     function-local `import simvla_data_generator` would be caught. importlib.import_module(<str>)
#     is not an Import node, so the composition logic stays unit-testable on a laptop, which is the
#     whole point of splitting authoring (geometry) from composition (which skill, which prim).
#
#   * Importing simvla_data_generator injects the six authoring plan() bodies: each @register_planner
#     runs at its import, and validate_planners() at the bottom of that module fails the import if one
#     is missing. So the injection is a side effect of the import. BUT the module's first line is
#     `from pxr import ...`: it assumes Isaac Sim is ALREADY booted, and does NOT bring the app up
#     itself. _authoring_app() therefore boots AppLauncher (headless, once) before importing it — the
#     same order goal_generator uses. Importing it cold, without that boot, raises
#     ModuleNotFoundError: pxr (which is exactly how the CLI's headless path was found to be unrun).
#
# The parts that do NOT need Omniverse — arg parsing, the kitchen list, the --out guard, loading and
# validating the template — are factored into the helpers below and tested directly. main() itself
# is never unit-tested: importing it is free, but running it boots Omniverse.


def parse_kitchens(spec: str) -> list[int]:
    """'813,422' -> [813, 422]. Whitespace tolerated; an empty or non-integer entry is refused
    loudly, never dropped — a silently-skipped kitchen number reads as 'that one had no object'."""
    nums: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nums.append(int(part))
        except ValueError:
            raise ValueError(
                f"--kitchens got {part!r}, which is not an integer kitchen number. Expected a "
                f"comma-separated list like '813,422'."
            ) from None
    if not nums:
        raise ValueError(
            f"--kitchens {spec!r} names no kitchen. Expected a comma-separated list like '813,422'."
        )
    return nums


def parse_subs(spec: str | None) -> list[int] | None:
    """Parse an optional rotation filter. Omission preserves full kitchen fan-out."""
    if spec is None:
        return None
    subs: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError("--subs contains an empty entry; expected rotation numbers 0 through 11")
        try:
            sub = int(part)
        except ValueError:
            raise ValueError(f"--subs got {part!r}; expected comma-separated rotation numbers 0 through 11") from None
        if not 0 <= sub <= 11:
            raise ValueError(f"--subs rotation {sub} is outside the supported range 0 through 11")
        if sub in subs:
            raise ValueError(f"--subs repeats rotation {sub}")
        subs.append(sub)
    if not subs:
        raise ValueError("--subs must name at least one rotation")
    return subs


def load_template(path) -> TaskTemplate:
    """Read a template JSON and hold it to BOTH validators before a single kitchen is touched. emit()
    re-runs them (they are derive_run_config's stated precondition), but doing it here means the CLI
    refuses an incoherent script in milliseconds, before it has spent three minutes booting
    Omniverse to discover the composition never made sense.

    The condition's non-fatal traps are surfaced here too (surface_warnings): a hand-written
    template used to reach the writer with none of the warnings the composer shows its author."""
    t = from_json(Path(path).read_text())
    validate_template(t)
    validate_sequence(t)
    surface_warnings(t, str(path))
    return t


def guard_out_dir(out) -> Path:
    """The --out directory, refused if it resolves into the real goals corpus. emit() refuses it too
    — this is the same check, run up front so `--out scripts/simvla/goals` fails before Omniverse
    boots, not after."""
    out = Path(out)
    _refuse_the_goals_corpus(out)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="task_emit.py",
        description="Compose a task template into v2 goal files, one per kitchen. Boots Omniverse: "
        "run it on a GPU node inside the conda env.",
    )
    p.add_argument("--template", required=True,
                   help="path to a task template JSON (e.g. src/simvla/templates/*.json)")
    p.add_argument("--kitchens", required=True,
                   help="comma-separated kitchen numbers, e.g. 813,422 (each fans out over every "
                        "rotation on disk: v<n>-00 .. v<n>-11)")
    p.add_argument("--subs", type=parse_subs, default=None,
                   help="optional comma-separated rotation filter 0..11; omit to process every rotation")
    p.add_argument("--out", required=True,
                   help="output directory for the generated goal files (NOT the goals corpus)")
    return p


def kitchen_task_name(num: int, sub: int = 0) -> str:
    """The registered task id for one (kitchen, rotation). --kitchens 813 fans out over every rotation
    that exists on disk, so a kitchen yields Isaac-Kitchen-v813-00 .. -v813-11, not just -00. sub
    defaults to 0 so callers naming 'the kitchen' get its first rotation."""
    return f"Isaac-Kitchen-v{num:02d}-{sub:02d}"


def kitchen_subs(num: int, kitchen_dir=None) -> list[int]:
    """The rotation sub-numbers that PHYSICALLY EXIST for kitchen <num>, ascending.

    Each is a separate kitchen_<num>_<sub>.usd (a 30*sub-degree rotation of the same kitchen); the
    reference corpus carries 00..11 per kitchen and one goal file per rotation. An emit fans out over
    exactly the rotations on disk — so it neither invents a rotation that has no USD (which would
    fail to open) nor silently stops at -00 (which would drop 11/12 of the dataset). Returns [] when
    the kitchen has no USD at all, which the caller reports as a skipped kitchen."""
    kitchen_dir = Path(kitchen_dir) if kitchen_dir is not None else KITCHEN_DIR
    pat = re.compile(rf"kitchen_{num:02d}_(\d+)\.usd$")
    subs = []
    for p in kitchen_dir.glob(f"kitchen_{num:02d}_*.usd"):
        m = pat.match(p.name)
        if m:
            subs.append(int(m.group(1)))
    return sorted(subs)


# ------------------------------------------------------------------------------------------------
# GPU-ONLY. Everything below needs the USD stage, so it is reached only from main(), only through
# importlib, and is never exercised by a test on this login node. See the report: the stage-to-
# KitchenScene enumeration and the authoring-app construction are the two pieces that cannot be
# verified here and must be confirmed by the controller's GPU run (Step 3 of the brief).
# ------------------------------------------------------------------------------------------------

#: kitchen_build.OBJECT_TYPES, the rigid-object families a role may bind to. Imported lazily inside
#: the GPU path; named here so _scene_from_stage reads top-to-bottom.
_OBJECT_TYPE_RE = re.compile(r"\d+$")
_ARTICULATION_FEATURE_RE = re.compile(r"^(drawer|door)(_\d+_\d+)?$")


def _scene_from_stage(stage, object_types) -> KitchenScene:
    """Enumerate an opened kitchen USD stage into the KitchenScene bind_roles consumes.

    GPU-ONLY, and the ONE piece of this CLI with no pre-existing reference to copy — so it is written
    to the conventions the rest of the pipeline already assumes and MUST be confirmed by a GPU run:

      objects        every prim at /world/<name> (a depth-3 path, as _generate_env_config already
                     enumerates them) whose name, with a trailing index stripped ('bowl0' -> 'bowl'),
                     is one of kitchen_build.OBJECT_TYPES. This is well-defined and low-risk;
                     object_index feeds nothing in the run config (task_bind.BoundPrim docstring),
                     only the bind-time ambiguity check.

      articulations  every descendant prim named like a drawer/door segment ('drawer_0_0'), with
                     feature 'drawer'/'door', plus 'handle' when it has a 'door_handle' child. This
                     is the reference task's '/world/base_cabinet/drawer_0_0' -> ['drawer','handle'].
                     THIS heuristic is the risk: if a scene names its segments differently the role
                     binds to nothing and the kitchen is (correctly, loudly) skipped.
    """
    UsdPrim = importlib.import_module("pxr").Usd  # noqa: F841 — kept for parity with the stage API

    objects: list[tuple[str, str]] = []
    articulations: dict[str, list[str]] = {}

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        parts = path.split("/")
        name = parts[-1]

        if len(parts) == 3 and parts[1] == "world":
            base = _OBJECT_TYPE_RE.sub("", name)
            if base in object_types:
                objects.append((base, path))

        if _ARTICULATION_FEATURE_RE.match(name):
            feature = name.split("_", 1)[0]
            features = [feature]
            if stage.GetPrimAtPath(path + "/door_handle"):
                features.append("handle")
            # The top-level furniture this articulation descends from (parts[2], the direct
            # /world/<furniture> child), as an ADDITIONAL feature -- lets a role bind by WHICH
            # PIECE OF FURNITURE carries it ("refrigerator"), not only by what kind of segment it
            # is ("door"/"drawer"). Needed now that a BARE "door"/"drawer" segment (no _N_N
            # suffix) matches too: base_cabinet's drawer and a bare refrigerator door are both
            # just "door"/"drawer" by kind alone. Purely additive to every existing entry -- it
            # never removes 'drawer'/'door'/'handle', so a role matching on those alone (every
            # articulation_with in this repo today) is unaffected; see task-3-report.md for the
            # before/after enumeration this claim was checked against, not just argued.
            if len(parts) > 2 and parts[1] == "world":
                features.append(parts[2])
            articulations[path] = features

    return KitchenScene(objects=objects, articulations=articulations)


def _boot_app() -> None:
    """Boot Isaac Sim (headless), exactly once. GPU-ONLY.

    MUST run before torch/numpy are imported. Isaac Sim's extensions bind to their own bundled numpy;
    if a conda-env numpy is already resident (which `import skills` pulls in through torch), it shadows
    the bundled one and the extensions fail to load ('cannot import broadcast_to from
    numpy.lib.stride_tricks', 'numpy.dtype size changed'). goal_generator uses exactly this order:
    launch the app first, import everything (skills, simvla_data_generator, torch) after. pxr/omni
    also only become importable once the app is up, so this also unblocks simvla_data_generator's
    top-level `from pxr import ...`. Reached through importlib to keep the module's AST-purity test
    green (a top-level `import isaacsim` would fail it and boot Omniverse on a plain import)."""
    global _APP_LAUNCHER
    if _APP_LAUNCHER is None:
        AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
        _APP_LAUNCHER = AppLauncher({"headless": True})
        _ = _APP_LAUNCHER.app                       # force the app up (pxr/omni + numpy bound)


def _authoring_app():
    """A GUI-free stand-in for the GoalGeneratorApp the plan() bodies read (app.stage, free_squares,
    kitchen_data, N_dir, log). GPU-ONLY. Constructed once and pointed at each kitchen in turn.
    Assumes _boot_app() has already run (main() calls it first, before any numpy import)."""
    _boot_app()
    sdg = importlib.import_module("simvla_data_generator")

    # A GUI-free stand-in for GoalGeneratorApp. The plan() bodies and _generate_env_config only read
    # stage / N_dir / free_squares / kitchen_data and call log() — none of which need tkinter — so we
    # subclass GoalGeneratorApp and SKIP its GUI __init__, reusing the real _generate_env_config /
    # _write_env_config_file / plan bodies as the single source of truth. No tk.Tk(), no X display:
    # this is what lets task_emit run headless on a GPU node. The interactive editor keeps using the
    # full GoalGeneratorApp unchanged. Defined here (not at module scope) because the base class needs
    # a booted app to import.
    class _HeadlessAuthoringApp(sdg.GoalGeneratorApp):
        def __init__(self):
            # Deliberately NOT super().__init__() (that builds the whole tkinter GUI). Set only the
            # state the non-GUI methods touch; per-kitchen values (stage, kitchen_data, free_squares)
            # are overwritten by _load_kitchen_scene, N_dir by the nav planner.
            self.stage = None
            self.N_dir = "N"
            self.free_squares = []
            self.kitchen_data = {}

        def log(self, message):
            print(f"[task_emit] {message}", flush=True)

        def _update_prim_list(self):
            pass  # GUI Scene Inspector only; the plan bodies never read all_prim_paths

    return _HeadlessAuthoringApp()


#: The one AppLauncher for the process. simvla_data_generator assumes a booted app at import; a second
#: launch hangs, so _authoring_app() boots exactly one and holds it here.
_APP_LAUNCHER = None


def _plan_fn_for(app) -> PlanFn:
    """plan_fn(skill_id, action, params) -> the authored goal, dispatched to the skill's real plan().

    This is exactly _process_goal_step's authored branch: `REGISTRY[skill].cls().plan(app, action,
    params)`. steps_for_kitchen only ever calls this for a NON-runtime skill (a runtime one carries
    goal=None and is never planned), so an authoring body is always what it reaches."""
    def plan_fn(skill_id: str, action: str, params: dict) -> Any:
        return REGISTRY[skill_id].cls().plan(app, action, params)
    return plan_fn


def _load_kitchen_scene(app, num: int, sub: int = 0):
    """Open kitchen <num>'s rotation <sub> USD, set app.stage / app.kitchen_data, and return the bound
    KitchenScene. GPU-ONLY. The env-config pass (which populates app.free_squares and writes the .py)
    is DELIBERATELY NOT run here — it moved to _write_kitchen_env_config, called by _generate AFTER
    bind_roles, because the env config's manipulated-object role must point at the prim the binding
    chose. plan_nav_to_prim reads free_squares and plan_arm_grasp reads kitchen_data, so that write
    still runs before any planning; it just runs post-bind.

    sub selects the rotation: it opens kitchen_<num>_<sub>.usd (a physically rotated kitchen) and sets
    kitchen_sub_num=sub, so BOTH the scene geometry and the grasp rotation match — the two must agree
    or the emitted grasp would be for the wrong orientation."""
    importlib.import_module("simvla_data_generator")
    omni_usd = importlib.import_module("omni.usd")
    kitchen_build = importlib.import_module("kitchen_build")

    num_str = f"{num:02d}"
    sub_str = f"{sub:02d}"
    usd_path = str(KITCHEN_DIR / f"kitchen_{num_str}_{sub_str}.usd")
    if not Path(usd_path).exists():
        raise BindingError(f"kitchen {num}: USD missing at {usd_path}")

    if not omni_usd.get_context().open_stage(usd_path):
        raise BindingError(f"kitchen {num}: failed to open USD {usd_path}")

    app.stage = omni_usd.get_context().get_stage()
    app.kitchen_data = {
        "task_name": "",
        "kitchen_num": num,
        "kitchen_type": "",
        "island_bound": [],
        "kitchen_sub_num": int(sub_str),
        "initial_pos_ranges": [],
        "initial_rot_yaw_range": [["yaw", -0.17453292519943295, 0.17453292519943295]],
        "goals": [],
    }
    app._update_prim_list()
    return _scene_from_stage(app.stage, set(kitchen_build.OBJECT_TYPES))


def _target_prim(t, bindings) -> str | None:
    """The manipulated object's prim name — the first rigid-body role that bound. The env config's
    success/OOB terminations and object randomizers attach to THIS prim: the template hardcodes it as
    'bowl0', and _write_env_config_file rebinds that name to whatever the scene actually manipulates.
    None when no rigid role bound (an all-articulation template) — the writer keeps its scene-derived
    default."""
    for r in t.roles:
        b = bindings.get(r.name)
        if b is not None and b.is_rigid_body:
            return b.name
    return None


def default_retry(t, bindings) -> dict:
    """The abort condition a template gets when it declares none.

    The operand is the template's FIRST RIGID-BODY ROLE — the same prim _target_prim picks to rebind
    the env config's randomizers — not a role that happens to be named 'target'. A template whose
    roles are all articulations has no such prim, and gets the robot-fell invariant alone.
    """
    manipulated = _target_prim(t, bindings)
    leaves = []
    if manipulated is not None:
        leaves.append({"obj_z": {"role": manipulated, "hi": 0.3}})
    leaves.append({"robot_fell": {"z": -0.1}})
    return {"any": leaves}


def resolved_condition(t, bindings) -> tuple[dict, dict]:
    """(success, retry) for ONE kitchen, with every '@role' replaced by the name that kitchen's
    SCENE actually carries. Pure — no stage, no app — so the resolution the env config is written
    from is testable on a laptop.

    THE OPERAND IS THE SCENE ENTITY, NOT THE BOUND PRIM. mdp.composed looks a spec's role/
    target_role up in env.scene.rigid_objects / env.scene.articulations, which hold FIRST-LEVEL
    prims only. A rigid role's bound prim is its own scene entity; an articulation role's is its
    owning articulation (/world/base_cabinet/drawer_0_0 -> 'base_cabinet'), which is also what the
    predicates' `joint`/`body` params are matched against. See BoundPrim.scene_entity.

    A template that declares no retry gets default_retry — which reads the bindings, not the spec.
    """
    name_of = {role: b.scene_entity for role, b in bindings.items()}
    success = resolve_roles(t.success, name_of)
    retry = resolve_roles(t.retry, name_of) if t.retry is not None else default_retry(t, bindings)
    return success, retry


def _py_literal(value) -> str:
    """`value`, rendered as PYTHON SOURCE — not the JSON text json.dumps alone would produce.

    STRUCTURAL, and it has to be. json.dumps writes JSON's true/false/null, which pasted into a .py
    file are three undefined NAMEs (the generated env config would NameError at import, not at
    emit, the first time a spec carries a Bool — obj_near_prim.xy_only is the one today). Rewriting
    them in the DUMPED TEXT is what this used to do, keyed on '": true' on the reasoning that a
    string value spelling 'true' dumps as ': "true"', with a quote where the pattern needs the
    literal 't'. The hole is the escape: json.dumps writes an embedded quote as \\", so a Text param
    valued `x": true` dumps as '"x\\": true"' and the rewrite fires INSIDE the string, round-tripping
    as `x": True` — valid Python, silently the wrong value, and no error anywhere. So the tree is
    walked and each literal emitted for what it IS.

    Output is byte-identical to json.dumps for every spec that carries no bool/None (same separators
    and the same repr for numbers and strings), so regenerating an existing config is still a no-op.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):                 # BEFORE int: bool is a subclass of int
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            # json.dumps writes Infinity/-Infinity/NaN (and json.loads reads them back), so a
            # hand-written template CAN carry one through validate_spec's number check. Those are
            # undefined names in Python too, and unlike true/false they have no correct rendering
            # here — `float('inf')` in a config is a bound that no state can satisfy or violate.
            # validate_spec refuses them at authoring; this is the backstop for anything that
            # reaches the writer without being validated.
            raise SpecError(
                f"a composed condition carries the non-finite number {value!r}. json.dumps writes "
                f"it as Infinity/NaN, which is an undefined NAME in Python — the generated env "
                f"config would NameError at import. Give the bound a finite value."
            )
        return repr(value)
    if isinstance(value, str):
        # json.dumps of a str is also a valid Python str literal with the same value (\", \\, \n,
        # \uXXXX all mean the same in both), and it keeps the double quotes the existing configs
        # are written with.
        return json.dumps(value)
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise SpecError(
                    f"a composed condition has the non-string key {key!r}. A spec is JSON: its "
                    f"keys are predicate ids and param names, and a config written with any other "
                    f"key would not survive a round trip through the template file."
                )
        return "{" + ", ".join(f"{json.dumps(k)}: {_py_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_py_literal(v) for v in value) + "]"
    raise SpecError(
        f"a composed condition carries {value!r} ({type(value).__name__}), which is not a JSON "
        f"value. A spec is data that has to survive a template file; only objects, arrays, "
        f"strings, finite numbers, booleans and null may reach the env config."
    )


def terminations_block(success_spec: dict, retry_spec: dict) -> str:
    """The body of TerminationsCfg, as source text.

    The spec is DATA in the term's params, so the block is identical for every task except for the
    literal it carries — which is the whole point: a radius change is a template edit, not a
    regeneration of twelve env configs.
    """
    return (
        f"success = DoneTerm(\n"
        f"\t\tfunc=mdp.composed,\n"
        f"\t\tparams={{\"spec\": {_py_literal(success_spec)}}}\n"
        f")\n"
        f"retry = DoneTerm(\n"
        f"\t\tfunc=mdp.composed,\n"
        f"\t\tparams={{\"spec\": {_py_literal(retry_spec)}}}\n"
        f")"
    )


def _write_kitchen_env_config(app, num: int, sub: int, primary_obj, t=None, bindings=None):
    """Write kitchen <num>_<sub>'s env-config .py and populate app.free_squares / kitchen_data,
    binding the config's manipulated-object role to `primary_obj`. Split out of _load_kitchen_scene so
    it runs AFTER bind_roles: only the binding knows which prim the task manipulates, and the success
    term must point at that prim. Returns kitchen_meta (kitchen_data minus goals) for the goal file.

    `t` and `bindings` are the template and THIS kitchen's bindings — given, the template's own
    success/retry condition is resolved (roles -> this kitchen's prim names) and written into the
    config in place of the mdp.task2/mdp.OOB default. Omitted (either), the config keeps emitting
    that default verbatim — the no-template GUI path _generate_env_config also serves."""
    success_spec = retry_spec = None
    if t is not None and bindings is not None:
        success_spec, retry_spec = resolved_condition(t, bindings)
    app._generate_env_config(f"{num:02d}", f"{sub:02d}", primary_obj=primary_obj,
                             success_spec=success_spec, retry_spec=retry_spec)
    return {k: v for k, v in app.kitchen_data.items() if k not in ("goals",)}


def _generate(t: TaskTemplate, kitchens: list[int], out_dir: Path,
              selected_subs: list[int] | None = None) -> tuple[list[str], list[str]]:
    """GPU-ONLY. For each requested kitchen, IN ONE PASS while its stage is live: open + enumerate,
    bind, plan, emit. Returns (written, skipped); every skip is named, none is swallowed.

    SINGLE pass on purpose. The plan() bodies read app.stage / app.free_squares / app.kitchen_data,
    which _load_kitchen_scene overwrites for each kitchen — so a kitchen MUST be planned before the
    next one's scene replaces that state. The previous two-pass form (enumerate every scene, then
    plan them all via the batch emit()) planned EVERY kitchen against the last-opened stage, so every
    emitted pose was computed against the wrong kitchen's geometry (verified on GPU: kitchen 813's nav
    used the last batch kitchen's bowl position). Binding stays per-kitchen too (bind_roles, not the
    batch bind_many) so the whole open->bind->plan->emit chain runs before the stage moves."""
    validate_template(t)
    validate_sequence(t)                       # refuse an incoherent script before any file is written
    out = Path(out_dir)
    _refuse_the_goals_corpus(out)
    out.mkdir(parents=True, exist_ok=True)

    app = _authoring_app()
    plan_fn = _plan_fn_for(app)

    written: list[str] = []
    skipped: list[str] = []

    for num in kitchens:
        available_subs = kitchen_subs(num)
        if not available_subs:
            # No rotation USD at all — the whole kitchen is missing, named once (not 12 times).
            skipped.append(f"{kitchen_task_name(num)}: no kitchen_{num:02d}_*.usd on disk ({KITCHEN_DIR})")
            continue
        subs = available_subs if selected_subs is None else [s for s in selected_subs if s in available_subs]
        if selected_subs is not None:
            missing = sorted(set(selected_subs) - set(available_subs))
            for sub in missing:
                skipped.append(f"{kitchen_task_name(num, sub)}: requested rotation USD is missing")
            if not subs:
                continue

        # Every rotation is its own task: a separate USD, its own grasp angle, its own goal file.
        # One rotation that can't plan must not lose the other eleven, so each is bound/planned/emitted
        # independently and every skip is named — exactly the per-kitchen contract, now per rotation.
        for sub in subs:
            task_name = kitchen_task_name(num, sub)
            try:
                scene = _load_kitchen_scene(app, num, sub)   # opens THIS rotation's stage
            except (BindingError, SpecError) as exc:
                skipped.append(f"{task_name}: {exc}")
                continue
            except Exception as exc:  # a scene config that cannot import (the mdp.sink landmine), etc.
                skipped.append(f"{task_name}: {type(exc).__name__}: {exc}")
                continue

            try:
                bindings = bind_roles(t, scene)
            except (BindingError, SpecError) as exc:
                skipped.append(f"{task_name}: {exc}")
                continue

            try:
                # Bind is done, so the env config can name the manipulated prim: write it (populating
                # free_squares) THEN plan + emit — all while THIS rotation's stage is still live.
                meta = _write_kitchen_env_config(app, num, sub, _target_prim(t, bindings),
                                                 t=t, bindings=bindings)
                path = emit_one(t, task_name, bindings, plan_fn, out, kitchen_meta=meta)
                written.append(path)
            except SCENE_FAILURES as exc:
                reason = f"{task_name}: {type(exc).__name__}: {exc}"
                skipped.append(reason)
                print(f"[task_emit] SKIP {reason}", flush=True)
            except Exception as exc:  # env-config generation itself failing is not a bad kitchen
                import traceback
                reason = f"{task_name}: {type(exc).__name__}: {exc}"
                skipped.append(reason)
                print(f"[task_emit] ERROR {reason}\n{traceback.format_exc()}", flush=True)

    return written, skipped


def _load_skill_registry() -> None:
    """Populate skill_contract.REGISTRY by importing the skill declarations.

    load_template -> validate_template checks every step's skill against REGISTRY, which the @skill
    decorators fill on `import skills`. Without this, main() validates against an EMPTY registry and
    refuses every real template ('no skill <id> in the registry') before Omniverse is even reached —
    which is why the CLI had never actually run a template end-to-end. Reached through importlib (not a
    top-level import) because `skills` pulls torch, which this module's AST-purity test forbids at
    import; it is safe on a login node (torch imports without a GPU) and needed only when running."""
    importlib.import_module("skills")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Cheap, numpy-free refusals FIRST — before three minutes of boot.
    kitchens = parse_kitchens(args.kitchens)
    selected_subs = args.subs
    out_dir = guard_out_dir(args.out)

    # Boot Isaac Sim BEFORE importing skills: skills pulls torch+numpy, and a numpy resident before
    # the app boots breaks Isaac's extension load (see _boot_app). This costs the fast skill-validation
    # refusal — the template's skills are now checked after the ~3-minute boot rather than before — but
    # the common CLI mistakes (bad --kitchens / --out) are still refused up front, and it is the only
    # order in which the app comes up at all.
    _boot_app()
    _load_skill_registry()          # now safe: numpy loads after Isaac has bound its own
    t = load_template(args.template)

    # Only now boot Omniverse (and inject the six plan() bodies via the import).
    written, skipped = _generate(t, kitchens, out_dir, selected_subs=selected_subs)

    print(f"{len(written)} written, {len(skipped)} skipped", flush=True)
    for path in written:
        print(f"  wrote  {path}", flush=True)
    for line in skipped:
        print(f"  SKIP   {line}", flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        import traceback
        print(traceback.format_exc(), flush=True)
    finally:
        importlib.import_module("workflow_status").record_status(exit_code)
        if _APP_LAUNCHER is not None:
            importlib.import_module("isaaclab.sim").SimulationContext.clear_instance()
            _APP_LAUNCHER.app.close()
    sys.exit(exit_code)
