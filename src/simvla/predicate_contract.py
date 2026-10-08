"""SimVLA: one place to declare a condition primitive.

Stdlib only (plus skill_contract's param types) — no torch, isaaclab or omni. The composer serves
this registry as a palette, exactly as it serves the skill registry, and neither may boot Omniverse.

Every success function in kitchen/mdp/terminations.py is an AND over the same eight sub-conditions,
re-inlined by copy-paste; only the parameters vary. `close_home` alone appears nine times with five
different radii. These eight declarations are those sub-conditions, uninlined — so a task states its
condition once, in the template, and the runtime, the composer and the eval all read that one
statement.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .skill_contract import Bool, Choice, Float

ROLE_PREFIX = "@"

PHYSICAL = "physical"
SCRIPT = "script"


class SpecError(Exception):
    """A composed condition cannot be resolved. Raised at emit, never at run time."""


# ---- param declarations. Bool/Choice/Float are reused from the skill contract so the composer
# ---- renders both palettes with one widget vocabulary. ----

@dataclass(frozen=True)
class RoleRef:
    """A '@role' reference, resolved to a bound prim name at emit."""
    name: str


@dataclass(frozen=True)
class OptFloat:
    """A bound that may be omitted entirely. `obj_z` with only `hi` is 'below hi'."""
    name: str


@dataclass(frozen=True)
class Text:
    """A body or joint NAME. A bare int is also accepted for `joint` and means a raw index.

    `required` is declared, not inferred from the type: joint_pos.joint MUST be given (a joint range
    with no joint means nothing), while obj_near_prim.body may be omitted and defaults to body 0 —
    which is how sink() and task1() measure to the corpus.
    """
    name: str
    default: str = ""
    required: bool = True


Param = RoleRef | OptFloat | Text | Bool | Choice | Float


@dataclass(frozen=True)
class PredicateSpec:
    id: str
    label: str
    params: list[Param]
    phase: str


REGISTRY: dict[str, PredicateSpec] = {}


def declare(*, id: str, label: str, params: list[Param], phase: str) -> None:
    if phase not in (PHYSICAL, SCRIPT):
        raise SpecError(f"predicate {id!r} declares phase {phase!r}, not {PHYSICAL!r}/{SCRIPT!r}")
    if id in REGISTRY:
        raise SpecError(f"duplicate predicate id {id!r}")
    REGISTRY[id] = PredicateSpec(id=id, label=label, params=list(params), phase=phase)


ARM = Choice("arm", ["right", "left", "both"], default="right")

declare(
    id="obj_z", label="Object height in a band", phase=PHYSICAL,
    params=[RoleRef("role"), OptFloat("lo"), OptFloat("hi")],
)
declare(
    id="obj_near_eef", label="Object near a gripper", phase=PHYSICAL,
    params=[RoleRef("role"), ARM, Float("radius", 0.15)],
)
declare(
    id="obj_between_eefs", label="Object sandwiched between both palms", phase=PHYSICAL,
    # No ARM here (see predicates_math.obj_between_eefs's docstring): the predicate is inherently
    # a relation over both palms, not a per-arm check with a "both" option. `near` carries no
    # physically-safe default (same reasoning as obj_near_eef's `radius`) -- 0.10 m is only a
    # composer starting value, in the middle of this repo's own measured per-object range
    # (0.071-0.1055 m for the six wide objects object_bimanual_lift.py targets); every real
    # caller must size it to the object at hand.
    params=[RoleRef("role"), Float("near", 0.10)],
)
declare(
    id="obj_near_prim", label="Object near another prim", phase=PHYSICAL,
    params=[
        RoleRef("role"), RoleRef("target_role"), Text("body", required=False),
        Choice("anchor", ["body", "door_midpoint"], default="body"),
        Float("radius", 0.12), OptFloat("z_override"), Bool("xy_only", False),
    ],
)
declare(
    id="eef_home", label="Gripper back at home", phase=PHYSICAL,
    params=[ARM, Float("radius", 0.12)],
)
declare(
    id="joint_pos", label="Articulation joint in a range", phase=PHYSICAL,
    params=[RoleRef("role"), Text("joint"), OptFloat("lo"), OptFloat("hi")],
)
declare(
    id="gripper_open", label="Gripper open", phase=PHYSICAL,
    params=[ARM, Float("gap", 0.079)],
)
declare(
    id="robot_fell", label="Robot base fell", phase=PHYSICAL,
    params=[Float("z", -0.1)],
)
declare(
    id="last_subtask", label="On the final script step", phase=SCRIPT, params=[],
)

#: Primitives whose parameters are a pair of optional bounds. At least one must be given, or the
#: leaf is unconditionally true — the same bug as an empty `all`.
_BOUNDED = {"obj_z": ("lo", "hi"), "joint_pos": ("lo", "hi")}


# ---- walking ----

def _is_leaf(spec) -> bool:
    return isinstance(spec, dict) and len(spec) == 1 and next(iter(spec)) not in ("all", "any", "not")


def leaves(spec) -> list[tuple[str, dict]]:
    """Every (predicate_id, params) in the tree, in order."""
    if not isinstance(spec, dict):
        return []
    if "all" in spec or "any" in spec:
        out = []
        for sub in spec.get("all", spec.get("any", [])) or []:
            out.extend(leaves(sub))
        return out
    if "not" in spec:
        return leaves(spec["not"])
    if _is_leaf(spec):
        pid, params = next(iter(spec.items()))
        return [(pid, params if isinstance(params, dict) else {})]
    return []


def role_ref(value) -> str | None:
    """'@target' -> 'target'. Anything else -> None."""
    if isinstance(value, str) and value.startswith(ROLE_PREFIX):
        return value[len(ROLE_PREFIX):]
    return None


# ---- validation ----

def omittable_params(declared: PredicateSpec) -> set[str]:
    """The params a spec may leave out of `declared`'s leaf.

    OptFloat and an explicitly-optional Text may be omitted outright (an omitted bound is
    unbounded; an omitted body means body 0). Choice and Bool are also omittable because both carry
    a self-consistent default: Choice.__post_init__ requires its default be one of its own options,
    and Bool's is always False. Float is deliberately NOT in this set even though it carries a
    `default` — that default is a composer starting value, not a physically safe fallback (a radius
    of 0.0 would make the check unsatisfiable), so a Float param must always be given explicitly.

    Named and public because THE IMPLEMENTATION MUST AGREE WITH IT. compile_spec passes only the
    params a spec actually carries, so every name in this set must have a Python-side default in
    predicates_math's function of the same id, with the same value where one is declared — or a
    spec validate_spec accepts raises TypeError at Isaac Lab env load, the one place no test can
    reach. predicates_math is torch-only and cannot import this module to find that out, so
    test_predicates_math holds the two together through this function.
    """
    return {
        p.name for p in declared.params
        if isinstance(p, OptFloat)
        or (isinstance(p, Text) and not p.required)
        or isinstance(p, Choice)
        or isinstance(p, Bool)
    }


def _validate_node(spec, role_names, problems, where) -> None:
    if not isinstance(spec, dict):
        problems.append(f"{where}: a condition must be an object, got {spec!r}")
        return

    for op in ("all", "any"):
        if op in spec:
            items = spec[op]
            if not isinstance(items, list) or not items:
                problems.append(
                    f"{where}: {op!r} is empty — an empty 'all' is vacuously true, which is a "
                    f"condition that always fires"
                )
                return
            for i, sub in enumerate(items):
                _validate_node(sub, role_names, problems, f"{where}.{op}[{i}]")
            return

    if "not" in spec:
        _validate_node(spec["not"], role_names, problems, f"{where}.not")
        return

    if len(spec) != 1:
        problems.append(f"{where}: a leaf must be exactly one predicate, got keys {sorted(spec)}")
        return

    pid, params = next(iter(spec.items()))
    declared = REGISTRY.get(pid)
    if declared is None:
        problems.append(
            f"{where}: no predicate {pid!r} in the registry (have: {', '.join(sorted(REGISTRY))})"
        )
        return
    if not isinstance(params, dict):
        problems.append(f"{where}: {pid!r} params must be an object, got {params!r}")
        return

    by_name = {p.name: p for p in declared.params}
    optional = omittable_params(declared)

    for key, value in params.items():
        p = by_name.get(key)
        if p is None:
            problems.append(
                f"{where}: {pid!r} declares no param {key!r} "
                f"(it declares {sorted(by_name) or ['nothing']})"
            )
            continue
        if isinstance(p, RoleRef):
            ref = role_ref(value)
            if ref is None:
                # A LITERAL prim/entity name, not a '@role'. Permitted, matching the two layers
                # around this one: task_emit's PrimPath resolution documents "a literal prim path:
                # a template may write one", and resolve_roles() already passes literals through
                # untouched. The runtime is not weakened by this -- composed.py raises a KeyError
                # naming every available entity when a literal matches nothing in the scene. What
                # a literal buys is addressing FIXTURES that exist on every kitchen under one name
                # (the first user: target_role="sink_cabinet") without inventing a role-binding
                # category for things that never vary.
                if not isinstance(value, str) or not value:
                    problems.append(
                        f"{where}: {pid}.{key} must be a '@role' reference or a literal entity "
                        f"name, got {value!r}")
            elif ref not in role_names:
                problems.append(f"{where}: {pid}.{key} is {value!r}, which names no declared role")
        elif isinstance(p, Choice):
            if value not in p.options:
                problems.append(f"{where}: {pid}.{key}={value!r} is not one of {p.options}")
        elif isinstance(p, Text):
            if not isinstance(value, (str, int)):
                problems.append(f"{where}: {pid}.{key} must be a name or an index, got {value!r}")
        elif isinstance(p, Bool):
            if not isinstance(value, bool):
                problems.append(f"{where}: {pid}.{key} must be true/false, got {value!r}")
        else:                                   # Float / OptFloat
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{where}: {pid}.{key} must be a number, got {value!r}")
            elif not math.isfinite(value):
                # json.loads ACCEPTS JSON's Infinity/-Infinity/NaN and json.dumps writes them back,
                # so a hand-written template can carry one all the way here — and it is a real
                # float, so the type check above passes. Pasted into the generated env config those
                # are three undefined Python NAMEs (the config NameErrors at import), and there is
                # no sane meaning to give them anyway: an infinite bound is a check nothing can
                # fail, a NaN bound is a check nothing can pass (every comparison is False).
                problems.append(
                    f"{where}: {pid}.{key} is {value!r} — a bound must be a FINITE number. JSON's "
                    f"Infinity/-Infinity/NaN survive a round trip through a template file, but "
                    f"they are undefined names in the generated env config, and neither bound can "
                    f"ever be meaningfully compared against."
                )

    for name, p in by_name.items():
        if name not in params and name not in optional:
            problems.append(f"{where}: {pid!r} requires {name!r}, which is not given")

    bounds = _BOUNDED.get(pid)
    if bounds and not any(b in params for b in bounds):
        problems.append(
            f"{where}: {pid!r} gives neither {bounds[0]!r} nor {bounds[1]!r} — with no bound it is "
            f"unconditionally true"
        )


def validate_spec(spec, role_names, kind: str) -> list[str]:
    """Every problem at once. `kind` is 'success' or 'retry'."""
    problems: list[str] = []
    _validate_node(spec, set(role_names), problems, kind)
    if problems:
        return problems

    if kind == "success":
        found = leaves(spec)
        if found and all(REGISTRY[pid].phase == SCRIPT for pid, _ in found):
            problems.append(
                f"success is made only of script-phase predicates "
                f"({', '.join(sorted({pid for pid, _ in found}))}) — that marks a demo successful "
                f"for merely finishing the script, with no physical check. Add a physical predicate."
            )
    return problems


def spec_warnings(spec, kind: str) -> list[str]:
    """Non-fatal traps. It cannot know the counter height, so it must not refuse."""
    if kind != "success":
        return []
    found = [pid for pid, _ in leaves(spec)]
    if "obj_near_eef" in found and "obj_z" not in found:
        return [
            "success checks obj_near_eef with no obj_z height check: proximity alone is a "
            "false-positive trap. An object resting on the ~0.82 m counter is within 0.2 m of a "
            "passing gripper — simvla_eval.py:695 records this scoring a non-grasping policy 4/20 "
            "before a lift requirement was added. Add obj_z with a `lo` above the support surface."
        ]
    return []


# ---- emit-time resolution ----

def resolve_roles(spec, bindings: dict[str, str]):
    """Replace every '@role' with the prim name it bound to. Returns a new spec; does not mutate."""
    out = copy.deepcopy(spec)
    _resolve_node(out, bindings)
    return out


def _resolve_node(spec, bindings) -> None:
    if not isinstance(spec, dict):
        return
    for op in ("all", "any"):
        if op in spec:
            for sub in spec[op]:
                _resolve_node(sub, bindings)
            return
    if "not" in spec:
        _resolve_node(spec["not"], bindings)
        return
    pid, params = next(iter(spec.items()))
    declared = REGISTRY.get(pid)
    if declared is None or not isinstance(params, dict):
        return
    for p in declared.params:
        if isinstance(p, RoleRef) and p.name in params:
            ref = role_ref(params[p.name])
            if ref is None:
                continue
            if ref not in bindings:
                raise SpecError(
                    f"{pid}.{p.name} references role {ref!r}, which did not bind in this kitchen"
                )
            params[p.name] = bindings[ref]


def unregistered_predicates(spec) -> list[str]:
    """Every leaf predicate id the registry does not declare, deduped and sorted.

    physical_only DROPS such a leaf, indistinguishably from how it drops a script-phase one — so a
    caller that only asks "is anything physical left?" scores a REDUCED condition and reports the
    loss as "script-phase leaves", while compile_spec raises KeyError for the very same spec. The
    generator and the eval must not disagree about what a condition means, so eval asks this first
    and refuses. (validate_spec already refuses an unknown id at authoring; this is for a spec that
    reaches a runtime from an env config written before the id was renamed or removed.)
    """
    return sorted({pid for pid, _ in leaves(spec) if pid not in REGISTRY})


def physical_only(spec):
    """The spec with script-phase leaves dropped, or None if nothing physical remains.

    `last_subtask` reads isaaclab.simvla.variable.env_goal_indices, which is data-generation state
    machine state the eval never advances — see simvla_eval.py:642.
    """
    out = _physical_node(spec)
    return out


def _physical_node(spec):
    if not isinstance(spec, dict):
        return None
    for op in ("all", "any"):
        if op in spec:
            kept = [k for k in (_physical_node(s) for s in spec[op]) if k is not None]
            return {op: kept} if kept else None
    if "not" in spec:
        inner = _physical_node(spec["not"])
        return {"not": inner} if inner is not None else None
    pid = next(iter(spec))
    declared = REGISTRY.get(pid)
    if declared is None or declared.phase == SCRIPT:
        return None
    return copy.deepcopy(spec)
