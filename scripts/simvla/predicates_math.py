"""SimVLA: the condition primitives' geometry, as pure functions of tensors.

torch only — no isaaclab, no omni. The adapter that turns a live `env` into an EvalContext lives in
kitchen/mdp/composed.py; everything below is a function of its arguments, and therefore testable.

Two reasons that split matters. The math inlined nine times in terminations.py has never been under
test because it needs a GPU and a live stage. And terminations.py:16 declares home_r/home_l as
`cuda:0` tensors at module import, pinning the device at load — here the home poses are arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EvalContext:
    """Everything the primitives may read. Filled by the adapter; a test builds only what it needs."""
    num_envs: int
    device: str = "cpu"
    obj_pos_w: dict = field(default_factory=dict)
    art_body_pos_w: dict = field(default_factory=dict)
    art_body_names: dict = field(default_factory=dict)
    art_joint_pos: dict = field(default_factory=dict)
    art_joint_names: dict = field(default_factory=dict)
    eef_pos_w: dict = field(default_factory=dict)
    eef_pos_base: dict = field(default_factory=dict)
    home: dict = field(default_factory=dict)
    gripper_gap: dict = field(default_factory=dict)
    base_pos_w: torch.Tensor | None = None
    goal_index: torch.Tensor | None = None
    max_steps: int | None = None
    #: {"base"|"grip_l"|"grip_r" -> Tensor(N,)}: contact force norm in newtons between that robot
    #: body and the refrigerator door. Empty when the scene has no contact sensors configured.
    #:
    #: A FACT, NOT AN INFERENCE. Every causality claim in this task was previously derived from
    #: door angle plus jaw gap, which certified 16 base collisions as successes.
    door_contact: dict = field(default_factory=dict)

    def true(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)


#: The default of predicate_contract's shared `arm` Choice, RESTATED (not imported): this module is
#: torch-only and must not import predicate_contract, which is what keeps the registry importable
#: without torch. validate_spec lets a Choice param be omitted precisely because it "takes its
#: declared default" — so the default has to exist on the Python side too, or a spec that validates
#: clean (`{"eef_home": {"radius": 0.18}}`) dies at Isaac Lab env load with
#: "missing 1 required keyword-only argument: 'arm'", which is the one place no test can reach.
#: test_predicates_math pins this against REGISTRY so the two declarations cannot drift, for these
#: three primitives and for any Choice/Bool param added later.
_DEFAULT_ARM = "right"


def _arms(arm: str) -> tuple[str, ...]:
    return ("right", "left") if arm == "both" else (arm,)


def _pos(ctx: EvalContext, name: str) -> torch.Tensor:
    try:
        return ctx.obj_pos_w[name]
    except KeyError:
        raise KeyError(
            f"no rigid object {name!r} in the scene (have: {', '.join(sorted(ctx.obj_pos_w))})"
        ) from None


def _anchor_pos(ctx: EvalContext, name: str, body, anchor: str) -> torch.Tensor:
    """The point of `name` to measure to."""
    if name in ctx.art_body_pos_w:
        bodies = ctx.art_body_pos_w[name]
        names = ctx.art_body_names.get(name, [])
        if anchor == "door_midpoint":
            # bodies[1] and bodies[2] are the doors. Whichever world axis they share is the axis
            # the corpus centre must be averaged along to land in the basin. Re-implemented in
            # sink(), task1(), task1_deploy() and task1_molmospace().
            out = bodies[:, 0, :].clone()
            if torch.allclose(bodies[:, 2, 0], bodies[:, 1, 0], rtol=1e-5, atol=1e-6):
                out[:, 0] = (out[:, 0] + bodies[:, 1, 0]) / 2
            elif torch.allclose(bodies[:, 2, 1], bodies[:, 1, 1], rtol=1e-5, atol=1e-6):
                out[:, 1] = (out[:, 1] + bodies[:, 1, 1]) / 2
            return out
        if body in (None, ""):
            return bodies[:, 0, :].clone()
        if isinstance(body, int):
            return bodies[:, body, :].clone()
        if body not in names:
            raise KeyError(f"{name!r} has no body {body!r} (have: {', '.join(names)})")
        return bodies[:, names.index(body), :].clone()
    return _pos(ctx, name).clone()


def obj_z(ctx, *, role, lo=None, hi=None):
    z = _pos(ctx, role)[:, 2]
    out = ctx.true()
    # Strict, matching the only precedent: task2's `inside = (obj_z > 0.6) & (obj_z < 0.8)`
    # (terminations.py:138). These primitives replace that inlined check, so the boundary
    # operator must match it exactly, not just agree in practice.
    if lo is not None:
        out = out & (z > lo)
    if hi is not None:
        out = out & (z < hi)
    return out


def obj_near_eef(ctx, *, role, radius, arm=_DEFAULT_ARM):
    obj = _pos(ctx, role)
    out = ctx.true()
    # Strict, matching the only precedent: task3/task3_molmospace's `bottle_check`/`mug_check`,
    # e.g. `torch.linalg.norm(bottle_pos - eef_r_pos_w, dim=-1) < 0.2` (terminations.py:89-90,
    # 123-124). Same reasoning as obj_z: reproduce the inlined comparison exactly.
    for a in _arms(arm):
        out = out & (torch.linalg.norm(obj - ctx.eef_pos_w[a], dim=-1) < radius)
    return out


#: Threshold on the normalised dot product between (object -> right palm) and (object -> left
#: palm). cos(180deg) = -1 is perfect opposition; cos(90deg) = 0 is a palm off to the side,
#: contributing nothing to a sandwich. A bare `< 0` would accept anything past perpendicular --
#: including two palms barely more than 90 degrees apart, which is not a hold. `< -0.5` requires
#: the angle BETWEEN the two "object -> palm" vectors to exceed 120 degrees: genuine opposition,
#: with 60 degrees of slack off a perfect 180 to absorb a real squeeze's asymmetry -- the two
#: palms are placed at an equal-and-opposite HORIZONTAL offset from the object's centre
#: (object_bimanual_lift.py's own plan_arm_squeeze arithmetic) but at the SAME fixed height,
#: which is only exactly the object's own centre height by coincidence, so the two vectors are
#: never exactly antiparallel even in a genuine two-handed hold.
_OPPOSITION_MAX_COS = -0.5

#: Below this, a "vector from object to palm" is numerically zero -- not a small number, an
#: undefined DIRECTION. Dividing by it (even after clamping the denominator below, which is what
#: stops the NaN) produces a cosine that looks like a real float but encodes no actual angle.
#: Treated as an automatic fail: a palm sitting exactly on the object's own centre point is a
#: degenerate reading, not evidence of a hold.
_EEF_DEGENERATE_EPS = 1e-6


def obj_between_eefs(ctx, *, role, near):
    """Both palms actually SANDWICH the object -- opposite sides, both within contact range.

    obj_near_eef's radius-only test is satisfied by both palms on the SAME side of the object
    (near it, never opposing it), by one palm resting on the object while the other drifts
    anywhere within a generous sphere, or by a palm coincident with the object's own centre --
    none of those is a two-handed hold. This predicate adds what a radius alone cannot express:
    the vector from the object to the right palm and the vector from the object to the left palm
    must point in substantially OPPOSITE directions (_OPPOSITION_MAX_COS), AND each palm must sit
    within `near` of the object's own centre.

    `near` is deliberately NOT a fixed radius baked in here -- a single fixed sphere across every
    object size is exactly the defect this predicate replaces. The caller sizes it to the
    object's own half-width plus the robot's own pad offset (object_bimanual_lift.py derives one
    `near` per object from its measured AABB).

    No `arm` Choice, unlike obj_near_eef/eef_home/gripper_open: "between" is inherently a relation
    over BOTH palms at once, so there is no single-arm reading of "sandwiched" to choose between.
    """
    obj = _pos(ctx, role)
    to_r = ctx.eef_pos_w["right"] - obj
    to_l = ctx.eef_pos_w["left"] - obj
    r_norm = torch.linalg.norm(to_r, dim=-1)
    l_norm = torch.linalg.norm(to_l, dim=-1)

    # A zero-length vector (a palm at the object's own centre) has no direction to compare.
    # clamp_min below stops the literal NaN, but "cos == 0" from a 0/eps division is a plausible-
    # looking number that would otherwise silently fall out of the opposed test on its own --
    # this makes the fail explicit and independent of exactly where _OPPOSITION_MAX_COS sits.
    degenerate = (r_norm < _EEF_DEGENERATE_EPS) | (l_norm < _EEF_DEGENERATE_EPS)
    cos = (to_r * to_l).sum(dim=-1) / (r_norm * l_norm).clamp_min(_EEF_DEGENERATE_EPS)
    opposed = cos < _OPPOSITION_MAX_COS

    return opposed & (r_norm < near) & (l_norm < near) & ~degenerate


def obj_near_prim(ctx, *, role, target_role, radius, body=None, anchor="body",
                  z_override=None, xy_only=False):
    obj = _pos(ctx, role)
    tgt = _anchor_pos(ctx, target_role, body, anchor)
    if z_override is not None:
        tgt[:, 2] = z_override
    if xy_only:
        return torch.linalg.norm(obj[:, :2] - tgt[:, :2], dim=-1) <= radius
    return torch.linalg.norm(obj - tgt, dim=-1) <= radius


def eef_home(ctx, *, radius, arm=_DEFAULT_ARM):
    out = ctx.true()
    for a in _arms(arm):
        home = ctx.home[a].to(ctx.eef_pos_base[a].device)
        out = out & (torch.linalg.norm(ctx.eef_pos_base[a] - home, dim=-1) < radius)
    return out


def joint_pos(ctx, *, role, joint, lo=None, hi=None):
    q = ctx.art_joint_pos[role]
    if isinstance(joint, int):
        idx = joint
    else:
        names = ctx.art_joint_names.get(role, [])
        if joint not in names:
            raise KeyError(f"{role!r} has no joint {joint!r} (have: {', '.join(names)})")
        idx = names.index(joint)
    value = q[:, idx]
    out = ctx.true()
    # Strict, matching the only precedent: task2's
    # `drawer_closed = env.scene["base_cabinet"].data.joint_pos[:, 0] < 0.03` (terminations.py:164).
    if lo is not None:
        out = out & (value > lo)
    if hi is not None:
        out = out & (value < hi)
    return out


def gripper_open(ctx, *, gap, arm=_DEFAULT_ARM):
    out = ctx.true()
    for a in _arms(arm):
        out = out & (ctx.gripper_gap[a] > gap)
    return out


def robot_fell(ctx, *, z):
    return ctx.base_pos_w[:, 2] < z


def last_subtask(ctx):
    if ctx.goal_index is None or ctx.max_steps is None:
        raise ValueError(
            "last_subtask needs the script state (goal_index / max_steps), which this caller does "
            "not have. During eval, drop script-phase leaves with predicate_contract.physical_only."
        )
    return ctx.goal_index == (ctx.max_steps - 1)


IMPL = {
    "obj_z": obj_z,
    "obj_near_eef": obj_near_eef,
    "obj_between_eefs": obj_between_eefs,
    "obj_near_prim": obj_near_prim,
    "eef_home": eef_home,
    "joint_pos": joint_pos,
    "gripper_open": gripper_open,
    "robot_fell": robot_fell,
    "last_subtask": last_subtask,
}


def compile_spec(spec):
    """Spec tree -> a callable(ctx) -> (N,) bool. Compiled once, called every step."""
    if not isinstance(spec, dict):
        raise TypeError(f"a condition must be an object, got {spec!r}")

    if "all" in spec:
        subs = [compile_spec(s) for s in spec["all"]]
        def _all(ctx):
            out = ctx.true()
            for fn in subs:
                out = out & fn(ctx)
            return out
        return _all

    if "any" in spec:
        subs = [compile_spec(s) for s in spec["any"]]
        def _any(ctx):
            out = torch.zeros(ctx.num_envs, dtype=torch.bool, device=ctx.device)
            for fn in subs:
                out = out | fn(ctx)
            return out
        return _any

    if "not" in spec:
        inner = compile_spec(spec["not"])
        return lambda ctx: ~inner(ctx)

    pid, params = next(iter(spec.items()))
    if pid not in IMPL:
        raise KeyError(f"no predicate {pid!r} (have: {', '.join(sorted(IMPL))})")
    fn = IMPL[pid]
    kwargs = dict(params or {})
    return lambda ctx: fn(ctx, **kwargs)
