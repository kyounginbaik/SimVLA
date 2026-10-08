"""SimVLA: evaluate a composed success/retry condition against a live env.

The thin half of the predicates_math split. Everything here needs a real `env`; everything that can
be tested without one lives in scripts/simvla/predicates_math.py.

The spec arrives as DATA in the DoneTerm's params, with roles already resolved to prim names by
task_emit — so this function is the same for every task and every kitchen.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from isaaclab.simvla import variable
from isaaclab.utils.math import quat_rotate_inverse


def _simvla_scripts_dir() -> str:
    root = os.environ.get("SIMVLA_REPO_ROOT")
    if not root:
        # Walk for the repo-root pyproject.toml, same convention as the shim in
        # kitchen_scene_generator.py. Unlike that file (which lives two levels below the repo
        # root), this module lives deep under source/isaaclab_tasks/ — and source/isaaclab_tasks,
        # source/isaaclab, source/isaaclab_assets and source/isaaclab_rl each ship their OWN
        # pyproject.toml (they're independently pip-installable). A bare "first pyproject.toml
        # wins" walk stops at source/isaaclab_tasks/pyproject.toml and never reaches the real
        # root, so scripts/simvla must also exist alongside the candidate for it to count.
        for p in Path(__file__).resolve().parents:
            if (p / "pyproject.toml").is_file() and (p / "scripts" / "simvla").is_dir():
                root = str(p)
                break
    if not root:
        raise RuntimeError("Cannot find repo root; set SIMVLA_REPO_ROOT env var")
    return str(Path(root) / "scripts" / "simvla")


if _simvla_scripts_dir() not in sys.path:
    sys.path.insert(0, _simvla_scripts_dir())

from predicates_math import EvalContext, _anchor_pos, compile_spec  # noqa: E402

# Base-frame home EE positions are PER ROBOT -- see homes.py for why the previous module
# constants (Anubis's pose, hardcoded) silently disabled eef_home on RB-Y1. Resolved per call, on
# the env's device, from the loaded articulation's joint names.
from .homes import home_for

_EEF_BODY = {"right": "ee_link1", "left": "ee_link2"}


def _resolve_home(robot, eef_pos_base, device):
    """The loaded robot's home, as device tensors.

    SIMVLA_PRINT_EEF_HOME=1 is the measurement mode homes.py's docstring names: it prints env 0's
    base-frame EEF positions at evaluation time -- at the first evaluations the arms are still at
    the home pose, so the printed numbers ARE the constants to bake. In this mode an unmeasured
    robot falls back to Anubis's tuple as a dummy (the measurement must be printable before the
    constant exists); outside it, home_for raises, because a wrong home silently disables every
    success that carries eef_home.
    """
    if os.environ.get("SIMVLA_PRINT_EEF_HOME"):
        print(f"[eef_home] measured right={[round(v, 4) for v in eef_pos_base['right'][0].tolist()]} "
              f"left={[round(v, 4) for v in eef_pos_base['left'][0].tolist()]}", flush=True)
        try:
            home_r, home_l = home_for(robot.joint_names)
        except KeyError:
            from .homes import ANUBIS
            home_r, home_l = ANUBIS
    else:
        home_r, home_l = home_for(robot.joint_names)
    return {"right": torch.tensor(home_r, device=device),
            "left": torch.tensor(home_l, device=device)}

#: Jaw bodies per arm, as CANDIDATE namings resolved against the articulation the scene actually
#: holds. ee_link1/ee_link2 happen to be spelled the same on every robot in this repo; the jaws are
#: not. Anubis calls them gripper1R/gripper1L, RB-Y1 ee_finger_r1/ee_finger_r2. This runs inside the
#: TERMINATION manager, i.e. every step of every episode, so a name the robot lacks is not a
#: degraded signal but a hard ValueError out of find_bodies the moment the first env steps.
#: Anubis's spelling stays first, so nothing about an Anubis run changes.
#: AI WORKER (FFW_SG2) is the third pair. Its two DISTAL pads are what close on the object, and
#: their separation is the jaw gap -- 0.1147 m measured at the open pose, the two pads sitting at
#: y = -+0.05736 relative to arm_?_link7. Note gripper_r_* is the RIGHT HAND while _r2/_l2 are
#: its two fingers, which reads confusingly but is the asset's own naming.
#:
#: THIS IS A SECOND TABLE, not the one simvla_video.jaw_bodies() uses. They hold the same pairs
#: for the same reason and neither imports the other, so a robot added to one and not the other
#: raises here, inside the TERMINATION manager, on the first step of the first episode.
_FINGER_CANDIDATES = {
    "right": (("gripper1R", "gripper1L"), ("ee_finger_r1", "ee_finger_r2"),
              ("gripper_r_rh_p12_rn_r2", "gripper_r_rh_p12_rn_l2")),
    "left": (("gripper2R", "gripper2L"), ("ee_finger_l1", "ee_finger_l2"),
             ("gripper_l_rh_p12_rn_r2", "gripper_l_rh_p12_rn_l2")),
}


def _fingers(robot, arm: str) -> tuple[str, str]:
    names = set(robot.body_names)
    for pair in _FINGER_CANDIDATES[arm]:
        if names.issuperset(pair):
            return pair
    raise ValueError(
        f"no known {arm}-hand jaw bodies on this robot; tried "
        f"{_FINGER_CANDIDATES[arm]}, articulation has {sorted(names)}"
    )

#: spec id() -> (spec, compiled closure). Keying on id() alone is unsafe: once a spec dict is
#: garbage collected, CPython is free to reuse its address for an unrelated object, and a bare
#: id()-keyed cache would then silently return the WRONG task's compiled condition for that address
#: — no exception, no size cap, just a wrong success signal. Storing the spec object alongside the
#: closure keeps it alive for as long as its cache entry exists, which makes that address reuse
#: unreachable: `entry[0] is spec` can only be True while `spec` itself is still live, so a stale
#: hit is impossible by construction, not just unlikely. On a miss (including an id() collision
#: with a *different* live spec, which cannot happen with plain dicts today but costs nothing to
#: guard) it recompiles and overwrites the entry.
_COMPILED: dict[int, tuple[object, object]] = {}


def build_context(env, spec, *, with_script_state: bool = True) -> EvalContext:
    """Everything the spec's primitives read, pulled once per call.

    The copy-pasted versions in terminations.py recompute the EEF/base transforms inside every
    sub-condition; this does it once.
    """
    from predicate_contract import leaves

    device = env.device
    robot = env.scene.articulations["robot"]
    base_idx = robot.find_bodies("base_link")[0][0]
    base_pos_w = robot.data.body_pos_w[:, base_idx]
    base_quat_w = robot.data.body_quat_w[:, base_idx]

    eef_pos_w, eef_pos_base, gripper_gap = {}, {}, {}
    for arm, body in _EEF_BODY.items():
        idx = robot.find_bodies(body)[0][0]
        pos_w = robot.data.body_pos_w[:, idx]
        eef_pos_w[arm] = pos_w
        eef_pos_base[arm] = quat_rotate_inverse(base_quat_w, pos_w - base_pos_w)
        a, b = _fingers(robot, arm)
        ai = robot.find_bodies(a)[0][0]
        bi = robot.find_bodies(b)[0][0]
        gripper_gap[arm] = torch.linalg.norm(
            robot.data.body_pos_w[:, ai] - robot.data.body_pos_w[:, bi], dim=-1
        )

    obj_pos_w, art_body_pos_w, art_body_names = {}, {}, {}
    art_joint_pos, art_joint_names = {}, {}
    wanted = {params[key] for _, params in leaves(spec)
              for key in ("role", "target_role") if isinstance(params.get(key), str)}
    for name in wanted:
        if name in env.scene.rigid_objects:
            obj_pos_w[name] = env.scene.rigid_objects[name].data.body_pos_w.squeeze(1)
        elif name in env.scene.articulations:
            art = env.scene.articulations[name]
            art_body_pos_w[name] = art.data.body_pos_w
            art_body_names[name] = list(art.body_names)
            art_joint_pos[name] = art.data.joint_pos
            art_joint_names[name] = list(art.joint_names)
        else:
            raise KeyError(
                f"the composed condition references {name!r}, which this scene has neither as a "
                f"rigid object nor an articulation. Rigid: "
                f"{', '.join(sorted(env.scene.rigid_objects.keys()))}. Articulations: "
                f"{', '.join(sorted(env.scene.articulations.keys()))}."
            )

    # WHO IS TOUCHING THE DOOR. Absent sensors leave this empty rather than zero: a 0.0 reads as
    # "measured, no contact" and would certify a base collision as a clean pull.
    door_contact = {}
    for key, sensor in (("base", "touch_base"), ("grip_l", "touch_grip_l"),
                        ("grip_r", "touch_grip_r")):
        s = env.scene.sensors.get(sensor)
        if s is None:
            continue
        fm = s.data.force_matrix_w          # (N, bodies, filters, 3)
        if fm is None:
            continue
        door_contact[key] = torch.linalg.norm(fm, dim=-1).flatten(start_dim=1).amax(dim=1)

    # Some externally supplied predicate adapters use an EvalContext with base orientation;
    # others do not. Pass it only when the installed context accepts the field.
    optional = {}
    if "base_quat_w" in getattr(EvalContext, "__dataclass_fields__", {}):
        optional["base_quat_w"] = base_quat_w

    return EvalContext(
        num_envs=env.num_envs,
        device=device,
        obj_pos_w=obj_pos_w,
        art_body_pos_w=art_body_pos_w,
        art_body_names=art_body_names,
        art_joint_pos=art_joint_pos,
        art_joint_names=art_joint_names,
        eef_pos_w=eef_pos_w,
        eef_pos_base=eef_pos_base,
        home=_resolve_home(robot, eef_pos_base, device),
        gripper_gap=gripper_gap,
        base_pos_w=base_pos_w,
        goal_index=getattr(variable, "env_goal_indices", None) if with_script_state else None,
        max_steps=getattr(variable, "max_sequence_length", None) if with_script_state else None,
        door_contact=door_contact,
        **optional,
    )


#: Per-SPEC call counters for the term print below. Module-level because composed() is a DoneTerm
#: function and has nowhere else to keep state.
#:
#: KEYED BY SPEC, not global. Both DoneTerms in a kitchen cfg -- `success` and `retry` -- call
#: composed() once per step, so a single shared counter with a fixed modulus lands on the same
#: spec every time (the calls alternate, so every 400th is the same parity) and the other spec is
#: never printed at all. That is exactly the success condition this was added to inspect.
_TERM_DBG_N: dict = {}


def composed(env, spec) -> torch.Tensor:
    """A DoneTerm function whose condition is data. One implementation for every task."""
    key = id(spec)
    entry = _COMPILED.get(key)
    if entry is None or entry[0] is not spec:
        entry = (spec, compile_spec(spec))
        _COMPILED[key] = entry
    ctx = build_context(env, spec)
    result = entry[1](ctx)

    # THE TRACE. Off unless SIMVLA_DOOR_TRACE names a path; see door_open_trace.py for why every
    # column is there. Placed beside the [terms] print because both want the same `ctx`, and
    # separate because [terms] reports env 0 only.
    try:
        import door_open_trace
        door_open_trace.record(ctx, spec)
    except Exception:
        pass

    # WHICH TERM IS FALSE. SIMVLA_PRINT_TERMS=<interval> prints every leaf of the condition
    # separately for env 0.
    #
    # WHY. The sink task's success is `mug near the basin AND both EEFs home AND last subtask`,
    # and a run that reports nothing but "fail_but_done" cannot say which of the three refused --
    # or whether the placement is geometrically reachable at all. arm.bowl_place is a BLIND
    # relative move (+0.23 m forward, -0.05 m down from wherever the hand happens to be) while the
    # success wants the mug inside 0.12 m of the basin centre, so "the grasp worked and the place
    # still missed" is a real possibility that no existing signal distinguishes from a failed
    # grasp. One line per leaf settles it. Off by default, byte-identical when unset.
    _iv = int(os.environ.get("SIMVLA_PRINT_TERMS", "0") or 0)
    if _iv > 0:
        _k = id(spec)
        _TERM_DBG_N[_k] = _TERM_DBG_N.get(_k, 0) + 1
        # Capture true terminal conditions BEFORE ManagerBasedRLEnv auto-resets.
        # The collector's later retry log otherwise sees the new episode's poses.
        if _TERM_DBG_N[_k] % _iv == 0 or bool(result[0]):
            # Local, exactly as build_context does it: leaves lives in predicate_contract, and
            # this module's sys.path insert only happens at import time above.
            from predicate_contract import leaves
            parts = []
            for lname, lparams in leaves(spec):
                try:
                    sub = compile_spec({lname: lparams})(ctx)
                    flag = 'T' if bool(sub[0]) else 'F'
                    # THE MEASURED VALUE, not just the verdict. A bare T/F cannot distinguish
                    # "the door did not move" from "the door moved 45 degrees and the threshold
                    # is 60" -- and those need opposite fixes. Chasing that difference through a
                    # boolean cost several GPU runs on the open-the-fridge task.
                    extra = ""
                    try:
                        if lname == "obj_near_prim":
                            role, target_role = lparams["role"], lparams["target_role"]
                            obj = ctx.obj_pos_w[role][0]
                            target = _anchor_pos(
                                ctx, target_role, lparams.get("body"),
                                lparams.get("anchor", "body"),
                            )[0].clone()
                            if lparams.get("z_override") is not None:
                                target[2] = lparams["z_override"]
                            axes = slice(0, 2) if lparams.get("xy_only", False) else slice(0, 3)
                            dist = torch.linalg.norm(obj[axes] - target[axes]).item()
                            obj_xyz = ",".join(f"{v:.3f}" for v in obj.tolist())
                            target_xyz = ",".join(f"{v:.3f}" for v in target.tolist())
                            extra = (
                                f"({dist:.3f}m/{lparams['radius']:.3f}m;"
                                f"obj=[{obj_xyz}],target=[{target_xyz}])"
                            )
                        elif lname == "obj_z":
                            extra = f"(z={float(ctx.obj_pos_w[lparams['role']][0, 2]):.4f})"
                        elif lname == "eef_home":
                            arm = lparams.get("arm", "right")
                            arms = ("right", "left") if arm == "both" else (arm,)
                            distances = []
                            for side in arms:
                                home = ctx.home[side].to(ctx.eef_pos_base[side].device)
                                actual = ctx.eef_pos_base[side][0]
                                dist = torch.linalg.norm(actual - home).item()
                                pos = ",".join(f"{v:.3f}" for v in actual.tolist())
                                goal = ",".join(f"{v:.3f}" for v in home.tolist())
                                distances.append(
                                    f"{side}={dist:.3f}m(actual={pos},home={goal})"
                                )
                            extra = "(" + ",".join(distances) + ")"
                        elif lname == "joint_pos":
                            role = lparams.get("role")
                            names = ctx.art_joint_names.get(role, [])
                            j = lparams.get("joint")
                            idx = j if isinstance(j, int) else names.index(j)
                            val = float(ctx.art_joint_pos[role][0, idx])
                            lo = lparams.get("lo")
                            extra = (f"({math.degrees(val):.1f}deg"
                                     + (f" vs {math.degrees(lo):.0f})" if lo is not None else ")"))
                    except Exception:
                        extra = ""
                    parts.append(f"{lname}={flag}{extra}")
                except Exception as exc:                      # a leaf that cannot stand alone
                    parts.append(f"{lname}=ERR({type(exc).__name__})")
            print(f"[terms] n={_TERM_DBG_N[_k]} env0 result={'T' if bool(result[0]) else 'F'} "
                  + " ".join(parts), flush=True)
    return result
