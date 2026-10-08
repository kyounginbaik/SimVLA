"""A per-frame, per-env CSV of the open-the-fridge attempt. Diagnostic only, off by default.

WHY THIS EXISTS. Run 2105971's episode 1 held the refrigerator door at joint_pos=0.0deg from frame
200 to frame 1400 and then timed out. The run log could say the door had not moved and nothing
else: not whether the gripper was closed, not whether the hand was still on the bar, not whether
the base went where it was told. Those three failures need three different fixes and the existing
signals cannot tell them apart.

WHY PER ENV. SIMVLA_PRINT_TERMS prints env 0 only. Three other envs ran the same episode and none
of them was reported.

WHY ACTUAL *AND* COMMANDED BASE POSITION. Two sessions were spent arguing about the retreat
direction from commanded values, against a user who was reading actual footage. The user was
right. One row carrying both numbers ends that class of argument permanently.

SIMVLA_DOOR_TRACE=<path> turns it on. Unset, record() returns before touching anything.
"""
from __future__ import annotations

import csv
import os

import torch

COLUMNS = (
    "frame", "env",
    "door_deg",            # the quantity being scored
    "jaw_gap_l_m", "jaw_gap_r_m",   # is the gripper actually closed
    "hand_l_to_door_m", "hand_r_to_door_m",  # has the grasp slid off the bar -- EACH hand, never
                                              # combined; see _hand_to_door's docstring for why
    "touch_base_n", "touch_grip_l_n", "touch_grip_r_n",   # WHO is touching the door, in newtons.
                                                          # Blank = no sensor. A 0.0 means measured
                                                          # and not touching; the two must not be
                                                          # conflated.
    "base_x", "base_y",    # what the base DID
    "goal_x", "goal_y", "goal_yaw",  # what it was TOLD
    "step_idx",
)

#: env index -> (goal_x, goal_y, goal_yaw), last commanded by nav.open_door_arc. Module level
#: because the resolver and the termination manager are different call stacks in one process, and
#: threading a handle between them would touch every skill's signature.
_GOALS: dict[int, tuple[float, float, float]] = {}
_FRAME: int = 0
_WRITER = None
_FH = None


def reset() -> None:
    """Forget everything. Tests call this between cases; a run never needs it."""
    global _FRAME, _WRITER, _FH
    _GOALS.clear()
    _FRAME = 0
    if _FH is not None:
        _FH.close()
    _WRITER, _FH = None, None


def publish_goal(env_ids: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> None:
    """Record what nav.open_door_arc just commanded, for the envs it resolved."""
    if not os.environ.get("SIMVLA_DOOR_TRACE"):
        return
    for k, e in enumerate(env_ids.tolist()):
        _GOALS[int(e)] = (float(xy[k, 0]), float(xy[k, 1]), float(yaw[k]))


def _door_joint(spec):
    """(role, joint) of the spec's joint_pos leaf, or None.

    Read from the SPEC rather than hardcoded, exactly as the [terms] printer does: task_emit has
    already resolved @fridge to the prim name the scene actually holds, and that name differs
    between kitchens.
    """
    try:
        from predicate_contract import leaves
        for name, params in leaves(spec):
            if name == "joint_pos":
                return params.get("role"), params.get("joint")
    except Exception:
        pass
    return None


def _door_deg(ctx, spec):
    got = _door_joint(spec)
    if got is None:
        return None
    role, joint = got
    try:
        names = ctx.art_joint_names.get(role, [])
        idx = joint if isinstance(joint, int) else names.index(joint)
        return torch.rad2deg(ctx.art_joint_pos[role][:, idx])
    except (KeyError, ValueError, IndexError):
        return None


def _hand_to_door(ctx, spec, arm):
    """|hand - door body origin|, per env, for exactly ONE named hand ("left" or "right"). Never
    combine the two -- see below.

    The door prim's frame origin is its hinge (measured: the door Xform pivot and the revolute
    joint's body0 anchor agree to 0.3 mm on this asset). A hand closed on the handle therefore
    stays a FIXED distance from it through the whole swing, whatever the door angle. Watching this
    number CHANGE is how a grasp sliding off a smooth bar is distinguished from a base that never
    moved -- the two failures the 0.0-degree episode could not tell apart.

    THIS USED TO RETURN min(left, right) OVER BOTH HANDS, with a docstring claiming "the caller
    does not know which one holds, and the wrong one is harmless extra columns rather than a wrong
    number." That was false and it cost a wrong diagnosis. On the diagonal-sweep task only the
    LEFT arm grasps the handle (it sits on the robot's left). Anubis's idle RIGHT hand, parked at
    its home pose, sat 1.1499 m from the door hinge -- within a few millimetres of where the
    GRASPING hand should read at the handle itself (~1.1525 m, computed from the measured handle
    and hinge world positions; see fridge_diag_sweep.HANDLE_RADIUS_M) -- so torch.minimum(left,
    right) silently picked the idle hand's distance every single frame. The old single
    "hand_to_door_m" column read a constant 1.1545 m for 300 frames across two arm steps with a
    stationary base, and a reader concluded from that number that the grasping hand was on the
    handle the whole time. It was not: the column was reporting the idle right hand's home-pose
    distance, and the grasping left hand was never inspected at all. A minimum over two sensors is
    not a measurement -- it reports whichever is closer and hides which one that was. Report each
    hand's own distance in its own column, always, and let the caller decide which hand is
    supposed to be grasping (see fridge_diag_sweep.score's `grasp_arm` parameter).
    """
    got = _door_joint(spec)
    if got is None:
        return None
    role, joint = got
    # THE BODY NAME COMES FROM THE JOINT, not the literal "door".
    #
    # This used to do names.index("door"), which is the refrigerator's body name and nothing else's.
    # A cabinet's is door_0_0 and a dishwasher's door_0_1, so on every task but the fridge the
    # lookup raised and this column silently went BLANK -- and a blank column is exactly the
    # signal you need when asking whether the hand is still on the handle. The exporter names the
    # joint <body0>_to_<body1>, so the body is the tail of the joint name; "door" remains the
    # fallback for a joint that is not named that way.
    candidates = []
    if isinstance(joint, str) and "_to_" in joint:
        candidates.append(joint.split("_to_", 1)[1])
    candidates.append("door")
    names = ctx.art_body_names.get(role, [])
    door = None
    for cand in candidates:
        try:
            door = ctx.art_body_pos_w[role][:, names.index(cand), :]
            break
        except (KeyError, ValueError, IndexError):
            continue
    if door is None:
        return None
    hand = ctx.eef_pos_w.get(arm)
    if hand is None:
        return None
    return torch.linalg.norm(hand - door, dim=1)


def _f(t, i):
    """One element as a float, or "" when the tensor is absent. A BLANK, never a 0.0: a zero reads
    as a real measurement of zero, and the scorer would treat the two identically.

    Rounded to 6 decimal places: a float32 tensor element widened to Python float carries garbage
    past the 7th digit (torch.tensor([3.1])[0] -> 3.0999999046325684), which is noise no reader of
    this CSV wants and no test should have to pytest.approx() around. 6 decimals is sub-micron for
    meters and sub-micronewton for the contact-force columns -- far finer than anything measured
    here, so nothing real is lost."""
    return "" if t is None else round(float(t[i]), 6)


def record(ctx, spec) -> None:
    """Append one row per env. Never raises: this runs inside the termination manager, so an
    exception here would take the whole episode with it, and the trace is diagnostic."""
    global _FRAME, _WRITER, _FH
    path = os.environ.get("SIMVLA_DOOR_TRACE")
    if not path:
        return
    try:
        if _WRITER is None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _FH = open(path, "w", newline="")
            _WRITER = csv.writer(_FH)
            _WRITER.writerow(COLUMNS)
        door = _door_deg(ctx, spec)
        hand_l = _hand_to_door(ctx, spec, "left")
        hand_r = _hand_to_door(ctx, spec, "right")
        gap_l = ctx.gripper_gap.get("left")
        gap_r = ctx.gripper_gap.get("right")
        contact = getattr(ctx, "door_contact", None) or {}
        t_base = contact.get("base")
        t_gl = contact.get("grip_l")
        t_gr = contact.get("grip_r")
        base = ctx.base_pos_w
        gidx = ctx.goal_index
        for e in range(ctx.num_envs):
            g = _GOALS.get(e)
            _WRITER.writerow([
                _FRAME, e,
                _f(door, e), _f(gap_l, e), _f(gap_r, e), _f(hand_l, e), _f(hand_r, e),
                _f(t_base, e), _f(t_gl, e), _f(t_gr, e),
                _f(None if base is None else base[:, 0], e),
                _f(None if base is None else base[:, 1], e),
                "" if g is None else g[0], "" if g is None else g[1],
                "" if g is None else g[2],
                _f(gidx, e),
            ])
        _FH.flush()
        _FRAME += 1
    except Exception as exc:                      # never load-bearing
        print(f"[door_trace] disabled after {type(exc).__name__}: {exc}", flush=True)
        os.environ.pop("SIMVLA_DOOR_TRACE", None)
