"""SimVLA: the fifteen skills, in the one module BOTH processes can import.

The declarations used to live in simvla_data_generator.py, which imports pxr, omni.usd,
omni.physx, tkinter, PIL, shapely and scipy — the authoring stack. simvla_gen.py, the executor,
cannot import any of that, so it never imported the module, so nothing ever ran the resolver
registrations at the bottom of it. In the executor's process the runtime-resolver table was
therefore literally `{}`, and every goal file with a runtime step — 3,905 `N` steps alone — died
on the lookup with `KeyError: -0.25`. (Both the table and the float it was keyed by are gone now;
the executor calls a skill's resolve() through the REGISTRY, by name. See skill_runtime.py.)

A registry that only one of its two consumers can populate is not a registry. So the skills live
here, in a module whose imports are torch, isaaclab.utils.math, skill_contract, skill_runtime and
stdlib — nothing that boots Omniverse or opens a window. The GUI imports it. The executor imports
it. They see the same REGISTRY, which also means SKILL_ID()'s sorted-order ints mean the same
thing on both sides — the property the executor's dispatch depends on.

WHAT COULD NOT COME ALONG. Six plan() bodies read the USD stage (bbox caches, xform caches, a
physx raycast) or pick grasps from a Tk thumbnail chooser. Those are authoring-time facts and the
executor has no use for them, but they cannot be expressed without pxr/omni/scipy. Rather than
drag the GUI stack in here — which would re-break the very import this module exists to make
possible — the class is declared here and its body stays in simvla_data_generator.py, injected:

    # skills.py
    @skill(id="arm.place", ...)
    class ArmPlace:
        plan = _authored("arm.place")     # body lives in the authoring process

    # simvla_data_generator.py
    @register_planner("arm.place")
    def plan_arm_place(app, action, params):
        ...UsdGeom.BBoxCache(...)...      # verbatim, where the USD imports already are

Both processes see the full 15-skill REGISTRY. Only the authoring process can plan() the six; in
the executor, calling one raises AuthoringStackRequired instead of ImportError-ing at startup.
validate_planners() (called at the bottom of simvla_data_generator.py) fails the import if the
authoring side forgets to supply one — the same half-wired-skill check the contract already makes,
applied to the half that is injected.
"""

from __future__ import annotations

import math
import os as _os
from typing import Any, Callable, Dict


from .skill_contract import (
    REGISTRY,
    Bool,
    Choice,
    Float,
    PrimPath,
    skill,
    validate_registry,
)


# =============================================================================
# plan() bodies that need the authoring stack, injected from the authoring process
# =============================================================================

class AuthoringStackRequired(RuntimeError):
    """plan() was called for a skill whose body lives in the authoring process."""


#: skill id -> plan(app, action, params). Filled by simvla_data_generator.py at import.
AUTHOR_PLANNERS: Dict[str, Callable[..., Any]] = {}

#: The skills that declare plan() but keep the body over there. validate_planners() checks it.
AUTHORING_ONLY: set[str] = set()


def register_planner(skill_id: str):
    """Supply the authoring-side body of `skill_id`'s plan(). Called from simvla_data_generator."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if skill_id not in AUTHORING_ONLY:
            raise KeyError(
                f"{skill_id!r} does not declare plan = _authored({skill_id!r}), so nothing will "
                f"ever call this body."
            )
        AUTHOR_PLANNERS[skill_id] = fn
        return fn
    return decorator


def _authored(skill_id: str):
    """plan() for a skill whose geometry cannot leave the authoring stack. See the module docstring."""
    AUTHORING_ONLY.add(skill_id)

    def plan(self, app, action: str, params: dict):
        fn = AUTHOR_PLANNERS.get(skill_id)
        if fn is None:
            raise AuthoringStackRequired(
                f"{skill_id!r} is planned while authoring: its body reads the USD stage (and, for "
                f"arm.grasp, a Tk thumbnail chooser), so it lives in simvla_data_generator.py and "
                f"is injected with register_planner(). Nothing in that module has been imported, "
                f"so this is a process that cannot author — the executor. It should be reading the "
                f"goal this skill already planned, not planning one."
            )
        return fn(app, action, params)

    plan.__qualname__ = f"{skill_id}.plan"
    return plan


def validate_planners() -> None:
    """Every _authored() skill has a body. Call at the end of the authoring module's import."""
    missing = sorted(AUTHORING_ONLY - set(AUTHOR_PLANNERS))
    if missing:
        raise AuthoringStackRequired(
            f"declared plan = _authored(...) but no @register_planner body was supplied: {missing}. "
            f"The skill would raise at author time, in the GUI, with the scene loaded."
        )


# =============================================================================
# Quaternion helpers. Shared by the resolvers below and by arm.grasp's authoring body,
# which imports them back from here.
# =============================================================================

def quat_normalize_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    import torch
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def rotate_vec_by_quat_wxyz(q_l2w_wxyz, v_local_xyz):
    """
    q_l2w_wxyz: (...,4) wxyz, local -> world
    v_local_xyz: (...,3) local vector
    return: (...,3) world vector
    """
    import torch
    from isaaclab.utils.math import quat_conjugate, quat_mul
    q = quat_normalize_wxyz(q_l2w_wxyz)
    v_quat = torch.cat([torch.zeros_like(v_local_xyz[..., :1]), v_local_xyz], dim=-1)  # (0,vx,vy,vz)
    out = quat_mul(quat_mul(q, v_quat), quat_conjugate(q))
    return out[..., 1:]  # xyz

def gripper_y_up_down(q_wxyz,
                      q_is_world_to_local: bool,
                      world_up=(0.0, 0.0, 1.0),
                      eps=1e-6):
    """
    q_wxyz: (...,4) wxyz on CUDA
    q_is_world_to_local:
      - True  => q maps world -> local
      - False => q maps local -> world
    Returns: dot, is_up, is_down, y_world
    """
    import torch
    from isaaclab.utils.math import quat_conjugate
    q = quat_normalize_wxyz(q_wxyz)

    # we need local->world to express local axes in world
    q_l2w = quat_conjugate(q) if q_is_world_to_local else q

    device, dtype = q_l2w.device, q_l2w.dtype
    y_local = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(q_l2w.shape[:-1] + (3,))
    y_world = rotate_vec_by_quat_wxyz(q_l2w, y_local)

    up = torch.tensor(world_up, device=device, dtype=dtype).expand_as(y_world)
    dot = (y_world * up).sum(dim=-1)

    is_up = dot > eps
    is_down = dot < -eps
    return dot, is_up, is_down, y_world


def local_z_plus_180_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Apply +180° rotation about LOCAL z-axis.
    q_wxyz: (..., 4) torch.Tensor on CUDA (wxyz)
    Returns: (..., 4) wxyz
    """
    import torch
    from isaaclab.utils.math import quat_mul
    q_wxyz = quat_normalize_wxyz(q_wxyz)

    # +180° about local z: angle = pi, half-angle = pi/2
    half = math.pi / 2
    q_delta = torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=q_wxyz.dtype, device=q_wxyz.device
    )  # (4,) wxyz	~= [0,0,0,1]

    # local-axis rotation => RIGHT multiply: q_new = q ⊗ q_delta
    q_new = quat_mul(q_wxyz, q_delta.expand_as(q_wxyz))
    return quat_normalize_wxyz(q_new)


# -----------------------------------------------------------------------------------------------
# BODex grasp pose -> ee_link1 target, PER ROBOT.
#
# arm.grasp's authoring body turns a BODex hand pose into an orientation for the arm's `ee_link1`
# body. Both quantities below are properties of THAT LINK'S FRAME, and the two robots' `ee_link1`
# frames are not the same frame, so a goal authored for one is wrong for the other. Until this
# table existed the offset was the literal (90, 90, 0) and the handedness rule was the literal
# "+Y must point down", neither consulting the robot -- so every goal file in the corpus is in
# Anubis's frame, and an RB-Y1 run reading one would drive the hand to a pose rotated 180 degrees
# about its own tool axis, i.e. reaching PAST the object with the back of the wrist.
#
# What the two frames are, read off the URDFs:
#
#   Anubis (anubis/anubis_final.urdf): ee_fixed_joint1 puts `ee_link1` at xyz="0 0 +0.10956"
#     from `gripper_base_link`, rpy 0; the jaws (gripper1L/gripper1R) slide on that link's +-X.
#     So in ee_link1: APPROACH (wrist -> object) is +Z, jaw travel is +-X.
#   RB-Y1 (rby1m/urdf/model.urdf): ee_fixed_joint1 puts `ee_link1` at xyz="0 0 -0.113" from
#     `ee_right`, rpy 0; the jaws (ee_finger_r1/r2) slide on +-X of the same frame.
#     So in ee_link1: APPROACH is -Z, jaw travel is +-X.
#
# Write a grasp as a physical triad: `a` the approach unit vector and `c` the jaw-travel unit
# vector, both in world. Then, as rotation matrices whose columns are the local axes in world,
#
#     R_anubis = [ c,      a x c,  a  ]        (X=jaw, Z=approach)
#     R_rby1   = [ c,    (-a) x c, -a  ]       (X=jaw, Z=-approach)
#     M := R_anubis^T R_rby1 = diag(1, -1, -1) = 180 degrees about the tool's own X axis,
#
# and M is CONSTANT -- independent of the grasp, as it must be for a fixed pair of links. Hence
# q_offset[rby1] = q_offset[anubis] (x) (0, 1, 0, 0). Two consequences, both encoded below:
#
#   * M negates the local +Y column, so the "+Y down" disambiguator INVERTS to "+Y up". Applying
#     Anubis's test to an RB-Y1 quaternion would pick the opposite jaw labelling.
#   * M and the local-Z-180 flip commute into each other (Rz180 . Rx180 = Ry180), so the same
#     branch of the test yields the same physical grasp on both robots. That is what makes one
#     right-multiplication the whole correction.
#
#: y_down: keep the composed quaternion when the tool frame's local +Y points DOWN in world
#: (True), or when it points UP (False); the other branch takes local_z_plus_180_wxyz.
GRASP_TOOL_FRAME = {
    # (90, 90, 0) as euler_to_quat writes it, i.e. the constant the authoring body used to inline.
    "anubis": {"q_offset": (0.5, 0.5, 0.5, -0.5), "y_down": True},
    # anubis (x) (0, 1, 0, 0), normalised sign-free; see the derivation above.
    "rby1":   {"q_offset": (-0.5, 0.5, -0.5, -0.5), "y_down": False},
    # AI WORKER (FFW_SG2), DERIVED with the construction above rather than copied.
    #
    # ITS ee_link1 IS ANUBIS-SHAPED, NOT RB-Y1-SHAPED, and that is the whole answer. Measured
    # off Robots/MM/aiworker/ffw_sg2.usd, which already carries the body: ee_link1 sits at
    # (0, 0, -0.178) from arm_r_link7 rotated 180 degrees about X, so its +Z points along
    # link7's -Z -- ALONG THE APPROACH, exactly like Anubis's. (An earlier pass here assumed
    # the tool frame was link7 itself, whose -Z is the approach, and derived an RB-Y1-shaped
    # constant from that. It was wrong, and nothing would have caught it before a video of the
    # hand closing on air.)
    #
    # What differs from Anubis is the JAW AXIS. The pads travel along link7's +-Y, which is
    # ee_link1's -+Y, so the jaw axis is the ee frame's Y where Anubis's is its X:
    #
    #     R_anubis   = [ c,     a x c, a ]        (X=jaw, Z=approach)
    #     R_aiworker = [ -(a x c),  c, a ]        (Y=jaw, Z=approach)
    #     M := R_anubis^T R_aiworker = Rz(-90 deg),  constant over random grasp triads (checked)
    #
    # hence q_offset[aiworker] = q_offset[anubis] (x) (cos45, 0, 0, -sin45) = (0, 0, 1, -1)/sqrt2.
    # The same construction reproduces the rby1 row above from the anubis one, which is what
    # says the method is right rather than merely plausible; the test runs both.
    #
    # For this robot use +X (the wrist camera bracket), not jaw-travel +Y, to
    # select the upward roll. A horizontal jaw axis has near-zero vertical
    # projection, so a Y-axis sign test can flip the camera into the counter.
    "aiworker": {"q_offset": (0.0, 0.0, 0.7071068, -0.7071068), "y_down": False,
                 "roll_axis": 0},
}


def grasp_tool_frame(robot: str) -> dict:
    """The ee_link1 grasp convention for `robot`. Raises rather than defaulting: a silent fall back
    to Anubis is the failure this table exists to prevent, and it would be invisible until the
    video showed the hand closing on air."""
    try:
        return GRASP_TOOL_FRAME[robot]
    except KeyError:
        raise KeyError(
            f"no ee_link1 grasp frame for robot {robot!r}; known: {sorted(GRASP_TOOL_FRAME)}. "
            f"Add one by measuring where that robot's ee_link1 sits relative to its jaws."
        ) from None


def preferred_grasp_roll(q_wxyz, robot: str, eps: float = 1e-6) -> bool:
    """Choose the parallel-jaw roll branch using an embodiment's asymmetric tool axis.

    Anubis/RB-Y1 retain their historical Y-axis rule. AI Worker's camera bracket
    extends along tool +X: keep that axis up. Testing its jaw-travel Y axis instead
    lets minute noise in a horizontal grasp flip the camera underneath the wrist.
    This chooses between equivalent jaw labels; it does not change the approach axis.
    """
    import math

    if len(q_wxyz) != 4 or not all(math.isfinite(float(v)) for v in q_wxyz):
        raise ValueError("grasp quaternion must have four finite components")
    norm = math.sqrt(sum(float(v)**2 for v in q_wxyz))
    if norm < 1e-8:
        raise ValueError("grasp quaternion must be nonzero")
    w, x, y, z = (float(v) / norm for v in q_wxyz)
    frame = grasp_tool_frame(robot)
    if frame.get("roll_axis", 1) == 0:
        return 2 * (x*z - w*y) > eps
    dot = 2 * (y*z + w*x)
    return dot < -eps if frame["y_down"] else dot > eps


@skill(
    id="nav.to_prim",
    actions=("N_s",),
    label="Move to prim",
    params=[
        PrimPath("prim_path"),
        # "Both" first: a Choice's first option is the GUI's default, and this one is load-bearing.
        # nav.to_prim reads which_arm as arm_bias (Left +5 cm, Right -5 cm, Both 0), and the old
        # hand-written dropdown defaulted to "Both". Ordering it ["Left", ...] would move every
        # newly authored base pose by 5 cm without anyone touching the geometry.
        Choice("which_arm", ["Both", "Left", "Right"], default="Both"),
        Float("safety", default=0.12),
    ],
)
class NavToPrim:
    plan = _authored("nav.to_prim")


@skill(
    id="nav.to_door_handle",
    actions=("N_s",),
    label="Park beside a door handle, clear of the door's own swing",
    params=[
        PrimPath("prim_path"),
        # HOW FAR ALONG THE DOOR, AWAY FROM THE HINGE, past the handle.
        #
        # A STATIC base parked at the handle's own lateral coordinate -- where nav.to_prim puts it
        # -- sits inside the arc the door sweeps, and the door stops against it: 24.0 deg on
        # base_cabinet/door_0_0 and 19.5-26.5 on sink_cabinet/door_0_1, by OBB/SAT against Anubis's
        # real 0.465 x 0.492 x 0.779 m base box (never a circle -- that model errs in both
        # directions). The refrigerator has the same defect at 10.5 deg, which is what
        # SIMVLA_FRIDGE_PARK_PAST_M exists to fix there.
        #
        # THAT IS NOT THE BINDING CONSTRAINT FOR A CABINET DOOR, and believing it was cost a GPU
        # run. The base does not hold still: it retreats 0.10 m per swing step and the door opens
        # only because it does, so by the time the door reaches 21 deg the base has already given up
        # its first 0.10 m. Under that coupled motion even past_m = 0 clears. What binds instead is
        # ARM REACH -- job 2110287 failed 21 times out of 21 at arm.fridge_handle_grasp with the
        # base 0.515 m from the handle, while the pregrasp 0.082 m nearer succeeded every time.
        #
        # So keep this small. It is margin against a slipping grip or a lagging door, and every
        # centimetre of it is a centimetre of lateral arm reach spent.
        Float("past_m", default=0.09),
        # Distance from THE HANDLE'S CENTRE -- not from the furniture's bounding face, which is what
        # nav.to_prim measures from and which differs by however far the handle stands off it.
        #
        # THIS IS THE KNOB THAT MATTERS, because the constraint that binds is arm reach and reach is
        # a distance from the handle. nav.to_prim cannot express it at all: its standoff is fixed at
        # safety + wheel_r off the handle prim's bbox MIN, so "park 0.39 m from the handle" is not
        # something it can be asked for. cabinet_kitchen.pick_park_pose chooses the value against
        # measured clearance and measured reach together.
        Float("standoff_m", default=0.39),
        # NO which_arm, and its absence is deliberate.
        #
        # nav.to_prim reads which_arm as a +-5 cm lateral arm_bias. Here the lateral position is
        # what `past_m` sets, it is the quantity the OBB/SAT search actually solved for, and 5 cm
        # of it is half the clearance budget the chosen pose has (0.108 m). A bias applied on the
        # same axis would move the base off the only pose anything has verified, so this skill
        # would have to either ignore the parameter or invalidate its own measurement. Which arm
        # grasps the handle is said by the ACTION CHANNEL of the arm steps that follow, which is
        # where it cannot disagree with itself.
    ],
)
class NavToDoorHandle:
    """Park in front of a door handle, offset along the door AWAY FROM ITS HINGE.

    AUTHORED, not runtime: the door is shut at t=0, so where to stand is a fact about the scene,
    knowable from the USD before the episode starts. The body reads the revolute joint's anchor and
    lives in simvla_data_generator with the other USD-reading planners.

    IT ALSO SETS app.N_dir, which is not incidental. arm.handle_pregrasp and arm.handle_grasp pick
    their approach offset AND their approach quaternion by compass direction off that attribute, so
    a nav skill that parked correctly without setting it would leave the following grasp reaching
    along whatever direction the previous step happened to leave behind.
    """
    plan = _authored("nav.to_door_handle")


@skill(
    id="nav.push_prim",
    actions=("N_s",),
    label="Push a floor object toward another prim",
    params=[
        PrimPath("prim_path"),
        PrimPath("toward"),
        Float("offset_m", default=0.0),
        Float("safety", default=0.12),
    ],
)
class NavPushPrim:
    """Park behind a floor-standing object and drive it at `toward`.

    AUTHORED, not runtime, and that is load-bearing: simvla_video.py is the legacy v1 executor plus
    a v2 adapter that re-encodes sentinels for arm.reset and arm.bowl_place only, so a runtime
    skill's goal=null flattens to an all-zero payload and the robot drives to the env-frame origin
    without erroring. An N_s step carrying a literal [x, y, yaw] is native to that executor.

    WHY NOT nav.to_prim. Its planner branches on height, and a floor-standing prim (z < 0.01) takes
    the CASE 1 path, which picks the standoff side by snapping the prim's OWN quaternion to one of
    four canonical orientations. A chair yawed to face its table matches none of them, and the
    fallback leaves N_pos at torch.zeros(3). Measured on kitchen 1215: both chairs sit at yaw
    +/-131.17 deg. Here the axis is computed from the prim->target vector instead, so there is no
    orientation to fail to recognise and no side to guess.

    Two steps make one push, and they share ONE axis: offset_m=0.0 parks behind the object, and a
    positive offset_m is the same pose slid that far toward the target.
    """

    plan = _authored("nav.push_prim")


@skill(
    id="arm.push_pose",
    actions=("A_b",),
    label="Hold both hands out to push a floor object",
    params=[
        PrimPath("prim_path"),
        PrimPath("toward"),
        #: Every default here is a MEASUREMENT on this task's chair and this robot, not a taste.
        #: pushchair_task.py states each one's derivation beside the constant it passes; what
        #: follows is the short form, and it is repeated here because a GUI author who never opens
        #: that file gets these values.
        #:
        #: height_m 0.0 MEANS "MEASURE IT", and that is the default because a constant here can
        #: only ever be right about half the problem. A push height has to be two things at once:
        #: on the thing being pushed, and somewhere the arm can hold the push orientation. The
        #: shipped 0.60 was the first (the chair's backrest panel spans 0.475-0.678) and 0.22 m
        #: under the second -- 0.25 m in front of base_link Anubis cannot hold this orientation
        #: below 0.82 m (push_prim_geometry.ARM_REACH_FLOOR_M, measured by IK off the URDF). And
        #: nothing refuses it: simvla_gen sets cuRobo's position and rotation thresholds to 10 m /
        #: 10 rad, so motion-gen returns the nearest configuration it can reach -- elbow over the
        #: shoulder, palm pitched 17 degrees over, the arm diving at the chair. At 0.0 the planner
        #: reads the prim's own triangles, takes the HIGHEST band where BOTH hands land on the face
        #: the push arrives at, and refuses (naming the standoff that would work) if that band is
        #: still under the arm's floor. A POSITIVE VALUE IS AN OVERRIDE and is used verbatim -- it
        #: is how pushchair_task.SHIPPED_DATASET reproduces the 32 shipped episodes bit for bit.
        Float("height_m", default=0.0),
        #: safety_m -- nav.push_prim's `safety` FOR THE SAME LEG, so the two cannot disagree.
        #: The hands sit wheel_r + safety - clearance_m in front of base_link and the half-extent
        #: cancels, so this is the one number that decides whether the derived height is reachable.
        #: It used to be a 0.12 literal in the planner's log line, with a comment admitting the
        #: planner could not see the nav step's value; now that the height DEPENDS on it, an
        #: assumed value would be a silent wrong answer instead of a slightly wrong log.
        Float("safety_m", default=0.12),
        #: top_margin_m -- how far below the prim's own top the hands are held, so they cannot ride
        #: over the top edge. Clamped into the pressable band afterwards: on a chair whose only
        #: pressable face is its top 3 cm, the hands land in those 3 cm rather than under them.
        Float("top_margin_m", default=0.05),
        #: span_m 0.16 -- the chair's exported footprint is 0.383 x 0.383 m and the backrest's
        #: solid band is about 6 of the 12 grid columns across, i.e. ~0.19 m, so +-0.08 lands both
        #: hands on it with ~0.015 m to spare. The reference file's 0.20 is +-0.10 and sits on the
        #: edge of that band.
        Float("span_m", default=0.16),
        #: clearance_m 0.10 -- how far PAST the prim's AABB the ee frames are authored, and it is
        #: bounded BELOW by the gripper rather than chosen: ee_link1 sits 0.0474 m behind the
        #: fingertips (anubis_final.urdf, finger collision box 0.018 x 0.035 x 0.093 centred at
        #: local z 0.11054 against ee_fixed_joint1's 0.10956), so anything under ~0.05 authors the
        #: FINGERS inside the chair even though the ee origin is outside it, and cuRobo refuses the
        #: goal. 0.10 leaves the fingertips 0.053 m clear.
        Float("clearance_m", default=0.10),
    ],
)
class ArmPushPose:
    """Both hands out in front of the base, square to the push, just clear of the thing to push.

    THE PUSH IS STILL THE BASE'S. This step only holds the hands where the chair's backrest will
    arrive; the nav.push_prim leg that follows drives the base forward and presses them into it.
    That order -- A_b then N_s -- is goals/Isaac-SceneSmith-001.json's, and it is forced: cuRobo's
    world contains the chair, so a hand pose authored IN it plans nothing.

    WHY BOTH HANDS AND NOT THE CHASSIS. Kitchen 1218 measured what the chassis push delivers to a
    chair's centroid -- 0.665 of the base's travel on chair_0, 0.817 on chair_1 -- because the base
    meets a chair LEG off the centre line and a third of the motion goes into yawing the chair
    rather than moving it. Two hands symmetric about the push axis, on the backrest, apply no such
    couple. (Measured once, on chair_0 of job 2108832: 0.862 of the post-contact travel reaches the
    centroid, against the chassis template's pessimistic 0.817 -- see pushchair_task's
    PUSH_TRANSFER_BIMANUAL, and note it is ONE sample.)

    AUTHORED, not runtime, for the reason NavPushPrim gives at length: simvla_video.py's v2 adapter
    re-encodes runtime sentinels for arm.reset and arm.bowl_place ONLY, so any other goal=null step
    flattens to an all-zero payload -- and on A_b that is fourteen zeros, i.e. both hands commanded
    to the env origin with an ALL-ZERO quaternion -- not the identity, a degenerate one -- silently.
    An A_b step carrying a literal 14-float pose is native to that executor.
    """

    plan = _authored("arm.push_pose")


@skill(
    id="arm.grasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to grasp",
    params=[PrimPath("prim_path")],
)
class ArmGrasp:
    plan = _authored("arm.grasp")


@skill(
    id="arm.bar_handle_pregrasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to pre-grasp a horizontal bar handle",
    params=[PrimPath("prim_path")],
)
class ArmBarHandlePregrasp:
    plan = _authored("arm.bar_handle_pregrasp")


@skill(
    id="arm.bar_handle_grasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to grasp a horizontal bar handle",
    params=[PrimPath("prim_path")],
)
class ArmBarHandleGrasp:
    """A HORIZONTAL bar, read from its bounding box.

    NEITHER EXISTING HANDLE PLANNER FITS ONE, and the dishwasher is where that shows.

      arm.handle_grasp        right quaternions for a flat bar, WRONG POSITION SOURCE. It reads the
                              handle prim's ExtractTranslation(), which on a dishwasher door is the
                              DOOR'S PIVOT at floor level -- z = 0.056 against a handle at 0.733. It
                              works for a drawer only because a drawer's handle prim happens to sit
                              on its handle.
      arm.fridge_handle_grasp right position source (the bbox centre, which door_geometry showed is
                              the only trustworthy one) but quaternions for an UPRIGHT bar.

    So this is the missing pairing: bbox centre, and jaws closing VERTICALLY across a bar whose axis
    is horizontal. The approach frame is derived rather than copied -- see _BAR_DIR, whose entries
    are checked to be right-handed rotations mapping local +Z onto the approach and local +X onto
    world +Z.
    """
    plan = _authored("arm.bar_handle_grasp")


@skill(
    id="arm.door_arc_pull",
    actions=("A_r", "A_l", "A_b"),
    label="Carry the hand along a door's arc (horizontal hinge)",
    params=[PrimPath("prim_path"), Float("to_deg", default=7.0),
            # Carry the WRIST round the hinge too, keeping the jaws square to the tilting bar. Set
            # False when the arm cannot achieve that while gripping -- measured on Anubis's left
            # arm, which converges in position to millimetres and leaves ~19 deg of rotation.
            Bool("rotate", default=True)],
)
class ArmDoorArcPull:
    """Open a DROP-DOWN door by moving the HAND along the arc, not the base along a line.

    WHY nav.open_articulation CANNOT DO THIS, measured on kitchen 1221's dishwasher (run 2111809).
    That skill backs the base up along -yaw, which drags the hand horizontally at a constant
    height. It is the right motion for a drawer, and for a cabinet door it is close enough that
    nav.open_door_arc's diagonal fixes the rest. A bottom-hinged door is different in kind: its
    handle travels on a VERTICAL circle, so it must come toward the robot AND fall.

        angle   handle toward robot   handle DOWN
          10deg        0.1175 m          0.0103 m
          30deg        0.3384 m          0.0907 m
          60deg        0.5862 m          0.3384 m

    A constant-height pull therefore fights the door's own geometry the moment it leaves vertical.
    It cannot slip "a little": the bar is FLUSH against the door panel (measured clearance behind
    it: 0.0000 m), so there is nothing to hook and the grip is friction-only, held across the bar's
    0.050 m height while the pull runs perpendicular to the jaw axis -- exactly the direction the
    bar escapes. Run 2111809 gripped the bar in 4 of 8 envs (jaw 0.046-0.051 m on a 0.050 m bar)
    and still moved the door 0.0000 deg in every one, the bar sliding out within ~2 s of the pull.

    WHY IT ONLY HAS TO REACH ~20 deg. At exactly closed the door is VERTICAL, so its weight passes
    through the hinge and gravity's torque is ZERO -- an unstable equilibrium, which is why nothing
    happens until something pulls it off vertical. Past that, weight does the work: the door's own
    torque grows as sin(angle) while the sampled joint damping is all that resists. So the arm's
    job is to break the equilibrium, not to carry the door down; author a few small steps and then
    RELEASE.

    Author one step per waypoint, with `to_deg` the ABSOLUTE door angle each should reach, and keep
    the gripper closed across all of them. Refuses a vertical hinge -- use nav.open_door_arc for
    those, which arcs the BASE and is what a cabinet door needs.
    """
    plan = _authored("arm.door_arc_pull")


@skill(
    id="arm.squeeze",
    actions=("A_r", "A_l", "A_b"),
    label="Squeeze wide object (bimanual, size-heuristic)",
    params=[PrimPath("prim_path")],
)
class ArmSqueeze:
    # A bimanual heuristic grasp: reads the object's LIVE world pose + AABB from the stage (never
    # BODex, never a hardcoded size) and places each hand on the opposite transverse face at the wide
    # lower body. For wide, smooth objects a single parallel-jaw gripper cannot span (>~9 cm) and two
    # cannot pinch on opposite faces without colliding (<~20 cm) — the two palms press and hold by
    # friction. Author it on BOTH arms (one A_r step, one A_l step) grasping the same object.
    plan = _authored("arm.squeeze")


# Pull articulation (Task2)
@skill(
    id="arm.handle_pregrasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to pre-grasp handle of pull articulation",
    params=[PrimPath("prim_path")],
)
class ArmHandlePregrasp:
    plan = _authored("arm.handle_pregrasp")

@skill(
    id="arm.handle_grasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to grasp handle of pull articulation",
    params=[PrimPath("prim_path")],
)
class ArmHandleGrasp:
    plan = _authored("arm.handle_grasp")


# =============================================================================
# Runtime-resolved skills
# =============================================================================
# These skills' goals cannot be computed while authoring: they depend on where the robot actually
# is when the step runs. Each defines resolve() instead of plan(), and the goal file records
# `goal: null` rather than a magic float in a positional slot.
#
# The resolver bodies below are lifted VERBATIM from the free `resolve_*` functions that used to
# sit at the bottom of this file, keyed by sentinel value. The only change is that they now live
# on the skill they implement, which is the point.
#
# Signature: resolve(ctx, envs, params, eef_idx=None)
#   ctx      RuntimeContext (skill_runtime.py), built once per goal-eval pass by simvla_gen.
#   envs     Long tensor selecting which envs need this step resolved.
#   params   The step's authored params (e.g. back_off_m).
#   eef_idx  Body index of the end-effector this step drives — r_eef_idx for A_r, l_eef_idx for
#            A_l. Nav skills drive the base and ignore it.

def _back_off(params, back: torch.Tensor) -> torch.Tensor:
    """params["back_off_m"] as an (N, 1) tensor that broadcasts over the (N, 2) back-off direction.

    It arrives as a plain float when a single step is resolved (the GUI, the tests) and as an
    (N,) tensor when the executor resolves a whole batch of envs at once — those envs
    can be on DIFFERENT steps of the script, so their back-offs can differ, and collapsing them to
    one scalar would silently give a fridge step a drawer's 0.25 m.
    """
    import torch
    value = torch.as_tensor(params["back_off_m"], device=back.device, dtype=back.dtype)
    return value.reshape(-1, 1)


@skill(
    id="nav.open_articulation",
    actions=("N",),
    label="Move robot to open articulation",
    params=[
        Float("back_off_m", default=0.25),
        # "Both" first: a Choice's first option is the GUI's default, and this one is load-bearing.
        # nav.to_prim reads which_arm as arm_bias (Left +5 cm, Right -5 cm, Both 0), and the old
        # hand-written dropdown defaulted to "Both". Ordering it ["Left", ...] would move every
        # newly authored base pose by 5 cm without anyone touching the geometry.
        Choice("which_arm", ["Both", "Left", "Right"], default="Both"),
    ],
)
class NavOpenArticulation:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Back the base off from where it is now, along -yaw. The arm is already holding the
        handle, so backing up is what swings the door open. Returns (xy, yaw).

        A refrigerator door sweeps further than a drawer, so the fridge step passes
        back_off_m=0.4. That is the thing the old code tried to say by registering a SECOND "N"
        skill returning [-0.4, -0.4, -0.4] — which was not a distance at all but a lookup key that
        matched no sentinel, fell through, and drove the robot to the env-frame corner
        (-0.4, -0.4) at heading -0.4 rad, in 73 goal files.
        """
        import torch
        root_pos_w = ctx.robot.data.body_pos_w[envs, ctx.base_link_idx, :2]
        yaw = _base_yaw(ctx, envs)
        back = torch.stack([-torch.cos(yaw), -torch.sin(yaw)], dim=1)
        nav_env_origins = ctx.env_origins[envs, :2]
        src = root_pos_w - nav_env_origins + _back_off(params, back) * back
        return src, yaw


@skill(
    id="nav.close_articulation",
    actions=("N",),
    label="Move robot to close articulation",
    params=[
        Float("back_off_m", default=0.25),
        # "Both" first: a Choice's first option is the GUI's default, and this one is load-bearing.
        # nav.to_prim reads which_arm as arm_bias (Left +5 cm, Right -5 cm, Both 0), and the old
        # hand-written dropdown defaulted to "Both". Ordering it ["Left", ...] would move every
        # newly authored base pose by 5 cm without anyone touching the geometry.
        Choice("which_arm", ["Both", "Left", "Right"], default="Both"),
    ],
)
class NavCloseArticulation:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Push the base forward from where it is now — nav.open_articulation with the sign
        flipped. Returns (xy, yaw)."""
        import torch
        root_pos_w = ctx.robot.data.body_pos_w[envs, ctx.base_link_idx, :2]
        yaw = _base_yaw(ctx, envs)
        back = torch.stack([-torch.cos(yaw), -torch.sin(yaw)], dim=1)
        nav_env_origins = ctx.env_origins[envs, :2]
        src = root_pos_w - nav_env_origins - _back_off(params, back) * back
        return src, yaw


def _arc_param(params, key, ref: torch.Tensor) -> torch.Tensor:
    """params[key] as a (N,) tensor on `ref`'s device and dtype.

    Same shape discipline as _back_off, for the same reason: a value arrives as a plain float when
    ONE step is resolved (the GUI, the tests) and as an (N,) tensor when the executor resolves a
    whole batch at once. Those envs can be on DIFFERENT steps of the script, so collapsing them to
    a single scalar would hand every env the first env's sweep.
    """
    import torch
    return torch.as_tensor(params[key], device=ref.device, dtype=ref.dtype).reshape(-1)


@skill(
    id="nav.open_door_arc",
    actions=("N",),
    label="Swing a door open by arcing the base about its hinge",
    params=[
        # 0.7172872526895057 m on the kitchen 1300/1400 refrigerator, measured by
        # door_geometry.measure_door. The default is a plausible fridge, NOT a licence to skip
        # measuring: a wrong radius puts the pivot in the wrong place and the hand walks off the
        # handle within one step.
        Float("radius_m", default=0.72),
        # "Right" first: a Choice's first option is the GUI's default, and Right is what kitchen
        # 1300's fridge measures as for a robot facing its door.
        Choice("hinge_side", ["Right", "Left"], default="Right"),
        Float("sweep_deg", default=30.0),
        # A RADIAL BACK-OFF ON TOP OF THE ROTATION, so each step moves the base DIAGONALLY out
        # rather than purely tangentially. Two reasons, one physical and one practical. A door
        # handle's arc is only tangential in the limit; a robot dragging one wants to give ground
        # as it goes round, or it ends up fighting its own arm. And a purely tangential step keeps
        # the base at a constant radius from the hinge, which is precisely where the door and the
        # robot are closest.
        #
        # From the user, watching the base pull and rotate: "pull back a bit after move
        # 대각선 way back" -- diagonal.
        Float("back_off_m", default=0.05),
        # THE RETREAT DIRECTION, in degrees off straight-back, rotated toward the HINGE side --
        # equivalently, away from the side the handle starts on. 0 is a pure reverse; 90 a pure
        # sideways strafe.
        #
        # 45 IS NOT A TASTE. Put the hinge at the origin, robot-forward +x, robot-left +y, and a
        # handle on the left starts at (0, +r). The door opens toward the robot, so the handle
        # travels to (-r, 0) at 90 degrees: back by r AND toward the hinge side by r. The chord is
        # therefore at exactly 45 degrees, and the hand is closed on the handle, so the base must
        # follow that chord.
        #
        # No single value is right at every instant -- the chord from t1 to t2 lies at (t1+t2)/2 --
        # which is why fridge_diag_sweep.py measures 10 through 90 instead of trusting this.
        Float("diag_deg", default=45.0),
    ],
)
class NavOpenDoorArc:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Rotate the base — position AND heading — about the door's hinge. Returns (xy, yaw).

        WHY THIS IS NOT nav.open_articulation WITH A BIGGER back_off_m. That skill translates the
        base straight back along -yaw. A drawer travels in a straight line, so a translation is
        exactly right for it. A DOOR does not: its handle is pinned to a circle about the hinge —
        0.717 m on this asset — and a straight pull drags the gripper off that circle along a
        chord. The grasp is friction on a smooth bar, so what that produces is a hand sliding off
        the handle, not a door opening.

        Rotating the base about the hinge instead moves the robot and the door as ONE RIGID BODY.
        The hand's distance from the hinge is then unchanged by construction rather than by
        tolerance, which is the property test_open_door_arc pins.

        WHERE THE HINGE COMES FROM. Not from the scene — RuntimeContext deliberately exposes only
        the robot, and adding env.scene.articulations here would be a contract change touching
        every resolver. It comes from the HAND: the gripper is closed on the handle, so the handle
        IS at the eef, and the hinge lies `radius_m` from it along the door plane — i.e. straight
        out to the robot's left or right, which is what `hinge_side` names. At yaw 0 the robot's
        right is (0, -1); in general (sin yaw, -cos yaw).

        WHICH HAND IS ON THE HANDLE IS DERIVED, NOT PASSED. The handle sits at the door's free
        edge, which is by construction opposite the hinge, so hinge on the right => handle on the
        left => read l_eef_idx. Taking it as a fourth param would create two facts that can
        disagree, and ACTION_WIDTH["N"] is 3 anyway. `eef_idx` is ignored: the executor's nav
        branch passes none, because a nav step drives the base.

        `hinge_sign` (+1 Right / -1 Left), not the declared `hinge_side` string: params reach a
        runtime skill as FLOATS through payloads_tensor, and executor_dispatch.encode_payload is
        what maps the authored string onto the sign. See its docstring for why they ride in tensor
        slots at all.
        """
        import torch
        yaw = _base_yaw(ctx, envs)
        sign = _arc_param(params, "hinge_sign", yaw)
        radius = _arc_param(params, "radius_m", yaw)
        sweep = _arc_param(params, "sweep_deg", yaw)

        # SIMVLA_ARC_SWEEP_DEG restores the base ROTATION at runtime, without a re-emit.
        #
        # WHY IT IS WORTH HAVING. With sweep_deg = 0 the base holds its heading, so a diagonal
        # picked in the robot's frame is a FIXED direction in the world -- and no fixed direction
        # is right for the whole swing, because the handle's chord over a step from t1 to t2 lies
        # (t1+t2)/2 off straight-back (about 7.5 degrees on the first of six steps, 82.5 on the
        # last). Let the base turn with the door and its heading carries that ramp for free: one
        # constant robot-frame diagonal tracks the handle the whole way round.
        #
        # An env var, matching SIMVLA_ARC_DIAG_DEG beside it, because sweep_deg is authored into
        # the goal file by fridge_template and changing it there costs a GPU emit per value. This
        # exists to be swept: one emit, many runs.
        _sw = _os.environ.get("SIMVLA_ARC_SWEEP_DEG")
        if _sw is not None and _sw != "":
            sweep = torch.full_like(sweep, float(_sw))

        # The robot's right, in world. (N, 2).
        right = torch.stack([torch.sin(yaw), -torch.cos(yaw)], dim=1)

        # The hand on the handle: opposite the hinge. BOTH are read and then selected, rather than
        # gathering a per-env body index, because envs on this step can differ in hinge_sign.
        l_xy = ctx.robot.data.body_pos_w[envs, ctx.l_eef_idx, :2]
        r_xy = ctx.robot.data.body_pos_w[envs, ctx.r_eef_idx, :2]
        hand = torch.where((sign > 0).unsqueeze(1), l_xy, r_xy)

        hinge = hand + (sign * radius).unsqueeze(1) * right

        theta = sign * sweep * (math.pi / 180.0)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        base = ctx.robot.data.body_pos_w[envs, ctx.base_link_idx, :2]
        d = base - hinge
        turned = torch.stack(
            [cos_t * d[:, 0] - sin_t * d[:, 1],
             sin_t * d[:, 0] + cos_t * d[:, 1]], dim=1
        )

        # Diagonal, not tangential: each step moves the base back AND toward the hinge side, which
        # is where the handle is going. `back_off_m` may be absent on a goal authored before it
        # existed; `diag_deg` may be absent for the same reason and takes its declared 45.
        back = _arc_param(params, "back_off_m", yaw) if "back_off_m" in params else None
        if back is not None:
            # SIMVLA_ARC_STEP_M overrides how FAR each swing step retreats. The door's free edge
            # sweeps a 0.62 m radius; if the base does not clear that, the door stops against the
            # robot.
            _stepm = _os.environ.get("SIMVLA_ARC_STEP_M")
            if _stepm is not None and _stepm != "":
                back = torch.full_like(back, float(_stepm))

            diag = (_arc_param(params, "diag_deg", yaw) if "diag_deg" in params
                    else torch.full_like(back, 45.0))

            # SIMVLA_ARC_DIAG_DEG overrides the param. A SCALAR sets every env; a COMMA LIST maps
            # by ENV ID -- vals[env_id % len(vals)] -- cycling if it is shorter. The list form is
            # what lets one GPU allocation carry all nine sweep angles: they then see identical
            # geometry, identical physics and identical seed, which nine separate jobs cannot
            # promise.
            #
            # BY ENV ID, NOT BY POSITION IN THIS CALL'S BATCH. `envs` here is only the subset of
            # envs currently sitting on an arc step (simvla_gen calls this with
            # `nav_rows[arc_mask]`), so position i equals env id only while every env is on the
            # arc step in lockstep -- and envs desync as soon as they finish steps at different
            # times. Mapping by position would then silently reassign env 0's angle to env 5, and
            # the nine-angle sweep would become unattributable: no way to say which angle produced
            # which door opening.
            #
            # An env var at all, rather than only the param, because the param is baked into the
            # goal JSON at emit time and an emit is a 40-minute GPU job. One emit, many runs.
            _diag = _os.environ.get("SIMVLA_ARC_DIAG_DEG")
            if _diag is not None and _diag != "":
                vals = [float(v) for v in _diag.split(",") if v.strip() != ""]
                if vals:
                    # An empty list (e.g. SIMVLA_ARC_DIAG_DEG=",") must not fall through to a
                    # ZeroDivisionError on `% len(vals)`, and must not silently become 0.0 either --
                    # 0.0 is a pure straight-back pull, exactly the broken default this task
                    # removed. So an empty list is ignored and the param value stands.
                    _ids = envs.tolist() if hasattr(envs, "tolist") else list(envs)
                    diag = torch.tensor(
                        [vals[int(e) % len(vals)] for e in _ids],
                        device=diag.device, dtype=diag.dtype,
                    )

            # ONE PATH, AND THE SIGN CARRIES BOTH HANDEDNESSES.
            #
            # A positive rotation carries straight-back toward the robot's RIGHT, and sign is +1
            # when the hinge is on the right -- i.e. when the handle is on the LEFT. So
            # `radians(diag) * sign` is:
            #
            #     handle LEFT  (sign +1) -> back and RIGHT
            #     handle RIGHT (sign -1) -> back and LEFT
            #
            # which is the rule, both rows, with no branch.
            #
            # WHAT WAS DELETED HERE, so it is not reinvented. There used to be a second branch for
            # the case where SIMVLA_ARC_DIAG_DEG was unset. It computed the tangent to the
            # handle's arc, perpendicular to hinge->hand -- which sounds self-adjusting and is not,
            # because the hinge is RECONSTRUCTED as `hand + sign*radius*right` and so lies exactly
            # along `right` by construction. Its perpendicular is therefore always exactly `back`.
            # Its own docstring conceded the point: "AND IT DEGENERATES TO EXACTLY STRAIGHT BACK,
            # ALWAYS." The shipped default was a straight pull with no diagonal at all, and the
            # default before it retreated toward the side the handle was LEAVING. Both were
            # observed on video by the user; neither opened the door.
            ang = torch.deg2rad(diag) * sign
            bx, by = -torch.cos(yaw), -torch.sin(yaw)      # straight back, in world
            ca, sa = torch.cos(ang), torch.sin(ang)
            radial = torch.stack([ca * bx - sa * by, sa * bx + ca * by], dim=1)
            turned = turned + back.unsqueeze(1) * radial

            # WHICH WAY DID IT ACTUALLY GO. Two sessions were spent disagreeing about this from
            # video alone, so the run states it in the log: the step resolved into the robot's own
            # back/left/right, which is the frame the question is asked in.
            if _os.environ.get("SIMVLA_ARC_DEBUG"):
                r_vec = hand - hinge
                step_xy = (hinge + turned) - base
                back_hat = torch.stack([-torch.cos(yaw), -torch.sin(yaw)], dim=1)
                right_hat = torch.stack([torch.sin(yaw), -torch.cos(yaw)], dim=1)
                b = (step_xy * back_hat).sum(dim=1)
                l = (step_xy * right_hat).sum(dim=1)
                side = (r_vec * right_hat).sum(dim=1)
                # ABSOLUTE base xy too, not just the step. The step is what was ASKED for; only
                # consecutive absolute positions show what the base actually DID, and that is the
                # only thing that answers "which way did it go". Every run so far was killed
                # before its HDF5 closed, so the recorded trajectories were unreadable and the
                # commanded direction was all there was to argue from -- against a human watching
                # the video, who kept saying left.
                for i in range(min(1, step_xy.shape[0])):
                    print(f"[arc] base=({base[i, 0]:+.3f},{base[i, 1]:+.3f}) "
                          f"yaw={float(yaw[i]) * 180.0 / math.pi:+.1f}deg  handle on the "
                          f"{'RIGHT' if side[i] > 0 else 'LEFT'}; commanded step = back "
                          f"{b[i]:+.3f} m, {'RIGHT' if l[i] > 0 else 'LEFT'} {abs(l[i]):.3f} m",
                          flush=True)

        src = hinge + turned - ctx.env_origins[envs, :2]
        return src, _wrap_to_pi(yaw + theta)


def _place_knob(params, name, env_var, default, *, device, dtype):
    """One arm.bowl_place knob, as a tensor that broadcasts over the (N, ...) envs being resolved.

    SHAPE, for the same reason _back_off has the same shape. The value arrives as a plain float
    when one step is resolved (the GUI, the tests) and as an (N,) tensor when the executor resolves
    a whole batch at once -- those envs can be on DIFFERENT steps of the script, so their knobs can
    legitimately differ, and collapsing them to one scalar would give one env another's geometry.

    THE ENV VAR, when set, overrides every env at once. The params are baked into the goal file at
    emit time and an emit is a 40-minute GPU job, so sweeping a release height through the
    environment is the difference between a re-run and a re-emit. Same idiom as SIMVLA_ARC_DIAG_DEG
    and SIMVLA_PLACE_ABOVE. Unset, it is byte-identical to reading the param.
    """
    import torch
    override = _os.environ.get(env_var, "")
    value = float(override) if override.strip() else params.get(name, default)
    return torch.as_tensor(value, device=device, dtype=dtype)


@skill(
    id="arm.bowl_place",
    actions=("A_r", "A_l", "A_b"),
    label="Place object",
    #: EVERY DEFAULT REPRODUCES THE CONSTANTS THIS SKILL WAS BORN WITH, so the sink and drawer
    #: templates that pass no params at all are unchanged to the bit. The knobs exist because the
    #: same move has to serve two incompatible jobs: DUMPING a mug into a sink basin at 0.88 m,
    #: 0.07 m below the counter the hand carries at -- where a 0.05 m descent and a 20 deg tilt are
    #: exactly right -- and SETTING a bowl down on a 0.74 m table, where that same move releases
    #: 0.25 m up and tilted, and the bowl lands on its rim.
    params=[
        Float("forward_m", default=0.23),
        Float("down_m", default=0.05),
        #: -inf, not None: Float carries a float, and `maximum(z, -inf) == z` makes "no floor" the
        #: arithmetic identity rather than a branch. It also never reaches a goal file -- a
        #: template that wants no floor omits the param, so JSON never has to encode an infinity.
        Float("min_eef_z", default=-math.inf),
        Float("roll_deg", default=-20.0),
        Float("lateral_m", default=0.0),
    ],
)
class ArmBowlPlace:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """`forward_m` along the base's cardinal yaw, down toward `min_eef_z` but never further
        than `down_m`, and a `roll_deg` wrist roll about world X relative to the current eef quat.

        THE DESCENT IS A FLOOR, NOT A DISTANCE, and that is the whole point of the clamp. A blind
        "drop 0.27 m" has to assume the carry height (1.077 m, measured once); if the hand arrives
        8 cm lower the target lands under the tabletop and cuRobo refuses the plan. Asking instead
        for a descent deeper than needed and pinning the floor to `table_top + gripper_offset +
        clearance` puts the release height under the control of the TABLE, which is the thing that
        actually determines whether the object is set down or dropped.

        The floor is quoted in the ENV frame -- the same frame as the returned goal, the goal file,
        and the 0.74 m table height. env_origins carries a z, so a world-frame floor would land in
        a different place in every env.
        """
        import torch
        from isaaclab.utils.math import quat_mul
        current_eef_pos = ctx.robot.data.body_pos_w[:, eef_idx]
        current_eef_quat = ctx.robot.data.body_quat_w[:, eef_idx]

        yaw = _base_yaw(ctx, envs)
        rad, deg, idx = _closest_cardinal_yaw(yaw)

        dtype = current_eef_pos.dtype
        dev = idx.device
        forward = _place_knob(params, "forward_m", "SIMVLA_PLACE_FORWARD", 0.23,
                              device=dev, dtype=dtype)
        lateral = _place_knob(params, "lateral_m", "SIMVLA_PLACE_LATERAL", 0.,
                              device=dev, dtype=dtype)
        down = _place_knob(params, "down_m", "SIMVLA_PLACE_DOWN", 0.05,
                           device=dev, dtype=dtype)
        floor_z = _place_knob(params, "min_eef_z", "SIMVLA_PLACE_MIN_Z", -math.inf,
                              device=dev, dtype=dtype)
        roll_deg = _place_knob(params, "roll_deg", "SIMVLA_PLACE_ROLL_DEG", -20.0,
                               device=dev, dtype=dtype)

        # The forward DIRECTION per cardinal quadrant, unit length. Kept separate from the distance
        # so that a per-env (N,) forward_m multiplies cleanly; writing the distance into the table
        # the way the old constants did only works while it is a scalar.
        fwd = torch.zeros((idx.shape[0], 2), device=dev, dtype=dtype)
        fwd[idx == 0] = torch.tensor([1.0, 0.0], device=dev, dtype=dtype)
        fwd[idx == 1] = torch.tensor([0.0, 1.0], device=dev, dtype=dtype)
        fwd[idx == 2] = torch.tensor([0.0, -1.0], device=dev, dtype=dtype)
        fwd[idx == 3] = torch.tensor([-1.0, 0.0], device=dev, dtype=dtype)

        origins = ctx.env_origins[envs]
        xy_e = current_eef_pos[envs, :2] + fwd * forward.reshape(-1, 1) - origins[:, :2]
        # Positive lateral is left of the same cardinal heading, allowing a
        # wall-clear parking pose without forcing the base beside the basin.
        left = torch.stack((-fwd[:, 1], fwd[:, 0]), dim=1)
        xy_e = xy_e + left * lateral.reshape(-1, 1)

        z_e = current_eef_pos[envs, 2] - origins[:, 2]
        # Descend by down_m, stop at the floor, and never ascend: a floor is "do not go below
        # this", so a hand already under it stays where it is rather than lifting the object.
        z_out = torch.minimum(z_e, torch.maximum(z_e - down.reshape(-1), floor_z.reshape(-1)))

        pos_e = torch.cat([xy_e, z_out.reshape(-1, 1)], dim=1)

        # WHAT THE PLACE ACTUALLY ASKED FOR. The floor is applied here, but the object was still
        # released 0.13 m up (job 2106014 env23) with the arm supposedly gated to 0.05 m of its
        # goal -- so either this goal is not what the arm was driven to, or the object does not sit
        # where eef_above_base says. One line settles which. SIMVLA_PLACE_DBG=1 to enable.
        if _os.environ.get("SIMVLA_PLACE_DBG"):
            for _n, _i in enumerate(envs.tolist()):
                print(f"[place] env{_i} eef_z_now={float(z_e[_n]):.4f} "
                      f"goal_z={float(z_out[_n]):.4f} floor={float(floor_z.reshape(-1)[0]):.4f} "
                      f"down={float(down.reshape(-1)[0]):.3f}", flush=True)

        base_q = current_eef_quat[envs]
        half = 0.5 * torch.deg2rad(roll_deg.reshape(-1)).to(base_q.dtype)
        zeros = torch.zeros_like(half)
        qx = torch.stack([torch.cos(half), torch.sin(half), zeros, zeros], dim=-1)
        qx = qx.expand_as(base_q)
        q_out = quat_normalize_wxyz(quat_mul(base_q, qx))
        return pos_e, q_out


@skill(
    id="arm.mug_to_position",
    actions=("A_r", "A_l", "A_b"),
    label="Mug to position",
    params=[],
)
class ArmMugToPosition:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Yaw-quadrant ±5 cm offset; quat unchanged from current.
        Branches on has_sweet_potato (z=5 cm) vs not (z=0)."""
        import torch
        current_eef_pos = ctx.robot.data.body_pos_w[:, eef_idx]
        current_eef_quat = ctx.robot.data.body_quat_w[:, eef_idx]

        yaw = _base_yaw(ctx, envs)
        rad, deg, idx = _closest_cardinal_yaw(yaw)

        dtype = current_eef_pos.dtype
        offsets = torch.zeros((idx.shape[0], 3), device=idx.device, dtype=dtype)
        if ctx.has_sweet_potato:
            offsets[idx == 0] = torch.tensor([0.0, -0.05, 0.05], device=idx.device, dtype=dtype)
            offsets[idx == 1] = torch.tensor([0.05, 0.0, 0.05], device=idx.device, dtype=dtype)
            offsets[idx == 2] = torch.tensor([-0.05, 0.0, 0.05], device=idx.device, dtype=dtype)
            offsets[idx == 3] = torch.tensor([0.0, 0.05, 0.05], device=idx.device, dtype=dtype)
        else:
            offsets[idx == 0] = torch.tensor([0.0, -0.05, 0.0], device=idx.device, dtype=dtype)
            offsets[idx == 1] = torch.tensor([0.05, 0.0, 0.0], device=idx.device, dtype=dtype)
            offsets[idx == 2] = torch.tensor([-0.05, 0.0, 0.0], device=idx.device, dtype=dtype)
            offsets[idx == 3] = torch.tensor([0.0, 0.05, 0.0], device=idx.device, dtype=dtype)

        pos_w = current_eef_pos[envs] + offsets
        pos_e = pos_w - ctx.env_origins[envs]
        return pos_e, current_eef_quat[envs]


@skill(
    id="arm.bottle_to_position",
    actions=("A_r", "A_l", "A_b"),
    label="Bottle to position",
    params=[],
)
class ArmBottleToPosition:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Yaw-quadrant offset + wrist roll. Branches on has_sweet_potato:
        sweet_potato → 1-2 cm xy offset @ z=0, -30° wrist roll about world Z;
        otherwise   → 1-2 cm xy offset @ z=3 cm, +25° wrist roll about world Y."""
        import torch
        from isaaclab.utils.math import quat_mul
        current_eef_pos = ctx.robot.data.body_pos_w[:, eef_idx]
        current_eef_quat = ctx.robot.data.body_quat_w[:, eef_idx]

        yaw = _base_yaw(ctx, envs)
        rad, deg, idx = _closest_cardinal_yaw(yaw)

        dtype = current_eef_pos.dtype
        offsets = torch.zeros((idx.shape[0], 3), device=idx.device, dtype=dtype)
        base_q = current_eef_quat[envs]

        if ctx.has_sweet_potato:
            offsets[idx == 0] = torch.tensor([0.01, 0.02, 0.0], device=idx.device, dtype=dtype)
            offsets[idx == 1] = torch.tensor([-0.02, 0.01, 0.0], device=idx.device, dtype=dtype)
            offsets[idx == 2] = torch.tensor([0.02, -0.01, 0.0], device=idx.device, dtype=dtype)
            offsets[idx == 3] = torch.tensor([-0.01, -0.02, 0.0], device=idx.device, dtype=dtype)
            angle = -30.0 * math.pi / 180.0
            half = 0.5 * angle
            qx = torch.tensor(
                [math.cos(half), 0.0, 0.0, math.sin(half)],
                device=base_q.device, dtype=base_q.dtype,
            )
        else:
            offsets[idx == 0] = torch.tensor([0.01, 0.02, 0.03], device=idx.device, dtype=dtype)
            offsets[idx == 1] = torch.tensor([-0.02, 0.01, 0.03], device=idx.device, dtype=dtype)
            offsets[idx == 2] = torch.tensor([0.02, -0.01, 0.03], device=idx.device, dtype=dtype)
            offsets[idx == 3] = torch.tensor([-0.01, -0.02, 0.03], device=idx.device, dtype=dtype)
            angle = 25.0 * math.pi / 180.0
            half = 0.5 * angle
            qx = torch.tensor(
                [math.cos(half), 0.0, math.sin(half), 0.0],
                device=base_q.device, dtype=base_q.dtype,
            )

        pos_w = current_eef_pos[envs] + offsets
        pos_e = pos_w - ctx.env_origins[envs]

        qx = qx.expand_as(base_q)
        q_out = quat_normalize_wxyz(quat_mul(base_q, qx))
        return pos_e, q_out


@skill(
    id="arm.bottle_pour",
    actions=("A_r", "A_l", "A_b"),
    label="Bottle pour",
    params=[],
)
class ArmBottlePour:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """No xy/z offset; ±60° wrist roll. Sign depends on has_ramen:
        has_ramen → +60° about world X (for bowl);
        otherwise → −60° about world Z (for bottle)."""
        import torch
        from isaaclab.utils.math import quat_mul
        current_eef_pos = ctx.robot.data.body_pos_w[:, eef_idx]
        current_eef_quat = ctx.robot.data.body_quat_w[:, eef_idx]

        yaw = _base_yaw(ctx, envs)
        _closest_cardinal_yaw(yaw)  # preserved for side-effect parity (no-op)

        pos_w = current_eef_pos[envs]
        pos_e = pos_w - ctx.env_origins[envs]

        base_q = current_eef_quat[envs]
        if ctx.has_ramen:
            angle = 60.0 * math.pi / 180.0
            half = 0.5 * angle
            qx = torch.tensor(
                [math.cos(half), math.sin(half), 0.0, 0.0],
                device=base_q.device, dtype=base_q.dtype,
            )
        else:
            angle = -60.0 * math.pi / 180.0
            half = 0.5 * angle
            qx = torch.tensor(
                [math.cos(half), 0.0, 0.0, math.sin(half)],
                device=base_q.device, dtype=base_q.dtype,
            )
        qx = qx.expand_as(base_q)
        q_out = quat_normalize_wxyz(quat_mul(base_q, qx))
        return pos_e, q_out


@skill(
    id="arm.pause",
    actions=("A_r", "A_l", "A_b"),
    label="Pause",
    params=[],
)
class ArmPause:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Use the current eef pose (in env frame) as the goal target.

        Instead of moving the arm, the goal slot is filled at runtime with whatever the eef
        happens to be at, so the planner produces a no-op trajectory.

        Returns:
            (pos_env, quat_wxyz) — tensors shaped (N, 3) and (N, 4) where
            N == envs.shape[0]. Caller writes these into payloads_tensor.
        """
        pos_w = ctx.robot.data.body_pos_w[:, eef_idx][envs]
        quat = ctx.robot.data.body_quat_w[:, eef_idx][envs]
        pos_e = pos_w - ctx.env_origins[envs]
        return pos_e, quat


# Refrigerator
@skill(
    id="arm.fridge_handle_pregrasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to pre-grasp refrigerator handle",
    params=[PrimPath("prim_path")],
)
class ArmFridgeHandlePregrasp:
    # Pairs with arm.fridge_handle_grasp exactly as arm.handle_pregrasp pairs with
    # arm.handle_grasp: same target, a larger outward standoff, so the hand arrives IN FRONT of
    # the bar instead of sweeping through the door panel on the way in.
    #
    # The two PAIRS are not interchangeable, and mixing them is a documented trap (see
    # hero_template.py's docstring): a cabinet pull is a small HORIZONTAL knob and a fridge bar is
    # a 30 cm VERTICAL rail, so the jaws close across different axes and the quaternions differ.
    plan = _authored("arm.fridge_handle_pregrasp")


@skill(
    id="arm.fridge_handle_grasp",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to grasp refrigerator handle",
    params=[PrimPath("prim_path")],
)
class ArmFridgeHandleGrasp:
    plan = _authored("arm.fridge_handle_grasp")


# There is no refrigerator skill. There never was one: opening a fridge is
# nav.open_articulation with a larger back-off, which is what
#
#     @register_skill("N", "Move robot to open refrigerator")
#     def skill_pull_articulation_pull(app, action, params):   # shadowed the real one, too
#         return [-0.4, -0.4, -0.4]
#
# was trying to say. v1's "open articulation" key was the float -0.25 and its resolver backed off a
# hardcoded 0.25 m, so the lookup key read exactly like a distance; someone wanting 40 cm wrote
# -0.4, it matched no key, and the robot drove to the env-frame corner (-0.4, -0.4) instead. The
# fridge is now the same skill with back_off_m=0.4.


# Reset skill
@skill(
    id="arm.reset",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to reset",
    params=[],
)
class ArmReset:
    def resolve(self, ctx, envs, params, eef_idx=None):
        """Return the arm to the pose it held before the last grasp.

        Runtime-resolved: the executor overwrites the WHOLE 7-vector (simvla_gen.py:2181,
        `arm_goals[reset, :7] = reset_r[...]`), so nothing an author could write here would
        survive. That is why the v1 payload's [1:7] — a hardcoded pose that has sat in this file
        since the beginning — never reached the robot, and why the v2 goal is `null`.

        The arithmetic itself is NOT lifted here, and deliberately so — but the reason is stronger
        than "the state has no home". THERE IS NO SINGLE RESET RESOLVER. The executor resets an arm
        three different ways, one per channel:

          A_r  pose AND quat from `reset_r` (the eef pose remembered at the last non-reset arm
               step), minus the env origin — except that with ramen or a sweet potato on the scene
               the quaternion comes from the LIVE eef instead.
          A_l  pose and quat from `reset_l`, minus the env origin. No live-quat branch.
          A_b  quaternion from `reset_l` / `reset_r`, but the POSITION from `home_l` / `home_r`
               through base2world — a fixed home pose, not the remembered one.

        Collapsing those into one resolve() would change what at least two of the three channels do,
        and a silent behaviour change in an arm reset is indistinguishable from a port error. So
        Task 6 wired the DISPATCH (an `arm.reset` step now selects the reset branch by skill id,
        not by a 999.0 in slot 0) and left the three bodies exactly where they are. Unifying them
        is a behaviour question — which of the three is right? — not a refactor, and it belongs
        with the spec's out-of-scope "collapse the near-duplicate A_r/A_l/A_b blocks" work.

        The skill declares resolve() because the goal genuinely IS runtime-resolved: the executor
        overwrites all 7 slots, so nothing an author could write here would survive, and the v2
        goal is `null`.
        """
        raise NotImplementedError(
            "arm.reset is resolved inside simvla_gen.py's three reset branches, and they do not "
            "agree: A_r restores `reset_r` (live quat if ramen/sweet_potato), A_l restores "
            "`reset_l`, and A_b takes the quaternion from those buffers but the position from "
            "home_l/home_r via base2world. There is no one body to put here. See the docstring."
        )


@skill(
    id="arm.place",
    actions=("A_r", "A_l", "A_b"),
    label="Move arm to place",
    #: GENTLE PLACE, same contract as arm.bowl_place. `clearance_m` is how far the object's
    #: UNDERSIDE ends up above the support; `eef_above_base_m` is the MEASURED gripper-to-object-
    #: base offset that turns it into an end-effector height (median grasp-candidate z minus the
    #: object's spawn z, pooled over a kitchen's rotations -- see table_template.KNOWN_OBJ).
    #:
    #: Both default to 0.0, which leaves the authored goal at SIMVLA_PLACE_ABOVE (0.15 m) above the
    #: target's bbox top exactly as before. Set them and the goal becomes
    #: `bbox_top + eef_above_base_m + clearance_m`, i.e. a set-down whose height is decided by the
    #: SUPPORT and the object, not by one constant that knew about neither: 0.15 m released a mug
    #: from 0.06 m and an apple from 0.09 m and neither number was chosen.
    params=[
        PrimPath("prim_path"),
        Float("clearance_m", default=0.0),
        Float("eef_above_base_m", default=0.0),
        # HOW FAR IN FROM THE TARGET'S BBOX EDGE the release point sits, along the nav direction.
        # 0.1 is the constant plan_arm_place hardcoded, so an existing step that omits this is
        # byte-identical. It was chosen for /world/table/top, which is 1.1 x 0.7 m. A PLATE is
        # 0.142 m across, so the same inset puts the release 29 mm off the plate's centre -- inside
        # its 71 mm radius, but with no margin for the placed object's own footprint. Pass 0.0 to
        # release over the target's centre.
        Float("inset_m", default=0.1),
    ],
)
class ArmPlace:
    plan = _authored("arm.place")


@skill(
    id="gripper.set",
    actions=("G_r", "G_l"),
    label="Gripper open/close",
    params=[Bool("grasp", default=True)],
)
class GripperSet:
    def plan(self, app: "GoalGeneratorApp", action: str, params: dict):
        """
        Gripper on/off.
        True = grasp/close, False = release/open.
        """
        return bool(params.get("grasp"))


# =============================================================================
# Yaw helpers, and the two resolvers that have no skill to live on
# =============================================================================
# resolve_move_front and resolve_grasp_target are v1-only. No @skill declares them, the GUI cannot
# author one and the v2 writer cannot emit one — but the corpus has one step of each, and the
# executor still has a branch for each (executor_dispatch.LEGACY_ONLY_SKILLS). So they stay as
# free functions and simvla_gen calls them directly, by name, rather than through the REGISTRY.
# They are the last two bodies in this file that are not attached to a skill; when those two goal
# files are regenerated as v2 they go, and so does LEGACY_ONLY_SKILLS.
#
# Signature, like every resolve(): inputs come from `ctx` (RuntimeContext), the per-arm body index
# arrives as `eef_idx`, and the return is (pos_e, quat) which the caller writes into
# payloads_tensor.

def _wrap_to_pi(angle):
    import torch
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def _closest_cardinal_yaw(yaw):
    """Return (rad, deg, idx) where idx is 0:0°, 1:90°, 2:-90°, 3:180°.

    Duplicates simvla_gen.py's closest_cardinal_yaw so resolvers can live
    in this file without a circular import. Behavior identical.
    """
    import torch
    device = yaw.device
    targets = torch.tensor(
        [0.0, torch.pi / 2, -torch.pi / 2, torch.pi], device=device
    )
    diff = _wrap_to_pi(yaw.unsqueeze(-1) - targets)
    dist = torch.abs(diff)
    idx = torch.argmin(dist, dim=-1)
    rad = targets[idx]
    deg = rad * 180 / torch.pi
    return rad, deg, idx


def _base_yaw(ctx, env_idx):
    """Yaw of the robot base_link for the given envs (wxyz → yaw)."""
    import torch
    w, x, y, z = ctx.robot.data.body_quat_w[env_idx, ctx.base_link_idx].unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def resolve_move_front(ctx, env_idx, eef_idx):
    """Yaw-quadrant ±20 cm offset (only applied if has_sweet_potato is true,
    otherwise offsets stay 0). Quat unchanged."""
    import torch
    current_eef_pos = ctx.robot.data.body_pos_w[:, eef_idx]
    current_eef_quat = ctx.robot.data.body_quat_w[:, eef_idx]

    yaw = _base_yaw(ctx, env_idx)
    rad, deg, idx = _closest_cardinal_yaw(yaw)

    dtype = current_eef_pos.dtype
    offsets = torch.zeros((idx.shape[0], 3), device=idx.device, dtype=dtype)
    if ctx.has_sweet_potato:
        offsets[idx == 0] = torch.tensor([0.2, 0.0, 0.03], device=idx.device, dtype=dtype)
        offsets[idx == 1] = torch.tensor([0.0, -0.2, 0.03], device=idx.device, dtype=dtype)
        offsets[idx == 2] = torch.tensor([-0.2, 0.0, 0.03], device=idx.device, dtype=dtype)
        offsets[idx == 3] = torch.tensor([0.0, 0.2, 0.03], device=idx.device, dtype=dtype)

    pos_w = current_eef_pos[env_idx] + offsets
    pos_e = pos_w - ctx.env_origins[env_idx]
    return pos_e, current_eef_quat[env_idx]


def resolve_grasp_target(ctx, env_idx, eef_idx):
    """Position = sweet_potato0 obj pos − env origin.
    Quat = fixed cardinal-yaw lookup (only populated if has_sweet_potato,
    otherwise zero-quat). Reads `ctx.rigid_objects["sweet_potato0"]`."""
    import torch
    obj_pos = (
        ctx.rigid_objects["sweet_potato0"].data.body_pos_w[env_idx].squeeze(1)
    )
    pos_e = obj_pos - ctx.env_origins[env_idx]

    yaw = _base_yaw(ctx, env_idx)
    rad, deg, idx = _closest_cardinal_yaw(yaw)

    dtype = ctx.robot.data.body_pos_w.dtype
    quats = torch.zeros((idx.shape[0], 4), device=idx.device, dtype=dtype)
    if ctx.has_sweet_potato:
        fixed = torch.tensor(
            [0.26296, -0.36909, 0.87852, -0.15111],
            device=idx.device, dtype=dtype,
        )
        quats[idx == 0] = fixed
        quats[idx == 1] = fixed
        quats[idx == 2] = fixed
        quats[idx == 3] = fixed
    return pos_e, quats


# =============================================================================
# The v1 bridge — one shim left, and it is a READER
# =============================================================================
# The registry above is the truth. Nothing writes v1 any more: the GUI names the skill, the writer
# emits v2, and the executor dispatches on skill ids. What survives is the *reading* side, because
# the files on disk did not change when the code did:
#
#   * this module: reloadable GUI templates whose steps name a skill only by a free-text `usage`
#     string. _legacy_skill_for turns one back into a SkillSpec.
#   * goal_format.py: v1 goal files, whose steps name a skill only by the magic float in a
#     coordinate slot. Its quarantined legacy reader owns those values now — they are not
#     importable from anywhere else, and no dispatch anywhere reads one.
#
# Both are archaeology: read once, at the edge, and immediately converted into a skill id.

#: GUI usages that are no longer skills. The refrigerator was never a skill — see NavOpenArticulation.
LEGACY_USAGE_ALIASES = {
    ("N", "Move robot to open refrigerator"): ("nav.open_articulation", {"back_off_m": 0.4}),
}

#: What a `usage`-less legacy step MEANS on an action that SEVERAL skills now offer.
#:
#: The released twins encode which skill they meant in one of two ways, and which one depends on a
#: fact about the registry AT THE TIME THEY WERE WRITTEN: if the action had more than one offerer
#: they carry a `usage` label, and if it had exactly one they carry nothing and rely on the lookup
#: below being unambiguous. Measured over the 7,905 goals/*.reloadable.json twins on disk, the split
#: is exact — A_b (18), A_l (10,741), A_r (32,893) and N (3,903) steps ALL carry a `usage`; G_l
#: (4,465), G_r (13,505) and N_s (14,020) steps NEVER do.
#:
#: So "exactly one offerer" is load-bearing corpus state, not an implementation detail, and adding a
#: skill to one of those three channels silently invalidates every twin that uses it. nav.push_prim
#: is the first to do it: it joins nav.to_prim on N_s, which made all 14,020 of those steps raise
#: "it is ambiguous which one was meant" — i.e. made every one of the 7,905 templates unloadable in
#: the GUI.
#:
#: This table pins the pre-existing reading rather than guessing a new one: the value is the skill
#: that WAS the action's sole offerer, so a twin resolves to exactly what it resolved to before.
#: G_l/G_r are still single-offerer (gripper.set) and so are deliberately absent — an entry for them
#: would be inert today and would suppress the loud error that is the right answer for a NEWLY
#: written file. Add one only alongside a second skill on that channel, as here.
LEGACY_DEFAULT_SKILL = {
    "N_s": "nav.to_prim",
}


def _legacy_skill_for(action, params):
    """(action, usage) -> (SkillSpec, params). The labels ARE the old usage strings, so this is a
    lookup, not a translation table. Goes away with the `usage` key in Task 4."""
    usage = params.get("usage")

    alias = LEGACY_USAGE_ALIASES.get((action, usage))
    if alias is not None:
        skill_id, extra = alias
        return REGISTRY[skill_id], {**params, **extra}

    offered = [s for s in REGISTRY.values() if action in s.actions]
    if usage is None:
        if len(offered) == 1:
            return offered[0], params
        # The action gained an offerer after this file was written; LEGACY_DEFAULT_SKILL says which
        # one it was written against. Checked only when `usage` is absent, so it can never override
        # a label a file actually carries.
        default = LEGACY_DEFAULT_SKILL.get(action)
        if default is not None:
            return REGISTRY[default], params
        raise ValueError(
            f"action {action!r} has no 'usage' and {len(offered)} skills offer it "
            f"({sorted(s.id for s in offered)}); it is ambiguous which one was meant."
        )
    for spec in offered:
        if spec.label == usage:
            return spec, params
    raise ValueError(
        f"no skill registered for action {action!r} with label {usage!r}. "
        f"{action!r} offers: {sorted(s.label for s in offered)}."
    )


# =============================================================================
# Every skill is declared. Check that none of them is half-wired.
# =============================================================================
#: The effector channels simvla_gen actually implements — its TASK_IDS (simvla_gen.py:1538),
#: transcribed. There is no "G_b": the GUI offered it, TASK_IDS had never heard of it, and any goal
#: file containing a G_b step died with KeyError while loading. So gripper.set declares G_r and G_l
#: only, and this set is what proves it: put "G_b" back into either place and the other one fails
#: the import. (Both grippers = two steps. The arms need A_b because a two-arm motion is planned as
#: one; two grippers are two independent binary channels.)
EXECUTOR_ACTIONS = {"N", "A_l", "A_r", "G_l", "G_r", "N_s", "A_b"}

# Fails at import, not an hour into a GPU run, if a skill defines neither plan() nor resolve() (or
# both), or names an action the executor has no channel for.
validate_registry(EXECUTOR_ACTIONS)
