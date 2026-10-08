"""Turn an emitted ANUBIS kitchen env-cfg into an AI WORKER (FFW_SG2) one, and register it.

The same shape as rby1_kitchen_cfg.py and for the same reason: everything an emitted env-cfg
says about the KITCHEN is robot-independent, so forking the 570-line template would duplicate
all of it and immediately drift. What is robot-dependent is a short list of blocks, each
replaced here, and every substitution is checked -- an anchor that does not appear exactly the
expected number of times raises. A silent no-op would emit a file that spawns an Anubis while
the planner runs AI Worker kinematics.

CPU-only: pure text, no Omniverse. Run it AFTER task_emit has written kitchen_<n>_<sub>.py.

WHERE THE CAMERA NUMBERS COME FROM. All measured off the robot's own USDs, not ported from
another machine:

  * The WRIST cameras already exist in the asset. Robots/FFW_SG2.usd carries
    arm_?_link7/camera_?_bottom_screw_frame/camera_?_link, the real D405 mounts, and measured
    relative to arm_?_link7 they are IDENTICAL on both sides -- same position, same rotation.
    In IsaacLab's `opengl` convention a camera looks down its own -Z, and link7's -Z is the
    tool approach axis, so one rotation about +Y reproduces both the asset's optical axis
    (5.5 degrees off the tool) and its roll.
  * The HEAD camera does NOT exist in the asset: head_link2 carries visuals and collisions
    only, so this mount is authored. 40 degrees down is setup K4 -- at that pitch the optical
    axis meets a 0.95 m counter 0.95 m from the robot origin, which is where it parks and
    reaches. head_link2's shell front face is +0.062 from its origin, so x=0.10 clears it.
  * THE HEAD JOINTS ARE NOT USED TO AIM IT. head_joint1's down-limit is 0.6951 rad (39.8
    degrees), already short of 40 -- but the real reason is the trap rby1_kitchen_cfg.py
    records: reset writes joint STATE while joint_pos_target initialises to ZEROS and no
    action term covers the head, so a nonzero head joint is dragged to 0 in the opening frames
    of every episode, on camera. Both head joints are 0 in AIWORKER_CFG and the pitch lives
    here, in the offset.
  * THE WRIST LENS IS 11.04 mm, NOT THE TEMPLATE'S 25. From the D405's measured mount the
    grasp centre sits 38.3 degrees off the optical axis and the jaw tips 33.3; a 25 mm pinhole
    at this aperture is a 22.7 degree half-angle, so the hand and anything in it are outside
    the frame at every pose. A real D405 is ~87 degrees wide, which is focal_length 11.04 at
    horizontal_aperture 20.955. Rendered both ways to confirm before changing it.

Both wrist views include the user-selected camera-local Z=-90 degree optical roll.
This is the upside-down version of the Z=+90 previews, not a change to the gripper
or grasp orientation. Mount positions and optical viewing axes remain unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SIMVLA_REPO_ROOT = os.environ.get(
    "SIMVLA_REPO_ROOT", str(Path(__file__).resolve().parents[2])
)
KITCHEN_PKG = Path(SIMVLA_REPO_ROOT) / (
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen"
)

# --------------------------------------------------------------------------------------
# The robot-dependent blocks. (anchor, replacement, expected_count)
# --------------------------------------------------------------------------------------

_ARM_NAMES_ANUBIS = '''right_arm_joint_names = [
\t"arm1_base_link_joint",
\t"link11_joint",
\t"link12_joint",
\t"link13_joint",
\t"link14_joint",
\t"link15_joint",
]

left_arm_joint_names = [
\t"arm2_base_link_joint",
\t"link21_joint",
\t"link22_joint",
\t"link23_joint",
\t"link24_joint",
\t"link25_joint",
]'''

_ARM_NAMES_AIWORKER = '''right_arm_joint_names = [
\t"arm_r_joint1",
\t"arm_r_joint2",
\t"arm_r_joint3",
\t"arm_r_joint4",
\t"arm_r_joint5",
\t"arm_r_joint6",
\t"arm_r_joint7",
]

left_arm_joint_names = [
\t"arm_l_joint1",
\t"arm_l_joint2",
\t"arm_l_joint3",
\t"arm_l_joint4",
\t"arm_l_joint5",
\t"arm_l_joint6",
\t"arm_l_joint7",
]'''

#: THE FRAME-TRANSFORMER OFFSET, which on this robot is very nearly nothing.
#:
#: Anubis's ee_link1 sits 0.10956 m short of its jaws, so its offset carries the hand out to
#: them. AI WORKER'S ee_link1 IS ALREADY AT THE JAWS: the MM USD places it at (0, 0, -0.178)
#: from arm_?_link7 rotated 180 degrees about X, and the pads' contact patch measures -0.1620
#: from that same link -- so in ee_link1's own frame the patch is 0.016 m away, and that is the
#: whole offset. (The 180-degree roll is why it is -0.016 and not +0.016: the frame's +Z runs
#: along the approach.)
#:
#: Nothing in the run path reads this FrameTransformer -- the only consumers in mdp/rewards.py
#: are commented out and observations.py reads body_pos_w directly -- so this is documentation
#: as much as configuration. It is still written correctly, because the next person to wire a
#: consumer to it will trust it.
_EE_OFFSET_ANUBIS = "\t\t\t\t\tpos=(0.0, 0.0, 0.1034),"
_EE_OFFSET_AIWORKER = "\t\t\t\t\tpos=(0.0, 0.0, -0.0160),"

#: HEAD CAMERA. Parented to head_link2 rather than base_link: base_link is on the floor, and a
#: camera hung off it at the template's z=1.18 sits inside this robot's torso shell
#: (arm_base_link spans z 1.142-1.553), which renders backfaces for the whole episode. That is
#: the failure RB-Y1 hit and diagnosed. head_link2's origin is (0.0662, 0, 1.5901) and its
#: shell reaches +0.062 forward of that, so (0.10, 0, 0.02) clears it by ~0.04 m and puts the
#: lens at world (0.166, 0, 1.610).
_HEAD_CAM_PRIM_ANUBIS = 'prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",'
_HEAD_CAM_PRIM_AIWORKER = 'prim_path="{ENV_REGEX_NS}/Robot/head_link2/head_cam",'
_HEAD_CAM_ANUBIS = "\t\t\tpos=(0.075, 0.0, 1.18),"
_HEAD_CAM_AIWORKER = "\t\t\tpos=(0.10, 0.0, 0.02),"
#: 40 degrees down about +Y -- setup K4. Where the axis lands, from the robot origin:
#: a 0.95 m counter at 0.95 m, a 0.74 m table at 1.20 m, a 0.35 m shelf at 1.67 m.
#: RB-Y1 uses 50 degrees from a head 0.114 m lower; the same framing on this robot is 54, and
#: 40 is the deliberately wider choice so the frame carries the drive-up as well as the grasp.
_HEAD_CAM_ROT_ANUBIS = "\t\t\trot=(0.9396926, 0, 0.3420201, 0),"
_HEAD_CAM_ROT_AIWORKER = "\t\t\trot=(0.93969, 0, 0.34202, 0),"

#: WRIST CAMERAS. The template's two blocks are byte-identical apart from their prim_path, so
#: the prim path is what distinguishes them and each substitution carries its own -- which also
#: makes a template reordering fail loudly rather than silently swap the two cameras.
_WRIST_R_ANUBIS = 'prim_path="{ENV_REGEX_NS}/Robot/ee_link1/ee_r_camera",'
_WRIST_L_ANUBIS = 'prim_path="{ENV_REGEX_NS}/Robot/ee_link2/ee_l_camera",'
_WRIST_R_AIWORKER = 'prim_path="{ENV_REGEX_NS}/Robot/arm_r_link7/wrist_r_cam",'
_WRIST_L_AIWORKER = 'prim_path="{ENV_REGEX_NS}/Robot/arm_l_link7/wrist_l_cam",'
#: Asset D405 position; orientation is normalized mount quaternion * local Z(-90).
#: User selected the upside-down version of both Z(+90) preview images.
_WRIST_OFF_ANUBIS = "\t\t\tpos=(0.0, -0.11, -0.13),\n\t\t\trot=(0.2164396,0.976296, 0.0, 0.0),"
_WRIST_OFF_AIWORKER = (
    "\t\t\tpos=(0.09824, 0.0, -0.07249),\n\t\t\trot=(0.7062927455, -0.0339198709, 0.0339198709, -0.7062927455),"
)
#: See the module docstring: at 25 mm this camera cannot see its own gripper.
_WRIST_LENS_ANUBIS = "\t\t\tfocal_length=25.0,"
_WRIST_LENS_AIWORKER = "\t\t\tfocal_length=11.04,"

#: A third-person "CCTV" camera, for WATCHING the robot rather than training on it. The head and
#: wrist cameras are what a policy consumes; neither shows what the arm is doing from outside,
#: and a failure you cannot see is a failure you cannot diagnose. It is also what
#: SIMVLA_RECORD_CAM=cctv records -- without this block that variable names a sensor the env does
#: not have, and the run writes no video at all.
#:
#: Mounted on base_link so it FOLLOWS the robot: this task drives from the counter to the table,
#: and a world-fixed camera loses the robot the moment the base moves.
#:
#: OVERHEAD, not behind: the earlier (-1.2, 0, 3.02) mount was behind kitchen 813's wall
#: after the base parked at the mug, so job 2395078 filmed only wall texture. At (0, 0, 3.02)
#: the camera stays above the robot and aims at its 1.05 m grasp plane 0.55 m ahead. The
#: 74.4-degree downward pitch is a quaternion about +Y in the camera's world convention.
#: This is a review-only camera; policy inputs still come from head and wrists.
#:
#: Sensor only -- no observation term reads it, so the policy input is unchanged.
_CCTV_ANCHOR = "\tfront = TiledCameraCfg("
_CCTV_AIWORKER = (
    '\tcctv = TiledCameraCfg(\n'
    '\t\tprim_path="{ENV_REGEX_NS}/Robot/base_link/cctv_cam",\n'
    '\t\tupdate_period=1/20,\n'
    '\t\theight=240,\n'
    '\t\twidth=320,\n'
    '\t\tdata_types=["rgb"],\n'
    '\t\tspawn = sim_utils.PinholeCameraCfg(\n'
    '\t\t\tfocal_length=14.0,\n'
    '\t\t\tfocus_distance=400.0,\n'
    '\t\t\thorizontal_aperture=20.955,\n'
    '\t\t\tclipping_range=(0.05, 1.0e5),\n'
    '\t\t),\n'
    '\t\toffset=TiledCameraCfg.OffsetCfg(\n'
    '\t\t\tpos=(0.0, 0.0, 3.02),\n'
    '\t\t\trot=(0.7967, 0.0, 0.6044, 0.0),\n'
    '\t\t\tconvention="world"\n'
    '\t\t),\n'
    '\t)\n'
    '\n'
    '\tfront = TiledCameraCfg('
)

_ACTIONS = [
    ('joint_names=["link2.*", "arm2.*"],', 'joint_names=["arm_l_joint.*"],', 1),
    ('joint_names=["link1.*", "arm1.*"],', 'joint_names=["arm_r_joint.*"],', 1),
    # Turn OFF the differential-IK dead zone, for both arms (hence expect=2).
    #
    # The controller discards any commanded joint delta under delta_joint_deadzone, per joint,
    # and that threshold does not survive redundancy: differential IK spreads one Cartesian
    # step across every joint the arm has, and this arm has SEVEN where Anubis's has six. RB-Y1
    # measured the consequence on its own seven-joint arm -- a 0.0137 rad command had all seven
    # joints fall under the 1e-2 default and the arm was issued 0.0, while the desired pose it
    # was chasing kept advancing. Its trajectories ended a median 0.071 m short of goal.
    (
        '\t\t\t\tik_method="dls",',
        '\t\t\t\tik_method="dls",\n\t\t\t\tdelta_joint_deadzone=0.0,',
        2,
    ),
    # GRIPPERS. Anubis's are prismatic and open at 0.04 / shut at 0.0. The RH-P12-RN is a
    # revolute linkage running the other way: gripper_?_joint1 and _joint3 (the proximal pair)
    # run [0, 1.1002 rad] and _joint2/_joint4 (the distal pair) [0, 1.0 rad], with ZERO being
    # OPEN. Measured at zero, the two pad faces sit 0.1114 m apart.
    #
    # The contact-holding action stops the commanded close at bilateral mug contact.
    (
        '\t\t\tjoint_names=["gripper2.*"],\n'
        '\t\t\topen_command_expr={"gripper2.*": 0.04},\n'
        '\t\t\tclose_command_expr={"gripper2.*": 0.0},',
        '\t\t\tjoint_names=["gripper_l_joint.*"],\n'
        '\t\t\topen_command_expr={"gripper_l_joint.*": 0.0},\n'
        '\t\t\tclose_command_expr={"gripper_l_joint1": 1.1002, "gripper_l_joint2": 1.0, '
        '"gripper_l_joint3": 1.1002, "gripper_l_joint4": 1.0},\n'
        '\t\t\tcontact_sensor_names=("touch_mug_l_r2", "touch_mug_l_l2"),\n'
        '\t\t\tadditional_contact_sensor_names=(("touch_mug_l_r1",), ("touch_mug_l_l1",)),\n'
        '\t\t\tpad_joint_names=(("gripper_l_joint1", "gripper_l_joint2"), ("gripper_l_joint3", "gripper_l_joint4")),',
        1,
    ),
    (
        '\t\t\tjoint_names=["gripper1.*"],\n'
        '\t\t\topen_command_expr={"gripper1.*": 0.04},\n'
        '\t\t\tclose_command_expr={"gripper1.*": 0.0},',
        '\t\t\tjoint_names=["gripper_r_joint.*"],\n'
        '\t\t\topen_command_expr={"gripper_r_joint.*": 0.0},\n'
        '\t\t\tclose_command_expr={"gripper_r_joint1": 1.1002, "gripper_r_joint2": 1.0, '
        '"gripper_r_joint3": 1.1002, "gripper_r_joint4": 1.0},\n'
        '\t\t\tcontact_sensor_names=("touch_mug_r_r2", "touch_mug_r_l2"),\n'
        '\t\t\tadditional_contact_sensor_names=(("touch_mug_r_r1",), ("touch_mug_r_l1",)),\n'
        '\t\t\tpad_joint_names=(("gripper_r_joint1", "gripper_r_joint2"), ("gripper_r_joint3", "gripper_r_joint4")),',
        1,
    ),
]

#: CONTACT SENSORS (Task 6). touch_grip_l/touch_grip_r each watch ONE representative pad per
#: hand -- index 0 of composed._FINGER_CANDIDATES's Anubis pair: gripper2R for the LEFT hand,
#: gripper1R for the RIGHT. This robot's pad in that SAME pair position is
#: gripper_l_rh_p12_rn_r2 / gripper_r_rh_p12_rn_r2 (composed._FINGER_CANDIDATES's third
#: tuple, index 0 again). touch_base watches base_link, which is shared naming between the
#: two robots and needs no substitution.
#:
#: The explanatory COMMENT above the sensor block spells out "gripper2R"/"gripper1R" too, so
#: it is translated here as well -- substituting only the prim_path would leave those two
#: substrings sitting in a comment and still fail
#: test_grippers_are_the_revolute_rh_p12_rn_not_a_prismatic_pair's "no gripper1/gripper2
#: survives" check.
_TOUCH_SENSORS = [
    (
        "\t# gripper2R/2L are the LEFT hand, gripper1R/1L the RIGHT (composed._FINGER_CANDIDATES).\n",
        "\t# gripper_l_rh_p12_rn_r2/_l2 are the LEFT hand, gripper_r_rh_p12_rn_r2/_l2 the RIGHT\n"
        "\t# (composed._FINGER_CANDIDATES).\n",
        1,
    ),
    (
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper2R",\n',
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_l_rh_p12_rn_r2",\n',
        1,
    ),
    (
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper1R",\n',
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_r_rh_p12_rn_r2",\n',
        1,
    ),
]

_AIWORKER_MUG_CONTACT_SENSORS = '''\t# Both segments contribute to each adaptive finger's object-contact measurement.
\ttouch_mug_l_r1: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_l_rh_p12_rn_r1",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_l_l1: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_l_rh_p12_rn_l1",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_r_r1: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_r_rh_p12_rn_r1",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_r_l1: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_r_rh_p12_rn_l1",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\t# Per-pad mug contact for grasp retention and contact-sized closing.
\ttouch_mug_l_r2: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_l_rh_p12_rn_r2",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_l_l2: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_l_rh_p12_rn_l2",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_r_r2: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_r_rh_p12_rn_r2",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_r_l2: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper_r_rh_p12_rn_l2",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
'''

#: PHYSICS RATE, taken from RB-Y1's finding rather than rediscovered. The template runs contact
#: at 20 Hz -- decimation 1 with sim.dt 1/20, a 50 ms step -- and RB-Y1 measured what that does
#: to a grasp: the object is held for about three frames and then extruded, because at 50 ms a
#: jaw can cross its whole travel inside ONE timestep and the two-point friction that has to
#: carry the object through the retract is resolved once per 50 ms. That diagnosis was about
#: the SOLVER, not about RB-Y1, so it applies here unchanged.
#:
#: decimation 6 x dt 1/120 leaves the ENV step at 6/120 = 0.05 s. Control rate, render
#: interval, episode_length_s, frames per episode and every recorded observation are unchanged
#: -- only the contact solve underneath them gets 6x finer.
_DECIMATION_ANUBIS = "\t\tself.decimation = 1\n"
_DECIMATION_AIWORKER = "\t\tself.decimation = 6\n"
_SIMDT_ANUBIS = "\t\tself.sim.dt = 1 / 20  # 20Hz\n"
_SIMDT_AIWORKER = (
    "\t\tself.sim.dt = 1 / 120  # 120 Hz physics; decimation 6 keeps control+render at 20 Hz\n"
)

#: reset_joints_by_offset over the Anubis arm + gripper joints. AI Worker keeps the ARMS only:
#: the lift and head are LOCKED at 0 in the cuRobo configs, so perturbing them would plan
#: against a pose the planner does not know about, and a +-0.1 rad offset on a jaw linkage
#: whose whole travel is 1.1 rad is a different grasp, not a perturbation of one.
_JOINT_INIT_ANUBIS = (
    "\"asset_cfg\": SceneEntityCfg(name=\"robot\", joint_names=['arm1_base_link_joint', "
    "'arm2_base_link_joint', 'link11_joint', 'link21_joint', 'link12_joint', 'link22_joint', "
    "'link13_joint', 'link23_joint', 'link14_joint', 'link24_joint', 'link15_joint', "
    "'link25_joint', 'gripper1R_joint', 'gripper1_joint', 'gripper2R_joint', 'gripper2_joint']),"
)
_JOINT_INIT_AIWORKER = (
    "\"asset_cfg\": SceneEntityCfg(name=\"robot\", joint_names=['arm_r_joint1', 'arm_r_joint2', "
    "'arm_r_joint3', 'arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', "
    "'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4', 'arm_l_joint5', "
    "'arm_l_joint6', 'arm_l_joint7']),"
)

#: Two randomize_visual_color events addressing Anubis mesh prims by path. This robot's USD has
#: no such prims and randomize_visual_color raises on a mesh_name it cannot resolve. They are
#: pure visual domain randomisation with no bearing on the grasp, so they are DROPPED rather
#: than guessed at. The object-colour and physics-material randomisers are untouched.
_VISUAL_EVENTS = re.compile(
    r"\trobot_link_material = EventTerm\(.*?\n\t\)\n\trobot_gripper_material = EventTerm\(.*?\n\t\)\n",
    re.DOTALL,
)

#: Every (anchor, replacement, count) that is a plain substitution, exposed so the test can
#: check each anchor against the real template without running the transform.
SUBSTITUTIONS = [
    ("from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG  # isort:skip",
     "from isaaclab_assets.robots.aiworker import AIWORKER_CFG  # isort:skip", 1),
    (_ARM_NAMES_ANUBIS, _ARM_NAMES_AIWORKER, 1),
    ("ANUBIS_CFG", "AIWORKER_CFG", 3),
    (_EE_OFFSET_ANUBIS, _EE_OFFSET_AIWORKER, 2),
    # INSERTED, not substituted: the template has no such camera.
    (_CCTV_ANCHOR, _CCTV_AIWORKER, 1),
    (_HEAD_CAM_PRIM_ANUBIS, _HEAD_CAM_PRIM_AIWORKER, 1),
    (_HEAD_CAM_ANUBIS, _HEAD_CAM_AIWORKER, 1),
    (_HEAD_CAM_ROT_ANUBIS, _HEAD_CAM_ROT_AIWORKER, 1),
    (_WRIST_R_ANUBIS, _WRIST_R_AIWORKER, 1),
    (_WRIST_L_ANUBIS, _WRIST_L_AIWORKER, 1),
    (_WRIST_LENS_ANUBIS, _WRIST_LENS_AIWORKER, 2),
    *_ACTIONS,
    *_TOUCH_SENSORS,
    (_DECIMATION_ANUBIS, _DECIMATION_AIWORKER, 1),
    (_SIMDT_ANUBIS, _SIMDT_AIWORKER, 1),
    (_JOINT_INIT_ANUBIS, _JOINT_INIT_AIWORKER, 1),
    ("class AnubisKitchenEnvCfg(ManagerBasedRLEnvCfg):",
     "class AIWorkerKitchenEnvCfg(ManagerBasedRLEnvCfg):", 1),
]


def transform(text: str) -> str:
    """Anubis kitchen env-cfg source -> AI Worker kitchen env-cfg source."""

    def sub(anchor: str, repl: str, expect: int = 1) -> None:
        nonlocal text
        n = text.count(anchor)
        if n != expect:
            raise SystemExit(
                f"[aiwcfg] anchor appeared {n}x, expected {expect}x:\n---\n{anchor[:300]}\n---"
            )
        text = text.replace(anchor, repl)

    # The wrist OFFSET blocks are byte-identical between the two cameras, so they cannot be
    # told apart by content -- only by which prim_path precedes them. Substitute them BEFORE
    # the prim paths are rewritten, while the Anubis anchors are still there to search from.
    def sub_after(anchor, old, repl):
        nonlocal text
        at = text.find(anchor)
        if at < 0:
            raise SystemExit(f"[aiwcfg] wrist anchor not found:\n---\n{anchor}\n---")
        rel = text.find(old, at)
        if rel < 0:
            raise SystemExit(
                f"[aiwcfg] no offset block after {anchor}: the template's wrist camera layout "
                f"changed, and blindly substituting would aim a camera at the wrong arm"
            )
        text = text[:rel] + repl + text[rel + len(old):]

    sub_after(_WRIST_R_ANUBIS, _WRIST_OFF_ANUBIS, _WRIST_OFF_AIWORKER)
    sub_after(_WRIST_L_ANUBIS, _WRIST_OFF_ANUBIS, _WRIST_OFF_AIWORKER)
    # Both wrist blocks now carry the same offset; the convention must move with it.
    n_conv = text.count('\t\t\tconvention="opengl",')
    if n_conv != 2:
        raise SystemExit(f"[aiwcfg] expected 2 opengl wrist conventions, found {n_conv}")

    for anchor, repl, expect in SUBSTITUTIONS:
        if (anchor, repl, expect) in _TOUCH_SENSORS:
            n = text.count(anchor)
            if n not in (0, 1):
                raise SystemExit(f"[aiwcfg] optional sensor anchor appeared {n}x, expected 0 or 1x")
            if n:
                text = text.replace(anchor, repl)
        else:
            sub(anchor, repl, expect)

    for arm in ("L", "R"):
        sub(
            f"\tgripper{arm}_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(",
            f"\tgripper{arm}_action: mdp.ContactHoldingBinaryJointPositionActionCfg = "
            "mdp.ContactHoldingBinaryJointPositionActionCfg(",
        )
    light_anchor = "\n\t# light\n"
    if text.count(light_anchor) != 1:
        raise SystemExit("[aiwcfg] light anchor must appear exactly once before mug sensors")
    text = text.replace(light_anchor, "\n" + _AIWORKER_MUG_CONTACT_SENSORS + light_anchor)

    text, n = _VISUAL_EVENTS.subn("", text)
    if n != 1:
        raise SystemExit(f"[aiwcfg] robot_link/gripper_material block matched {n}x, expected 1x")

    if "ANUBIS" in text or "anubis" in text:
        raise SystemExit("[aiwcfg] Anubis references survive the transform")
    return text


_REGISTER = '''
gym.register(
    id="Isaac-Kitchen-v{knum}a-{sub:02d}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.kitchen_{knum}a_{sub:02d}:AIWorkerKitchenEnvCfg",
    }},
)
'''


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aiworker_kitchen_cfg.py", description=__doc__)
    p.add_argument("--kitchen", required=True, type=int)
    p.add_argument("--subs", default="", help="comma-separated rotations; default = all on disk")
    p.add_argument(
        "--goals-dir",
        type=Path,
        help="rename AI Worker-authored v<kitchen>-NN goals to registered v<kitchen>a-NN ids",
    )
    args = p.parse_args(argv)

    knum = args.kitchen
    if args.subs:
        subs = [int(s) for s in args.subs.split(",")]
    else:
        subs = sorted(
            int(m.group(1))
            for m in (
                re.match(rf"kitchen_{knum}_(\d+)\.py$", f.name)
                for f in KITCHEN_PKG.glob(f"kitchen_{knum}_*.py")
            )
            if m
        )
    if not subs:
        raise SystemExit(f"[aiwcfg] no emitted kitchen_{knum}_*.py to transform")

    init_file = KITCHEN_PKG / "__init__.py"
    init_text = init_file.read_text()
    added = []
    for sub in subs:
        src = KITCHEN_PKG / f"kitchen_{knum}_{sub:02d}.py"
        dst = KITCHEN_PKG / f"kitchen_{knum}a_{sub:02d}.py"
        dst.write_text(transform(src.read_text()))
        print(f"[aiwcfg] wrote {dst}")
        task_id = f"Isaac-Kitchen-v{knum}a-{sub:02d}"
        if f'id="{task_id}"' not in init_text:
            init_text += _REGISTER.format(knum=knum, sub=sub)
            added.append(task_id)
    init_file.write_text(init_text)
    print(f"[aiwcfg] registered {len(added)} task ids: {added}")
    if args.goals_dir is not None:
        _rename_goals(args.goals_dir, knum, subs)
    return 0


def _rename_goals(directory: Path, knum: int, subs: list[int]) -> list[Path]:
    """Align AI Worker-authored goal filenames with the registered ``v<n>a-NN`` task ids."""
    directory = directory.expanduser().resolve()
    renamed = []
    for sub in subs:
        source = directory / f"Isaac-Kitchen-v{knum}-{sub:02d}.json"
        target = directory / f"Isaac-Kitchen-v{knum}a-{sub:02d}.json"
        if target.exists():
            if source.exists():
                raise FileExistsError(f"refusing to overwrite existing AI Worker goal: {target}")
            renamed.append(target)  # Idempotent rerun after the source was already renamed.
            continue
        if not source.is_file():
            raise FileNotFoundError(
                f"missing AI Worker-authored goal {source}; emit with SIMVLA_ROBOT=aiworker first"
            )
        source.rename(target)
        renamed.append(target)
        print(f"[aiwcfg] renamed {source.name} -> {target.name}")
    return renamed


if __name__ == "__main__":
    sys.exit(main())
