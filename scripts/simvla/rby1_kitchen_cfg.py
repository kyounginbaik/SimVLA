"""Turn an emitted ANUBIS kitchen env-cfg into an RB-Y1 one, and register it.

Why a text transform rather than a second `kitchen_env_cfg_source.py`:

  * Everything a kitchen env-cfg says about the KITCHEN -- the Change block (kitchen USD +
    one RigidObjectCfg per placed object), the baked lighting/material literals, the
    terminations composed from the task template -- is produced by
    `simvla_data_generator._write_env_config_file` and is robot-independent. Forking the
    570-line template would duplicate all of it and immediately drift.
  * The robot-dependent part is nine blocks, listed below. Each is replaced by the
    corresponding block from `molmospace_rby1.py`, which is the repo's one existing,
    hand-written RB-Y1 env-cfg -- so this script ports a working configuration rather than
    inventing one.

Every substitution is checked: an anchor that does not appear exactly once raises. A silent
no-op here would emit a file that still spawns an Anubis while the planner runs RB-Y1
kinematics, which is the exact failure the RB-Y1 gate report warned about.

CPU-only: pure text, no Omniverse. Run it AFTER task_emit has written kitchen_<n>_<sub>.py.
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
# The nine robot-dependent blocks. (anchor, replacement, expected_count)
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

_ARM_NAMES_RBY1 = '''right_arm_joint_names = [
\t"right_arm_0",
\t"right_arm_1",
\t"right_arm_2",
\t"right_arm_3",
\t"right_arm_4",
\t"right_arm_5",
\t"right_arm_6",
]

left_arm_joint_names = [
\t"left_arm_0",
\t"left_arm_1",
\t"left_arm_2",
\t"left_arm_3",
\t"left_arm_4",
\t"left_arm_5",
\t"left_arm_6",
]'''

#: Anubis's ee_link1 sits +0.10956 m along its own +Z from gripper_base_link, i.e. the tool
#: points along +Z. RB-Y1's ee_link1 sits -0.113 m along +Z from ee_right, so its tool points
#: along -Z. Nothing in the run path reads this FrameTransformer (the only consumers in
#: mdp/rewards.py are commented out; observations.py reads body_pos_w directly), so the
#: magnitude is carried over unchanged and only the SIGN is corrected, to keep the frame on
#: the finger side of ee_link1 rather than behind the wrist.
_EE_OFFSET_ANUBIS = "\t\t\t\t\tpos=(0.0, 0.0, 0.1034),"
_EE_OFFSET_RBY1 = "\t\t\t\t\tpos=(0.0, 0.0, -0.1034),"

#: CAMERAS. The kitchen template's camera block and molmospace_rby1.py's are byte-identical, so
#: nothing here was ever ported -- both carry Anubis-tuned offsets, and on RB-Y1 they render
#: nothing usable. Measured, from the RB-Y1 URDF with every link resolved to base_link:
#:
#:   * head_cam hangs off base_link at pos=(0.075, 0, 1.18). RB-Y1's base_link is on the FLOOR
#:     (z=0) and its torso shell reaches x=0.073 at link_torso_4 (z=0.996) and x=0.110 at
#:     link_torso_5 (z=1.306). So the camera sits INSIDE the torso: every one of the 2000 frames
#:     of a recorded episode is a flat grey/black wedge with a moving boundary -- the inside of
#:     the robot's own shell as the torso flexes. Moved forward to x=0.20, clear of the widest
#:     point by 0.09 m, and to z=1.30 near the torso top. The ROTATION is untouched: the 40-degree
#:     downward pitch is what aims it at the worktop, and only the mount was wrong.
#:
#:   * The wrist cameras needed BOTH a sign correction and a PER-SIDE transform, and the second
#:     half is why the first was not enough. Anubis's tool points along +Z and RB-Y1's along -Z
#:     (the same fact that makes GRASP_TOOL_FRAME a 180-degree flip), so the template's z=-0.13
#:     puts the camera past the fingertips on RB-Y1. Flipping that sign alone still left both
#:     views useless, because the template also shares ONE offset between the two arms and
#:     RB-Y1's LEFT end-effector frame is MIRRORED: the same transform that aims the right camera
#:     at the work aims the left one into the arm.
#:
#:     Rendered head to head at the extended pose (campose_probe, job 2056088): MolmoBot's own
#:     values (0,-0.11,-0.13) point the camera back down its own arm -- their file carries
#:     Anubis's numbers unchanged, so this was never solved upstream and there is nothing to
#:     copy. The template rotation at +0.13 looks sideways past the work. What renders the
#:     workspace with the fingertips framing the bottom corners on BOTH sides is a mirrored pair:
#:         right  pos (0, -0.11, 0.13)  rot +25 deg about X
#:         left   pos (0, +0.11, 0.13)  rot -25 deg about X
#:     so the substitution below is per side rather than one anchor applied twice.
#: THE PARENT LINK IS THE FIX, not the offset. This file previously kept head_cam on base_link
#: and only pushed it up and forward to escape the torso -- a workaround for the wrong mount.
#: molmospace_rby1.py, the repo's hand-written RB-Y1 env, had already diagnosed and solved it
#: with measurements (rby1_probe.py job 2049683): base_link at z=1.18 clears ANUBIS (1.19 m tall)
#: but sits INSIDE RB-Y1's torso (1.49 m tall), 0.149 m from link_torso_5, rendering backfaces
#: for 81-90% of every episode -- intermittently, because the camera hung off base_link while the
#: TORSO joints swung the shell in and out of the lens. Parenting to link_head_2 (world z 1.446)
#: at 0.18 m forward / 0.05 m up clears head_top (z 1.486) by ~0.16 m AND makes the camera follow
#: the head as a real head camera does, which is also how MolmoBot's training data was collected.
#: The rotation is untouched in both: the same 40-degree downward pitch.
_HEAD_CAM_PRIM_ANUBIS = 'prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",'
_HEAD_CAM_PRIM_RBY1 = 'prim_path="{ENV_REGEX_NS}/Robot/link_head_2/head_cam",'
_HEAD_CAM_ANUBIS = "\t\t\tpos=(0.075, 0.0, 1.18),"
_HEAD_CAM_RBY1 = "\t\t\tpos=(0.18, 0.0, 0.05),"

#: 50 DEGREES DOWN, THE SAME AIM THE DEPLOY ENV USES. `observation.images.front` in every LeRobot
#: dataset is rendered by THIS config; the policy that consumes those frames runs against
#: molmospace_rby1.py. The two must aim alike or the policy is trained on one view and served
#: another -- and for a while they did not: this was 74 degrees while deployment sat at 40.
#:
#: WHY 74 EXISTED, because the reasoning was sound and the conclusion was not. head_1 is the only
#: nonzero init joint with no action term covering it: reset writes joint STATE (0.6) but
#: joint_pos_target initialises to ZEROS (articulation.py:1243) and nothing ever writes the head,
#: so its actuator dragged it 0.6 -> 0 in the first moments of every episode -- on camera, the
#: scene starting at the floor and swinging up to the worktop. head_1 is now 0 in rby1.py (the
#: value physics settles at, so no swing), and the 34 degrees the tilt used to contribute was
#: folded into this static offset to match how MolmoBot's data was said to be collected:
#: 40 + 34 = 74.
#:
#: WHAT THAT COST. The fold-in reproduced the transient, not the steady state. The same comment
#: that justified it also recorded what the two pitches see -- 74 is "the floor", 40 is "the
#: worktop" -- and 74 is what shipped. Measured: at 74 degrees the optical axis meets a 0.74 m
#: table 0.20 m ahead of the head, i.e. the robot's own feet, while it parks 0.35-0.50 m off and
#: reaches ~0.50 m, so the whole manipulation zone sat crushed against the top of the frame. At 40
#: the axis meets that table 0.84 m ahead and the counter 0.59 m ahead, spanning the reach.
#:
#: 40 was the first correction (the camera's own design pitch, matching deploy); 30 is the second,
#: asked for after seeing 40 on video -- at 40 the axis still meets the counter only 0.59 m out,
#: barely past the 0.35-0.50 m parking standoff, so the near field still dominated. At 30 it meets
#: the counter 0.86 m and the table 1.22 m ahead, so the frame carries the approach as well as the
#: workspace. The lens is a 200-degree fisheye, so widening the aim costs no coverage of the hand.
#:
#: Tuned on video, not on paper: 74 (floor) -> 40 -> 30 (too far forward) -> 35 (still too
#: forward) -> 45 -> 50. At 45 the axis meets a 0.74 m table 0.71 m ahead of the head and the 0.95 m
#: counter 0.50 m ahead, which is where the robot parks and reaches.
#: Quaternion about +Y, cos/sin of 25 degrees. test_rby1_head_cam.py pins this to the deploy env's
#: value so the two cannot drift apart again silently -- change one and that test fails.
_HEAD_CAM_ROT_ANUBIS = "\t\t\trot=(0.9396926, 0, 0.3420201, 0),"
_HEAD_CAM_ROT_RBY1 = "\t\t\trot=(0.9063078, 0, 0.4226183, 0),"
#: The template's two wrist blocks are byte-identical, so they cannot be told apart by their
#: offset line alone -- the PRIM PATH is what distinguishes them. Each substitution therefore
#: carries its block's prim_path with it, which also makes a template reordering fail loudly
#: rather than silently swap the two cameras.
_WRIST_R_ANUBIS = ('prim_path="{ENV_REGEX_NS}/Robot/ee_link1/ee_r_camera",')
_WRIST_L_ANUBIS = ('prim_path="{ENV_REGEX_NS}/Robot/ee_link2/ee_l_camera",')
_WRIST_OFF_ANUBIS = "\t\t\tpos=(0.0, -0.11, -0.13),\n\t\t\trot=(0.2164396,0.976296, 0.0, 0.0),"
# User-selected candidate 07: upper-side offset 10 cm, setback 16 cm, pitch 20 deg.
# Original right roll; mirrored left includes local Z(pi) for an upright image.
# These are fixed wrist-local mounts, not world-stabilized overhead cameras.
_WRIST_OFF_R_RBY1 = "\t\t\tpos=(0.0, 0.1, 0.16),\n\t\t\trot=(0.984807753012208, -0.17364817766693033, 0.0, 0.0),"
_WRIST_OFF_L_RBY1 = "\t\t\tpos=(0.0, -0.1, 0.16),\n\t\t\trot=(0.0, 0.0, -0.17364817766693033, 0.984807753012208),"

#: A third-person "CCTV" camera, for WATCHING the robot rather than training on it. The head
#: and wrist cameras are what a policy consumes; neither shows what the arm is doing from
#: outside, and a failure you cannot see is a failure you cannot diagnose.
#:
#: Mounted on base_link rather than the world, so it FOLLOWS the robot: this task navigates
#: before it grasps, and a world-fixed camera loses the robot the moment the base drives.
#:
#: OVERHEAD-BEHIND, not the wide standoff a CCTV shot suggests. The first attempt put it 1.6 m
#: behind and 1.2 m to the side, and every frame came back a blurry cream surface -- the inside
#: of a wall. These kitchens are built with 0.20-0.75 m walkways and the robot parks 0.38 m off
#: the counter it is reaching into, so there is no 1.6 m of free floor behind it. Height is the
#: one direction that is always free: the room shell is four walls and NO CEILING
#: (ROOM_SHELL_DIMENSIONS in kitchen_build.py), so a camera ABOVE the walls looks down into the
#: room with nothing to clip against. So: 1.2 m back, 2.9 m up, aimed at (0.55, 0, 1.05) --
#: the grasp height at the counter -- which is 46.6 degrees down, written as
#: a unit quaternion about +Y. Compared side by side against a closer 0.35 m / 2.25 m mount,
#: which frames only the robot's shoulders: at 1.2 m back the whole robot, the counter run and
#: the mug are all in shot, which is what makes a failed grasp watchable.
#: a unit quaternion about +Y in the same "world" convention head_cam uses. A 14 mm lens
#: against head_cam's fisheye, because this one is for a human to read, not for coverage.
#: SIMVLA_CCTV_POS / SIMVLA_CCTV_ROT override it per run without re-emitting.
#:
#: INSERTED, not substituted: the Anubis template has no such camera, so this is the one entry
#: here that adds a block rather than porting one. Sensor only -- no observation term reads it,
#: so the policy input is unchanged.
_CCTV_ANCHOR = "\tfront = TiledCameraCfg("
_CCTV_RBY1 = (
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
    '\t\t\tpos=(-1.2, 0.0, 2.9),\n'
    '\t\t\trot=(0.9184, 0.0, 0.3957, 0.0),\n'
    '\t\t\tconvention="world"\n'
    '\t\t),\n'
    '\t)\n'
    '\n'
    '\tfront = TiledCameraCfg('
)

_ACTIONS = [
    ('joint_names=["link2.*", "arm2.*"],', 'joint_names=["left_arm_.*"],', 1),
    ('joint_names=["link1.*", "arm1.*"],', 'joint_names=["right_arm_.*"],', 1),
    # Turn OFF the differential-IK dead zone, for both arms (hence expect=2).
    #
    # The controller discards any commanded joint delta under delta_joint_deadzone, per joint.
    # That threshold does not survive redundancy: differential IK spreads one Cartesian step
    # across every joint the arm has, and RB-Y1's arms have seven where Anubis's have six.
    # Measured with SIMVLA_IKPROBE on this exact chain, a 0.0137 rad command had ALL SEVEN of
    # its joints fall under the 1e-2 default and the arm was issued 0.0 -- while the desired
    # pose it was chasing kept advancing, one plan index per step, waiting for nobody. Its
    # trajectories ended a median 0.071 m short of goal and 53 of them tripped the 0.05 m reset.
    #
    # Anubis keeps the default; this is why the substitution lives here and not in the shared
    # controller.
    (
        '\t\t\t\tik_method="dls",',
        '\t\t\t\tik_method="dls",\n\t\t\t\tdelta_joint_deadzone=0.0,',
        2,
    ),
    # RB-Y1's two jaws travel on OPPOSITE signs (gripper_finger_r1 in [-0.05, 0],
    # gripper_finger_r2 in [0, 0.05]), so a single regex value cannot express "open".
    # Values from molmospace_rby1.py: open = +-0.04 -> 86 mm between the jaw faces,
    # closed = 0 -> 6 mm. The mug body this chain grasps measures 68.7 mm.
    (
        '\t\t\tjoint_names=["gripper2.*"],\n'
        '\t\t\topen_command_expr={"gripper2.*": 0.04},\n'
        '\t\t\tclose_command_expr={"gripper2.*": 0.0},',
        '\t\t\tjoint_names=["gripper_finger_l.*"],\n'
        '\t\t\topen_command_expr={"gripper_finger_l1": -0.04, "gripper_finger_l2": 0.04},\n'
        '\t\t\tclose_command_expr={"gripper_finger_l1": 0.0, "gripper_finger_l2": 0.0},\n'
        '\t\t\tcontact_sensor_names=("touch_mug_l_pad1", "touch_mug_l_pad2"),\n'
        '\t\t\tpad_joint_names=(("gripper_finger_l1",), ("gripper_finger_l2",)),',
        1,
    ),
    (
        '\t\t\tjoint_names=["gripper1.*"],\n'
        '\t\t\topen_command_expr={"gripper1.*": 0.04},\n'
        '\t\t\tclose_command_expr={"gripper1.*": 0.0},',
        '\t\t\tjoint_names=["gripper_finger_r.*"],\n'
        '\t\t\topen_command_expr={"gripper_finger_r1": -0.04, "gripper_finger_r2": 0.04},\n'
        '\t\t\tclose_command_expr={"gripper_finger_r1": 0.0, "gripper_finger_r2": 0.0},\n'
        '\t\t\tcontact_sensor_names=("touch_mug_r_pad1", "touch_mug_r_pad2"),\n'
        '\t\t\tpad_joint_names=(("gripper_finger_r1",), ("gripper_finger_r2",)),',
        1,
    ),
]

#: CONTACT SENSORS (Task 6). touch_grip_l/touch_grip_r each watch ONE representative pad per
#: hand -- index 0 of composed._FINGER_CANDIDATES's Anubis pair: gripper2R for the LEFT hand,
#: gripper1R for the RIGHT. This robot's pad in that SAME pair position is
#: ee_finger_l1 / ee_finger_r1 (composed._FINGER_CANDIDATES's second tuple, index 0 again).
#: touch_base watches base_link, which is shared naming between the two robots (see the
#: head-cam comment above: 'every link resolved to base_link') and needs no substitution.
#:
#: The explanatory COMMENT above the sensor block spells out "gripper2R"/"gripper1R" too, so
#: it is translated here as well -- substituting only the prim_path would leave those two
#: substrings sitting in a comment, an ANUBIS name baked into an RB-Y1 cfg with nothing to
#: catch it (unlike aiworker_kitchen_cfg.py, no test here runs transform() over the real
#: template and checks what survives).
_TOUCH_SENSORS = [
    (
        "\t# gripper2R/2L are the LEFT hand, gripper1R/1L the RIGHT (composed._FINGER_CANDIDATES).\n",
        "\t# ee_finger_l1/l2 are the LEFT hand, ee_finger_r1/r2 the RIGHT\n"
        "\t# (composed._FINGER_CANDIDATES).\n",
        1,
    ),
    (
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper2R",\n',
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/ee_finger_l1",\n',
        1,
    ),
    (
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/gripper1R",\n',
        '\t\tprim_path="{ENV_REGEX_NS}/Robot/ee_finger_r1",\n',
        1,
    ),
]

#: PHYSICS RATE. The template runs contact at 20 Hz -- `decimation = 1` with `sim.dt = 1/20`, i.e.
#: a 50 ms physics step -- and that is where the RB-Y1 grasp dies. It is NOT a planning, targeting
#: or gripper-timing failure; all three were measured clean first:
#:
#:   * Geometry. Read off the URDF and EE_FINGER.dae, jaw separation is 0.006 + the face gap, so a
#:     grip on this 0.0687 m mug reads fingsep = 0.0747. All ten arm.grasp candidates in the 1201
#:     goal file put the jaw PAD CENTRE inside the mug body (radial offset 0.007-0.032 m against a
#:     0.034 m radius), and the fingmid those candidates predict at goal (median 0.067 m) matches
#:     what the run measures (median 0.085 m). The hand arrives and the jaws straddle the mug.
#:   * The grasp itself SUCCEEDS, for about three frames. From the run_2049853 trace:
#:         t=390 mug_z=0.957 fingsep=0.089     closing
#:         t=405 mug_z=0.961 fingsep=0.074     gripped -- 0.074 IS the mug's width
#:         t=420 mug_z=0.959 fingsep=0.074     still gripped, mug carried 9 mm up
#:         t=435 mug_z=0.933 fingsep=0.071     mug now BELOW its own rest height
#:         t=450 mug_z=0.607 fingsep=0.007     mug on the floor, jaws shut on air
#:     So the mug is held and then extruded. That is a RETENTION failure.
#:   * Statics say it should hold: the implicit PD jaws close from 0.0344 m of error at 250 N/m,
#:     ~8.6 N per jaw, against a 0.1-0.5 kg mug (1-5 N) at a combined friction of ~0.5. A 2-8x
#:     margin. Nothing about the force budget explains the loss.
#:
#: What does explain it is the step. At 50 ms and velocity_limit_sim=2, a jaw can cross its whole
#: 0.04 m travel INSIDE ONE TIMESTEP -- the gripper does not close on the mug, it swats it -- and
#: the two-point friction that has to carry the mug through the retract is resolved once per 50 ms
#: at 4 position and 0 velocity iterations.
#:
#: decimation 6 x dt 1/120 leaves the ENV step at 6/120 = 0.05 s. Control rate, render interval,
#: episode_length_s, frames per episode and therefore every recorded observation are unchanged --
#: only the contact solve underneath them gets 6x finer. Scoped to RB-Y1 so the Anubis numbers
#: stay a valid control.
_DECIMATION_ANUBIS = "\t\tself.decimation = 1\n"
_DECIMATION_RBY1 = "\t\tself.decimation = 6\n"
_SIMDT_ANUBIS = "\t\tself.sim.dt = 1 / 20  # 20Hz\n"
_SIMDT_RBY1 = (
    "\t\tself.sim.dt = 1 / 120  # 120 Hz physics; decimation 6 keeps control+render at 20 Hz\n"
)

#: A door-filtered contact sensor cannot establish that a grasp touched the mug. Instrument both
#: pads for either arm on emitted RB-Y1 kitchen configs so collection can require real two-sided
#: contact instead of inferring success from link-root proximity. The regex intentionally covers
#: the procedural mug0/... variants while remaining limited to the Kitchen prim tree.
_RBY1_MUG_CONTACT_SENSORS = '''\t# Per-pad mug contact for pre-lift and retention checks.
\ttouch_mug_r_pad1: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/ee_finger_r1",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_r_pad2: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/ee_finger_r2",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_l_pad1: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/ee_finger_l1",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
\ttouch_mug_l_pad2: ContactSensorCfg = ContactSensorCfg(
\t\tprim_path="{ENV_REGEX_NS}/Robot/ee_finger_l2",
\t\tfilter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
\t\tupdate_period=0.0, history_length=0, debug_vis=False,
\t)
'''

#: reset_joints_by_offset over the Anubis arm + gripper joints. RB-Y1 keeps the ARMS only:
#: the torso is LOCKED at 0 in the cuRobo config (rby1_right_arm.yml lock_joints), so
#: perturbing it would plan against a torso pose the planner does not know about; and a
#: +-0.1 rad offset on a jaw whose whole travel is 0.05 m is meaningless.
_JOINT_INIT_ANUBIS = (
    "\"asset_cfg\": SceneEntityCfg(name=\"robot\", joint_names=['arm1_base_link_joint', "
    "'arm2_base_link_joint', 'link11_joint', 'link21_joint', 'link12_joint', 'link22_joint', "
    "'link13_joint', 'link23_joint', 'link14_joint', 'link24_joint', 'link15_joint', "
    "'link25_joint', 'gripper1R_joint', 'gripper1_joint', 'gripper2R_joint', 'gripper2_joint']),"
)
_JOINT_INIT_RBY1 = (
    "\"asset_cfg\": SceneEntityCfg(name=\"robot\", joint_names=['right_arm_0', 'right_arm_1', "
    "'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6', 'left_arm_0', "
    "'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6']),"
)

#: Two randomize_visual_color events addressing Anubis mesh prims by path
#: (gripper_base_link/visuals/..., gripper1L/visuals/fingerL/mesh). RB-Y1's USD has no such
#: prims, and randomize_visual_color raises on a mesh_name it cannot resolve. They are pure
#: visual domain randomisation with no bearing on the grasp, so they are DROPPED rather than
#: guessed at. The object-colour and physics-material randomisers are untouched.
_VISUAL_EVENTS = re.compile(
    r"\trobot_link_material = EventTerm\(.*?\n\t\)\n\trobot_gripper_material = EventTerm\(.*?\n\t\)\n",
    re.DOTALL,
)


def transform(text: str) -> str:
    """Anubis kitchen env-cfg source -> RB-Y1 kitchen env-cfg source."""

    def sub(anchor: str, repl: str, expect: int = 1) -> None:
        nonlocal text
        n = text.count(anchor)
        if n != expect:
            raise SystemExit(
                f"[rby1cfg] anchor appeared {n}x, expected {expect}x:\n---\n{anchor[:300]}\n---"
            )
        text = text.replace(anchor, repl)

    sub(
        "from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG  # isort:skip",
        "from isaaclab_assets.robots.rby1 import RBY1_CFG  # isort:skip",
    )
    sub(_ARM_NAMES_ANUBIS, _ARM_NAMES_RBY1)
    # 3x: the scene's robot, and the two DifferentialIK init_joint_pos lookups.
    sub("ANUBIS_CFG", "RBY1_CFG", 3)
    sub(_EE_OFFSET_ANUBIS, _EE_OFFSET_RBY1, 2)
    sub(_CCTV_ANCHOR, _CCTV_RBY1, 1)
    sub(_HEAD_CAM_PRIM_ANUBIS, _HEAD_CAM_PRIM_RBY1, 1)
    sub(_HEAD_CAM_ANUBIS, _HEAD_CAM_RBY1, 1)
    sub(_HEAD_CAM_ROT_ANUBIS, _HEAD_CAM_ROT_RBY1, 1)
    def sub_after(anchor, old, repl):
        """Replace the first `old` that follows `anchor`. Needed because the two wrist blocks are
        byte-identical apart from their prim_path, so a plain replace cannot tell them apart --
        and they now take DIFFERENT values (RB-Y1's left EE frame is mirrored)."""
        nonlocal text
        at = text.find(anchor)
        if at < 0:
            raise SystemExit(f"[rby1cfg] wrist anchor not found:\n---\n{anchor}\n---")
        rel = text.find(old, at)
        if rel < 0:
            raise SystemExit(
                f"[rby1cfg] no offset block after {anchor}: the template's wrist camera layout "
                f"changed, and blindly substituting would aim a camera at the wrong arm"
            )
        text = text[:rel] + repl + text[rel + len(old):]

    sub_after(_WRIST_R_ANUBIS, _WRIST_OFF_ANUBIS, _WRIST_OFF_R_RBY1)
    sub_after(_WRIST_L_ANUBIS, _WRIST_OFF_ANUBIS, _WRIST_OFF_L_RBY1)
    for anchor, repl, expect in _ACTIONS:
        sub(anchor, repl, expect)
    for arm in ("L", "R"):
        sub(
            f"\tgripper{arm}_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(",
            f"\tgripper{arm}_action: mdp.ContactHoldingBinaryJointPositionActionCfg = "
            "mdp.ContactHoldingBinaryJointPositionActionCfg(",
        )
    # Older emitted kitchen configs predate the optional contact-sensor block. Translate it
    # when present, but don't make task registration depend on that newer instrumentation.
    for anchor, repl, _expect in _TOUCH_SENSORS:
        n = text.count(anchor)
        if n not in (0, 1):
            raise SystemExit(f"[rby1cfg] optional sensor anchor appeared {n}x, expected 0 or 1x")
        if n:
            text = text.replace(anchor, repl)
    light_anchor = "\n\t# light\n"
    if text.count(light_anchor) != 1:
        raise SystemExit("[rby1cfg] light anchor must appear exactly once before mug sensors")
    text = text.replace(light_anchor, "\n" + _RBY1_MUG_CONTACT_SENSORS + light_anchor)
    sub(_DECIMATION_ANUBIS, _DECIMATION_RBY1)
    sub(_SIMDT_ANUBIS, _SIMDT_RBY1)
    sub(_JOINT_INIT_ANUBIS, _JOINT_INIT_RBY1)
    text, n = _VISUAL_EVENTS.subn("", text)
    if n != 1:
        raise SystemExit(f"[rby1cfg] robot_link/gripper_material block matched {n}x, expected 1x")
    sub(
        "class AnubisKitchenEnvCfg(ManagerBasedRLEnvCfg):",
        "class RBY1KitchenEnvCfg(ManagerBasedRLEnvCfg):",
    )
    if "ANUBIS" in text or "anubis" in text:
        raise SystemExit("[rby1cfg] Anubis references survive the transform")
    return text


_REGISTER = '''
gym.register(
    id="Isaac-Kitchen-v{knum}r-{sub:02d}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.kitchen_{knum}r_{sub:02d}:RBY1KitchenEnvCfg",
    }},
)
'''


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rby1_kitchen_cfg.py", description=__doc__)
    p.add_argument("--kitchen", required=True, type=int)
    p.add_argument("--subs", default="", help="comma-separated rotations; default = all on disk")
    p.add_argument("--goals-dir", type=Path,
                   help="rename emitted goals to the registered RB-Y1 task ids")
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
        raise SystemExit(f"[rby1cfg] no emitted kitchen_{knum}_*.py to transform")

    init_file = KITCHEN_PKG / "__init__.py"
    init_text = init_file.read_text()
    added = []
    for sub in subs:
        src = KITCHEN_PKG / f"kitchen_{knum}_{sub:02d}.py"
        dst = KITCHEN_PKG / f"kitchen_{knum}r_{sub:02d}.py"
        dst.write_text(transform(src.read_text()))
        print(f"[rby1cfg] wrote {dst}")
        task_id = f"Isaac-Kitchen-v{knum}r-{sub:02d}"
        if f'id="{task_id}"' not in init_text:
            init_text += _REGISTER.format(knum=knum, sub=sub)
            added.append(task_id)
    init_file.write_text(init_text)
    print(f"[rby1cfg] registered {len(added)} task ids: {added}")
    if args.goals_dir is not None:
        _rename_goals(args.goals_dir, knum, subs)
    return 0


def _rename_goals(directory: Path, knum: int, subs: list[int]) -> list[Path]:
    """Align authored goals with the registered ``v<n>r-NN`` task ids."""
    directory = directory.expanduser().resolve()
    renamed = []
    for sub in subs:
        source = directory / f"Isaac-Kitchen-v{knum}-{sub:02d}.json"
        target = directory / f"Isaac-Kitchen-v{knum}r-{sub:02d}.json"
        if target.exists():
            if source.exists():
                raise FileExistsError(f"refusing to overwrite existing RB-Y1 goal: {target}")
            renamed.append(target)
            continue
        if not source.is_file():
            raise FileNotFoundError(
                f"missing RB-Y1-authored goal {source}; emit with SIMVLA_ROBOT=rby1 first"
            )
        source.rename(target)
        renamed.append(target)
        print(f"[rby1cfg] renamed {source.name} -> {target.name}")
    return renamed


if __name__ == "__main__":
    sys.exit(main())
