# exaFLOPs

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from .aiworker_collisions import spawn_aiworker_from_usd

##
# Configuration
##


#: THE LIFT COLUMN'S PARKED HEIGHT (2026-09-24). `lift_joint` is [-0.5, 0]; 0 is the TOP and
#: puts arm_base_link at z 1.4316, where a 0.85-0.95 m counter grasp sits ~0.55 m below the
#: shoulder -- 0.72 m of a 0.82 m arm. The mug-to-sink matrix smoke measured what that costs
#: (most grasp plans IK_FAIL or stop 0.4 m short). The measured -0.30 m home is the default;
#: SIMVLA_AIWORKER_LIFT_M=0 restores the original top-lift C2 pose.
#: THREE readers must agree on it: this init state, cuRobo's lock_joints (simvla_gen.py patches
#: the yml value from the same variable) and the joint target the generator re-asserts every
#: step (joint_pos_target starts at zero and nothing else writes the lift's slot).
AIWORKER_LIFT_M = float(os.environ.get("SIMVLA_AIWORKER_LIFT_M", "-0.30") or -0.30)
if not -0.5 <= AIWORKER_LIFT_M <= 0.0:
    raise ValueError(f"SIMVLA_AIWORKER_LIFT_M={AIWORKER_LIFT_M} is outside lift_joint's [-0.5, 0]")

# The jaw motor is deliberately tunable for close-before-lift experiments. At 1 rad/s, the public
# mug diagnostics measured a large force imbalance between the two pads and lateral mug motion
# during close. A slower close keeps the default unchanged while allowing a reproducible test of
# whether the object is being pushed out of the pinch before the collector starts lifting.
AIWORKER_GRIPPER_SPEED_RAD_S = float(
    os.environ.get("SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S", "1.0"))
if (not math.isfinite(AIWORKER_GRIPPER_SPEED_RAD_S)
        or not 0.1 <= AIWORKER_GRIPPER_SPEED_RAD_S <= 1.0):
    raise ValueError(
        "SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S must be finite and within [0.1, 1.0] rad/s"
    )
# Optional symmetric wrist-gain experiment for loaded home return. Keep the
# reference value until a new collection AND replay validate an alternative.
AIWORKER_WRIST_STIFFNESS = float(os.environ.get("SIMVLA_AIWORKER_WRIST_STIFFNESS", "252"))
if not math.isfinite(AIWORKER_WRIST_STIFFNESS) or not 252 <= AIWORKER_WRIST_STIFFNESS <= 1500:
    raise ValueError("SIMVLA_AIWORKER_WRIST_STIFFNESS must be finite and within [252, 1500]")
#: THE ARM HOME PER LIFT. At the top: C2, the BG2 teleop capture. Lowering the column with C2's
#: joints would sink the resting hands into the worktop. The default -0.30 m home is the user's
#: selected symmetric pose option 1; -0.35 m retains its earlier solved home. Both use the SG2
#: mirror rule (+,-,-,+,-,+,-). Keyed by lift value as a string with two decimals.
AIWORKER_LIFT_HOMES: dict[str, dict[str, float]] = {
    # The selected symmetric home (pose option 1), rendered from six views on 2026-10-01.
    # The right-arm joints are mirrored onto the left with (+,-,-,+,-,+,-).
    "-0.30": {
        "arm_r_joint1": 0.9455975,
        "arm_r_joint2": -0.8036000,
        "arm_r_joint3": -0.9631013,
        "arm_r_joint4": -2.7254865,
        "arm_r_joint5": -1.3239744,
        "arm_r_joint6": 0.5481815,
        "arm_r_joint7": -0.7211956,
        "arm_l_joint1": 0.9455975,
        "arm_l_joint2": 0.8036000,
        "arm_l_joint3": 0.9631013,
        "arm_l_joint4": -2.7254865,
        "arm_l_joint5": 1.3239744,
        "arm_l_joint6": 0.5481815,
        "arm_l_joint7": 0.7211956,
    },
    # The optional -0.35 home uses the same exact mirror rule.
    "-0.35": {
        "arm_r_joint1": 0.6724,
        "arm_r_joint2": -0.8662,
        "arm_r_joint3": -0.9821,
        "arm_r_joint4": -2.5193,
        "arm_r_joint5": -1.5797,
        "arm_r_joint6": 0.6565,
        "arm_r_joint7": -1.0316,
        "arm_l_joint1": 0.6724,
        "arm_l_joint2": 0.8662,
        "arm_l_joint3": 0.9821,
        "arm_l_joint4": -2.5193,
        "arm_l_joint5": 1.5797,
        "arm_l_joint6": 0.6565,
        "arm_l_joint7": 1.0316,
    },
}

# The public FFW-SG2 USD has visual meshes on its distal jaw links but no collision geometry
# applied to those meshes. Collection can therefore close through a mug even when the measured
# jaw frame is centered correctly. `scripts/tools/patch_aiworker_gripper_collisions.py` creates a
# local overlay with convex colliders on those same meshes; keep the downloaded asset immutable
# and select the prepared overlay explicitly for runs that need physical grasp contacts.
AIWORKER_USD_PATH = os.environ.get(
    "SIMVLA_AIWORKER_USD_PATH",
    f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/MM/aiworker/ffw_sg2.usd",
)
if "SIMVLA_AIWORKER_USD_PATH" in os.environ and not os.path.isfile(AIWORKER_USD_PATH):
    raise FileNotFoundError(
        "SIMVLA_AIWORKER_USD_PATH does not exist: " + AIWORKER_USD_PATH
    )


def _arm_home_for_lift(lift_m: float) -> dict[str, float]:
    if lift_m == 0.0:
        return dict(AIWORKER_C2_ARMS)     # the literals in AIWORKER_CFG below
    key = f"{lift_m:.2f}"
    if key not in AIWORKER_LIFT_HOMES:
        raise ValueError(f"SIMVLA_AIWORKER_LIFT_M={lift_m}: no arm home measured for that column "
                         f"height; run scripts/tools/probe_aiworker_lift_home.py --lift {lift_m} "
                         f"and bake its joints into AIWORKER_LIFT_HOMES (have: "
                         f"{sorted(AIWORKER_LIFT_HOMES) or 'none'})")
    return dict(AIWORKER_LIFT_HOMES[key])


AIWORKER_CFG = ArticulationCfg(
	spawn=sim_utils.UsdFileCfg(
		func=spawn_aiworker_from_usd,
		usd_path=AIWORKER_USD_PATH,
		activate_contact_sensors=False,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(
			rigid_body_enabled=True,
			kinematic_enabled=False,
			disable_gravity=False,
			max_linear_velocity=20.0,
			max_angular_velocity=20.0,
			max_depenetration_velocity=5.0,
			max_contact_impulse=20.0,
			stabilization_threshold=0.001,
		),
		# SOLVER ITERATIONS: 4, NOT 8, AND THE ROBOT CANNOT DRIVE AT 8.
		#
		# At solver_position_iteration_count=8 the two PRISMATIC base joints are dead --
		# commanded 0.20 m/s, achieved 0.0000 -- while base_revolute_z_joint drives normally at
		# 83% of command. The robot turns on the spot and never translates, which reads as a nav
		# bug and is not one.
		#
		# ISOLATED BY EXPERIMENT, one robot in an empty scene with the base joints commanded
		# directly (scripts/tools/probe_aiworker_base_drive.py), across a 2x2:
		#
		#   self_col  pos  vel   prismatic x / y
		#   True       8    2    0.0% / 0.0%     <- as this file shipped
		#   False      8    2    0.0% / 0.0%
		#   False      8    0    0.0% / 0.0%     <- so it is not the velocity iterations
		#   False      4    0   88.3% / 70.1%
		#   True       4    0   90.8% / 71.5%    <- so it is not the self-collision flag either
		#
		# WHY THE PRISMATIC JOINTS SPECIFICALLY. They act through base_x and base_y, dummy links
		# of 1e-7 kg carrying 94 kg of robot -- a mass ratio of 1e-9. The base drive is a
		# velocity PD (stiffness 0.01745, damping 1745), and more position iterations give the
		# solver more opportunity to resolve that ill-conditioned chain against the drive. The
		# revolute joint acts on base_link itself, 36 kg, and is untouched at any setting.
		#
		# Anubis and RB-Y1 both ship 4/0, which is why neither has ever shown this.
		# self_collisions stays True: it is measured harmless here, and this robot has two arms
		# that can reach each other.
		articulation_props=sim_utils.ArticulationRootPropertiesCfg(
			enabled_self_collisions=True,
			solver_position_iteration_count=4,
			# Resolve contact velocity and friction impulses during the mug lift.
			solver_velocity_iteration_count=4,
		),
	), # --/renderer/shadercache/driverDiskCache/flush=true
	init_state=ArticulationCfg.InitialStateCfg(
		# TEN MILLIMETRES OFF THE FLOOR, AND WITHOUT IT THE ROBOT CANNOT TRANSLATE AT ALL.
		#
		# The lowest collision geometry on this robot -- left_wheel_drive_link/collisions -- sits
		# at exactly z = 0.00000, so spawned at z = 0 the wheels rest precisely ON the floor
		# plane. That is fatal here in a way it is not for a normal robot, because THIS
		# ARTICULATION IS FIXED-BASE: the whole 94 kg hangs off base_prismatic_x/y_joint and
		# base_revolute_z_joint, and the root has no vertical degree of freedom to settle with.
		# PhysX therefore resolves the wheel/floor contact against a body it cannot move, drives
		# the normal force arbitrarily high, and the friction that comes with it pins the two
		# prismatic joints. Yaw survives because rotating about the vertical axis sweeps the
		# contact patch roughly in place.
		#
		# MEASURED (probe_aiworker_base_stall.py, in the real kitchen env, at a pose verified
		# clear of all geometry, with the ActionManager bypassed):
		#
		#   lift      travelled of 0.398 m expected     drive force on the y joint
		#   0.000 m   0.0019   (0.5%)   HELD            +1960 N  <- full command, joint immobile
		#   0.005 m   0.3981   (100.0%) DRIVEN             -2 N
		#   0.020 m   0.3981   (100.0%) DRIVEN             -2 N
		#   0.050 m   0.3981   (100.0%) DRIVEN             -3 N
		#   0.200 m   0.3981   (100.0%) DRIVEN             -3 N
		#   1.000 m   0.3981   (100.0%) DRIVEN             -3 N
		#
		# The drive was never short of force -- it was applying 1960 N into an immovable
		# constraint. That is why raising the damping from 1745 to 10000 fixed the yaw and did
		# nothing whatever for translation, and why 50000 would not have helped either.
		#
		# 0.005 is the smallest value measured good; 0.010 doubles that margin and is 0.7% of
		# this robot's height, so it is invisible on camera. Being fixed-base, it hovers there
		# rather than falling. Anubis's lowest collider is also at z = 0.00000 but it has never
		# shown this, so the clearance goes on THIS robot rather than into the shared floor.
		pos=(0.0, 0.0, 0.010),
		joint_pos={
			# The planar dummy triple nav actually drives. The OTHER FFW_SG2 in this tree
			# (Robots/FFW_SG2.usd) has a three-wheel swerve base instead and none of these
			# joints, so it cannot be driven by this pipeline at all.
			"base_prismatic_x_joint": 0.0,
			"base_prismatic_y_joint": 0.0,
			"base_revolute_z_joint": 0.0,

			# Top-lift C2 fallback. After this cfg is built, SIMVLA_AIWORKER_LIFT_M rewrites
			# this joint and both arm homes; the measured default is -0.30 m for counter reach.
			"lift_joint": 0.0,

			# C2 -- the BG2 teleop capture, ported joint for joint. The arm chain is
			# bit-identical between the two variants: every arm and head link agrees to
			# 3.7e-9 m when resolved against arm_base_link, measured across both USDs. So
			# these are this robot's own numbers rather than a neighbour's. All fourteen were
			# checked against SG2's own limits -- several of which are asymmetric and two
			# one-sided -- and test_aiworker_cfg.py pins that.
			#
			# Measured result, right hand: grasp centre (+0.259, -0.397, 1.031), reach
			# 0.474 m, tool 20.8 degrees nose-down.
			"arm_l_joint1": 0.5075439214706421,
			"arm_l_joint2": 0.34410303831100464,
			"arm_l_joint3": 0.2006279081106186,
			"arm_l_joint4": -1.7388510704040527,
			"arm_l_joint5": 0.4914371073246002,
			"arm_l_joint6": -0.07154582440853119,
			"arm_l_joint7": -1.1643078327178955,
			"arm_r_joint1": 0.7028987407684326,
			"arm_r_joint2": -0.349675714969635,
			"arm_r_joint3": -0.5260835289955139,
			"arm_r_joint4": -1.8550262451171875,
			"arm_r_joint5": -0.3125845491886139,
			"arm_r_joint6": 0.015327823348343372,
			"arm_r_joint7": 0.9495005011558533,

			# Grippers open.
			**{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
			**{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

			# THE HEAD JOINTS STAY AT ZERO, and this is load-bearing rather than lazy.
			# head_joint1's down-limit is 0.6951 rad (39.8 deg), already less than the 40 deg
			# the head camera wants -- but the real reason is that reset writes joint STATE
			# while joint_pos_target initialises to ZEROS (articulation.py:1243) and no action
			# term covers the head. A nonzero value here is dragged to 0 in the opening frames
			# of every episode, on camera, with the view swinging as it goes. RB-Y1 lost 34
			# degrees of head tilt exactly this way; see the long note in rby1_kitchen_cfg.py.
			# The camera pitch is baked into the TiledCameraCfg offset instead.
			"head_joint1": 0.0,
			"head_joint2": 0.0,
		},
	),


	actuators={
		# Actuators for swerve base
		"base": ImplicitActuatorCfg(
			joint_names_expr=["base_prismatic_x_joint","base_prismatic_y_joint", "base_revolute_z_joint"],
			# DAMPING 10000, NOT 1745, AND THE ARITHMETIC IS THE WHOLE ARGUMENT.
			#
			# This is a VELOCITY PD with stiffness ~0, so the force it can ever deliver is
			# damping * (v_target - v): at rest against a 0.20 m/s command that is
			# 1745 * 0.20 = 349 N, and the effort limit (1e5) never comes into it.
			#
			# The robot weighs 94.22 kg and its three wheels have NO JOINTS in this USD -- 28
			# actuatable joints and not one is a wheel -- so they are rigidly fixed and SCRUB
			# across the floor rather than rolling. Neither robot nor floor carries a physics
			# material, so PhysX's default mu = 0.5 applies:
			#
			#   AI Worker  94.22 kg -> needs 0.5 * 94.22 * 9.81 = 462 N  >  349 N   STUCK
			#   Anubis     57.08 kg -> needs 0.5 * 57.08 * 9.81 = 280 N  <  349 N   moves
			#
			# which is exactly what was measured: 0.0% of a commanded 0.463 rad/s on the ground,
			# 95.6% lifted 5 m clear of the floor.
			#
			# At 10000 the drive delivers 2000 N at that command and the steady state is
			# v_target - f/damping = 0.20 - 0.046 = 0.154 m/s, 77% of command. Raising DAMPING
			# rather than the effort limit is deliberate: the limit was never the binding
			# constraint, the PD's own gain was.
			effort_limit_sim=1e5,
			velocity_limit_sim=1e2,
			stiffness=0.01745,
			damping=10000,
#			 friction=0.75,
		),

		# Actuator for vertical lift joint
		"lift": ImplicitActuatorCfg(
			joint_names_expr=["lift_joint"],
			velocity_limit_sim=1.0,
			effort_limit_sim=400.0,
			stiffness=50000.0,
			damping=10000.0,
		),

		# ARM GAINS, PORTED FROM BG2 RATHER THAN INVENTED.
		#
		# aiworker_BG2.py carries per-joint gains tuned on the IDENTICAL arm chain -- same
		# links, same anchors, same limits, every one agreeing to 3.7e-9 m when both stages
		# are resolved against arm_base_link. The flat 400/80 that used to stand here was a
		# placeholder for a robot nobody had run, and it threw that tuning away by averaging
		# seven very different joints into one number: the elbow (arms_4) wants 880/66 and the
		# upper-arm roll (arms_3) wants 990/964, which is a 15x spread in damping alone.
		#
		# ONLY THE ARM TRANSFERS. BG2's 12-joint dexterous hand does not -- see "grippers".
		"arms_1": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint1", "arm_r_joint1"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0, stiffness=569, damping=178,
		),
		"arms_2": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint2", "arm_r_joint2"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0, stiffness=569, damping=178,
		),
		"arms_3": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint3", "arm_r_joint3"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0, stiffness=990, damping=964.0,
		),
		"arms_4": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint4", "arm_r_joint4"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0, stiffness=880.0, damping=66.0,
		),
		"arms_5": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint5", "arm_r_joint5"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0, stiffness=963.0, damping=269.0,
		),
		"arms_6": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint6", "arm_r_joint6"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0, stiffness=444.0, damping=57.0,
		),
		"arms_7": ImplicitActuatorCfg(
			joint_names_expr=["arm_l_joint7", "arm_r_joint7"],
			velocity_limit_sim=10.0, effort_limit_sim=1000.0,
			stiffness=AIWORKER_WRIST_STIFFNESS, damping=278.0,
		),

		# Actuators for grippers
		"grippers": ImplicitActuatorCfg(
			joint_names_expr=[
				"gripper_l_joint[1-4]",
				"gripper_r_joint[1-4]",
			],
			# THE JAWS WERE SLAMMING SHUT AND BATTING THE BOTTLE AWAY.
			#
			# At velocity_limit_sim=5.0 with damping=100 this hand closed 1.1 rad of travel
			# essentially instantly. Measured consequence: with the pads centred on the bottle to
			# within 0.017 m -- well inside its 0.032 m radius -- the gap still shut to 0.0102 m
			# and the bottle was left standing, displaced by up to 0.03 m and in one env knocked
			# onto the floor at obj_z=0.387. Not a positioning failure; the object was struck out
			# of the way before the jaws could close on it.
			#
			# Anubis, which grips reliably on this pipeline, uses velocity_limit_sim=0.1 with
			# damping=1e3 -- fifty times slower and ten times better damped. Its jaws are
			# PRISMATIC though, so 0.1 there is metres per second and 0.1 here would be radians:
			# 1.1 rad of travel would take 11 s, and the close-detection watches for the gap
			# changing by less than 0.001 m per step, so it would read a crawling jaw as already
			# settled and finish the step before anything closed.
			#
			# 1.0 rad/s closes in about a second (22 env steps at 20 Hz) and still moves the gap
			# ~0.005 m per step, comfortably above that threshold. damping takes Anubis's 1e3.
			velocity_limit_sim=AIWORKER_GRIPPER_SPEED_RAD_S,
			effort_limit_sim=1000.0,
			stiffness=60000.0,
			damping=1000.0,
		),

		# Actuators for head joints
		"head": ImplicitActuatorCfg(
			joint_names_expr=["head_joint1", "head_joint2"],
			velocity_limit_sim=5.0,
			effort_limit_sim=300.0,
			stiffness=200.0,
			damping=50.0,
		),
	}
)


#: C2 as the cfg literal spells it (the tests read the literal; this reads the cfg).
AIWORKER_C2_ARMS: dict[str, float] = {
    k: v for k, v in AIWORKER_CFG.init_state.joint_pos.items() if k.startswith("arm_")}

# A LOWERED COLUMN IS APPLIED AFTER THE FACT, on purpose: the joint_pos literal above stays the
# shipped C2 at lift 0 (test_aiworker_cfg reads those literals with ast), and the env knob
# rewrites the lift and the arm home on the constructed cfg only when it is set.
if AIWORKER_LIFT_M != 0.0:
    AIWORKER_CFG.init_state.joint_pos["lift_joint"] = AIWORKER_LIFT_M
    AIWORKER_CFG.init_state.joint_pos.update(_arm_home_for_lift(AIWORKER_LIFT_M))
