# exaFLOPs

import isaaclab.sim as sim_utils
from .mobile_base import spawn_mobile_base_from_usd
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import json
import os
import math

_GRIPPER_STIFFNESS = float(os.environ.get("SIMVLA_RBY1_GRIPPER_STIFFNESS", "250"))
if not math.isfinite(_GRIPPER_STIFFNESS) or _GRIPPER_STIFFNESS <= 0:
	raise ValueError("SIMVLA_RBY1_GRIPPER_STIFFNESS must be finite and positive")

##
# Configuration
##
# SIMVLA_RBY1_GAINS=<json> overrides the arm actuator gains below at import time with
# {"stiffness": {joint: K}, "damping": {joint: D}} (the format calibration/rby1_identified_gains.json
# uses), for A/B runs of one gain set against another without editing this file. Unset = the
# values written below.
def _arm_gains(arm: str, prop: str, default: dict) -> dict:
	path = os.environ.get("SIMVLA_RBY1_GAINS", "")
	if not path:
		return default
	with open(path, "r", encoding="utf-8") as f:
		g = json.load(f)
	out = {j: float(v) for j, v in g.get(prop, {}).items() if j.startswith(f"{arm}_arm_")}
	if set(out) != set(default):
		raise ValueError(f"SIMVLA_RBY1_GAINS {path}: {prop} must list exactly {sorted(default)}; got {sorted(out)}")
	return out


RBY1_CFG = ArticulationCfg(
	spawn=sim_utils.UsdFileCfg(
		func=spawn_mobile_base_from_usd,
		# THE BLACK WRAPPER, not the raw asset. On the real RB-Y1 the gripper, the tool adapter
		# above it, the whole mobile base (platform, forks and wheels) and the head unit are BLACK;
		# the shipped USD renders all of them light. This layer REFERENCES the original by absolute
		# path and adds nothing but one UsdPreviewSurface material and 55 bindings, so the geometry,
		# joints and physics are still the vendor's (verified: 46 joints, articulation root
		# /RBY1_M/root_joint, 46 rigid bodies).
		#
		# THE ARMS AND TORSO ARE DELIBERATELY UNTOUCHED. Their meshes already split into a dark
		# material for the shoulder, elbow and waist housings and a light one for the tubes, chest
		# panel and spine, which is what the manufacturer's photographs show. That includes arm_5,
		# the wrist housing: an earlier round painted it and arm_6 a flat WristGrey 0.474 and got it
		# wrong in BOTH directions -- the housing should keep the vendor's white-plus-dark-band
		# split, and arm_6 with the FT sensor should be black, not grey. Those two are the ring
		# directly above the hand, and the joints between them are named `tool_right` and
		# `FT_Sensor_END_right`, which is how they were identified rather than by mesh name.
		#
		# IT HAS TO BE A USD, not a runtime event. `randomize_visual_color` was tried first and
		# measured NOT to work here: the four fingers carry no direct material (their colours
		# come from two GeomSubsets each) and every material is shared far beyond its link --
		# the wrist shares with link_*_arm_0..3. Filmed A/B on kitchen 1700, wrist camera:
		#   wrapper  gripper (28.9, 27.3, 26.7)  wrist (125.1, 121.7, 120.4)
		#   events   gripper (51.5, 51.0, 51.5)  wrist (142.7, 138.3, 139.5)
		#   target   gripper (27, 29, 28)        wrist (121, 123, 120)
		# The events arm is indistinguishable from the pre-change clip (59.6, 58.9, 59.7).
		usd_path=f"{os.environ['SIMVLA_RBY1M_DIR']}/models/rby1m/urdf/model/model_simvla_black_gripper.usd",
		activate_contact_sensors=False,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(
			rigid_body_enabled=True,
			kinematic_enabled=False,
			disable_gravity=False,
			max_linear_velocity=20.0,
			max_angular_velocity=20.0,
			max_depenetration_velocity=20.0,
			max_contact_impulse=20.0,
			stabilization_threshold=0.001,
		),
		articulation_props=sim_utils.ArticulationRootPropertiesCfg(
			enabled_self_collisions=False,
			solver_position_iteration_count=4,
			# Resolve contact velocity and friction impulses during the mug lift.
			solver_velocity_iteration_count=4,
			# fix_root_link=True,
		),
	), # --/renderer/shadercache/driverDiskCache/flush=true
	init_state=ArticulationCfg.InitialStateCfg(
		joint_pos={
			# base
			"base_prismatic_x_joint": 0.0,
			"base_prismatic_y_joint": 0.0,
			"base_revolute_z_joint": 0.0,

			"torso_0": 0.0,
			"torso_1": 0.0,
			"torso_2": 0.0,
			"torso_3": 0.0,
			"torso_4": 0.0,
			"torso_5": 0.0,

			# MolmoBot's home pose, verbatim. Source: allenai/molmospaces,
			# molmo_spaces/configs/robot_configs.py, RBY1Config.init_qpos --
			#     "right_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0]   (left identical)
			#     "head": [0.0, 0.6]     "torso": six zeros     grippers open at -0.05
			# This is the pose their published RB-Y1 simulation results were produced with, which
			# makes it the reference rather than one more thing for us to tune.
			#
			# MEASURED AGAINST THIS REPO'S KITCHENS, NOT ADOPTED WHOLESALE.
			#
			# MolmoBot's arm values do not transfer. Their arm_0=0.5 / arm_5=-0.5 was run here on
			# kitchen 1201, 32 envs: 179 refusals, every one "Start state is colliding with world",
			# and not a single successful plan. A second attempt keeping arm_0=0.5 and moving
			# arm_5 to +0.5 -- chosen to rest the hands mid-way between the 0.95 m counter and the
			# wall cabinets this builder samples at uniform(1.25, 1.35) -- failed the same way,
			# 129 refusals and no plans. So the hand's HEIGHT is not what cuRobo objects to; the
			# in-run penetration probe agrees, scoring that pose the CLEANEST of the three
			# (225 link-steps at 0.009 m worst depth, against 654 at 0.025 m for the pose that
			# plans fine). What the two failures share is arm_0=0.5, and what works is arm_0=1.0.
			# The mechanism behind that is still unexplained -- recorded here as a measurement
			# rather than dressed up as a reason.
			#
			# POSE F01, chosen 2026-08-25. REPLACES A01. Two independent requirements, and A01
			# only met the first:
			#   (1) the JAWS must be HORIZONTAL, so the gripper closes across a standing mug rather
			#       than along it. A01 met this -- 1.7 degrees from horizontal.
			#   (2) the APPROACH AXIS (ee_link1 local -Z, wrist -> fingertips) must point FORWARD.
			#       A01 aimed it OUTBOARD. Jaw roll and approach direction are independent; treating
			#       a flat jaw as sufficient is what let A01 through.
			# The visible consequence of (2): A01 put the hands at x 0.137 while the head camera sits
			# at x 0.18, so both grippers were 4 cm BEHIND the lens plane and 0.51 m out to either
			# side -- out of frame at any pitch or FOV. F01 puts them at (0.354, -+0.244, 1.077),
			# both fully inside the head camera.
			#
			# HOW F01 WAS FOUND. 3M poses sampled over the seven ARM joints with the torso pinned at
			# six zeros (it is locked there in this cfg and in rby1_right_arm.yml, so a search that
			# leaves it free invents poses the robot never holds), filtered on forward-ness and jaw
			# tilt, then the survivors RENDERED AND MEASURED IN SIM. Forward kinematics alone was not
			# trusted to rank them: predicted and measured hand positions differ by 0.023 m to
			# 0.151 m, growing with extension, as the arms sag against the actuator gains.
			#
			# MEASURED AT REST (5 s settle):  fwd 0.994   jaw 0.3 deg right / 0.8 deg left
			#                                 hand (0.354, -+0.244, 1.077)   reach 0.354
			#
			# THE POSE MUST BE AT REST, AND MIRROR SYMMETRY DOES NOT PROVE THAT. F06 and F07 passed
			# the mirror check at 0.75 s and had drifted 0.060 m and 0.078 m by 5 s -- both arms fall
			# symmetrically. F01 reads identically at both durations, drift 0.000. Any future
			# candidate has to clear the same two-duration test.
			#
			# THE MIRROR RULE PREVIOUSLY RECORDED HERE WAS WRONG. It said the left arm mirrors by
			# negating arm_4 alone. Solving all 128 sign patterns against the LEFT chain's own
			# kinematics over five probe poses gives, at error 0.00000:
			#     left = right * (+1, -1, -1, +1, -1, +1, -1)
			# The arm_4-only rule agreed with this only because A01 held arm_1, arm_2 and arm_6 at
			# zero, which hid three of the flips. F01 is nonzero on all seven, so the full pattern
			# matters; the left table below is that pattern applied to the right one.
			#
			# OPEN: arm_0 IS 0.717, NOT 1.0. The measurement above says arm_0=0.5 drew 179 planner
			# refusals on kitchen 1201 and arm_0=1.0 plans fine, with the mechanism unexplained.
			# 0.717 sits between them, and the rest of F01's configuration differs from the poses
			# that finding came from, so it does not transfer directly. The refusal count on the next
			# grasp-isolation run is what settles it -- if refusals spike, arm_0 is the first suspect.
			"right_arm_0": 0.717,
			"right_arm_1": -0.316,
			"right_arm_2": 0.19,
			"right_arm_3": -2.19,
			"right_arm_4": -0.282,
			"right_arm_5": -0.103,
			"right_arm_6": -1.484,

			# Left arm -- the right arm through (+1, -1, -1, +1, -1, +1, -1). NOT a copy.
			"left_arm_0": 0.717,
			"left_arm_1": 0.316,
			"left_arm_2": -0.19,
			"left_arm_3": -2.19,
			"left_arm_4": 0.282,
			"left_arm_5": -0.103,
			"left_arm_6": 1.484,

			# finger
			# Both hands start OPEN, matching MolmoBot's "left_gripper"/"right_gripper" = [-0.05].
			# They carry one value per hand; the URDF splits each hand into two prismatic joints
			# travelling on OPPOSITE signs (gripper_finger_r1 in [-0.05, 0], r2 in [0, 0.05]), so
			# their single -0.05 is this -+0.05 pair. The right pair used to start at 0.0 -- fully
			# closed -- while the left started open, so the hand that does the grasping approached
			# the mug with its jaws shut and could only knock it.
			"gripper_finger_r1": -0.05,
			"gripper_finger_r2": 0.05,
			"gripper_finger_l1": -0.05,
			"gripper_finger_l2": 0.05,
            
            # head
			# MolmoBot's head: [0.0, 0.6] -- pan 0 is forward, tilt 0.6 looks down at the
			# work surface. This repo had 0.4.
			# head_1 = 0.0, NOT MolmoBot's 0.6 (changed 2026-08-19). head_1 is the ONLY nonzero
			# init joint that NO action term covers -- the action space is arms, grippers and
			# base. reset_scene_to_default writes joint STATE (0.6), but joint_pos_target is
			# initialised to ZEROS (articulation.py:1243) and nothing ever writes the head, so its
			# implicit actuator (stiffness 250) drags it 0.6 -> 0 over the first moments of every
			# episode. Visible in the head camera as the scene starting at the floor and swinging
			# up to the worktop: the aim is head tilt PLUS the camera's own 40-degree offset, so
			# 34+40 = 74 degrees down at t=0 settling to 40.
			#
			# 0.6 was therefore never in force except as a transient. Setting it to the value
			# physics settles at makes the initial frame equal the steady frame -- no swing, and
			# the recorded data no longer opens on a second of floor. The AIM now lives entirely
			# in the camera offset, which is static and always honoured.
			"head_0": 0.0,
			"head_1": 0.0,
		},
		joint_vel={".*": 0.0},
	),

	actuators={
		"base": ImplicitActuatorCfg(
			# TRANSLATION ONLY. base_revolute_z_joint is deliberately NOT in this group any more:
			# it was never the joint that was stuck. The same [navtr] trace that measured the slides
			# at 5% of command measured the yaw at 88% (commanded 0.463 rad/s, achieved 0.410), so
			# 1e2 was ample there and raising it 20x would only buy an instant snap-to-speed on a
			# machine that turns while holding a mug. Split so the fix reaches the broken axis and
			# nothing else. See the "base_yaw" group below for the untouched rotation gains.
			joint_names_expr=["base_prismatic_x_joint","base_prismatic_y_joint"],
			# EFFORT LIMIT. 1e2 was the shipped value and it is what stopped this robot driving.
			#
			# MEASURED, from the sink pilot's own [navtr] trace: commanded 0.20 m/s, achieved
			# 0.011-0.020 m/s -- 5% of command, on 93% of the frames the base was told to move.
			# The base was not steering wrong, it was CREEPING, and every episode timed out short
			# of its nav goal with goal_step still 0. That is the whole reason the mug-to-sink task
			# never produced a demo: nothing downstream of "arrive at the mug" ever ran.
			#
			# WHY 1e2 CANNOT WORK HERE. model.urdf's 42 links total 158.9 kg, so the machine weighs
			# 1559 N. The drive is a velocity PD with stiffness ~0 and damping 1745, so the force it
			# asks for is 1745*(v_target - v), i.e. 349 N at rest for a 0.2 m/s target -- already
			# 3.5x past a 100 N ceiling before any load at all. Merely accelerating 158.9 kg to
			# 0.2 m/s in a quarter second is 127 N on its own. The limit bound every frame, and a
			# permanently saturated drive is exactly what "moves at 5% of command" looks like.
			#
			# 2e3 is chosen to STOP BINDING rather than to be large: it sits above the 349 N the PD
			# term can ever demand at this target speed, so the actuator delivers the gain it was
			# tuned for instead of a clipped constant. The damping then does the limiting, as
			# intended -- steady state is v_target - f/1745, so a ~100 N rolling load costs 0.057 m/s
			# and the base settles near 0.14 m/s rather than 0.015.
			#
			# The arms next to it are 1000/300/200; 1e2 for the joint that carries all 158.9 kg was
			# out of scale with them, not a deliberate weak base.
			effort_limit_sim=2e3,
			velocity_limit_sim=100,
			stiffness=0.01745,
			damping=1745,  # tip:: For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints. This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
#			 friction=0.75,
		),
		#: base_revolute_z_joint, at EXACTLY the numbers the whole base group carried before the
		#: translation limit was raised. Measured at 88% of commanded yaw rate, so there is nothing
		#: to fix here; this group exists only so that raising translation does not drag rotation
		#: up with it.
		"base_yaw": ImplicitActuatorCfg(
			joint_names_expr=["base_revolute_z_joint"],
			effort_limit_sim=1e2,
			velocity_limit_sim=100,
			stiffness=0.01745,
			damping=1745,
		),
		"torso": ImplicitActuatorCfg(
			joint_names_expr=["torso_.*"],
			effort_limit_sim=1000,
			velocity_limit_sim=10,
			stiffness= 3520,
			damping= 139,
		),
		# ARM STIFFNESS: IDENTIFIED FROM THE REAL ROBOT, 2026-08-31. Replaces the per-arm 552 (right)
		# and 708 (left) that shipped before; damping stays at the old 139 / 247 (see below). Two real teleop logs (/lustre/exaflops/rby1_data,
		# ~100 s) were replayed as absolute joint targets into 1024 envs each at THIS physics rate
		# (120 Hz, decimation 6); each joint's (K, D) is the median of the best 5% by its own tracking
		# error, pooled over both logs. The fixed-gain replay tracks the real follower to 0.009 rad
		# RMS per joint. Procedure and per-joint confidence: docs/superpowers/plans/
		# 2026-08-31-rby1-sysid-findings.md; values also in calibration/rby1_identified_gains.json.
		#
		# STIFFNESS is the identified per-joint set. DAMPING IS NOT -- it is the old per-arm 139/247.
		# The identified damping (arm_0 ~6, arm_3 ~14-400) went into a kitchen A/B on 2026-09-01
		# (kitchen 1550 mug->table, 32 envs, 8000 frames, jobs 2134517/2134919/2134920/2134921):
		#     old K + old D            9 demos / 430 episodes
		#     identified K + old D    10 / 425   (and 97 "grasp pose not reached" vs 127 -- reaches better)
		#     identified K, D ~6       0 / 462   (shoulders K 15000 / D 6)
		#     shoulders K 4000, D ~6   3 / 470
		# so the near-zero shoulder damping is what killed the grasp, not the stiffness. A strict
		# identifiability pass on 2026-09-01 (2048 envs/log, systemid_report.py --strict) found
		# 0 of 28 (joint, parameter) pairs identified: the damping objective varies only 1.3-2.0x
		# over three decades, and where it does vary it wants the shoulders UNDERDAMPED (D <= 63-126)
		# because low damping tracks a fast free-space trajectory better -- which is precisely what
		# fails under contact. (An earlier note here blamed the 300 N.m effort limit; measured
		# saturation peaks at 2.8% of steps, so that was wrong. The flat region is a model-error
		# floor.) The stiffness below satisfies every lower bound the data does support. Keep this
		# damping until data that loads the joints -- contact, payload, or recorded torques -- pins it.
		#
		# Per-joint dicts, not per-arm scalars: the shoulders (arm_0, arm_3) are ~40x stiffer than the
		# wrist roll (arm_6). arm_4 is the least excited joint in the logs. Every joint of the group
		# must appear here: IsaacLab's resolver raises on a joint matching two keys (so the regex
		# cannot stay alongside exact names) and silently keeps the USD default for a joint matching
		# none. SIMVLA_RBY1_GAINS=<json> overrides the whole set (see _arm_gains above); the raw
		# identified set incl. its damping is calibration/rby1_identified_gains.json.
		"right_arm": ImplicitActuatorCfg(
			joint_names_expr=["right_arm.*"],
			effort_limit_sim=300,
			velocity_limit_sim=10,
			stiffness=_arm_gains("right", "stiffness", {"right_arm_0": 14852, "right_arm_1": 4482, "right_arm_2": 4986, "right_arm_3": 6530, "right_arm_4": 1495, "right_arm_5": 3601, "right_arm_6": 406}),
			damping=_arm_gains("right", "damping", {"right_arm_0": 139, "right_arm_1": 139, "right_arm_2": 139, "right_arm_3": 139, "right_arm_4": 139, "right_arm_5": 139, "right_arm_6": 139}),
		),
		"left_arm": ImplicitActuatorCfg(
			joint_names_expr=["left_arm.*"],
			effort_limit_sim=300,
			velocity_limit_sim=10,
			stiffness=_arm_gains("left", "stiffness", {"left_arm_0": 15607, "left_arm_1": 3683, "left_arm_2": 3841, "left_arm_3": 12629, "left_arm_4": 1515, "left_arm_5": 2447, "left_arm_6": 371}),
			damping=_arm_gains("left", "damping", {"left_arm_0": 247, "left_arm_1": 247, "left_arm_2": 247, "left_arm_3": 247, "left_arm_4": 247, "left_arm_5": 247, "left_arm_6": 247}),
		),
		"right_hand": ImplicitActuatorCfg(
			joint_names_expr=["gripper_finger_r.*"],
			effort_limit_sim=200.0,
			velocity_limit_sim=2,
			stiffness=_GRIPPER_STIFFNESS,
			damping=10,
			friction= 1.0,
		),
		"left_hand": ImplicitActuatorCfg(
			joint_names_expr=["gripper_finger_l.*"],
			effort_limit_sim=50.0,
			velocity_limit_sim=1,
			stiffness=_GRIPPER_STIFFNESS,
			damping=10,
			friction= 1.0,
		),
		"head": ImplicitActuatorCfg(
			joint_names_expr=["head.*"],
			effort_limit_sim=50.0,
			velocity_limit_sim=1,
			stiffness=250,
			damping=10,
			friction= 1.0,
		),
	},
)
