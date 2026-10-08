# exaFLOPs

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

AIWORKER_BG2_CFG = ArticulationCfg(
	spawn=sim_utils.UsdFileCfg(
		usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/MM/aiworker/ffw_bg2.usd",
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
		articulation_props=sim_utils.ArticulationRootPropertiesCfg(
			enabled_self_collisions=True,
			fix_root_link=True,
			solver_position_iteration_count=8,
			solver_velocity_iteration_count=2,
		),
	),
	init_state=ArticulationCfg.InitialStateCfg(
		joint_pos={
			# Left arm joints
			"arm_l_joint1": 0.5075439214706421,
			"arm_l_joint2": 0.34410303831100464,
			"arm_l_joint3": 0.2006279081106186,
			"arm_l_joint4": -1.7388510704040527,
			"arm_l_joint5": 0.4914371073246002,
			"arm_l_joint6": -0.07154582440853119,
			"arm_l_joint7": -1.1643078327178955,
			# Right arm joints
			"arm_r_joint1": 0.7028987407684326,
			"arm_r_joint2": -0.349675714969635,
			"arm_r_joint3": -0.5260835289955139,
			"arm_r_joint4": -1.8550262451171875,
			"arm_r_joint5": -0.3125845491886139,
			"arm_r_joint6": 0.015327823348343372,
			"arm_r_joint7": 0.9495005011558533,

			# Head joints (rad) -- NOTE: head_joint1 (0.887) exceeds the URDF upper limit (0.6951 rad)
			"head_joint1": 0.4866409063339233,
			"head_joint2": -2.0694557179012918e-13,

			# Vertical lift (m)
			"lift_joint": 0.0,

			# Left dexterous hand: gripper_left_0..11 -> _1_1.._3_4 (finger-major), source degrees
			"gripper_l_j_dg_1_1": math.radians(30.799999237060547),
			"gripper_l_j_dg_1_2": math.radians(-0.20000000298023224),
			"gripper_l_j_dg_1_3": math.radians(140.3000030517578),
			"gripper_l_j_dg_1_4": math.radians(-65.19999694824219),
			"gripper_l_j_dg_2_1": math.radians(57.5),
			"gripper_l_j_dg_2_2": math.radians(82.80000305175781),
			"gripper_l_j_dg_2_3": math.radians(157.6999969482422),
			"gripper_l_j_dg_2_4": math.radians(83.0999984741211),
			"gripper_l_j_dg_3_1": math.radians(32.0),
			"gripper_l_j_dg_3_2": math.radians(0.0),
			"gripper_l_j_dg_3_3": math.radians(137.89999389648438),
			"gripper_l_j_dg_3_4": math.radians(-56.900001525878906),

			# Right dexterous hand: gripper_right_0..11 -> _1_1.._3_4 (finger-major), source degrees
			"gripper_r_j_dg_1_1": math.radians(-29.799999237060547),
			"gripper_r_j_dg_1_2": math.radians(-4.699999809265137),
			"gripper_r_j_dg_1_3": math.radians(140.89999389648438),
			"gripper_r_j_dg_1_4": math.radians(-64.5999984741211),
			"gripper_r_j_dg_2_1": math.radians(-28.0),
			"gripper_r_j_dg_2_2": math.radians(-0.30000001192092896),
			"gripper_r_j_dg_2_3": math.radians(139.1999969482422),
			"gripper_r_j_dg_2_4": math.radians(-56.900001525878906),
			"gripper_r_j_dg_3_1": math.radians(-57.20000076293945),
			"gripper_r_j_dg_3_2": math.radians(-84.19999694824219),
			"gripper_r_j_dg_3_3": math.radians(157.1999969482422),
			"gripper_r_j_dg_3_4": math.radians(82.5999984741211),
		},
	),


	actuators={
		# Actuator for vertical lift joint
		"lift": ImplicitActuatorCfg(
			joint_names_expr=["lift_joint"],
			velocity_limit_sim=1.0,
			effort_limit_sim=400.0,
			stiffness=50000.0,
			damping=100.0,
		),

		# Actuators for both arms
		"arms_1": ImplicitActuatorCfg(
			joint_names_expr=[
				"arm_l_joint1",
				"arm_r_joint1",
			],
			velocity_limit_sim=10.0,
			effort_limit_sim=1000.0,
			stiffness=569,
			damping=178,
		),
		"arms_2": ImplicitActuatorCfg(
			joint_names_expr=[
				"arm_l_joint2",
				"arm_r_joint2",
			],
			velocity_limit_sim=10.0,
			effort_limit_sim=1000.0,
			stiffness=569,
			damping=178,
		),
		"arms_3": ImplicitActuatorCfg(
			joint_names_expr=[
				"arm_l_joint3",
				"arm_r_joint3",
			],
			velocity_limit_sim=10.0,
			effort_limit_sim=1000.0,
			stiffness=990,
			damping=964.0,
        ),
        "arms_4": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint4",
                "arm_r_joint4",
            ],
            velocity_limit_sim=10.0,
            effort_limit_sim=1000.0,
            stiffness=880.0,
            damping=66.0,
		),
        "arms_5": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint5",
                "arm_r_joint5",
            ],
            velocity_limit_sim=10.0,
            effort_limit_sim=1000.0,
            stiffness=963.0,
            damping=269.0,
		),
        "arms_6": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint6",
                "arm_r_joint6",
            ],
            velocity_limit_sim=10.0,
            effort_limit_sim=1000.0,
            stiffness=444.0,
            damping=57.0,
		),
        "arms_7": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=10.0,
            effort_limit_sim=1000.0,
            stiffness=252.0,
            damping=278.0,
		),

		# Actuators for the two 3-finger dexterous grippers
		"grippers": ImplicitActuatorCfg(
			joint_names_expr=[
				"gripper_l_j_dg_[1-3]_[1-4]",
				"gripper_r_j_dg_[1-3]_[1-4]",
			],
			velocity_limit_sim=5.0,
			effort_limit_sim=100.0,
			stiffness=20.0,
			damping=2.0,
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
