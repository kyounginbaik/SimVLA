# exaFLOPs

import isaaclab.sim as sim_utils
from .mobile_base import spawn_mobile_base_from_usd
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##
ANUBIS_CFG = ArticulationCfg(
	spawn=sim_utils.UsdFileCfg(
		func=spawn_mobile_base_from_usd,
#usd_path="${ISAACLAB_REPO_ROOT}/source/isaaclab_assets/data/Robots/MM/anubis/anubis_omni.usd",
#		usd_path="${ISAACLAB_REPO_ROOT}/anubis/anubis_omni_without_wheels/anubis_omni_without_wheels.usd",
	#	usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/anubis_for_random_parts.usd",
		usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/anubis_simvla.usd",
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
			solver_velocity_iteration_count=0,
			# fix_root_link=True,
		),
	), # --/renderer/shadercache/driverDiskCache/flush=true
	init_state=ArticulationCfg.InitialStateCfg(
		joint_pos={
			# base
			"base_prismatic_x_joint": 0.0,
			"base_prismatic_y_joint": 0.0,
			"base_revolute_z_joint": 0.0,

			# arm <-> base (rad)
			"arm1_base_link_joint": 0.0,
			"arm2_base_link_joint": 0.0,

			# Right arm
			"link11_joint": -0.5,
			"link12_joint": 2.356048653,
			"link13_joint": -0.07679449,
			"link14_joint": 0.52359878,
			"link15_joint": -0.17453293,
			
			# Left arm
			"link21_joint": -0.5,
			"link22_joint": 2.356048653,
			"link23_joint": -0.07679449,
			"link24_joint": -0.52359878,
			"link25_joint": 0.17453293,
			
			# finger
			"gripper1_joint": 0.04,
			"gripper1R_joint": 0.04,
			"gripper2_joint": 0.04,
			"gripper2R_joint": 0.04,
		},
		joint_vel={".*": 0.0},
	),

	actuators={
		"base": ImplicitActuatorCfg(
			joint_names_expr=["base_prismatic_x_joint","base_prismatic_y_joint", "base_revolute_z_joint"],
			effort_limit_sim=1e2,
			velocity_limit_sim=100,
			stiffness=0.01745,
			damping=1745,  # tip:: For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints. This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
#			 friction=0.75,
		),
		"arm_base1": ImplicitActuatorCfg(
			joint_names_expr=["arm1_base_link_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 352,
			damping= 139,
		),
		"arm_base2": ImplicitActuatorCfg(
			joint_names_expr=["arm2_base_link_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 352,
			damping= 139,
		),
		"arm_link11": ImplicitActuatorCfg(
			joint_names_expr=["link11_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 708,
			damping= 247,

		),
		"arm_link21": ImplicitActuatorCfg(
			joint_names_expr=["link21_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 708,
			damping= 247,
		),
		"arm_link12": ImplicitActuatorCfg(
			joint_names_expr=["link12_joint"],
			effort_limit_sim=39,
			velocity_limit_sim=10,
			stiffness= 859,
			damping= 314,
		),
		"arm_link22": ImplicitActuatorCfg(
			joint_names_expr=["link22_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 859,
			damping= 314,
		),
		"arm_link13": ImplicitActuatorCfg(
			joint_names_expr=["link13_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 272,
			damping= 69,
		),
		"arm_link23": ImplicitActuatorCfg(
			joint_names_expr=["link23_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 272,
			damping= 69,
		),
		"arm_link14": ImplicitActuatorCfg(
			joint_names_expr=["link14_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 922,
			damping= 139,
		),
		"arm_link24": ImplicitActuatorCfg(
			joint_names_expr=["link24_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 922,
			damping= 139,
		),
		"arm_link15": ImplicitActuatorCfg(
			joint_names_expr=["link15_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 984,
			damping= 197,
		),
		"arm_link25": ImplicitActuatorCfg(
			joint_names_expr=["link25_joint"],
			effort_limit_sim=50,
			velocity_limit_sim=10,
			stiffness= 984,
			damping= 197,
		),
		"anubis_right_hand": ImplicitActuatorCfg(
			joint_names_expr=["gripper1.*"],
			effort_limit_sim=200.0,
			velocity_limit_sim=2,
			stiffness=250,
			damping=10,
			friction= 1.0,
		),
		"anubis_left_hand": ImplicitActuatorCfg(
			joint_names_expr=["gripper2.*"],
			effort_limit_sim=50.0,
			velocity_limit_sim=1,
			stiffness=250,
			damping=10,
			friction= 1.0,
		),
	},
)



ANUBIS_PD_CFG = ANUBIS_CFG.copy()
# ANUBIS_PD_CFG.spawn.rigid_props.disable_gravity = True
# ANUBIS_PD_CFG.actuators["arm_link"].stiffness = 400.0
# ANUBIS_PD_CFG.actuators["arm_link"].damping ==400.0
# ANUBIS_PD_CFG.actuators["arm_base"].stiffness = 400.0
# ANUBIS_PD_CFG.actuators["arm_base"].damping ==400.0
