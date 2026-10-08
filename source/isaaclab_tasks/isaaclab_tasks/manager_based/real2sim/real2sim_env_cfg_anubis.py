from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import TiledCameraCfg

from . import mdp
import torch
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
# Scene definition
##
right_arm_joint_names = [
    "arm1_base_link_joint",
    "link11_joint",
    "link12_joint",
    "link13_joint",
    "link14_joint",
    "link15_joint",
]

left_arm_joint_names = [
    "arm2_base_link_joint",
    "link21_joint",
    "link22_joint",
    "link23_joint",
    "link24_joint",
    "link25_joint",
]

@configclass
class Real2SimSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ANUBIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_R_frame :  FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RightEndEffectorFrameTransformer_R"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_base_link",
                name="ee_tcp",
#                offset=OffsetCfg(
#                    pos=(0.0, 0.0, 0.0),
#                   rot=(0.7071, 0, 0, -0.7071)
#                ),
            ),
    ],
    )
    ee_L_frame : FrameTransformerCfg= FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LeftEndEffectorFrameTransformer_L"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper2_base_link",
                name="ee_tcp",
#                offset=OffsetCfg(
#                   pos=(0.0, 0.0, 0.027),
#                    rot=(0.7071, 0, 0, -0.7071)
#                ),
            ),
        ],
    )
    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=1/20,
        height=960,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyeKannalaBrandtK3",
            fisheye_max_fov=200,
            focal_length=1.93,
            focus_distance=0.5,
            horizontal_aperture=3.896,
            vertical_aperture=2.453,
            clipping_range=(0.01, 1000000),
            fisheye_polynomial_b=0.4
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.075, 0.0, 1.18),
            rot=(0.9396926, 0, 0.3420201, 0),
            convention="world"
        ),
    ) 

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    armL_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link2.*", "arm2.*"],
            body_name="gripper2_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                init_joint_pos=[ANUBIS_CFG.init_state.joint_pos[name] for name in left_arm_joint_names],
            ),
            scale=1.0,
#            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.027], rot=[0.7071, 0, 0, -0.7071]),
        )


    armR_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            body_name="gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                init_joint_pos=[ANUBIS_CFG.init_state.joint_pos[name] for name in right_arm_joint_names],
            ),
            scale=1.0,
        )

    gripperL_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper2.*"],
            open_command_expr={"gripper2.*": 0.04},
            close_command_expr={"gripper2.*": 0.0},
    )
    gripperR_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper1.*"],
            open_command_expr={"gripper1.*": 0.04},
            close_command_expr={"gripper1.*": 0.0},
    )

    base_action: mdp.JointVelocityActionCfg = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["base_prismatic_x_joint","base_prismatic_y_joint", "base_revolute_z_joint"],
    )

    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    robot_joint_stiffness_and_damping1 = EventTerm(
         func=mdp.randomize_actuator_gains,
         mode="reset",
         params={
             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm1_base_link_joint", "arm2_base_link_joint")),
             "stiffness_distribution_params": (10, 1000),
             "damping_distribution_params": (10, 1000),
             "operation": "abs",
             "distribution": "uniform",
         },
     )

    robot_joint_stiffness_and_damping2 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("link11_joint","link21_joint")),
            "stiffness_distribution_params": (10, 1000),
            "damping_distribution_params": (10, 1000),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping3 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("link12_joint","link22_joint")),
            "stiffness_distribution_params": (10, 1000),
            "damping_distribution_params": (10, 1000),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping4 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("link13_joint","link23_joint")),
            "stiffness_distribution_params": (10, 1000),
            "damping_distribution_params": (10, 1000),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping5 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("link14_joint","link24_joint")),
            "stiffness_distribution_params": (10, 1000),
            "damping_distribution_params": (10, 1000),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping6 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("link15_joint","link25_joint")),
            "stiffness_distribution_params": (10, 1000),
            "damping_distribution_params": (10, 1000),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    gripper_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="gripper.*"),
            "stiffness_distribution_params": (1.0, 300),
            "damping_distribution_params": (1.0, 300),
            "operation": "abs",
            "distribution": "uniform",
        },
    ) 

@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

##
# Environment configuration
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    sim2ruin = DoneTerm(
        func=mdp.sim2ruin, 
        params={
            "qpos": torch.zeros(1),
            "eefpos_thres": 0.06,
            "base_vel_thres": 2.0,
            "qpos_thres": 2.0,
            "rot_thres": 2.0,
            })

@configclass 
class Real2SimEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Real2SimSceneCfg = Real2SimSceneCfg(num_envs=4096, env_spacing=2.8)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 20  
        self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625


