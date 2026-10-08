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
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg 
from isaaclab.sensors import TiledCameraCfg

from . import mdp
import torch
import math
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.aiworker_BG2 import AIWORKER_BG2_CFG

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
# Scene definition
##
right_arm_joint_names = [
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
]

left_arm_joint_names = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
]

@configclass
class Real2SimSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = AIWORKER_BG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=1/30,
        height=720,
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
            pos=(0.15, 0.0, 1.35),
            rot=(0.8910065, 0, 0.4539905, 0),
            convention="world"
        ),
    )
#    camera = TiledCameraCfg(
#        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
#        update_period=1/30,
#        height=720,
#        width=1280,
#        data_types=["rgb"],
#        spawn=sim_utils.FisheyeCameraCfg(
#            projection_type="fisheyeKannalaBrandtK3",
#            fisheye_max_fov=200,
#            focal_length=1.93,
#            focus_distance=0.5,
#            horizontal_aperture=3.896,
#            vertical_aperture=2.453,
#            clipping_range=(0.01, 1000000),
#            fisheye_polynomial_b=0.4
#        ),
#        offset=TiledCameraCfg.OffsetCfg(
#            pos=(1.3, 0.0, 1.6),
#            rot=(0, -0.258819, 0.0, 0.9659258),
#            convention="world"
#        ),
#    ) 

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
    armL_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm_l_.*"],
            scale=1.0,
            use_default_offset=False,
        )
    armR_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm_r_.*"],
            scale=1.0,
            use_default_offset=False,
        )
    gripperL_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_l_j.*"],
            open_command_expr={
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
            },
            close_command_expr={
                "gripper_l_j_dg_1_1": math.radians(30.799999237060547),
                "gripper_l_j_dg_1_2": math.radians(-0.20000000298023224),
                "gripper_l_j_dg_1_3": math.radians(150.1999969482422),
                "gripper_l_j_dg_1_4": math.radians(-58.0),
                "gripper_l_j_dg_2_1": math.radians(57.599998474121094),
                "gripper_l_j_dg_2_2": math.radians(82.80000305175781),
                "gripper_l_j_dg_2_3": math.radians(158.39999389648438),
                "gripper_l_j_dg_2_4": math.radians(83.0999984741211),
                "gripper_l_j_dg_3_1": math.radians(32.0),
                "gripper_l_j_dg_3_2": math.radians(0.0),
                "gripper_l_j_dg_3_3": math.radians(147.6999969482422),
                "gripper_l_j_dg_3_4": math.radians(-49.599998474121094),
            }
    )
    gripperR_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_j.*"],
            open_command_expr={
                "gripper_r_j_dg_3_1": math.radians(-29.799999237060547),
                "gripper_r_j_dg_3_2": math.radians(-4.699999809265137),
                "gripper_r_j_dg_3_3": math.radians(140.89999389648438),
                "gripper_r_j_dg_3_4": math.radians(-64.5999984741211),
                "gripper_r_j_dg_1_1": math.radians(-28.0),
                "gripper_r_j_dg_1_2": math.radians(-0.30000001192092896),
                "gripper_r_j_dg_1_3": math.radians(139.1999969482422),
                "gripper_r_j_dg_1_4": math.radians(-56.900001525878906),
                "gripper_r_j_dg_2_1": math.radians(-57.20000076293945),
                "gripper_r_j_dg_2_2": math.radians(-84.19999694824219),
                "gripper_r_j_dg_2_3": math.radians(157.1999969482422),
                "gripper_r_j_dg_2_4": math.radians(82.5999984741211),
            },
            close_command_expr={
                "gripper_r_j_dg_3_1": math.radians(-29.700000762939453),
                "gripper_r_j_dg_3_2": math.radians(-4.5),
                "gripper_r_j_dg_3_3": math.radians(140.8000030517578),
                "gripper_r_j_dg_3_4": math.radians(-64.5999984741211),
                "gripper_r_j_dg_1_1": math.radians(-28.299999237060547),
                "gripper_r_j_dg_1_2": math.radians(0.0),
                "gripper_r_j_dg_1_3": math.radians(139.1999969482422),
                "gripper_r_j_dg_1_4": math.radians(-56.900001525878906),
                "gripper_r_j_dg_2_1": math.radians(-57.20000076293945),
                "gripper_r_j_dg_2_2": math.radians(-84.0),
                "gripper_r_j_dg_2_3": math.radians(157.1999969482422),
                "gripper_r_j_dg_2_4": math.radians(82.5),
            },
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
    
#    robot_joint_stiffness_and_damping1 = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint1", "arm_r_joint1")),
#             "stiffness_distribution_params": (10, 1000),
#             "damping_distribution_params": (10, 1000),
#             "operation": "abs",
#             "distribution": "uniform",
#         },
#     )
#
#    robot_joint_stiffness_and_damping2 = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint2", "arm_r_joint2")),
#            "stiffness_distribution_params": (10, 1000),
#            "damping_distribution_params": (10, 1000),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    )
#    robot_joint_stiffness_and_damping3 = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint3", "arm_r_joint3")),
#            "stiffness_distribution_params": (10, 1000),
#            "damping_distribution_params": (10, 1000),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    )
#    robot_joint_stiffness_and_damping4 = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint4", "arm_r_joint4")),
#            "stiffness_distribution_params": (10, 1000),
#            "damping_distribution_params": (10, 1000),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    )
#    robot_joint_stiffness_and_damping5 = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint5", "arm_r_joint5")),
#            "stiffness_distribution_params": (10, 1000),
#            "damping_distribution_params": (10, 1000),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    )
#    robot_joint_stiffness_and_damping6 = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint6", "arm_r_joint6")),
#            "stiffness_distribution_params": (10, 1000),
#            "damping_distribution_params": (10, 1000),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    )
#    robot_joint_stiffness_and_damping7 = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=("arm_l_joint7", "arm_r_joint7")),
#            "stiffness_distribution_params": (10, 1000),
#            "damping_distribution_params": (10, 1000),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    )
#    gripper_joint_stiffness_and_damping = EventTerm(
#        func=mdp.randomize_actuator_gains,
#        mode="reset",
#        params={
#            "asset_cfg": SceneEntityCfg("robot", joint_names="gripper.*"),
#            "stiffness_distribution_params": (1.0, 300),
#            "damping_distribution_params": (1.0, 300),
#            "operation": "abs",
#            "distribution": "uniform",
#        },
#    ) 

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
        func=mdp.sim2ruin_joint, 
        params={
            "qpos": torch.zeros(1),
            "qpos_thres": 2.5,
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
        self.sim.dt = 1 / 30  
        self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625


