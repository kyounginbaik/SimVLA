# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Environment Configuration from exaFLOPs

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import MdlFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
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
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
import isaacsim.core.utils.prims as prim_utils
from . import mdp
import random
import os

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG  # isort:skip
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR



FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
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
# floor material
floor_list = ["{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks.mdl", "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch_Planks.mdl", "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Parquet_Floor.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks.mdl" ]
floor_material = random.choice( floor_list)
# wall material
wall_list = ["{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Adobe_Brick.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Pavers.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Block.mdl"]
wall_material = random.choice(wall_list)

##
# Scene definition



@configclass
class KitchenSceneCfg(InteractiveSceneCfg): 
    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = ANUBIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # End-effector, Will be populated by agent env cfg
    ee_R_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RightEndEffectorFrameTransformer_R"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link1",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
            ],
        )

    ee_L_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LeftEndEffectorFrameTransformer_L"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link2",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
            ],
        )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # floor
    floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.8, -1.3, 0.000000001),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/floor.usd",
                scale=(6,6,1.0),
                visual_material=MdlFileCfg(mdl_path=floor_material),
            ),
            collision_group=-1,
    )
    
# -------Change-------
    
    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=sim_utils.UsdFileCfg(usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/Kitchen/kitchen_435_03.usd"),
    )
    base_cabinet = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(0.85, -0.0145, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0, 'corpus_to_door_2_1': 0.0, 'corpus_to_drawer_0_0': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1", "corpus_to_door_2_1", "corpus_to_drawer_0_0"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    bowl0 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/bowl0",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, -0.2991, 0.8457), rot=(0.7071, 0.0, 0.0, 0.7071)))
    countertop_base_cabinet = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/countertop_base_cabinet",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, -0.0145, 0.828), rot=(1.0, 0.0, 0.0, 0.0)))
    countertop_dishwasher = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/countertop_dishwasher",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(2.4336, -0.0145, 0.828), rot=(1.0, 0.0, 0.0, 0.0)))
    dishwasher = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/dishwasher",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(2.4336, -0.0145, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpse_to_bottom_basket': 0.0, 'corpse_to_top_basket': 0.0, 'corpus_to_door_0_1': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["corpse_to_bottom_basket", "corpse_to_top_basket", "corpus_to_door_0_1"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    mug0 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/mug0",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9599, -0.2991, 0.8457), rot=(1.0, 0.0, 0.0, 0.0)))
    range = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/range",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(1.7357, -0.0145, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_drawer_0_2': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_drawer_0_2"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    range_hood = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/range_hood",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(1.7357, 0.0814, 1.2625), rot=(1.0, 0.0, 0.0, 0.0)))
    refrigerator = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/refrigerator",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'door_joint': 0.0, 'freezer_door_joint': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["door_joint", "freezer_door_joint"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    sink_cabinet = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/sink_cabinet",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(3.1349, -0.0145, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    sink_cabinet = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/sink_cabinet",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(3.1349, -0.0145, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    sink_cabinet = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/sink_cabinet",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(3.1349, -0.0145, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0}),
                        actuators={
                            "default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1"],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)
                            })
    wall_01 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_01",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.850, -2.200, 1.500), rot=(0.70711, 0.70711, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/floor.usd",
            scale=(5.100, 3.0, 0.1),
            visual_material=MdlFileCfg(mdl_path=wall_material),
        ),
        collision_group=-1,
    )
    wall_02 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_02",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.400, -0.850, 1.500), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/floor.usd",
            scale=(2.700, 3.0, 0.1),
            visual_material=MdlFileCfg(mdl_path=wall_material),
        ),
        collision_group=-1,
    )
    wall_03 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_03",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.700, -0.850, 1.500), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/floor.usd",
            scale=(2.700, 3.0, 0.1),
            visual_material=MdlFileCfg(mdl_path=wall_material),
        ),
        collision_group=-1,
    )
    wall_04 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_04",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.850, 0.500, 1.500), rot=(0.70711, 0.70711, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/floor.usd",
            scale=(5.100, 3.0, 0.1),
            visual_material=MdlFileCfg(mdl_path=wall_material),
        ),
        collision_group=-1,
    )
        # -------Stop-------
    # camera

    front = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=1/20,
        height=240,
        width=320,
        data_types=["rgb", "instance_id_segmentation_fast"],
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
    
    wrist_right = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link1/ee_r_camera",
        update_period=1/20,
        height=240,
        width=320,
        data_types=["rgb", "instance_id_segmentation_fast"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=25.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, -0.11, -0.13),
            rot=(0.2164396,0.976296, 0.0, 0.0),
            convention="opengl",
        ),
    )

    wrist_left = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link2/ee_l_camera",
        update_period=1/20,
        height=240,
        width=320,
        data_types=["rgb", "instance_id_segmentation_fast"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=25.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, -0.11, -0.13),
            rot=(0.2164396,0.976296, 0.0, 0.0),
            convention="opengl",
        ),
    )
 
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    armL_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link2.*", "arm2.*"],
            body_name="ee_link2",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                init_joint_pos=[ANUBIS_CFG.init_state.joint_pos[name] for name in left_arm_joint_names],
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0]),
        )


    armR_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            body_name="ee_link1",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                init_joint_pos=[ANUBIS_CFG.init_state.joint_pos[name] for name in right_arm_joint_names],
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0]),
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
        mobile_base_world_frame = ObsTerm(func=mdp.mobile_base) 

        ee_6D_pos = ObsTerm(func=mdp.ee_6d_pos)
        base_vel = ObsTerm(func=mdp.base_vel)
        
        joint_angles = ObsTerm(func=mdp.joint_angles)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    obj_init_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bowl0"),
            "pose_range": {
                "x" : (0.0, 0.0),
                "y" : (0.0, 0.0),
                "z" : (0.0, 0.0),
                "roll" : (0.0, 0.0),
                "pitch" : (0.0, 0.0),
                "yaw" : (0.0, 0.0),
            },
            "velocity_range": {
                "x" : (0.0, 0.0),
                "y" : (0.0, 0.0),
                "z" : (0.0, 0.0),
                "roll" : (0.0, 0.0),
                "pitch" : (0.0, 0.0),
                "yaw" : (0.0, 0.0),
            }
        },
    )
    obj_physics_material = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("bowl0"),
          "static_friction_range": (0.2, 0.6),
          "dynamic_friction_range": (0.1, 0.5),
          "restitution_range": (0.3, 0.7),
          "num_buckets": 250,
      },
    )
    obj_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bowl0"),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "abs",
            "distribution": "uniform",
        }
    )
    robot_init_pos= EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x" : (0.0, 0.0),
                "y" : (0.0, 0.0),
                "z" : (0.0, 0.0),
                "roll" : (0.0, 0.0),
                "pitch" : (0.0, 0.0),
                "yaw" : (0.0, 0.0),
            },
            "velocity_range": {
                "x" : (0.0, 0.0),
                "y" : (0.0, 0.0),
                "z" : (0.0, 0.0),
                "roll" : (0.0, 0.0),
                "pitch" : (0.0, 0.0),
                "yaw" : (0.0, 0.0),
            }
        },
    )

    robot_joint_init = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=['arm1_base_link_joint', 'arm2_base_link_joint', 'link11_joint', 'link21_joint', 'link12_joint', 'link22_joint', 'link13_joint', 'link23_joint', 'link14_joint', 'link24_joint', 'link15_joint', 'link25_joint', 'gripper1R_joint', 'gripper1_joint', 'gripper2R_joint', 'gripper2_joint']),
            "position_range":(-0.1, 0.1),
            "velocity_range":(-0.01, 0.01),
        },
    )

    robot_physics_material = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
          "static_friction_range": (0.7, 1.0),
          "dynamic_friction_range": (0.5, 1.0),
          "restitution_range": (0.3, 0.5),
          "num_buckets": 250,
      },
    )
   
    robot_link_material = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "colors":{"r":(0.0, 0.1), "g":(0.0, 0.1), "b": (0.0 ,0.1)},
            "asset_cfg": SceneEntityCfg("robot"),
            "mesh_name": "gripper_base_link/visuals/gripper_base_link/mesh",
            "event_name": "rep_link_randomize_color",
        }
    )
    robot_gripper_material = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "colors":{"r":(0.0, 0.1), "g":(0.0, 0.1), "b": (0.0 ,0.1)},
            "asset_cfg": SceneEntityCfg("robot"),
            "mesh_name": "gripper1L/visuals/fingerL/mesh",
            "event_name": "rep_gripper_randomize_color",
        }
    )
    dishwasher_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dishwasher", joint_names=(".*")),
            "stiffness_distribution_params": (0, 1),
            "damping_distribution_params": (1, 25),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    bowl_material = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "colors":{"r":(0.0, 1.0), "g":(0.0, 1.0), "b": (0.0 ,1.0)},
            "asset_cfg": SceneEntityCfg("bowl0"),
            "mesh_name": "simplified_obj/simplified_obj",
            "event_name": "rep_bowl_randomize_color",
        }
    )
    mug_material = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "colors":{"r":(0.0, 1.0), "g":(0.0, 1.0), "b": (0.0 ,1.0)},
            "asset_cfg": SceneEntityCfg("mug0"),
            "mesh_name": "simplified_obj/simplified_obj",
            "event_name": "rep_mug_randomize_color",
        }
    )
    bowl_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bowl0"),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z":(-0.02, 0.02)}
        }
    )
@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
#TODO How to define success
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    success = DoneTerm(
            func=mdp.task2,
            params={"obj_cfg": SceneEntityCfg("bowl0")}
    )
    retry = DoneTerm(
            func=mdp.OOB,
            params={"obj_cfg": SceneEntityCfg("bowl0")}
    )

@configclass
class AnubisKitchenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Kitchen environment."""
    # Scene setting
    scene: KitchenSceneCfg = KitchenSceneCfg(env_spacing=6 , replicate_physics=False)
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
        self.episode_length_s = 16.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 20  # 60Hz
        self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
#       self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
