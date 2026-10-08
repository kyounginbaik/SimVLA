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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
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
from isaaclab_assets.robots.rby1 import RBY1_CFG  # isort:skip
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR



FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
right_arm_joint_names = [
	"right_arm_0",
	"right_arm_1",
	"right_arm_2",
	"right_arm_3",
	"right_arm_4",
	"right_arm_5",
	"right_arm_6",
]

left_arm_joint_names = [
	"left_arm_0",
	"left_arm_1",
	"left_arm_2",
	"left_arm_3",
	"left_arm_4",
	"left_arm_5",
	"left_arm_6",
]
# floor material
floor_list = ["{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks.mdl", "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch_Planks.mdl", "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak_Planks.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Parquet_Floor.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks.mdl" ]
# wall material
wall_list = ["{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Adobe_Brick.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Pavers.mdl","{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Block.mdl"]
#
# EVERYTHING BETWEEN THE MARKERS IS REPLACED AT EMISSION by env_cfg_emit.substitute_materials,
# which draws from its OWN random.Random(kitchen_num) -- a separate instance from the one
# lighting uses below, never the same stream -- and writes CONCRETE STRING LITERALS into the
# task file. The calls below are the template's own defaults; they must never survive into an
# emitted file. This module is copied as TEXT and never evaluated, so a sampling call left in a
# task file is evaluated at module IMPORT in the training/eval process, off the unseeded global
# `random`: one draw shared by every parallel env, a different draw on every process launch, and
# a replay finished unlike the episode it replays.
# -------Materials-------
# Baked in at emission, seeded from kitchen 1407
# -- see env_cfg_emit.materials_values. Literals, never a sampling call: this
# module is imported once per training/eval process, so a draw made here would
# share one floor/wall across every parallel env, repaint the scene on every
# process launch, and paint a replay differently from the episode it replays.
floor_material = '{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut.mdl'
wall_material = '{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Block.mdl'
# -------StopMaterials-------
# lighting
# Sampled per emitted task file, alongside floor_material and wall_material above, because
# IsaacLab has no light-randomization event term: mdp.events offers randomize_visual_texture_material
# and randomize_visual_color and nothing for lights. Every task file in the corpus before this
# carried byte-identical lighting.
#
# The ranges are deliberately narrow. This is lighting variation, not a coloured-light
# augmentation: the head camera is a 200 degree fisheye and a scene it cannot expose is a wasted
# episode. Sphere 1500-5000 is half to ~1.7x the old 3000; distant 500-2000 brackets the old 1000;
# colour stays a near-white tint around the old 0.75.
#
# EVERYTHING BETWEEN THE MARKERS IS REPLACED AT EMISSION by env_cfg_emit.substitute_lighting,
# which draws the same four ranges from its OWN random.Random(kitchen_num) -- a separate
# instance from the one the Materials block above uses, never the same stream -- and writes
# CONCRETE LITERALS into the task file. The calls below are the template's own defaults and the
# single place the ranges are written down; they must never survive into an emitted file. This
# module is copied as TEXT and never evaluated, so a sampling call left in a task file is
# evaluated at module IMPORT in the training/eval process, off the unseeded global `random`: one
# draw shared by every parallel env, a different draw on every process launch, and a replay lit
# unlike the episode it replays. Same defect class as the Materials block above, fixed the same
# way.
# -------Lighting-------
# Baked in at emission, seeded from kitchen 1407
# -- see env_cfg_emit.lighting_values. Literals, never a sampling call: this
# module is imported once per training/eval process, so a draw made here would
# re-light the scene on every launch, share one value across every parallel env,
# and light a replay differently from the episode it replays. The sub-variant
# number is not part of the seed, so all 12 rotations of this kitchen match.
sphere_light_intensity = 1812.415
distant_light_intensity = 1198.7863
sphere_light_color = (0.8143, 0.7642, 0.7267)
distant_light_color = (0.7578, 0.6675, 0.71)
# -------StopLighting-------

##
# Scene definition



@configclass
class KitchenSceneCfg(InteractiveSceneCfg):
	# robots, Will be populated by agent env cfg
	robot: ArticulationCfg = RBY1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
	robot.spawn.activate_contact_sensors = True
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
						pos=(0.0, 0.0, -0.1034),
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
						pos=(0.0, 0.0, -0.1034),
					),
				),
			],
		)

	# WHO MOVED THE DOOR. Filtered contacts require prim_path to resolve to exactly ONE primitive
	# per env (ContactSensorCfg's "attention" note), so this is three single-body sensors rather
	# than one regex over the robot. force_matrix_w is then (N, 1, 1, 3): this body against the
	# door, per env.
	touch_base: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/base_link",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/refrigerator/door"],
		update_period=0.0, history_length=0, debug_vis=False,
	)
	# ee_finger_l1/l2 are the LEFT hand, ee_finger_r1/r2 the RIGHT
	# (composed._FINGER_CANDIDATES).
	touch_grip_l: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_finger_l1",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/refrigerator/door"],
		update_period=0.0, history_length=0, debug_vis=False,
	)
	touch_grip_r: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_finger_r1",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/refrigerator/door"],
		update_period=0.0, history_length=0, debug_vis=False,
	)

	# Per-pad mug contact for pre-lift and retention checks.
	touch_mug_r_pad1: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_finger_r1",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
		update_period=0.0, history_length=0, debug_vis=False,
	)
	touch_mug_r_pad2: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_finger_r2",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
		update_period=0.0, history_length=0, debug_vis=False,
	)
	touch_mug_l_pad1: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_finger_l1",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
		update_period=0.0, history_length=0, debug_vis=False,
	)
	touch_mug_l_pad2: ContactSensorCfg = ContactSensorCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_finger_l2",
		filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"],
		update_period=0.0, history_length=0, debug_vis=False,
	)

	# light
	distantlight = AssetBaseCfg(
		prim_path="/World/distantlight",
		spawn=sim_utils.DistantLightCfg(color=distant_light_color, intensity=distant_light_intensity, angle = 0.53),
	)

	light = AssetBaseCfg(
			prim_path="{ENV_REGEX_NS}/light",
			spawn=sim_utils.SphereLightCfg(color=sphere_light_color, intensity=sphere_light_intensity),
			init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
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
		spawn=sim_utils.UsdFileCfg(usd_path=f"file:{ISAACLAB_ASSETS_DATA_DIR}/Kitchen/kitchen_1407_00.usd"),
	)
	base_cabinet = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(2.2323, -0.9684, -0.0), rot=(-0.7071, 0.0, 0.0, 0.7071), joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0, 'corpus_to_door_2_0': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0", "corpus_to_door_2_0"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	bottle0 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/bottle0",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(1.9535, -1.3684, 0.8572), rot=(1.0, 0.0, 0.0, 0.0)))
	bowl0 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/bowl0",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(2.4315, -1.3684, 0.8572), rot=(1.0, 0.0, 0.0, 0.0)))
	chair_0 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/chair_0",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0776, -4.2013, 0.0), rot=(0.9635, 0.0, 0.0, 0.2675)))
	chair_1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/chair_1",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(1.1039, -4.41, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071)))
	chair_2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/chair_2",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(2.1301, -4.2013, 0.0), rot=(0.2675, 0.0, 0.0, 0.9635)))
	corner = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/corner",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(2.1973, -0.035, 0.4038), rot=(1.0, 0.0, 0.0, 0.0)))
	countertop_base_cabinet = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/countertop_base_cabinet",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(2.2323, -0.9684, 0.8314), rot=(-0.7071, 0.0, 0.0, 0.7071)))
	countertop_corner = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/countertop_corner",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(2.1973, -0.035, 0.8314), rot=(1.0, 0.0, 0.0, 0.0)))
	countertop_dishwasher = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/countertop_dishwasher",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7467, 0.0, 0.8314), rot=(1.0, 0.0, 0.0, 0.0)))
	dishwasher = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/dishwasher",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(0.7467, 0.0, -0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpse_to_bottom_basket': 0.0, 'corpse_to_top_basket': 0.0, 'corpus_to_door_0_1': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["corpse_to_bottom_basket", "corpse_to_top_basket", "corpus_to_door_0_1"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	kitchen_island = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/kitchen_island",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(0.077, -1.6853, -0.0), rot=(0.7071, 0.0, 0.0, 0.7071), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0, 'corpus_to_drawer_0_0': 0.0, 'corpus_to_drawer_1_0': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1", "corpus_to_drawer_0_0", "corpus_to_drawer_1_0"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	microwave = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/microwave",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(2.4358, -0.9684, 0.8552), rot=(-0.7071, 0.0, 0.0, 0.7071), joint_pos={'door_joint': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["door_joint"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	mug0 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/mug0",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.033, -2.104, 0.8572), rot=(1.0, 0.0, 0.0, 0.0)))
	range = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/range",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(1.4172, 0.0, -0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_drawer_0_2': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_drawer_0_2"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	range_hood = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/range_hood",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4172, 0.1166, 1.2847), rot=(1.0, 0.0, 0.0, 0.0)))
	refrigerator = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/refrigerator",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(2.2721, -1.8261, -0.0), rot=(-0.7071, 0.0, 0.0, 0.7071), joint_pos={'door_joint': 0.0, 'freezer_door_joint': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["door_joint", "freezer_door_joint"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	sink_cabinet = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Kitchen/sink_cabinet",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0}),
						actuators={
							"default": ImplicitActuatorCfg(joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1"],effort_limit=87.0,velocity_limit=100.0,stiffness=300.0,damping=30.0)
							})
	table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Kitchen/table",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(1.1039, -3.5838, 0.4078), rot=(1.0, 0.0, 0.0, 0.0)))
		# -------Stop-------
	# camera

	cctv = TiledCameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/base_link/cctv_cam",
		update_period=1/20,
		height=240,
		width=320,
		data_types=["rgb"],
		spawn = sim_utils.PinholeCameraCfg(
			focal_length=14.0,
			focus_distance=400.0,
			horizontal_aperture=20.955,
			clipping_range=(0.05, 1.0e5),
		),
		offset=TiledCameraCfg.OffsetCfg(
			pos=(-1.2, 0.0, 2.9),
			rot=(0.9184, 0.0, 0.3957, 0.0),
			convention="world"
		),
	)

	front = TiledCameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/link_head_2/head_cam",
		update_period=1/20,
		height=240,
		width=320,
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
			pos=(0.18, 0.0, 0.05),
			rot=(0.9063078, 0, 0.4226183, 0),
			convention="world"
		),
	)

	# A WORLD-FIXED CAMERA, so a human can judge which way the robot went.
	#
	# `front` is bolted to Robot/base_link and is a 200-degree fisheye, so the robot is the one
	# thing in shot that never moves: when the base steps right, the whole kitchen sweeps left
	# across the frame. Two sessions were spent arguing about the retreat direction from that
	# footage, with the log and the geometry both saying "back and right" while the video read as
	# "left". Nothing about the motion was wrong; the camera was the wrong camera to ask.
	#
	# Parented to the ENV rather than the robot, and a plain pinhole rather than a fisheye, so
	# straight lines stay straight and the base's path across the floor is what it looks like.
	# Placed high in a corner INSIDE the room, looking down at the work area. Both parts of that
	# were learned the hard way, each costing a full run of flat brown frames:
	#
	#   1. a hand-written quaternion pointed it at nothing; the pose is now a computed look-at,
	#      eye -> (0.25,-0.55,0.95), the floor in front of the fridge.
	#   2. the eye was then placed at (-2.2,-3.2), which is OUTSIDE the room -- kitchen 1400's
	#      shell runs x in [-1.60, 2.26], y in [-3.12, 0.41] with a 2.65 m ceiling, so the camera
	#      sat behind two walls and filmed the back of one. Any new pose must be checked against
	#      those bounds, not eyeballed.
	#
	# convention="world" means the camera looks along its own +X with +Z up, so the rotation is
	# built from columns [forward, left, up]. Getting that convention wrong is the other way to
	# end up filming a wall.
	scene_cam = TiledCameraCfg(
		prim_path="{ENV_REGEX_NS}/scene_cam",
		update_period=1/20,
		height=480,
		width=640,
		data_types=["rgb"],
		spawn=sim_utils.PinholeCameraCfg(
			focal_length=14.0,
			clipping_range=(0.05, 1000000),
		),
		offset=TiledCameraCfg.OffsetCfg(
			# EVERYTHING BETWEEN THE MARKERS IS REPLACED AT EMISSION by
			# env_cfg_emit.substitute_scene_cam. The literals below are kitchen 1400's pose and
			# the corpus default; a kitchen listed in env_cfg_emit.SCENE_CAM_AIMS gets a look-at
			# computed for ITS room instead. They are not a fallback -- substitute_scene_cam
			# raises if these markers are gone, because a silent no-op re-emits 1400's camera
			# into a kitchen that overrode it and the only symptom is a video of the wrong half
			# of the room, found after a GPU run instead of before one.
			# -------SceneCam-------
			# scene_cam's default pose -- kitchen 1400's, and the corpus's. This kitchen has no
			# entry in env_cfg_emit.SCENE_CAM_AIMS, so the shipped literals are emitted
			# unchanged. If this kitchen's task is not on the counter run, it is not in shot:
			# check with check_scene_cam.py --num <n> --kitchen <usd> before filming it.
			pos=(-1.25, -2.55, 2.25),
			rot=(0.8688422, -0.1061994, 0.2123989, 0.4344211),
			# -------StopSceneCam-------
			convention="world"
		),
	)

	wrist_right = TiledCameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_link1/ee_r_camera",
		update_period=1/20,
		height=240,
		width=320,
		data_types=["rgb"],
		spawn = sim_utils.PinholeCameraCfg(
			focal_length=25.0,
			focus_distance=400.0,
			horizontal_aperture=20.955,
			clipping_range=(0.1, 1.0e5),
		),
		offset = CameraCfg.OffsetCfg(
			pos=(0.0, 0.1, 0.16),
			rot=(0.984807753012208, -0.17364817766693033, 0.0, 0.0),
			convention="opengl",
		),
	)

	wrist_left = TiledCameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/ee_link2/ee_l_camera",
		update_period=1/20,
		height=240,
		width=320,
		data_types=["rgb"],
		spawn = sim_utils.PinholeCameraCfg(
			focal_length=25.0,
			focus_distance=400.0,
			horizontal_aperture=20.955,
			clipping_range=(0.1, 1.0e5),
		),
		offset = CameraCfg.OffsetCfg(
			pos=(0.0, -0.1, 0.16),
			rot=(0.0, 0.0, -0.17364817766693033, 0.984807753012208),
			convention="opengl",
		),
	)

@configclass
class ActionsCfg:
	"""Action specifications for the MDP."""
	armL_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
			asset_name="robot",
			joint_names=["left_arm_.*"],
			body_name="ee_link2",
			controller=DifferentialIKControllerCfg(
				command_type="pose",
				use_relative_mode=True,
				ik_method="dls",
				delta_joint_deadzone=0.0,
				init_joint_pos=[RBY1_CFG.init_state.joint_pos[name] for name in left_arm_joint_names],
			),
			scale=1.0,
			body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0]),
		)


	armR_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
			asset_name="robot",
			joint_names=["right_arm_.*"],
			body_name="ee_link1",
			controller=DifferentialIKControllerCfg(
				command_type="pose",
				use_relative_mode=True,
				ik_method="dls",
				delta_joint_deadzone=0.0,
				init_joint_pos=[RBY1_CFG.init_state.joint_pos[name] for name in right_arm_joint_names],
			),
			scale=1.0,
			body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0]),
		)

	gripperL_action: mdp.ContactHoldingBinaryJointPositionActionCfg = mdp.ContactHoldingBinaryJointPositionActionCfg(
			asset_name="robot",
			joint_names=["gripper_finger_l.*"],
			open_command_expr={"gripper_finger_l1": -0.04, "gripper_finger_l2": 0.04},
			close_command_expr={"gripper_finger_l1": 0.0, "gripper_finger_l2": 0.0},
			contact_sensor_names=("touch_mug_l_pad1", "touch_mug_l_pad2"),
			pad_joint_names=(("gripper_finger_l1",), ("gripper_finger_l2",)),
	)
	gripperR_action: mdp.ContactHoldingBinaryJointPositionActionCfg = mdp.ContactHoldingBinaryJointPositionActionCfg(
			asset_name="robot",
			joint_names=["gripper_finger_r.*"],
			open_command_expr={"gripper_finger_r1": -0.04, "gripper_finger_r2": 0.04},
			close_command_expr={"gripper_finger_r1": 0.0, "gripper_finger_r2": 0.0},
			contact_sensor_names=("touch_mug_r_pad1", "touch_mug_r_pad2"),
			pad_joint_names=(("gripper_finger_r1",), ("gripper_finger_r2",)),
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

	light_init_pos = EventTerm(
		func=mdp.reset_root_state_uniform,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("light"),
			"pose_range": {
				"x" : (-1, 1),
				"y" : (-1, 1),
				"z" : (-0.3, 0.0),
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
	obj_init_pos = EventTerm(
		func=mdp.reset_root_state_uniform,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("bottle0"),
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
	obj_physics_material_ = EventTerm(
	  func=mdp.randomize_rigid_body_material,
	  mode="startup",
	  params={
		  "asset_cfg": SceneEntityCfg("bottle0"),
		  "static_friction_range": (0.8, 1.1),
		  "dynamic_friction_range": (0.7, 1.0),
		  "restitution_range": (0.0, 0.1),
		  "num_buckets": 250,
	  },
	)
	obj_mass_1 = EventTerm(
		func=mdp.randomize_rigid_body_mass,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("bottle0"),
			"mass_distribution_params": (0.35, 0.6),
			"operation": "abs",
			"distribution": "uniform",
		}
	)
	obj_physics_material = EventTerm(
	  func=mdp.randomize_rigid_body_material,
	  mode="startup",
	  params={
		  "asset_cfg": SceneEntityCfg("bottle0"),
		  "static_friction_range": (0.8, 1.1),
		  "dynamic_friction_range": (0.7, 1.0),
		  "restitution_range": (0.0, 0.1),
		  "num_buckets": 250,
	  },
	)
	obj_mass = EventTerm(
		func=mdp.randomize_rigid_body_mass,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("bottle0"),
			"mass_distribution_params": (0.35, 0.6),
			"operation": "abs",
			"distribution": "uniform",
		}
	)
	obj_com_low = EventTerm(
		func=mdp.randomize_rigid_body_com,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("bottle0"),
			# A CONTAINER WITH ANYTHING IN IT HAS A LOW CENTRE OF MASS.
			#
			# Measured on bottle0, which is the object this mattered for: its collision geometry
			# puts the CoM at about 0.083 m -- the mean height of a uniform 0.182 m shell -- while
			# its flat contact patch is only 0.0163 m in radius, because the mesh's base is
			# chamfered. That is a static tipping angle of atan(0.0163/0.083) = 11 degrees, and 299
			# closes in the AI Worker campaign went from upright at plan to flat at close.
			#
			# Dropping the CoM by 0.06 puts it at about 0.023 and the tipping angle at 35
			# degrees (was 0.04 / 20 degrees, which took toppling from 92% to 79% of closes). It is
			# the physically honest direction rather than a cheat: an empty shell is the unusual
			# case, and anything part-filled carries its mass in the bottom third. Mugs too.
			"com_range": {"z": (-0.055, -0.045)},
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
			"asset_cfg": SceneEntityCfg(name="robot", joint_names=['right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6', 'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6']),
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
		  "restitution_range": (0.0, 0.1),
		  "num_buckets": 250,
	  },
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
	obj_material_1 = EventTerm(
		func=mdp.randomize_visual_color,
		mode="startup",
		params={
			"colors":{"r":(0.0, 1.0), "g":(0.0, 1.0), "b": (0.0 ,1.0)},
			"asset_cfg": SceneEntityCfg("bottle0"),
			"mesh_name": "simplified_obj/simplified_obj",
			"event_name": "rep_mug_randomize_color_1",
		}
	)
	obj_com_1 = EventTerm(
		func=mdp.randomize_rigid_body_com,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("bottle0"),
			"com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z":(-0.005, 0.005)}
		}
	)
	obj_material = EventTerm(
		func=mdp.randomize_visual_color,
		mode="startup",
		params={
			"colors":{"r":(0.0, 1.0), "g":(0.0, 1.0), "b": (0.0 ,1.0)},
			"asset_cfg": SceneEntityCfg("bottle0"),
			"mesh_name": "simplified_obj/simplified_obj",
			"event_name": "rep_bowl_randomize_color",
		}
	)
	obj_com = EventTerm(
		func=mdp.randomize_rigid_body_com,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("bottle0"),
			"com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z":(-0.005, 0.005)}
		}
	)
@configclass
class RewardsCfg:
	action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
	joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
@configclass
class TerminationsCfg:
	"""Termination terms for the MDP."""

	# ----Terminations----
	success = DoneTerm(
			func=mdp.composed,
			params={"spec": {"all": [{"obj_near_prim": {"role": "mug0", "target_role": "table", "radius": 0.52, "xy_only": True}}, {"obj_z": {"role": "mug0", "lo": 0.82, "hi": 0.97}}, {"gripper_open": {"arm": "right", "gap": 0.079}}, {"not": {"obj_near_eef": {"role": "mug0", "arm": "right", "radius": 0.15}}}, {"eef_home": {"arm": "right", "radius": 0.12}}, {"last_subtask": {}}]}}
	)
	retry = DoneTerm(
			func=mdp.composed,
			params={"spec": {"any": [{"obj_z": {"role": "mug0", "hi": 0.3}}, {"robot_fell": {"z": -0.1}}]}}
	)
		# ----StopTerminations----

@configclass
class RBY1KitchenEnvCfg(ManagerBasedRLEnvCfg):
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
		self.decimation = 6
		self.episode_length_s = 16.0
		self.viewer.eye = (-2.0, 2.0, 2.0)
		self.viewer.lookat = (0.8, 0.0, 0.5)
		# simulation settings
		self.sim.dt = 1 / 120  # 120 Hz physics; decimation 6 keeps control+render at 20 Hz
		self.sim.render_interval = self.decimation
		# self.sim.physx.bounce_threshold_velocity = 0.2
		self.sim.physx.bounce_threshold_velocity = 0.01
		self.sim.physx.friction_correlation_distance = 0.00625
#		self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
