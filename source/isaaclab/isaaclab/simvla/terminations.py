from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv

def sink(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("mug0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
# Depends on  N_dir 
	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
	sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
	sink_pos[:,2] = 0.9
	radius = 0.14
   
	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)

	result = distance <= radius
	return result

from isaaclab.utils.math import euler_xyz_from_quat

def navigation(env, goal_pos):
	# robot root pose
	robot_pos = (env.scene.articulations["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]) 		 # (N, 3)
	robot_quat = env.scene.articulations["robot"].data.root_link_quat_w[:, :]	 # (N, 4) wxyz

	radius = 0.02			 # [m]
	max_yaw_err = 0.1		 # [rad] (~5.7 deg) â you called it "degree" but it's radians

	# --- position distance in the x-y plane ---
	# goal_pos is assumed to be (N, 3): [x, y, yaw]
	distance = torch.norm(goal_pos[:2] - robot_pos[:, :2], dim=1)	# (N,)

	# --- yaw distance with wrap-around ---
	r, p, yaw = euler_xyz_from_quat(robot_quat)  # each is (N,)
	goal_yaw = goal_pos[2]					  # (N,)

	# shortest signed angle difference in [-Ï, Ï]
	yaw_diff = torch.atan2(
		torch.sin(goal_yaw - yaw),
		torch.cos(goal_yaw - yaw),
	)											 # (N,)

	angle = torch.abs(yaw_diff)					 # (N,)

	# --- termination condition ---
	result = (distance <= radius) & (angle <= max_yaw_err)
	return result


def OOB(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("mug0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
	
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
# Depends on  N_dir 
	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
	sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
	sink_pos[:,2] = 0.9
	radius = 3.5

	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
	# Check if the distance is within the specified radius | nan | robot falling
	result = (distance > radius) | torch.isnan(distance) | (env.scene.articulations['robot'].data.root_pos_w[:,2] < -0.1 )
	true_indices = torch.nonzero(result, as_tuple=True)[0]

	# If  
	return result


