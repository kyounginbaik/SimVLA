from __future__ import annotations

import torch
import math
import os as _os
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.simvla import variable
from isaaclab.utils.math import quat_rotate_inverse
import ipdb
if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv

home_r = torch.tensor([ 0.2257, -0.0988,  1.0351], device='cuda:0')
home_l = torch.tensor([ 0.2203,  0.1080,  1.0342], device='cuda:0')


import logging
from pathlib import Path

def make_logger(log_path: str, name: str = "sink_logger"):
	log_path = Path(log_path)
	log_path.parent.mkdir(parents=True, exist_ok=True)

	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	logger.propagate = False  # root logger로 중복 전파 방지

	# 이미 핸들러가 붙어있으면 중복 추가 방지
	if not logger.handlers:
		fh = logging.FileHandler(log_path, mode="w")  # append면 "a"
		fmt = logging.Formatter(
			fmt="%(asctime)s | %(levelname)s | %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		fh.setFormatter(fmt)
		logger.addHandler(fh)

	return logger


LOGGER = make_logger("/scratch/simvla/logs/sink_debug.txt")

def task4(
	env: ManagerBasedRLEnv,
) -> bool:
   
	result = (variable.env_goal_indices == (variable.max_sequence_length-1)) 
	return result

def task3_rby1(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bowl0"),
) -> bool:

    result = variable.env_goal_indices == 4
    #print(bottle_check)
    #print(mug_check)

    return result
def task3(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bowl0"),
) -> bool:

    # 1. close home
    robot = env.scene.articulations["robot"]
    eef_r_idx = robot.find_bodies("ee_link1")[0][0]
    eef_l_idx = robot.find_bodies("ee_link2")[0][0]
    base_idx  = robot.find_bodies("base_link")[0][0]
    eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
    eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
    base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
    base_quat_w = robot.data.body_quat_w[:, base_idx]		   
    eef_r_rel_w = eef_r_pos_w - base_pos_w
    eef_l_rel_w = eef_r_pos_w - base_pos_w
    eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
    eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
    distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
    radius = 0.02
    close_home = ( distance_r < radius)
    #print(distance_r)
    # 2. Both Mug and Bottle in hand
    bottle_pos = env.scene.rigid_objects["bottle0"].data.body_pos_w[:].clone().squeeze(1)
    mug_pos = env.scene.rigid_objects["mug0"].data.body_pos_w[:].clone().squeeze(1)

    bottle_check = (torch.linalg.norm(bottle_pos - eef_r_pos_w, dim=-1) < 0.2)
    mug_check = (torch.linalg.norm(mug_pos - eef_l_pos_w, dim=-1) < 0.2)
    #result = (variable.env_goal_indices == (variable.max_sequence_length-1)) 
    result = close_home & bottle_check & mug_check & (variable.env_goal_indices == (variable.max_sequence_length-1)) 
    #print(bottle_check)
    #print(mug_check)

    return result

def task3_molmospace(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bowl0"),
) -> bool:
	# 1. close home
	robot = env.scene.articulations["robot"]
	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]
	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   
	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_r_pos_w - base_pos_w
	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	radius = 0.05
	close_home = ( distance_r < radius)
	#print(distance_r)
	# 2. Both Mug and Bottle in hand
	bottle_pos = env.scene.rigid_objects["bottle0"].data.body_pos_w[:].clone().squeeze(1)
	mug_pos = env.scene.rigid_objects["mug0"].data.body_pos_w[:].clone().squeeze(1)

	bottle_check = (torch.linalg.norm(bottle_pos - eef_r_pos_w, dim=-1) < 0.2)
	mug_check = (torch.linalg.norm(mug_pos - eef_l_pos_w, dim=-1) < 0.2)
	result = close_home & bottle_check & mug_check 

	return result

def task2(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bowl0"),
	base_cabinet: SceneEntityCfg = SceneEntityCfg("base_cabinet")
) -> bool:
	# 1. Obj in drawer — use the configured manipulated object (obj_cfg), not a hardcoded 'bowl0',
	# so the success check works for whatever object the task manipulates (cup, jar, ...).
	obj_pos = env.scene[obj_cfg.name].data.body_pos_w   # (num_envs, 1, 3)
	obj_z = obj_pos[:, 0, 2]					  # (num_envs,)
	inside = (obj_z > 0.6) & (obj_z < 0.8) 

	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   

	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_l_pos_w - base_pos_w

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

	radius = 0.18
	close_home = ( distance_r < radius)
	
	# check if the drawer is closed
	drawer_closed = env.scene["base_cabinet"].data.joint_pos[:,0] < 0.03
#	 print(inside)
#	 print(close_home)
#	 print(drawer_closed)
	result = inside & close_home  & (variable.env_goal_indices == (variable.max_sequence_length-1)) & drawer_closed
	return result

def task1_molmospace(
	env: ManagerBasedRLEnv,
	sink_pos: torch.Tensor,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("mug0"),
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1) - env.scene.env_origins

	# 1. Obj close to sink
	radius = 0.2
	sinks_pos = sink_pos.unsqueeze(0).repeat(env.scene.num_envs, 1)
	distance = torch.linalg.norm(sinks_pos[:,:2] - obj_pos[:,:2], dim=-1)
	close_sink = distance <= radius
	
	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   

	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_l_pos_w - base_pos_w

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

	radius_home = 0.1
	close_home = ( distance_r < radius_home) & ( distance_l < radius_home)
	robot = env.scene.articulations["robot"]
	r_gripperR_idx = robot.find_bodies("gripper1R")[0][0]
	r_gripperL_idx = robot.find_bodies("gripper1L")[0][0]
	open = torch.linalg.norm(robot.data.body_pos_w[:,r_gripperR_idx] - robot.data.body_pos_w[:,r_gripperL_idx], dim=1) > 0.079

	# 3. Last subtask
  #  result = close_home & close_sink & close_sink1 & (variable.env_goal_indices == (variable.max_sequence_length-1))
	result = close_home & close_sink & open
	return result
def task1_deploy(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
) -> bool:
	obj: RigidObject = env.scene["mug0"]
	obj_pos = obj.data.body_pos_w.squeeze(1)
#	 bowl = env.scene["bowl0"]
#	 bowl_pos = bowl.data.body_pos_w.squeeze(1)
# if door x same? -> N or S
	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:].clone()
	if torch.allclose(env.scene[sink.name].data.body_pos_w[:,2,0] ,env.scene[sink.name].data.body_pos_w[:,1,0] , rtol=1e-5, atol=1e-6):
		door_x = env.scene[sink.name].data.body_pos_w[:, 1, 0].clone()
		sink_x = sink_pos[:, 0].clone()
		sink_pos[:, 0] = (sink_x + door_x) / 2

# if door y same? -> E or W
	elif torch.allclose(env.scene[sink.name].data.body_pos_w[:,2,1] ,env.scene[sink.name].data.body_pos_w[:,1,1] , rtol=1e-5, atol=1e-6):
		door_y = env.scene[sink.name].data.body_pos_w[:, 1, 1].clone()
		sink_y = sink_pos[:, 1].clone()
		sink_pos[:, 1] = (sink_y + door_y) / 2

	# 1. Obj close to sink
	sink_pos[:,2] = 0.88
	radius = 0.12
	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
#	 print(obj_pos)
#	 print(sink_pos)
	#print(distance)
#	 distance1 = torch.linalg.norm(bowl_pos - sink_pos, dim=-1)
	close_sink = distance <= radius
 #	 close_sink1 = distance1 <= radius
	
	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   

	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_l_pos_w - base_pos_w

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

	close_home = ( distance_r < radius) & ( distance_l < radius)

	# 3. Last subtask
  #  result = close_home & close_sink & close_sink1 & (variable.env_goal_indices == (variable.max_sequence_length-1))
  #  result = close_home & close_sink & (variable.env_goal_indices == (variable.max_sequence_length-1))
	result = close_home & close_sink
	return result

def in_pot(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("ramen0"),
	pot: SceneEntityCfg = SceneEntityCfg("pot0")
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
	pot_pos = env.scene[pot.name].data.body_pos_w[:,0,:].clone()
	radius = 0.12
	distance = torch.linalg.norm(obj_pos - pot_pos, dim=-1)
#	 print(obj_pos)
#	 print(sink_pos)
	print(distance)
	close_range = distance <= radius
	
	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   

	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_l_pos_w - base_pos_w

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

	close_home = ( distance_r < radius) & ( distance_l < radius)
	close_home =  distance_l < radius

	result = close_home & close_range & (variable.env_goal_indices == (variable.max_sequence_length-1))
	return result
def pot(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
	range: SceneEntityCfg = SceneEntityCfg("range")
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
	range_pos = env.scene[range.name].data.body_pos_w[:,-1,:].clone()
	range_pos[:,2] = 0.85
	radius = 0.12
	distance = torch.linalg.norm(obj_pos - range_pos, dim=-1)
#	 print(obj_pos)
#	 print(sink_pos)
	#print(distance)
	close_range = distance <= radius
	
	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   

	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_l_pos_w - base_pos_w

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

	close_home = ( distance_r < radius) & ( distance_l < radius)

	result = close_home & close_range & (variable.env_goal_indices == (variable.max_sequence_length-1))
	return result
def sink(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet"),
	radius: float = 0.12,
) -> bool:
	"""Success when ``obj_cfg`` is inside the sink and both EEFs are back home.

	This is the generic, per-object version referenced by the generated kitchen
	task configs (``func=mdp.sink``). ``task1`` below is the same check with the
	object hardcoded to ``mug0``; the 0.12 default matches the tolerance ``task1``
	converged on (the pre-cleanup version of this function used 0.065, which is
	tight enough that successes are rare — pass ``params={"radius": 0.065}`` to
	restore it).
	"""
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)

	# if door x same? -> N or S
	sink_pos = env.scene[sink.name].data.body_pos_w[:, 0, :].clone()
	if torch.allclose(env.scene[sink.name].data.body_pos_w[:, 2, 0], env.scene[sink.name].data.body_pos_w[:, 1, 0], rtol=1e-5, atol=1e-6):
		door_x = env.scene[sink.name].data.body_pos_w[:, 1, 0].clone()
		sink_x = sink_pos[:, 0].clone()
		sink_pos[:, 0] = (sink_x + door_x) / 2

	# if door y same? -> E or W
	elif torch.allclose(env.scene[sink.name].data.body_pos_w[:, 2, 1], env.scene[sink.name].data.body_pos_w[:, 1, 1], rtol=1e-5, atol=1e-6):
		door_y = env.scene[sink.name].data.body_pos_w[:, 1, 1].clone()
		sink_y = sink_pos[:, 1].clone()
		sink_pos[:, 1] = (sink_y + door_y) / 2

	# 1. Obj close to sink
	sink_pos[:, 2] = 0.88
	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
	close_sink = distance <= radius

	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w = robot.data.body_pos_w[:, base_idx]
	base_quat_w = robot.data.body_quat_w[:, base_idx]

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_pos_w - base_pos_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_pos_w - base_pos_w)

	distance_r = torch.linalg.norm(eef_r_pos_base - home_r.to(obj_pos.device), dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l.to(obj_pos.device), dim=-1)

	close_home = (distance_r < radius) & (distance_l < radius)

	# 3. Last subtask
	last_subtask = variable.env_goal_indices == (variable.max_sequence_length - 1)

	# Opt-in diagnostic: this check ANDs three conditions, so a run that never succeeds says
	# nothing about which one is missing. Set SIMVLA_DEBUG_SINK=1 to print them per env.
	if _os.environ.get("SIMVLA_DEBUG_SINK"):
		# Does the gripper ever actually get to the object? If the base never drives to the
		# counter, the arm traces a correct-looking trajectory in the wrong place.
		d_eef_obj = torch.linalg.norm(eef_r_pos_w - obj_pos, dim=-1)
		# Command vs reality: the recorded gripper channel proves what was ASKED for, not
		# whether the fingers actually closed on the mug before the arm retracted.
		_gj = robot.find_joints("gripper1.*")[0]
		_grip_q = robot.data.joint_pos[:, _gj]
		_grip_tgt = robot.data.joint_pos_target[:, _gj]
		for i in range(obj_pos.shape[0]):
			print(
				f"[sink] env={i} goal={int(variable.env_goal_indices[i])}/"
				f"{variable.max_sequence_length - 1} "
				f"d_obj={float(distance[i]):.4f} d_eef_r={float(distance_r[i]):.4f} "
				f"d_eef_l={float(distance_l[i]):.4f} "
				f"d_eefR_to_obj={float(d_eef_obj[i]):.4f} "
				f"base=({float(base_pos_w[i,0]):.3f},{float(base_pos_w[i,1]):.3f}) "
				f"gripR_q={[round(float(v),4) for v in _grip_q[i]]} "
				f"gripR_tgt={[round(float(v),4) for v in _grip_tgt[i]]} "
				f"obj=({float(obj_pos[i,0]):.3f},{float(obj_pos[i,1]):.3f},{float(obj_pos[i,2]):.3f}) "
				f"radius={radius} "
				f"| sink={bool(close_sink[i])} home={bool(close_home[i])} "
				f"last={bool(last_subtask[i])}",
				flush=True,
			)

	return close_sink & close_home & last_subtask


def task1(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
) -> bool:
	obj: RigidObject = env.scene["mug0"]
	obj_pos = obj.data.body_pos_w.squeeze(1)
#	 bowl = env.scene["bowl0"]
#	 bowl_pos = bowl.data.body_pos_w.squeeze(1)
# if door x same? -> N or S
	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:].clone()
	if torch.allclose(env.scene[sink.name].data.body_pos_w[:,2,0] ,env.scene[sink.name].data.body_pos_w[:,1,0] , rtol=1e-5, atol=1e-6):
		door_x = env.scene[sink.name].data.body_pos_w[:, 1, 0].clone()
		sink_x = sink_pos[:, 0].clone()
		sink_pos[:, 0] = (sink_x + door_x) / 2

# if door y same? -> E or W
	elif torch.allclose(env.scene[sink.name].data.body_pos_w[:,2,1] ,env.scene[sink.name].data.body_pos_w[:,1,1] , rtol=1e-5, atol=1e-6):
		door_y = env.scene[sink.name].data.body_pos_w[:, 1, 1].clone()
		sink_y = sink_pos[:, 1].clone()
		sink_pos[:, 1] = (sink_y + door_y) / 2

	# 1. Obj close to sink
	sink_pos[:,2] = 0.88
	radius = 0.12
	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
#	 print(obj_pos)
#	 print(sink_pos)
	#print(distance)
#	 distance1 = torch.linalg.norm(bowl_pos - sink_pos, dim=-1)
	close_sink = distance <= radius
 #	 close_sink1 = distance1 <= radius
	
	# 2. EEF close to home
	robot = env.scene.articulations["robot"]

	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx  = robot.find_bodies("base_link")[0][0]

	eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
	eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
	base_pos_w	= robot.data.body_pos_w[:, base_idx]		   
	base_quat_w = robot.data.body_quat_w[:, base_idx]		   

	eef_r_rel_w = eef_r_pos_w - base_pos_w
	eef_l_rel_w = eef_l_pos_w - base_pos_w

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
	distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
	distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

	close_home = ( distance_r < radius) & ( distance_l < radius)

	# 3. Last subtask
  #  result = close_home & close_sink & close_sink1 & (variable.env_goal_indices == (variable.max_sequence_length-1))
	result = close_home & close_sink & (variable.env_goal_indices == (variable.max_sequence_length-1))
 #	 result = close_home & close_sink
	return result

from isaaclab.utils.math import quat_from_euler_xyz

def navigation(env, goal_pos):
	# robot root pose
	base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
	robot_pos = env.scene.articulations["robot"].data.body_pos_w[:,base_link_idx, :]		 # (N, 3)
	robot_quat = env.scene.articulations["robot"].data.body_quat_w[:,base_link_idx, :]	 # (N, 4) wxyz

	radius = 0.02			 # [m]
	max_yaw_err = 0.1		 # [rad] 

	# --- position distance in the x-y plane ---
	# goal_pos is assumed to be (N, 3): [x, y, yaw]
	distance = torch.norm(goal_pos[:2] - robot_pos[:, :2], dim=1)	# (N,)

	# --- yaw distance with wrap-around ---
	r, p, yaw = euler_xyz_from_quat(robot_quat)  # each is (N,)
	goal_yaw = goal_pos[2]					 # (N,)

	# shortest signed angle difference in [-Ï, Ï]
	yaw_diff = torch.atan2(
		torch.sin(goal_yaw - yaw),
		torch.cos(goal_yaw - yaw),
	)											 # (N,)

	angle = torch.abs(yaw_diff)					 # (N,)

	# --- termination condition ---
	result = (distance <= radius) & (angle <= max_yaw_err)
	return result

def OOB_molmospace(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
#sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
	
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
	result = (obj_pos[:,2] >2 )

	true_indices = torch.nonzero(result, as_tuple=True)[0]
	# If  
	return result
def OOB(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
#sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
	
) -> bool:
# Depends on  N_dir 
#	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
#	sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
#	sink_pos[:,2] = 0.9

#	 radius = 3.5

#	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
#	print(distance)

	# Check if the distance is within the specified radius | nan | robot falling
#result = (distance > radius) | torch.isnan(distance) | (env.scene.articulations['robot'].data.root_pos_w[:,2] < -0.1 )
	#base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
	#print(obj_pos)
#	result = (env.scene.articulations['robot'].data.body_pos_w[:,base_link_idx, 2] < -0.1 )
	result = (obj_pos[:,2] < 0.3 )

	true_indices = torch.nonzero(result, as_tuple=True)[0]
	# If  
	return result


