from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.simvla import variable
from isaaclab.utils.math import quat_rotate_inverse
import logging
from pathlib import Path
import ipdb

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

home_r = torch.tensor([ 0.2257, -0.0988,  1.0351], device='cuda:0')
home_l = torch.tensor([ 0.2203,  0.1080,  1.0342], device='cuda:0')


def make_logger(log_path: str, name: str = "sink_logger"):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w")
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",

        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


LOGGER = make_logger("/scratch/simvla/logs/sink_debug.txt")


# SceneSmith

def mug2sink(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> bool:
    # If sink pos and obj_cfg is close 
    radius_sink = 0.83
    radius = 0.05

    # sink
    sink_pos = env.scene.extras["sink"].get_world_poses()[0].clone()
    # mug 
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)
    #print(obj_pos)

    distance = torch.norm(sink_pos[:, :2] - obj_pos[:, :2], dim=1)   # (N,)
    close_sink = ( distance < radius_sink)

    robot = env.scene.articulations["robot"]
    eef_r_idx = robot.find_bodies("ee_link1")[0][0]
    eef_l_idx = robot.find_bodies("ee_link2")[0][0]
    base_idx  = robot.find_bodies("base_link")[0][0]
    eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
    eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
    base_pos_w  = robot.data.body_pos_w[:, base_idx] 
    base_quat_w = robot.data.body_quat_w[:, base_idx] 
    eef_r_rel_w = eef_r_pos_w - base_pos_w
    eef_l_rel_w = eef_r_pos_w - base_pos_w
    eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
    eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
    distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
    close_home = ( distance_r < radius)

    result = close_home  & close_sink & (variable.env_goal_indices == (variable.max_sequence_length-1)) 
    return result

def pushchair(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> bool:
    # If desk pos and obj_cfg is close 
    radius_chair = 0.75
    radius = 0.05

    # desk
    desk_pos = env.scene.extras["desk"].get_world_poses()[0].clone()
    # chair 
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)

    distance = torch.norm(desk_pos[:, :3] - obj_pos[:, :3], dim=1)   # (N,)
    close_chair = ( distance < radius_chair)

    robot = env.scene.articulations["robot"]
    eef_r_idx = robot.find_bodies("ee_link1")[0][0]
    eef_l_idx = robot.find_bodies("ee_link2")[0][0]
    base_idx  = robot.find_bodies("base_link")[0][0]
    eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
    eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
    base_pos_w  = robot.data.body_pos_w[:, base_idx] 
    base_quat_w = robot.data.body_quat_w[:, base_idx] 
    eef_r_rel_w = eef_r_pos_w - base_pos_w
    eef_l_rel_w = eef_r_pos_w - base_pos_w
    eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
    eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
    distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
    close_home = ( distance_r < radius)

    result = close_home  & close_chair & (variable.env_goal_indices == (variable.max_sequence_length-1)) 
    return result

def pushchair_2(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> bool:
    # If desk pos and obj_cfg is close 
    radius_chair = 0.52
    radius = 0.02

    # desk
    desk_pos = env.scene.extras["desk"].get_world_poses()
    desk_pos = env.scene.extras["desk"].get_world_poses()[0].clone()
    # chair 
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)

    distance = torch.norm(desk_pos[:, :3] - obj_pos[:, :3], dim=1)   # (N,)
    close_chair = ( distance < radius_chair)

    obj_1: RigidObject = env.scene[obj_cfg.name]
    obj_pos_1 = obj_1.data.body_pos_w.squeeze(1)

    distance_1 = torch.norm(desk_pos[:, :3] - obj_pos_1[:, :3], dim=1)   # (N,)
    close_chair_1 = ( distance_1 < radius_chair)

    robot = env.scene.articulations["robot"]
    eef_r_idx = robot.find_bodies("ee_link1")[0][0]
    eef_l_idx = robot.find_bodies("ee_link2")[0][0]
    base_idx  = robot.find_bodies("base_link")[0][0]
    eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
    eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
    base_pos_w  = robot.data.body_pos_w[:, base_idx] 
    base_quat_w = robot.data.body_quat_w[:, base_idx] 
    eef_r_rel_w = eef_r_pos_w - base_pos_w
    eef_l_rel_w = eef_r_pos_w - base_pos_w
    eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
    eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
    distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
    close_home = ( distance_r < radius)

    result = close_home & close_chair_1 & close_chair & (variable.env_goal_indices == (variable.max_sequence_length-1)) 
    return result
# Kitchen scene
def task4(
    env: ManagerBasedRLEnv,
) -> bool:
   
    result = (variable.env_goal_indices == (variable.max_sequence_length-1)) 
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
    base_pos_w  = robot.data.body_pos_w[:, base_idx]           
    base_quat_w = robot.data.body_quat_w[:, base_idx]          
    eef_r_rel_w = eef_r_pos_w - base_pos_w
    eef_l_rel_w = eef_r_pos_w - base_pos_w
    eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
    eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
    distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
    radius = 0.02
    close_home = ( distance_r < radius)
   
    # 2. Both Mug and Bottle in hand
    bottle_pos = env.scene.rigid_objects["bottle0"].data.body_pos_w[:].clone().squeeze(1)
    mug_pos = env.scene.rigid_objects["mug0"].data.body_pos_w[:].clone().squeeze(1)

    bottle_check = (torch.linalg.norm(bottle_pos - eef_r_pos_w, dim=-1) < 0.15)
    mug_check = (torch.linalg.norm(mug_pos - eef_l_pos_w, dim=-1) < 0.15)
    result = close_home & bottle_check & mug_check & (variable.env_goal_indices == (variable.max_sequence_length-1)) 
    print(close_home)
    print(bottle_check)
    print(mug_check)

    return result

def task2(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bowl0"),
    base_cabinet: SceneEntityCfg = SceneEntityCfg("base_cabinet")
) -> bool:
    # 1. Obj in drawer
    obj_pos = env.scene['bowl0'].data.body_pos_w   # (num_envs, 1, 3)
    obj_z = obj_pos[:, 0, 2]                      # (num_envs,)
    inside = (obj_z > 0.6) & (obj_z < 0.8) 

    # 2. EEF close to home
    robot = env.scene.articulations["robot"]

    eef_r_idx = robot.find_bodies("ee_link1")[0][0]
    eef_l_idx = robot.find_bodies("ee_link2")[0][0]
    base_idx  = robot.find_bodies("base_link")[0][0]

    eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
    eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
    base_pos_w  = robot.data.body_pos_w[:, base_idx]           
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
    result = inside & close_home  & (variable.env_goal_indices == (variable.max_sequence_length-1)) & drawer_closed
    return result

def task1(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
    sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
) -> bool:
#    bowl = env.scene["bowl0"]
#    bowl_pos = bowl.data.body_pos_w.squeeze(1)
# if door x same? -> N or S
    radius = 0.05
    sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
    sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
    sink_pos[:,2] = 0.85
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)
    #print(obj_pos)

    distance = torch.norm(sink_pos[:, :2] - obj_pos[:, :2], dim=1)   # (N,)
    close_sink = ( distance < radius)
    
    # 2. EEF close to home
    robot = env.scene.articulations["robot"]

    eef_r_idx = robot.find_bodies("ee_link1")[0][0]
    eef_l_idx = robot.find_bodies("ee_link2")[0][0]
    base_idx  = robot.find_bodies("base_link")[0][0]

    eef_r_pos_w = robot.data.body_pos_w[:, eef_r_idx]
    eef_l_pos_w = robot.data.body_pos_w[:, eef_l_idx]
    base_pos_w  = robot.data.body_pos_w[:, base_idx]           
    base_quat_w = robot.data.body_quat_w[:, base_idx]          

    eef_r_rel_w = eef_r_pos_w - base_pos_w
    eef_l_rel_w = eef_l_pos_w - base_pos_w

    eef_r_pos_base = quat_rotate_inverse(base_quat_w, eef_r_rel_w)
    eef_l_pos_base = quat_rotate_inverse(base_quat_w, eef_l_rel_w)
    distance_r = torch.linalg.norm(eef_r_pos_base - home_r, dim=-1)
    distance_l = torch.linalg.norm(eef_l_pos_base - home_l, dim=-1)

    close_home = ( distance_r > radius) & ( distance_l > radius)

    # 3. Last subtask
  #  result = close_home & close_sink & close_sink1 & (variable.env_goal_indices == (variable.max_sequence_length-1))
    result = close_home & close_sink & (variable.env_goal_indices == (variable.max_sequence_length-1))
   # result = close_home 
    return result

from isaaclab.utils.math import quat_from_euler_xyz

def navigation(env, goal_pos):
    # robot root pose
    base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
    robot_pos = env.scene.articulations["robot"].data.body_pos_w[:,base_link_idx, :]         # (N, 3)
    robot_quat = env.scene.articulations["robot"].data.body_quat_w[:,base_link_idx, :]   # (N, 4) wxyz

    radius = 0.02            # [m]
    max_yaw_err = 0.1        # [rad] 

    # --- position distance in the x-y plane ---
    # goal_pos is assumed to be (N, 3): [x, y, yaw]
    distance = torch.norm(goal_pos[:2] - robot_pos[:, :2], dim=1)   # (N,)

    # --- yaw distance with wrap-around ---
    r, p, yaw = euler_xyz_from_quat(robot_quat)  # each is (N,)
    goal_yaw = goal_pos[2]                   # (N,)

    # shortest signed angle difference in [-Ï, Ï]
    yaw_diff = torch.atan2(
        torch.sin(goal_yaw - yaw),
        torch.cos(goal_yaw - yaw),
    )                                            # (N,)

    angle = torch.abs(yaw_diff)                  # (N,)

    # --- termination condition ---
    result = (distance <= radius) & (angle <= max_yaw_err)
    return result

def OOB_chair(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> bool:
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)
    chair_z = (obj_pos[:,2] > -0.03)

    return chair_z

def OOB_chair_2(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> bool:
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)
    chair_z = (obj_pos[:,2] > 2)

    obj_1: RigidObject = env.scene["chair_1"]
    obj_pos_1 = obj_1.data.body_pos_w.squeeze(1)
    chair_z_1 = (obj_pos_1[:,2] > 0.1)
    chair_final = chair_z | chair_z_1
    return chair_final

def OOB(
    env: ManagerBasedRLEnv,
    obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
#sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
    
) -> bool:
# Depends on  N_dir 
#   sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
#   sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
#   sink_pos[:,2] = 0.9

#    radius = 3.5
    radius = 30.5

#   distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
#   print(distance)

    # Check if the distance is within the specified radius | nan | robot falling
#result = (distance > radius) | torch.isnan(distance) | (env.scene.articulations['robot'].data.root_pos_w[:,2] < -0.1 )
    #base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
    obj: RigidObject = env.scene[obj_cfg.name]
    obj_pos = obj.data.body_pos_w.squeeze(1)
#   result = (env.scene.articulations['robot'].data.body_pos_w[:,base_link_idx, 2] < -0.1 )
    #result = (obj_pos[:,2] < 0.3 )
    result = (obj_pos[:,2] > 3 )

    true_indices = torch.nonzero(result, as_tuple=True)[0]
    # If  
    return result


