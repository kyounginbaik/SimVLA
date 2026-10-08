import os
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import isaaclab.simvla as simvla  # IMPORTANT: module import, not "from ... import timestep"

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_rotate_inverse
from isaaclab.simvla.rot import sixd_to_quat_wxyz, quat_angle_error_rad

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------


def real_eef_pose(
    eefpos_t: torch.Tensor,
    arm: str,
    pos_offset_xyz,
) -> torch.Tensor:
    if eefpos_t.dim() == 1:
        eefpos_t = eefpos_t.unsqueeze(0)

    if arm.upper() == "L":
        pos = eefpos_t[:, 0:3]
        rot6d = eefpos_t[:, 3:9]
    elif arm.upper() == "R":
        pos = eefpos_t[:, 10:13]
        rot6d = eefpos_t[:, 13:19]
    else:
        raise ValueError("arm must be 'L' or 'R'")

    offset = torch.tensor(pos_offset_xyz, device=eefpos_t.device, dtype=eefpos_t.dtype).view(1, 3)
    pos = pos + offset
    quat = sixd_to_quat_wxyz(rot6d)
    return torch.cat([pos, quat], dim=-1)  # (B,7)




def _frame_pos_quat_local(env, frame_name: str):
    """Return (pos, quat) as (B,3),(B,4) and pos is in env-local frame."""
    pos_w = env.scene.sensors[frame_name].data.target_pos_w
    quat_w = env.scene.sensors[frame_name].data.target_quat_w
    
    # handle (B,1,3)/(B,1,4)
    if pos_w.dim() == 3:
        pos_w = pos_w[:, 0, :]
    if quat_w.dim() == 3:
        quat_w = quat_w[:, 0, :]

    pos = pos_w - env.scene.env_origins  # (B,3)
    return pos, quat_w


# check only at these per-env timesteps
CHECK_STEPS = tuple(range(1, 920))

def sim2ruin(
    env,
    qpos,
    eefpos,
    qpos_thres: float,
    base_vel_thres: float,
    eefpos_thres: float,
    rot_thres: float,
    termination_t: int,
    json_file: str,
    pos_offset_xyz,
) -> torch.Tensor:
    device = env.device

    # ------------------------------------------------------------
    # Ensure simvla.timestep exists and is on correct device/dtype
    # ------------------------------------------------------------
    if not hasattr(simvla, "timestep") or simvla.timestep is None:
        simvla.timestep = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    elif simvla.timestep.device != device:
        simvla.timestep = simvla.timestep.to(device)

    # ------------------------------------------------------------
    # Only CHECK on selected timesteps
    # ------------------------------------------------------------
    check_steps_t = torch.tensor(CHECK_STEPS, device=device, dtype=simvla.timestep.dtype)
    check_mask = (simvla.timestep.unsqueeze(-1) == check_steps_t.view(1, -1)).any(dim=-1)  # (B,) bool

    # default: no reset unless check step says so
    mask_reset = torch.zeros(env.num_envs, device=device, dtype=torch.bool)

    if check_mask.any():
        # Sim pose
        robot = env.scene.articulations["robot"]
        # Recorded EEF poses are relative to the mobile base. Comparing against
        # environment-world coordinates rejects every gain sample as the base turns.
        frame = env.scene.sensors["ee_R_frame"].data
        eef_r_sim_pos = frame.target_pos_source[:, 0, :]
        eef_r_sim_quat = frame.target_quat_source[:, 0, :]
        eef_r_sim = torch.cat([eef_r_sim_pos, eef_r_sim_quat], dim=-1)  # (B,7)

        # Sim qpos
        # Match qpos order
        # Change gripper range
        arm_qpos_l = robot.data.joint_pos[:, [4,6,8,10,12,14] ] # 12D
        arm_qpos_r = robot.data.joint_pos[:, [3,5,7,9,11,13] ] # 12D
        gripper_qpos_l = 0.1 - 1.7 * robot.data.joint_pos[:, [18] ]/0.04 # 2D
        gripper_qpos_r = 0.1 - 1.7 * robot.data.joint_pos[:, [16] ]/0.04 # 2D

        base_idx = robot.find_bodies("base_link")[0][0]
        base_quat_w = robot.data.body_quat_w[:, base_idx]      
        base_lin_vel_w = robot.data.body_lin_vel_w[:, base_idx] 
        base_ang_vel_w = robot.data.body_ang_vel_w[:, base_idx] 

        base_lin_vel_b = quat_rotate_inverse(base_quat_w, base_lin_vel_w)[:,:2]
        base_ang_vel_b = quat_rotate_inverse(base_quat_w, base_ang_vel_w)[:, 2:3]
        qpos_sim = torch.cat([arm_qpos_l, gripper_qpos_l, arm_qpos_r, gripper_qpos_r, base_lin_vel_b, base_ang_vel_b], dim=-1)

        # Real pose
        eefpos_t = torch.as_tensor(eefpos[simvla.timestep], device=eef_r_sim.device, dtype=eef_r_sim.dtype)
        eef_r_real = real_eef_pose(eefpos_t, arm="R", pos_offset_xyz=pos_offset_xyz)  # (B,7)
        
        # Real qpos
        qpos_t = torch.as_tensor(qpos[simvla.timestep], device=eef_r_sim.device, dtype=eef_r_sim.dtype)

        # qpos MSE 
        qpos_mse = (qpos_t[:,:-3] - qpos_sim[:,:-3]).pow(2).mean(dim=-1)  # (B,)
        print("qpos_mse: ", qpos_mse)
        
        # Base Vel MSE
        base_vel_mse = (qpos_t[:,-3:] - qpos_sim[:,-3:]).pow(2).mean(dim=-1)  # (B,)
        print("base_vel_mse: ", base_vel_mse)

        # EEF Rot Error
        rot_err = quat_angle_error_rad(eef_r_sim[:, 3:], eef_r_real[:, 3:])   # (B,)
        print("rot err: ", rot_err)

        # EEF Euclidean Distance 
        dp = eef_r_sim[:, :3] - eef_r_real[:, :3]   # (B,3)
        pos_l2 = dp.norm(dim=-1)                   # (B,)
        print("EEF l2:  ", pos_l2)

        # All mask 
        qpos_mask = (qpos_mse > qpos_thres) & check_mask
        base_vel_mask = (base_vel_mse > base_vel_thres) & check_mask
        eefpos_mask = (pos_l2 > eefpos_thres) & check_mask
        rot_mask = (rot_err > rot_thres) & check_mask
        #mask_reset = qpos_mask | base_vel_mask | rot_mask | eefpos_mask
        mask_reset = eefpos_mask

    # ------------------------------------------------------------
    # Advance timestep IN-PLACE (critical)
    # ------------------------------------------------------------
    simvla.timestep.add_(1)

    # done mask
    mask_done = (simvla.timestep == termination_t)
    done_ids = torch.where(mask_done)[0]

    # ------------------------------------------------------------
    # Save JSON on done
    # ------------------------------------------------------------
    if done_ids.numel() != 0:
        def load_dict_or_empty(path: Path) -> dict:
            if (not path.exists()) or path.stat().st_size == 0:
                return {}
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def save_json_atomic(path: Path, obj: dict):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        path = Path("./scripts/simvla") / json_file
        data = load_dict_or_empty(path)
        data.setdefault("stiffness", [])
        data.setdefault("damping", [])

        for i in done_ids.tolist():
            new_stiffness = env.scene.articulations["robot"].data.joint_stiffness[i].detach().cpu().tolist()
            new_damping = env.scene.articulations["robot"].data.joint_damping[i].detach().cpu().tolist()
            data["stiffness"].append(new_stiffness)
            data["damping"].append(new_damping)
        
        save_json_atomic(path, data)
        simvla.sysid_done_count = getattr(simvla, "sysid_done_count", 0) + int(done_ids.numel())

    # ------------------------------------------------------------
    # Final reset mask = error reset OR done
    # ------------------------------------------------------------
    mask_reset = mask_reset | mask_done
    reset_ids = torch.where(mask_reset)[0]

    # reset timestep IN-PLACE
    simvla.timestep[mask_reset] = 0

    # reset controllers
    if reset_ids.numel() != 0:
        env.action_manager.get_term("armL_action")._ik_controller.reset(env_ids=reset_ids)
        env.action_manager.get_term("armR_action")._ik_controller.reset(env_ids=reset_ids)

    return mask_reset


def sim2ruin_joint(
    env,
    qpos,
    qpos_thres: float,
    termination_t: int,
    json_file: str,
) -> torch.Tensor:
    device = env.device

    # ------------------------------------------------------------
    # Ensure simvla.timestep exists and is on correct device/dtype
    # ------------------------------------------------------------
    if not hasattr(simvla, "timestep") or simvla.timestep is None:
        simvla.timestep = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    elif simvla.timestep.device != device:
        simvla.timestep = simvla.timestep.to(device)

    # ------------------------------------------------------------
    # Only CHECK on selected timesteps
    # ------------------------------------------------------------
    check_steps_t = torch.tensor(CHECK_STEPS, device=device, dtype=simvla.timestep.dtype)
    check_mask = (simvla.timestep.unsqueeze(-1) == check_steps_t.view(1, -1)).any(dim=-1)  # (B,) bool

    # default: no reset unless check step says so
    mask_reset = torch.zeros(env.num_envs, device=device, dtype=torch.bool)

    if check_mask.any():
        robot = env.scene.articulations["robot"]
        # Sim qpos
        arm_qpos_l = robot.data.joint_pos[:, [1,4,7,9,11,13,15] ]
        arm_qpos_r = robot.data.joint_pos[:, [2,5,8,10,12,14,16] ] # 12D
        gripper_qpos_l = robot.data.joint_pos[:, [17,23,29,35,18,24,30,36,19,25,31,37] ]
        gripper_qpos_r = robot.data.joint_pos[:, [20,26,32,38,21,27,33,39,22,28,34,40] ]

        qpos_sim = torch.cat([arm_qpos_l, arm_qpos_r, gripper_qpos_l,gripper_qpos_r], dim=-1)

        # Real qpos
        qpos_t = torch.as_tensor(qpos[simvla.timestep], device=arm_qpos_l.device, dtype=arm_qpos_l.dtype)
        qpos_t = torch.cat([qpos_t[:, :14], torch.deg2rad(qpos_t[:, 17:])], dim=1)

        # qpos MSE 
        qpos_mse = (qpos_t[:,:14] - qpos_sim[:,:14]).pow(2).mean(dim=-1)  # (B,)
        qpos_abs = (qpos_t[:,:14] - qpos_sim[:,:14]).abs().sum(dim=-1)  # (B,)
        print(qpos_mse)
        print(qpos_abs)
        
        # All mask 
        qpos_mask = (qpos_abs > qpos_thres) & check_mask
        mask_reset = qpos_mask 

    # ------------------------------------------------------------
    # Advance timestep IN-PLACE (critical)
    # ------------------------------------------------------------
    simvla.timestep.add_(1)

    # done mask
    mask_done = (simvla.timestep == termination_t)
    done_ids = torch.where(mask_done)[0]

    # ------------------------------------------------------------
    # Save JSON on done
    # ------------------------------------------------------------
    if done_ids.numel() != 0:
        def load_dict_or_empty(path: Path) -> dict:
            if (not path.exists()) or path.stat().st_size == 0:
                return {}
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def save_json_atomic(path: Path, obj: dict):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        path = Path("./scripts/simvla") / json_file
        data = load_dict_or_empty(path)
        data.setdefault("stiffness", [])
        data.setdefault("damping", [])

        for i in done_ids.tolist():
            new_stiffness = env.scene.articulations["robot"].data.joint_stiffness[i].detach().cpu().tolist()
            new_damping = env.scene.articulations["robot"].data.joint_damping[i].detach().cpu().tolist()
            data["stiffness"].append(new_stiffness)
            data["damping"].append(new_damping)
        
        save_json_atomic(path, data)

    # ------------------------------------------------------------
    # Final reset mask = error reset OR done
    # ------------------------------------------------------------
    mask_reset = mask_reset | mask_done
    reset_ids = torch.where(mask_reset)[0]

    # reset timestep IN-PLACE
    simvla.timestep[mask_reset] = 0

    return mask_reset


def sim2ruin_named(
    env,
    qpos,
    joint_names,
    qpos_thres: float,
    termination_t: int,
    json_file: str,
    settle_steps: int = 20,
    gain_prior: dict | None = None,
    qpos_cols: list | None = None,
) -> torch.Tensor:
    """Joint-space sysid termination that resolves the compared joints BY NAME.

    `qpos` is (T, J) on the env device, one column per entry of `joint_names`, in that order.
    After env.step(action[t]) the sim holds the state 1/fps AFTER command t was applied, which
    is what the real recorder sampled at t+1 -- so the comparison target is qpos[t+1], not
    qpos[t] (the older terms compare against qpos[t], a systematic one-frame lag).

    `settle_steps` is a GRACE WINDOW after every reset during which the threshold is not
    enforced. On the reset frame the arm has just been teleported to the episode's first
    recorded pose and is compared against qpos[t+1], so the error is one frame (50 ms) of the
    REAL robot's motion -- 0.15-0.28 rad abs-sum on these logs, and completely independent of
    the gains. Enforcing a tight threshold there rejects every draw on a transient no gain can
    affect: at qpos_thres 0.2 every env failed at t=0 and re-drew forever, 0 survivors. The
    original sim2ruin sidesteps this by only checking CHECK_STEPS = range(1, 920).

    Per env: abs-sum error > qpos_thres resets it (a fresh gain draw); reaching termination_t
    appends its stiffness/damping and its mean-squared tracking error to the json and bumps
    `simvla.sysid_done_count`, which systemid.py's --max_done watches.
    """
    device = env.device
    robot = env.scene.articulations["robot"]

    if not hasattr(simvla, "timestep") or simvla.timestep is None:
        simvla.timestep = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    elif simvla.timestep.device != device:
        simvla.timestep = simvla.timestep.to(device)

    cache = getattr(simvla, "sysid_named", None)
    if cache is None or cache["num_envs"] != env.num_envs:
        idx, found = robot.find_joints(list(joint_names), preserve_order=True)
        if list(found) != list(joint_names):
            raise ValueError(f"sim2ruin_named: resolved {found} for requested {list(joint_names)}")
        cache = {
            "num_envs": env.num_envs,
            "idx": torch.tensor(idx, device=device, dtype=torch.long),
            "sq": torch.zeros(env.num_envs, len(joint_names), device=device),  # per joint
            # SATURATION BOOK-KEEPING. An implicit actuator computes tau = K*e + D*edot and PhysX
            # clips it at effort_limit_sim; while it is clipped the joint is a bang-bang torque
            # source and its motion no longer depends on K or D at all, so those steps carry NO
            # information about the gains. Counting them lets the fit be restricted to the steps
            # that do (see qpos_mse_joint_unsat), instead of averaging a flat region into an
            # apparently-precise median -- which is how a 2026-08-31 fit reported an arbitrary
            # damping for the shoulders and zeroed the kitchen grasp.
            "sat": torch.zeros(env.num_envs, len(joint_names), device=device),
            "sq_unsat": torch.zeros(env.num_envs, len(joint_names), device=device),
            "n_unsat": torch.zeros(env.num_envs, len(joint_names), device=device),
            "names": list(joint_names),
        }
        simvla.sysid_named = cache
        simvla.sysid_done_count = 0

    sim_q = robot.data.joint_pos[:, cache["idx"]]
    qpos_t = torch.as_tensor(qpos, device=device, dtype=sim_q.dtype)
    # When only some joints are scored (--fit_joints), take the matching qpos columns too.
    if qpos_cols is not None:
        qpos_t = qpos_t[:, torch.tensor(qpos_cols, device=device, dtype=torch.long)]
    target_step = torch.clamp(simvla.timestep + 1, max=qpos_t.shape[0] - 1)
    err = qpos_t[target_step] - sim_q                         # (B, J)
    abs_sum = err.abs().sum(dim=-1)                           # (B,)
    in_grace = simvla.timestep < settle_steps
    cache["sq"] += err.pow(2)

    # A joint is saturated on this step when PhysX had to clip the torque the gains asked for.
    sat = (robot.data.computed_torque[:, cache["idx"]]
           - robot.data.applied_torque[:, cache["idx"]]).abs() > 1e-3
    cache["sat"] += sat.float()
    cache["n_unsat"] += (~sat).float()
    cache["sq_unsat"] += torch.where(sat, torch.zeros_like(err), err.pow(2))
    mask_reset = (abs_sum > qpos_thres) & (~in_grace)

    if int(simvla.timestep[0]) % 100 == 0:
        settled = abs_sum[~in_grace]
        past = (f"past grace: min={settled.min():.3f} n={int(settled.numel())}"
                if settled.numel() else "all envs still in grace")
        print(f"[sim2ruin_named] t={int(simvla.timestep[0])} abs_sum min={abs_sum.min():.3f} "
              f"median={abs_sum.median():.3f} ({past}) "
              f"rejecting={int(mask_reset.sum())}/{env.num_envs} "
              f"done_so_far={simvla.sysid_done_count}")

    simvla.timestep.add_(1)
    mask_done = simvla.timestep == termination_t
    done_ids = torch.where(mask_done)[0]

    if done_ids.numel() != 0:
        def load_dict_or_empty(path: Path) -> dict:
            if (not path.exists()) or path.stat().st_size == 0:
                return {}
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def save_json_atomic(path: Path, obj: dict):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        p = Path(json_file)
        path = p if p.is_absolute() else Path("./scripts/simvla") / p
        data = load_dict_or_empty(path)
        data.setdefault("joint_names", list(robot.data.joint_names))
        data.setdefault("fit_joints", cache["names"])
        data.setdefault("termination_t", int(termination_t))
        if gain_prior is not None:
            data.setdefault("gain_prior", gain_prior)
        data.setdefault("stiffness", [])
        data.setdefault("damping", [])
        data.setdefault("qpos_mse", [])
        data.setdefault("qpos_mse_joint", [])
        data.setdefault("qpos_mse_joint_unsat", [])
        data.setdefault("sat_frac", [])
        data.setdefault("n_unsat", [])
        for i in done_ids.tolist():
            data["stiffness"].append(robot.data.joint_stiffness[i].detach().cpu().tolist())
            data["damping"].append(robot.data.joint_damping[i].detach().cpu().tolist())
            per_joint = (cache["sq"][i] / float(termination_t)).detach().cpu().tolist()
            data["qpos_mse"].append(float(sum(per_joint)) / len(per_joint))
            data["qpos_mse_joint"].append(per_joint)  # one entry per fit_joints column
            n_un = cache["n_unsat"][i]
            data["n_unsat"].append(n_un.detach().cpu().tolist())
            data["sat_frac"].append((cache["sat"][i] / float(termination_t)).detach().cpu().tolist())
            # NaN where the joint never left saturation: no evidence about its gains at all,
            # which a 0.0 would disguise as a perfect fit.
            mse_un = torch.where(n_un > 0, cache["sq_unsat"][i] / n_un.clamp(min=1),
                                 torch.full_like(n_un, float("nan")))
            data["qpos_mse_joint_unsat"].append(mse_un.detach().cpu().tolist())
        save_json_atomic(path, data)
        simvla.sysid_done_count += int(done_ids.numel())
        print(f"[sim2ruin_named] +{int(done_ids.numel())} survivors -> {path} "
              f"(total {len(data['qpos_mse'])}, best mse {min(data['qpos_mse']):.3e})")

    mask_reset = mask_reset | mask_done
    simvla.timestep[mask_reset] = 0
    cache["sq"][mask_reset] = 0.0
    cache["sat"][mask_reset] = 0.0
    cache["sq_unsat"][mask_reset] = 0.0
    cache["n_unsat"][mask_reset] = 0.0
    return mask_reset
