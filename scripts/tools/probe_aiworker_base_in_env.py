"""Why does the base drive in an empty scene but not in the kitchen?

probe_aiworker_base_drive.py established that the robot itself is fine: with
solver_position_iteration_count=4 the prismatic joints reach 88-91% of a commanded 0.20 m/s in
an empty scene. In kitchen 1215 the same robot achieves 0.000 and never leaves its spawn. Two
things could account for that and they need separating:

  A. THE SCENE. Contact with the floor, the kitchen, or the robot's own arms holds it.
  B. THE ACTION PATH. The nav's command never reaches the actuator, so nothing is ever asked of
     the base and the trace's `cmd` is only what the planner WANTED.

This builds the REAL env -- same cfg, same physics settings, same kitchen -- and drives the base
two ways in the same episode:

  1. DIRECT: robot.set_joint_velocity_target(), bypassing the ActionManager entirely.
  2. VIA ACTION: a zeroed action vector with only the base slots set, through env.step().

Read it as:
  * direct drives, action does not  -> B, the action path. The scene is innocent.
  * neither drives                  -> A, the scene. The robot is held.
  * both drive                      -> neither; the fault is in what the NAV computes.

    conda run --no-capture-output -n env_isaaclab ./isaaclab.sh -p \\
        scripts/tools/probe_aiworker_base_in_env.py --headless --task Isaac-Kitchen-v1215a-00
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v1215a-00")
parser.add_argument("--seconds", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The kitchen env cfg spawns four TiledCameras, and IsaacLab refuses to build it
# without this even though nothing here reads a frame.
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import gymnasium as gym                                          # noqa: E402
import torch                                                     # noqa: E402
import isaaclab_tasks                                            # noqa: E402  (registers the ids)
from isaaclab_tasks.utils import parse_env_cfg                    # noqa: E402

BASE_JOINTS = ["base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint"]
YAW_CMD = 0.463          # what the nav asks for in phase 0
LIN_CMD = 0.20


def base_pose(robot, bidx):
    """base_link's WORLD pose. Independent of the joint buffers -- a body pose comes from the
    physics view, so it cannot read stale in the way joint_pos can when the refresh is done by
    robot.update() outside env.step()."""
    p = robot.data.body_pos_w[0, bidx].tolist()
    q = robot.data.body_quat_w[0, bidx].tolist()
    return p, q


def report(tag, robot, ids, axis, target, before, bidx=None, base_before=None):
    v = float(robot.data.joint_vel[0, ids[axis]])
    moved = float(robot.data.joint_pos[0, ids[axis]]) - before
    if bidx is not None and base_before is not None:
        import math as _m
        p, q = base_pose(robot, bidx)
        dxy = _m.dist(p[:2], base_before[0][:2])
        # yaw from the wxyz quaternion
        def _yaw(qq):
            w, x, y, z = qq
            return _m.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dyaw = abs(_yaw(q) - _yaw(base_before[1]))
        print(f"[envprobe]   base_link world: moved {dxy:.4f} m, turned {dyaw:.4f} rad "
              f"(THIS is the ground truth; joint_pos can read stale)", flush=True)
    frac = v / target if target else 0.0
    # JUDGE ON `moved`, NOT ON THE VELOCITY SAMPLE. A joint held by contact oscillates: the
    # first run of this probe read -241% and +329% of command while the joint travelled 0.0008
    # in two seconds. Displacement cannot be faked that way.
    expected = target * args_cli.seconds
    ok = abs(moved) > 0.25 * abs(expected)
    print(f"[envprobe] {tag:14s} {BASE_JOINTS[axis]:26s} commanded {target:+.3f} -> "
          f"moved {moved:+.4f} of an expected {expected:+.3f} "
          f"({100 * moved / expected if expected else 0:6.1f}%), vel sample {v:+.4f} "
          f"[{'DRIVEN' if ok else 'HELD'}]", flush=True)
    return ok


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene.articulations["robot"]
    ids, _ = robot.find_joints(BASE_JOINTS, preserve_order=True)
    print(f"[envprobe] task={args_cli.task} base joint ids={ids} "
          f"decimation={env_cfg.decimation} dt={env_cfg.sim.dt}", flush=True)
    props = env_cfg.scene.robot.spawn.articulation_props
    print(f"[envprobe] articulation props: self_col={props.enabled_self_collisions} "
          f"pos_iter={props.solver_position_iteration_count} "
          f"vel_iter={props.solver_velocity_iteration_count}", flush=True)

    # THE GAINS AS THE ENV ACTUALLY BUILT THEM, not as AIWORKER_CFG declares them. The empty-scene
    # probe printed these and the first version of this one did not, which is the difference
    # between knowing and assuming that the two runs drive the same actuator.
    for name, act in robot.actuators.items():
        if "base" in name:
            print(f"[envprobe] actuator {name!r}: joints={act.joint_names} "
                  f"effort={act.effort_limit.flatten()[:3].tolist()} "
                  f"vel_limit={act.velocity_limit.flatten()[:3].tolist()} "
                  f"stiffness={act.stiffness.flatten()[:3].tolist()} "
                  f"damping={act.damping.flatten()[:3].tolist()}", flush=True)
    lim = robot.data.joint_limits[0, ids].tolist()
    print(f"[envprobe] base joint limits {lim}", flush=True)
    print(f"[envprobe] base joint_pos={robot.data.joint_pos[0, ids].tolist()} "
          f"default={robot.data.default_joint_pos[0, ids].tolist()}", flush=True)
    unmatched = [j for j in BASE_JOINTS
                 if not any(j in a.joint_names for a in robot.actuators.values())]
    print(f"[envprobe] base joints with NO actuator group: {unmatched}", flush=True)

    bidx = robot.find_bodies("base_link")[0][0]
    steps = max(1, int(args_cli.seconds / (env_cfg.sim.dt * env_cfg.decimation)))
    action_dim = env.action_manager.total_action_dim
    print(f"[envprobe] action dim={action_dim}; base is the LAST 3 slots", flush=True)

    # ---- 1. DIRECT: bypass the ActionManager entirely -------------------------------------
    for axis, target in ((2, YAW_CMD), (0, LIN_CMD)):
        env.reset()
        before = float(robot.data.joint_pos[0, ids[axis]])
        vt = torch.zeros((1, robot.num_joints), device=robot.device)
        vt[0, ids[axis]] = target
        for _ in range(steps * env_cfg.decimation):
            robot.set_joint_velocity_target(vt)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            # robot.update, NOT scene.update: the latter also updates the four TiledCameras,
            # which have no frame to read when the step did not render, and raise
            # AttributeError: 'TiledCamera' object has no attribute '_timestamp'.
            robot.update(env_cfg.sim.dt)
        report("DIRECT", robot, ids, axis, target, before)

    # ---- 2. THE SAME DRIVE, WITH THE ROBOT LIFTED CLEAR OF EVERYTHING ---------------------
    #
    # DIRECT already showed the base does not move in this scene even when the ActionManager is
    # bypassed, so the action path is innocent and something is HOLDING the robot. This asks
    # whether that something is CONTACT: teleport the whole articulation 1.5 m into the air,
    # where it can touch nothing, and drive again.
    #
    #   drives in the air, not on the ground -> contact. Find what it is resting in.
    #   still will not drive                 -> not contact; something else about this scene.
    # 5.0 m, NOT 1.5. The first version of this test lifted the robot 1.5 m and reported it
    # still HELD -- but the wall cabinets in these kitchens sit at 1.4-1.9 m, so that moved the
    # robot INTO geometry rather than clear of it, and the "not contact" conclusion it produced
    # was wrong. The room shell is four walls and NO CEILING, so up is the one direction that is
    # reliably empty.
    for lift in (0.0, 5.0):
        for axis, target in ((2, YAW_CMD), (0, LIN_CMD)):
            env.reset()
            root = robot.data.root_state_w.clone()
            root[:, 2] += lift
            robot.write_root_state_to_sim(root)
            robot.write_data_to_sim()
            for _ in range(10):
                env.sim.step(render=False)
                robot.update(env_cfg.sim.dt)
            pos = robot.data.root_state_w[0, :3].tolist()
            before = float(robot.data.joint_pos[0, ids[axis]])
            base_before = base_pose(robot, bidx)
            vt = torch.zeros((1, robot.num_joints), device=robot.device)
            vt[0, ids[axis]] = target
            for _ in range(steps * env_cfg.decimation):
                robot.set_joint_velocity_target(vt)
                robot.write_data_to_sim()
                env.sim.step(render=False)
                robot.update(env_cfg.sim.dt)
            tag = "AIRBORNE" if lift else "ON GROUND"
            print(f"[envprobe] base at ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})", flush=True)
            report(tag, robot, ids, axis, target, before, bidx, base_before)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
