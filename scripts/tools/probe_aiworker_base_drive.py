"""Can the AI Worker's base be driven at all? One robot, empty scene, no kitchen, no planner.

WHY THIS EXISTS. In kitchen 1215 the nav commands 0.463 rad/s of yaw and the base achieves
-0.015 -- 3% of command, wrong sign -- and never leaves its spawn pose. That symptom has at
least four causes and the kitchen run cannot tell them apart: the actuator may not be driving,
the action may not be reaching it, the robot may be wedged against the counter, or the solver
may be unstable at these gains. This strips away everything but the robot and the drive.

It commands each base joint in turn and reports commanded vs achieved. Read it as:

  * achieved ~= commanded            -> the robot and its actuators are fine; the kitchen run is
                                        blocked by geometry or the action never arrives.
  * achieved ~= 0 in an EMPTY scene  -> the robot config or the asset is at fault, and the
                                        kitchen is irrelevant.

    conda run --no-capture-output -n env_isaaclab ./isaaclab.sh -p \\
        scripts/tools/probe_aiworker_base_drive.py --headless

Needs a GPU (it steps physics) but no cameras, one env, and a few seconds of sim.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=2.0, help="sim seconds per commanded axis")
parser.add_argument("--variant", type=str, default="as_shipped",
                    help="as_shipped | anubis_props | selfcol_off_only | pos8_vel0 | selfcol_on_4_0")
parser.add_argument("--with-kitchen", type=str, default="",
                    help="also spawn this kitchen USD. The robot drives in an empty scene under "
                         "the env's own SimulationCfg but not inside the env, so this asks "
                         "whether the SCENE CONTENTS -- dozens of extra articulations and "
                         "hundreds of rigid bodies -- are what stops it.")
parser.add_argument("--env-sim", type=str, default="",
                    help="copy the SimulationCfg from this task's env cfg instead of using a "
                         "bare default one. The robot drives in a default sim and does not in "
                         "the kitchen env with identical actuators and articulation props, so "
                         "this asks whether the SIM SETTINGS are the difference.")
parser.add_argument("--pose", type=str, default="zero", choices=("zero", "c2"),
                    help="joint pose to drive FROM. zero hangs the arms straight down; c2 is the "
                         "pose the robot actually spawns in, with the arms folded -- which is the "
                         "one that matters if self-collision at C2 is what holds the base.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch                                                     # noqa: E402
import isaaclab.sim as sim_utils                                 # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg           # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass                           # noqa: E402
from isaaclab_assets.robots.aiworker import AIWORKER_CFG         # noqa: E402

BASE_JOINTS = ["base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint"]
#: (axis index, commanded velocity). The yaw rate is the one the kitchen run asks for.
COMMANDS = [(0, 0.20), (1, 0.20), (2, 0.463)]


#: The one configuration difference between this robot and the two that drive. Anubis and RB-Y1
#: both ship enabled_self_collisions=False with 4/0 solver iterations; AIWORKER_CFG shipped True
#: with 8/2. Tested rather than argued about.
VARIANTS = {
    "as_shipped": dict(enabled_self_collisions=True,
                       solver_position_iteration_count=8, solver_velocity_iteration_count=2),
    "anubis_props": dict(enabled_self_collisions=False,
                         solver_position_iteration_count=4, solver_velocity_iteration_count=0),
    "selfcol_off_only": dict(enabled_self_collisions=False,
                             solver_position_iteration_count=8, solver_velocity_iteration_count=2),
    # The 2x2 that separates the two knobs. anubis_props (off, 4/0) drives at 88%;
    # selfcol_off_only (off, 8/2) does not -- so the self-collision flag is NOT the cause and
    # one of the iteration counts is. These two say which.
    "pos8_vel0": dict(enabled_self_collisions=False,
                      solver_position_iteration_count=8, solver_velocity_iteration_count=0),
    "selfcol_on_4_0": dict(enabled_self_collisions=True,
                           solver_position_iteration_count=4, solver_velocity_iteration_count=0),
}


def scene_cfg_for(variant: str):
    cfg = AIWORKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.spawn = cfg.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(**VARIANTS[variant]))

    kitchen_usd = args_cli.with_kitchen

    @configclass
    class _Cfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome = AssetBaseCfg(prim_path="/World/light",
                            spawn=sim_utils.DomeLightCfg(intensity=1000.0))
        robot = cfg
        kitchen = (AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Kitchen",
                                spawn=sim_utils.UsdFileCfg(usd_path=kitchen_usd))
                   if kitchen_usd else None)

    return _Cfg


def main() -> None:
    # 120 Hz, matching what aiworker_kitchen_cfg.py sets for the real runs.
    if args_cli.env_sim:
        from isaaclab_tasks.utils import parse_env_cfg
        _ec = parse_env_cfg(args_cli.env_sim, device=args_cli.device, num_envs=1)
        sim_cfg = _ec.sim
        sim_cfg.device = args_cli.device
        print(f"[probe] SIM CFG COPIED FROM {args_cli.env_sim}: dt={sim_cfg.dt} "
              f"render_interval={sim_cfg.render_interval} "
              f"bounce={sim_cfg.physx.bounce_threshold_velocity} "
              f"friction_corr={sim_cfg.physx.friction_correlation_distance} "
              f"solver_type={getattr(sim_cfg.physx, 'solver_type', '?')} "
              f"min_pos_iter={getattr(sim_cfg.physx, 'min_position_iteration_count', '?')} "
              f"max_pos_iter={getattr(sim_cfg.physx, 'max_position_iteration_count', '?')} "
              f"min_vel_iter={getattr(sim_cfg.physx, 'min_velocity_iteration_count', '?')} "
              f"max_vel_iter={getattr(sim_cfg.physx, 'max_velocity_iteration_count', '?')}",
              flush=True)
    else:
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
        print(f"[probe] SIM CFG default: "
              f"min_pos_iter={getattr(sim_cfg.physx, 'min_position_iteration_count', '?')} "
              f"max_pos_iter={getattr(sim_cfg.physx, 'max_position_iteration_count', '?')} "
              f"min_vel_iter={getattr(sim_cfg.physx, 'min_velocity_iteration_count', '?')} "
              f"max_vel_iter={getattr(sim_cfg.physx, 'max_velocity_iteration_count', '?')}",
              flush=True)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[probe] VARIANT = {args_cli.variant}: {VARIANTS[args_cli.variant]}", flush=True)
    scene = InteractiveScene(scene_cfg_for(args_cli.variant)(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]

    ids, _ = robot.find_joints(BASE_JOINTS, preserve_order=True)
    print(f"[probe] base joint indices {ids}", flush=True)
    print(f"[probe] {robot.num_joints} joints, {robot.num_bodies} bodies", flush=True)
    for name, act in robot.actuators.items():
        if "base" in name:
            print(f"[probe] actuator {name!r}: effort_limit={act.effort_limit.flatten()[:3].tolist()} "
                  f"velocity_limit={act.velocity_limit.flatten()[:3].tolist()} "
                  f"stiffness={act.stiffness.flatten()[:3].tolist()} "
                  f"damping={act.damping.flatten()[:3].tolist()}", flush=True)

    steps = max(1, int(args_cli.seconds * 120))
    ok = True
    for axis, target in COMMANDS:
        # Settle, then hold one axis at `target` and read back what the joint actually does.
        #
        # THE POSE MATTERS AND ZERO IS NOT THE DEFAULT ONE. Driving from all-zeros hangs the arms
        # straight down; the robot actually SPAWNS at C2, arms folded, with
        # enabled_self_collisions=True. If contact at C2 is what holds the base, only --pose c2
        # can show it.
        q0 = (robot.data.default_joint_pos.clone() if args_cli.pose == "c2"
              else torch.zeros_like(robot.data.joint_pos))
        robot.write_joint_state_to_sim(q0, torch.zeros_like(robot.data.joint_vel))
        robot.reset()
        for _ in range(30):
            robot.write_data_to_sim(); sim.step(); scene.update(1.0 / 120.0)

        vt = torch.zeros((1, robot.num_joints), device=robot.device)
        vt[0, ids[axis]] = target
        achieved = []
        for _ in range(steps):
            robot.set_joint_velocity_target(vt)
            robot.write_data_to_sim()
            sim.step()
            scene.update(1.0 / 120.0)
            achieved.append(float(robot.data.joint_vel[0, ids[axis]]))
        settled = sum(achieved[-30:]) / min(30, len(achieved))
        moved = float(robot.data.joint_pos[0, ids[axis]])
        frac = settled / target if target else 0.0
        # JUDGE ON DISPLACEMENT, not on the settled velocity. A joint held by contact oscillates:
        # this probe reported 430.9% of command while the joint travelled -0.0008 m in two
        # seconds, and called it DRIVEN. Displacement cannot be faked that way.
        expected = target * args_cli.seconds
        driven = abs(moved) > 0.25 * abs(expected)
        verdict = "DRIVEN" if driven else "HELD"
        ok = ok and driven
        print(f"[probe] {BASE_JOINTS[axis]:26s} commanded {target:+.3f} -> travelled {moved:+.4f} "
              f"of an expected {expected:+.3f} ({100 * moved / expected if expected else 0:6.1f}%), "
              f"settled vel {settled:+.4f} ({frac * 100:5.1f}%) [{verdict}]", flush=True)

    print(f"[probe] VERDICT: the base {'CAN' if ok else 'CANNOT'} be driven in an empty scene",
          flush=True)


if __name__ == "__main__":
    main()
    app.close()
