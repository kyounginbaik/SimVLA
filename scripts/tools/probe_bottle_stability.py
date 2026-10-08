"""Does the bottle stay standing when nothing touches it?

WHY. A per-step tilt trace during a grasp shows the bottle going from 3.5 to 90.8 degrees in ten
steps while the end-effector is still 0.10-0.16 m away. The furthest point of the open jaws is
about 0.062 m from the end-effector, so at that range nothing is within 0.06 m of the bottle --
and the tilt curve accelerates the way a free topple about a pivot does, not the way a push does.
That says the object falls over on its own, and that every fix aimed at the arm was aimed at the
wrong system.

This checks it without the robot in the loop at all: build the real env, command NOTHING, and
watch the bottle for 600 steps.

  topples with the robot idle  -> asset stability. The collider is a convexDecomposition of a
                                  bottle mesh and evidently gives it no reliable flat base.
  stays upright                -> the robot's presence destabilises it (base contact, counter
                                  motion), and the arm is back in scope.

    conda run --no-capture-output -n env_isaaclab ./isaaclab.sh -p \\
        scripts/tools/probe_bottle_stability.py --headless --task Isaac-Kitchen-v1215a-00
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v1215a-00")
parser.add_argument("--obj", type=str, default="bottle0")
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import math                                                          # noqa: E402
import gymnasium as gym                                              # noqa: E402
import torch                                                         # noqa: E402
import isaaclab_tasks                                                # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg                        # noqa: E402


def tilt_of(q):
    """Degrees between the body's +Z and world up."""
    upz = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, upz))))


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    obj = env.scene.rigid_objects[args_cli.obj]
    robot = env.scene.articulations["robot"]

    print(f"[stab] task={args_cli.task} obj={args_cli.obj!r} -- NOTHING is commanded; the robot "
          f"holds its spawn pose for {args_cli.steps} steps", flush=True)
    worst = [0.0] * env.num_envs
    for k in range(args_cli.steps):
        # No action at all: hold the spawn joint targets, step physics only.
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env_cfg.sim.dt)
        obj.update(env_cfg.sim.dt)
        if k % 20 == 0 or k == args_cli.steps - 1:
            row = []
            for e in range(env.num_envs):
                t = tilt_of(obj.data.body_quat_w[e, 0])
                worst[e] = max(worst[e], t)
                z = float(obj.data.body_pos_w[e, 0, 2] - env.scene.env_origins[e, 2])
                row.append(f"env{e}: {t:5.1f}deg z={z:.4f}")
            print(f"[stab] step {k:4d}  " + "  ".join(row), flush=True)

    print(f"[stab] worst tilt per env: "
          + "  ".join(f"env{e}={w:.1f}deg" for e, w in enumerate(worst)), flush=True)
    fell = sum(1 for w in worst if w > 30.0)
    print(f"[stab] VERDICT: {fell} of {env.num_envs} toppled with NOTHING touching them -- "
          f"{'the object is unstable on its own' if fell else 'the object is stable; the robot destabilises it'}",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
