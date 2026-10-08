"""Why does yaw drive at 83% while the two PRISMATIC base joints sit at exactly 0.0?

That split is the signature of solver_position_iteration_count=8, measured in an empty scene
(scripts/tools/probe_aiworker_base_drive.py): at 8 the prismatic joints achieve 0.0% and the
revolute 81-84%; at 4 all three drive. AIWORKER_CFG asks for 4 and the ffw_sg2.usd now authors 4,
but NOTHING HAS EVER READ BACK THE VALUE PHYSX IS ACTUALLY USING -- every check so far read the
config object, which is a statement of intent, not of fact.

Two other things this closes, both of which invalidated earlier probes:

  * IT SPAWNS THE ROBOT WHERE THE RUN ACTUALLY SPAWNS IT. probe_aiworker_base_in_env.py left the
    root at the env origin, and (0, 0) in kitchen 1215 is INSIDE THE REFRIGERATOR
    (x[-0.373,+0.373] y[-0.398,+0.368] z[0,1.559]). Its "ON GROUND -> HELD" verdict was measured
    on a robot embedded in an appliance. This reproduces smoke 011's frozen pose exactly:
    base (+0.732,-1.421), yaw +1.397, which a world-AABB sweep says is clear of everything.
  * IT JUDGES ON base_link's WORLD DISPLACEMENT, never on a velocity sample.

Then it sweeps the candidate fixes in one job so there is no second round trip.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v1215a-00")
parser.add_argument("--seconds", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import math                                                      # noqa: E402
import gymnasium as gym                                          # noqa: E402
import torch                                                     # noqa: E402
import isaaclab_tasks                                            # noqa: E402  (registers the ids)
from isaaclab_tasks.utils import parse_env_cfg                    # noqa: E402

BASE = ["base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint"]
#: smoke 011's frozen pose, and the command it was refusing.
STUCK_XY, STUCK_YAW, VX = (0.732, -1.421), 1.397, 0.199


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def applied_solver_counts(robot):
    """What PhysX is USING, not what the config asked for."""
    view = robot.root_physx_view
    out = {}
    for label, fn in (("pos", "get_solver_position_iteration_counts"),
                      ("vel", "get_solver_velocity_iteration_counts")):
        try:
            out[label] = fn and [int(v) for v in getattr(view, fn)().flatten()[:1]]
        except Exception as exc:                                  # noqa: BLE001
            out[label] = f"unreadable ({type(exc).__name__}: {exc})"
    return out


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene.articulations["robot"]
    ids, _ = robot.find_joints(BASE, preserve_order=True)
    bidx = robot.find_bodies("base_link")[0][0]

    # THE ARTICULATION'S REAL JOINT ORDER. robot_schemas.AIWORKER_JOINTS was written from the
    # USD by hand and is marked PROVISIONAL; an exported LeRobot dataset whose action columns are
    # permuted against the sim is silently wrong, and nothing downstream would catch it. This is
    # the authority: it is the order PhysX actually uses.
    print(f"[joints] {robot.num_joints} joints, in articulation order:", flush=True)
    print(f"[joints] {robot.joint_names}", flush=True)

    props = env_cfg.scene.robot.spawn.articulation_props
    print(f"[stall] CONFIG ASKS FOR   pos_iter={props.solver_position_iteration_count} "
          f"vel_iter={props.solver_velocity_iteration_count}", flush=True)
    print(f"[stall] PHYSX IS USING    {applied_solver_counts(robot)}", flush=True)
    for name, act in robot.actuators.items():
        if "base" in name:
            print(f"[stall] base actuator      stiffness={act.stiffness.flatten()[0]:.5f} "
                  f"damping={act.damping.flatten()[0]:.1f} "
                  f"effort={act.effort_limit.flatten()[0]:.0f}", flush=True)

    steps = max(1, int(args_cli.seconds / env_cfg.sim.dt))

    def run(tag, pos_iter=None, damping=None, lift=0.0, pos_drive=False):
        """Put the robot in smoke 011's exact frozen pose, command +vx, report what MOVED."""
        env.reset()
        root = robot.data.default_root_state.clone()
        root[:, :2] = torch.tensor(STUCK_XY, device=robot.device)
        root[:, 2] = robot.data.default_root_state[:, 2] + lift
        half = STUCK_YAW / 2.0
        root[:, 3] = math.cos(half); root[:, 4] = 0.0
        root[:, 5] = 0.0;            root[:, 6] = math.sin(half)
        root[:, 7:] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                       torch.zeros_like(robot.data.joint_vel))
        robot.reset()

        if pos_iter is not None:
            try:
                n = torch.full((robot.num_instances,), pos_iter, dtype=torch.int32)
                robot.root_physx_view.set_solver_position_iteration_counts(n)
                print(f"[stall]   forced pos_iter={pos_iter} at runtime", flush=True)
            except Exception as exc:                              # noqa: BLE001
                print(f"[stall]   COULD NOT force pos_iter: {exc}", flush=True)
        if damping is not None:
            # write_joint_damping_to_sim, NOT `act.damping[:] = ...`. These are IMPLICIT
            # actuators: PhysX runs the PD itself from gains written into the simulation at
            # startup, so assigning to the Python actuator object changes nothing and the run
            # silently measures the OLD gain. A first pass "swept" damping 10000 -> 50000 that
            # way and reported no change, which was true but meaningless.
            robot.write_joint_damping_to_sim(
                torch.full((1, len(ids)), damping, device=robot.device), joint_ids=ids)
            print(f"[stall]   wrote base damping={damping} INTO THE SIM", flush=True)

        for _ in range(60):                                        # settle onto the floor
            robot.write_data_to_sim(); env.sim.step(render=False); robot.update(env_cfg.sim.dt)

        p0 = robot.data.body_pos_w[0, bidx].clone()
        vt = torch.zeros((1, robot.num_joints), device=robot.device)
        # +vx in the BASE frame, resolved to the world-axis prismatic joints by hand.
        vt[0, ids[0]] = VX * math.cos(STUCK_YAW)
        vt[0, ids[1]] = VX * math.sin(STUCK_YAW)
        for k in range(steps):
            if pos_drive:
                # A POSITION target instead of a velocity one. If the joint moves under this and
                # not under the velocity drive, the fault is the drive mode, not the joint.
                qt = robot.data.joint_pos.clone()
                qt[0, ids[0]] += 0.4 * math.cos(STUCK_YAW)
                qt[0, ids[1]] += 0.4 * math.sin(STUCK_YAW)
                robot.set_joint_position_target(qt)
            else:
                robot.set_joint_velocity_target(vt)
            robot.write_data_to_sim(); env.sim.step(render=False); robot.update(env_cfg.sim.dt)
            # THE ONE NUMBER THAT SPLITS THE REMAINING SPACE. If applied_torque on the prismatic
            # joints is thousands of newtons, the drive IS pushing and something rigid resists;
            # if it is ~0, the drive never pushed and no amount of damping was ever going to help
            # -- which is what damping 50000 moving the robot exactly 0.0000 m already hints at.
            if k in (0, steps // 2, steps - 1):
                tq = robot.data.applied_torque[0, ids]
                jt = robot.data.joint_vel_target[0, ids]
                jv = robot.data.joint_vel[0, ids]
                print(f"[stall]     k={k:4d} target=({float(jt[0]):+.4f},{float(jt[1]):+.4f},"
                      f"{float(jt[2]):+.4f}) applied_force=({float(tq[0]):+9.1f},"
                      f"{float(tq[1]):+9.1f},{float(tq[2]):+9.1f}) N  "
                      f"vel=({float(jv[0]):+.4f},{float(jv[1]):+.4f},{float(jv[2]):+.4f})",
                      flush=True)
        p1 = robot.data.body_pos_w[0, bidx]
        moved = float(torch.linalg.norm(p1[:2] - p0[:2]))
        want = VX * args_cli.seconds
        print(f"[stall] {tag:22s} base_link travelled {moved:.4f} m of an expected {want:.3f} "
              f"({100 * moved / want:5.1f}%)  z {float(p0[2]):.4f} -> {float(p1[2]):.4f}  "
              f"[{'DRIVEN' if moved > 0.25 * want else 'HELD'}]", flush=True)

    run("as-configured")
    # THE LIFT SWEEP. 5 m up the robot can touch nothing; if it drives there and not on the
    # floor, the fault is contact, and the HEIGHT AT WHICH IT FREES says what kind. Freeing at
    # 5 mm means the wheels are jammed a few millimetres into the floor plane; needing 0.2 m
    # means it is resting on something with real height. A single airborne test cannot tell
    # those apart, and the answer picks the fix.
    for lift in (0.005, 0.02, 0.05, 0.20, 1.0, 5.0):
        run(f"lift {lift:.3f} m", lift=lift)
    # If a POSITION target moves the joint that a velocity target cannot, the drive mode is the
    # fault rather than the joint or the contact.
    run("POSITION drive", pos_drive=True)
    run("damping=50000", damping=50000.0)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
