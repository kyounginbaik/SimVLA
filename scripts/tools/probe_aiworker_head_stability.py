"""Measure physical head jitter under repeatable arm and turn/stop excitation."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stiffness', type=float, default=200.)
parser.add_argument('--damping', type=float, default=50.)
parser.add_argument('--gravity-compensation', action='store_true')
parser.add_argument('--disable-self-collisions', action='store_true',
                    help='Diagnostic isolation only; does not change production defaults.')
parser.add_argument('--filter-head-torso', action='store_true')
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
exit_code = 0
try:
    import gymnasium as gym
    import torch
    import numpy as np
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg('Isaac-Kitchen-v813a-00', device=args.device, num_envs=1)
    cfg.seed = 0
    cfg.recorders = None
    cfg.eval_mode = True
    cfg.observations.policy.concatenate_terms = False
    cfg.decimation = 6
    cfg.sim.dt = 1/120
    cfg.sim.render_interval = 6
    cfg.events.robot_init_pos.params['pose_range'].update(x=(.75, .75), y=(-1.7, -1.7), yaw=(0., 0.))
    for name in vars(cfg.terminations):
        if not name.startswith('_'):
            setattr(cfg.terminations, name, None)
    cfg.scene.robot.actuators['head'].stiffness = args.stiffness
    cfg.scene.robot.actuators['head'].damping = args.damping
    if args.disable_self_collisions:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.filter_head_torso:
        from pxr import UsdPhysics
        original_spawn = cfg.scene.robot.spawn.func
        def spawn_with_head_filter(*spawn_args, **spawn_kwargs):
            prim = original_spawn(*spawn_args, **spawn_kwargs)
            head = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild('head_link2'))
            torso = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild('arm_base_link'))
            if not head.IsValid() or not torso.IsValid():
                raise RuntimeError(f'Cannot locate head/torso under {prim.GetPath()}')
            UsdPhysics.FilteredPairsAPI.Apply(head).CreateFilteredPairsRel().AddTarget(torso.GetPath())
            print('HEAD_FILTER', head.GetPath(), torso.GetPath(), flush=True)
            return prim
        cfg.scene.robot.spawn.func = spawn_with_head_filter
    env = gym.make('Isaac-Kitchen-v813a-00', cfg=cfg).unwrapped
    env.reset()
    robot = env.scene['robot']
    head = robot.find_joints(['head_joint1', 'head_joint2'], preserve_order=True)[0]
    arms = robot.find_joints(['arm_l_joint1', 'arm_l_joint4'], preserve_order=True)[0]
    home = robot.data.default_joint_pos.clone()
    limits = robot.data.joint_limits
    print('HEAD_SETUP', dict(default=home[0, head].cpu().tolist(),
                            limits=limits[0, head].cpu().tolist()), flush=True)
    samples = []
    efforts = []
    goal = torch.zeros(1, dtype=torch.long, device=env.device)
    with torch.inference_mode():
        for tick in range(300):
            # Smoothly extend/retract a small amount while yawing, then hold still.
            fraction = max(0., min(1., (tick-40)/40, (200-tick)/40))
            target = home.clone()
            target[:, arms[0]] += .25*fraction
            target[:, arms[1]] += .35*fraction
            target = torch.clamp(target, limits[..., 0], limits[..., 1])
            env.action_manager.set_joint_position_replay_targets(target)
            if args.gravity_compensation:
                gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, head]
                robot.set_joint_effort_target(gravity, joint_ids=head)
            action = torch.zeros(env.action_space.shape, device=env.device)
            action[:, -1] = .46 if 80 <= tick < 180 else 0.
            env.step(action, goal)
            samples.append(robot.data.joint_pos[0, head].cpu().tolist())
            efforts.append(robot.data.applied_torque[0, head].cpu().tolist())
            if tick % 50 == 0:
                print('HEAD_STABILITY_PROGRESS', tick, samples[-1], flush=True)
    values = np.asarray(samples)
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite head state')
    report = dict(stiffness=args.stiffness, damping=args.damping, fps=20,
                  gravity_compensation=args.gravity_compensation,
                  self_collisions=not args.disable_self_collisions,
                  head_torso_filtered=args.filter_head_torso,
                  head_applied_torque_nm=efforts,
                  head_joint_names=[robot.joint_names[i] for i in head], head_q_rad=samples,
                  max_frame_delta_deg_after_settle=float(np.abs(np.diff(values[40:], axis=0)).max()*180/np.pi),
                  stop_max_frame_delta_deg=float(np.abs(np.diff(values[180:], axis=0)).max()*180/np.pi),
                  final_second_span_deg=(np.ptp(values[-20:], axis=0)*180/np.pi).tolist(),
                  scope='Controlled physical excitation; requires follow-up collection/video validation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('HEAD_STABILITY_RESULT', json.dumps({k:v for k,v in report.items()
                                              if k not in ('head_q_rad', 'head_applied_torque_nm')}), flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    from isaaclab.sim import SimulationContext
    from workflow_status import close_with_status
    exit_code = close_with_status(exit_code, SimulationContext.clear_instance, app.close)
raise SystemExit(exit_code)
