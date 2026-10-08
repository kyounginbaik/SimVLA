"""Isolate an unloaded AI Worker wrist hold; diagnostic, never collection evidence."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
exit_code = 0
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    cfg = parse_env_cfg('Isaac-Kitchen-v813a-00', device=args.device, num_envs=1)
    cfg.seed = 0
    cfg.recorders = None
    cfg.eval_mode = True
    cfg.observations.policy.concatenate_terms = False
    if not args.enable_cameras:
        for name, sensor in vars(cfg.scene).items():
            if isinstance(sensor, (CameraCfg, TiledCameraCfg)):
                setattr(cfg.scene, name, None)
    cfg.decimation = 6
    cfg.sim.dt = 1 / 120
    cfg.sim.render_interval = 6
    cfg.events.robot_init_pos.params['pose_range'].update(x=(.75, .75), y=(-1.7, -1.7), yaw=(0., 0.))
    for name in vars(cfg.terminations):
        if not name.startswith('_'):
            setattr(cfg.terminations, name, None)
    env = gym.make('Isaac-Kitchen-v813a-00', cfg=cfg).unwrapped
    env.reset()
    robot = env.scene['robot']
    names = [f'arm_l_joint{i}' for i in range(1, 8)]
    ids = robot.find_joints(names, preserve_order=True)[0]
    q = robot.data.default_joint_pos.clone()
    q[:, ids] = torch.tensor([-.5298107, .0738088, -.4224597, -2.2541463,
                              .6243474, 1.2285243, 1.2874281], device=env.device)
    # Explicit state initialization is for this isolated diagnostic only.
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    targets = q.clone()
    targets[:, ids] = torch.tensor([-.5369181, .0724743, -.4198892, -2.2566712,
                                    .6066388, 1.2242862, 1.2083784], device=env.device)
    robot.set_joint_position_target(q)
    env.action_manager.set_joint_position_replay_targets(targets, joint_ids=ids, env_ids=[0])
    samples = []
    with torch.inference_mode():
        for tick in range(200):
            env.step(torch.zeros(env.action_space.shape, device=env.device),
                     torch.zeros(1, dtype=torch.long, device=env.device))
            if tick % 20 == 0 or tick == 199:
                row = dict(tick=tick, actual=robot.data.joint_pos[0, ids].tolist(),
                           target=robot.data.joint_pos_target[0, ids].tolist(),
                           velocity=robot.data.joint_vel[0, ids].tolist(),
                           velocity_target=robot.data.joint_vel_target[0, ids].tolist(),
                           feedforward=robot.data.joint_effort_target[0, ids].tolist(),
                           applied=robot.data.applied_torque[0, ids].tolist(),
                           gravity=robot.root_physx_view.get_gravity_compensation_forces()[0, ids].tolist())
                samples.append(row)
                print('WRIST_HOLD', json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(scope='Unloaded isolated motor hold; not task success.',
                       joint_names=names, samples=samples), stream, indent=2)
    env.close()
except Exception:
    exit_code = 1
    import traceback
    traceback.print_exc()
finally:
    from isaaclab.sim import SimulationContext
    from workflow_status import close_with_status
    exit_code = close_with_status(exit_code, SimulationContext.clear_instance, app.close)
raise SystemExit(exit_code)
