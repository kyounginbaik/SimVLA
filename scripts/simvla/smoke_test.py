"""Bounded GPU validation: physics, or a task's robot state and RGB cameras."""
import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default=None, help='Optional registered task; requires its external assets')
parser.add_argument('--robot', choices=('anubis', 'aiworker', 'rby1'), default='anubis')
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducible task initialization')
parser.add_argument('--goal_file', type=Path, help='Task goals containing a collision-free robot spawn band')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--wrist-axis-preview', action='store_true',
                    help='Render extra +90-degree camera-local X/Y/Z views for both wrists')
parser.add_argument('--rby1-wrist-candidates', action='store_true',
                    help='Render ten paired above-gripper camera mounts without changing defaults')
parser.add_argument('--output_dir', type=Path, default=Path('outputs/smoke'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error('--steps and --num_envs must be positive')
if args.rby1_wrist_candidates and (args.robot != 'rby1' or not args.task):
    parser.error('--rby1-wrist-candidates requires a task and --robot rby1')
if args.task:
    import simvla_paths
    simvla_paths.validate_robot_model_file(args.robot)
    args.enable_cameras = True
app = AppLauncher(args).app

exit_code = 0
try:
    import torch
    import isaaclab.sim as sim_utils
    if args.task:
        import gymnasium as gym
        import isaaclab_tasks  # Register tasks from this checkout.
        from isaaclab_tasks.utils import parse_env_cfg
        import simvla_paths
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        from sink_collision import configure_sink_collisions
        configure_sink_collisions(cfg)
        simvla_paths.validate_robot_asset(args.robot, cfg.scene.robot.spawn.usd_path)
        cfg.seed = args.seed
        # Collection initializes from the authored free-space bands. Spawning at
        # the config's default origin can put the robot inside a refrigerator.
        goal_file = args.goal_file or simvla_paths.goals_dir() / f'{args.task}.json'
        metadata = json.loads(goal_file.read_text())
        band = metadata['initial_pos_ranges'][0]
        for axis, lower, upper in band:
            center = (lower + upper) / 2
            cfg.events.robot_init_pos.params['pose_range'][axis] = (center, center)
        cfg.observations.policy.concatenate_terms = False
        cfg.recorders = None
        cfg.eval_mode = True
        preview_rotations = {}
        if args.wrist_axis_preview:
            from wrist_axis_preview import add_wrist_axis_previews
            preview_rotations = add_wrist_axis_previews(cfg.scene)
        wrist_candidates = {}
        if args.rby1_wrist_candidates:
            from rby1_wrist_candidates import add_rby1_wrist_candidates
            wrist_candidates = add_rby1_wrist_candidates(cfg.scene)
        # A smoke check only needs stable physics, robot state, and cameras. Generated tasks may
        # add task-specific success/retry predicates that require collector-only runtime state;
        # evaluating those would turn this infrastructure check into a partial task execution.
        for term_name in vars(cfg.terminations):
            if not term_name.startswith("_"):
                setattr(cfg.terminations, term_name, None)
        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()
        goal_indices = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        robot = env.scene['robot']
        lift_ids = None
        lift_target = None
        lift_initial = None
        if args.robot == 'aiworker':
            lift_ids = robot.find_joints(['lift_joint'])[0]
            lift_target = robot.data.default_joint_pos[:, lift_ids].clone()
            lift_initial = robot.data.joint_pos[:, lift_ids].clone()
        with torch.inference_mode():
            for _ in range(args.steps):
                if lift_ids is not None:
                    robot.set_joint_position_target(lift_target, joint_ids=lift_ids)
                # This research fork also accepts scripted goal indices and returns recorder data.
                env.step(torch.zeros(env.action_space.shape, device=env.device), goal_indices)
                if not torch.isfinite(robot.data.joint_pos).all():
                    raise RuntimeError('Robot state contains non-finite joint positions')
        cameras = {}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio
        for name, sensor in env.scene.sensors.items():
            outputs = getattr(sensor.data, 'output', {})
            if 'rgb' in outputs:
                rgb = outputs['rgb'][0, :, :, :3].cpu().numpy()
                iio.imwrite(args.output_dir / f'{name}.png', rgb)
                cameras[name] = {'shape': list(rgb.shape), 'std': float(rgb.std())}
        if not cameras or any(v['std'] <= 1 for name, v in cameras.items()
                              if name not in preview_rotations):
            raise RuntimeError(f'Missing or nearly constant camera output: {cameras}')
        report = {'task': args.task, 'robot': args.robot, 'seed': args.seed,
                  'steps': args.steps, 'cameras': cameras}
        if preview_rotations:
            report['preview_camera_local_plus90_wxyz'] = preview_rotations
            (args.output_dir / 'camera-axis-preview.json').write_text(json.dumps(report, indent=2))
        if wrist_candidates:
            from isaaclab.utils.math import quat_apply
            for name, mount in wrist_candidates.items():
                link = 'ee_link1' if mount['side'] == 'right' else 'ee_link2'
                body_id = robot.find_bodies(link)[0][0]
                delta = quat_apply(robot.data.body_quat_w[0, body_id].unsqueeze(0),
                                   torch.tensor([mount['pos']], device=env.device))
                mount['measured_height_above_wrist_m'] = float(delta[0, 2])
                mount['above_wrist_at_render_pose'] = float(delta[0, 2]) > 0
            report['wrist_candidates'] = wrist_candidates
            (args.output_dir / 'wrist-candidates.json').write_text(json.dumps(report, indent=2))
        if lift_ids is not None:
            lift_final = robot.data.joint_pos[:, lift_ids]
            report['lift'] = {
                'target_m': float(lift_target[0, 0]),
                'initial_m': float(lift_initial[0, 0]),
                'final_m': float(lift_final[0, 0]),
                'error_m': float((lift_final - lift_target).abs().max()),
            }
        env.close()
    else:
        from isaaclab.assets import RigidObject, RigidObjectCfg
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        obj = RigidObject(RigidObjectCfg(
            prim_path='/World/Cube',
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1),
                                     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                     collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 2))))
        sim.reset()
        obj.update(0.01)
        z0 = float(obj.data.root_pos_w[0, 2])
        for _ in range(args.steps):
            sim.step()
            obj.update(0.01)
        z1 = float(obj.data.root_pos_w[0, 2])
        if not z1 < z0 or not torch.isfinite(obj.data.root_pos_w).all():
            raise RuntimeError(f'Gravity check failed: z={z0} -> {z1}')
        report = {'steps': args.steps, 'z0': z0, 'z1': z1}
        sim.clear_all_callbacks()
    print('SIMVLA_SMOKE_PASS ' + json.dumps(report), flush=True)
except BaseException as exc:
    print('SIMVLA_SMOKE_FAIL ' + repr(exc) + '\n' + traceback.format_exc(), flush=True)
    exit_code = 1
finally:
    from workflow_status import close_with_status
    from isaaclab.sim import SimulationContext
    exit_code = close_with_status(exit_code, SimulationContext.clear_instance, app.close)
raise SystemExit(exit_code)
