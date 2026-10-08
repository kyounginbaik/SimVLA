"""Drop-test the sink cavity. This explicitly initialized probe is NOT collection evidence."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--approximation", choices=("authored", "convexDecomposition"), default="authored")
parser.add_argument("--leaf-collisions", action="store_true",
                    help="Remove redundant collision APIs from sink Xforms, retaining all shape colliders")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
enable_cameras = bool(getattr(args, "enable_cameras", False))
app = AppLauncher(args).app
exit_code = 0
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    import omni.usd
    from pxr import UsdGeom, UsdPhysics
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    cfg = parse_env_cfg("Isaac-Kitchen-v813a-00", device=args.device, num_envs=1)
    cfg.seed = 0
    cfg.recorders = None
    cfg.eval_mode = True
    cfg.observations.policy.concatenate_terms = False
    if not enable_cameras:
        for name, sensor in vars(cfg.scene).items():
            if isinstance(sensor, (CameraCfg, TiledCameraCfg)):
                setattr(cfg.scene, name, None)
        # The physics-only app has no Replicator execution graph. Skip visual
        # randomization, but retain every physical material/reset event.
        for name, event in vars(cfg.events).items():
            if getattr(getattr(event, "func", None), "__name__", "") == "randomize_visual_color":
                setattr(cfg.events, name, None)
    cfg.decimation = 6
    cfg.sim.dt = 1 / 120
    cfg.sim.render_interval = 6
    cfg.events.robot_init_pos.params["pose_range"].update(x=(.75, .75), y=(-1.7, -1.7), yaw=(0., 0.))
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    original_spawn = cfg.scene.kitchen.spawn.func
    changed = []
    removed_xform_apis = []
    def spawn_kitchen(*spawn_args, **spawn_kwargs):
        result = original_spawn(*spawn_args, **spawn_kwargs)
        if args.approximation != "authored" or args.leaf_collisions:
            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                if "/sink_cabinet/corpus/" in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh) and args.approximation != "authored":
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(args.approximation)
                    changed.append(str(prim.GetPath()))
                if (args.leaf_collisions and "/sink_cabinet" in str(prim.GetPath())
                        and prim.IsA(UsdGeom.Xform) and prim.HasAPI(UsdPhysics.CollisionAPI)):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    removed_xform_apis.append(str(prim.GetPath()))
        return result
    cfg.scene.kitchen.spawn.func = spawn_kitchen
    env = gym.make("Isaac-Kitchen-v813a-00", cfg=cfg).unwrapped
    env.reset()
    stage = env.sim.stage
    sink = stage.GetPrimAtPath("/World/envs/env_0/Kitchen/sink_cabinet/corpus/sink")
    bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                 UsdGeom.Tokens.proxy]).ComputeWorldBound(sink).ComputeAlignedBox()
    lower, upper = list(bounds.GetMin()), list(bounds.GetMax())
    center = [(lower[i] + upper[i]) / 2 for i in range(3)]
    mug = env.scene["mug0"]
    initial = mug.data.default_root_state.clone()
    initial[:, :3] = torch.tensor([[center[0], center[1], upper[2] + .30]], device=env.device)
    initial[:, 7:] = 0
    # Only the probe's initialization writes object state, never a manipulation run.
    mug.write_root_state_to_sim(initial)
    samples = []
    robot = env.scene["robot"]
    lift = robot.find_joints(["lift_joint"])[0]
    with torch.inference_mode():
        for tick in range(160):
            robot.set_joint_position_target(robot.data.default_joint_pos[:, lift], joint_ids=lift)
            env.step(torch.zeros(env.action_space.shape, device=env.device),
                     torch.zeros(1, dtype=torch.long, device=env.device))
            if tick % 20 == 0 or tick == 159:
                samples.append({"tick": tick, "mug_root": mug.data.root_pos_w[0].tolist()})
    report = {"scope": "Isolated initialized sink-drop physics probe, not a successful robot episode",
              "approximation": args.approximation, "changed_meshes": changed,
              "removed_xform_collision_apis": removed_xform_apis,
              "sink_bounds": [lower, upper], "samples": samples,
              "root_below_sink_rim": samples[-1]["mug_root"][2] < upper[2] - .03}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print("SINK_DROP_PROBE " + json.dumps(report), flush=True)
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
