"""Small, explicit entry points for independently usable workflows."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import __version__
from .task_template import TaskTemplateError
from .task_validate import SequenceError
from .scene_spec import SceneError
from .predicate_contract import SpecError

WORKFLOWS = {
    "generate": "generate_example.py",
    "goals": "task_emit.py",
    "identify": "systemid.py",
    "smoke": "smoke_test.py",
    "compose": "kitchen_scene_generator.py",
    "collect": "simvla_gen.py",
    "replay": "simvla_replay.py",
    "evaluate": "simvla_eval.py",
}


def _robot_assets(robot: str, root: Path) -> dict[str, str]:
    assets = Path(os.environ.get("SIMVLA_ASSETS_DIR", root / "source/isaaclab_assets/data"))
    models = Path(os.environ.get("SIMVLA_ROBOT_MODELS_DIR", root))
    if robot == "anubis":
        planner = assets / "curobo/robot"
        expected = {
            "usd": assets / "Robots/anubis_simvla.usd",
            "urdf": models / "anubis/anubis_final.urdf",
            "curobo_left": planner / "anubis_left_arm.yml",
            "curobo_right": planner / "anubis_right_arm.yml",
        }
    elif robot == "aiworker":
        expected = {
            "usd": assets / "Robots/MM/aiworker/ffw_sg2.usd",
            "urdf": Path(os.environ.get("SIMVLA_AIWORKER_URDF_PATH",
                models / "ai_worker_min/ffw_description/urdf/ffw_sg2_follower/ffw_sg2_follower.urdf")),
            "curobo_left": root / "configs/curobo/robot/aiworker_left_arm.yml",
            "curobo_right": root / "configs/curobo/robot/aiworker_right_arm.yml",
            "spheres_left": root / "configs/curobo/robot/spheres/aiworker_spheres_left.yml",
            "spheres_right": root / "configs/curobo/robot/spheres/aiworker_spheres_right.yml",
        }
    elif robot == "rby1":
        model_root = Path(os.environ.get("SIMVLA_RBY1M_DIR", models / "rby1m"))
        expected = {
            "usd": model_root / "models/rby1m/urdf/model/model_simvla_black_gripper.usd",
            "urdf": model_root / "models/rby1m/urdf/model.urdf",
            "curobo_left": root / "configs/curobo/robot/rby1_left_arm.yml",
            "curobo_right": root / "configs/curobo/robot/rby1_right_arm.yml",
            "spheres_left": root / "configs/curobo/robot/spheres/rby1_spheres_left.yml",
            "spheres_right": root / "configs/curobo/robot/spheres/rby1_spheres_right.yml",
        }
    else:
        raise ValueError(f"No asset manifest for robot {robot!r}; supported: anubis, aiworker, rby1")
    return {name: str(path) for name, path in expected.items()}


def doctor(*, root: Path | None = None, robot: str | None = None) -> dict:
    versions = {}
    for name in ("simvla", "scene-synthesizer", "torch", "isaacsim", "isaaclab"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
        gpu = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    planner_ready = False
    try:
        curobo = importlib.util.find_spec("curobo")
        if curobo and curobo.submodule_search_locations:
            helper = Path(next(iter(curobo.submodule_search_locations))) / "util/usd_helper.py"
            planner_ready = helper.is_file() and "get_obstacles_from_stage_simvla" in helper.read_text()
    except (ImportError, OSError, ValueError):
        pass
    report = {"python": sys.version.split()[0], "packages": versions, "gpu": gpu,
              "simulator_ready": bool(gpu and versions["isaacsim"] and versions["isaaclab"]),
              "planner_ready": planner_ready,
              "note": "Package presence is not an end-to-end simulator check; planner_ready checks the SimVLA cuRobo patch."}
    if robot:
        expected = _robot_assets(robot, (root or Path.cwd()).resolve())
        missing = [name for name, path in expected.items() if not Path(path).is_file()]
        report["robot"] = {"name": robot, "files": expected, "missing": missing,
                           "ready": not missing}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SimVLA: scenes, tasks and simulation workflows")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Inspect dependencies and GPU availability without booting Isaac Sim")
    d.add_argument("--require-sim", action="store_true")
    d.add_argument("--robot", choices=["anubis", "aiworker", "rby1"],
                   help="Also check this robot's required assets")
    d.add_argument("--repo-root", type=Path, default=None)
    s = sub.add_parser("scene", help="Generate a procedural kitchen GLB on CPU")
    s.add_argument("--layout", default="single_wall",
                   choices=["single_wall", "l_shaped", "u_shaped", "island", "peninsula"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("demo", help="Build a visible CPU quickstart bundle in one command")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--layout", default="single_wall",
                      choices=["single_wall", "l_shaped", "u_shaped", "island", "peninsula"])
    demo.add_argument("--seed", type=int, default=0)
    sub.add_parser("templates", help="List bundled task templates")
    i = sub.add_parser("init-task", help="Create an editable task JSON from a bundled template")
    i.add_argument("--from", dest="source", default="bowl_to_drawer",
                   help="Bundled template to copy (default: bowl_to_drawer)")
    i.add_argument("--name", help="Name for the new task (defaults to the output filename)")
    i.add_argument("--output", type=Path, required=True)
    k = sub.add_parser("skills", help="Inspect available skill APIs on CPU")
    k.add_argument("skill_id", nargs="?", help="Show one skill's actions and parameters")
    t = sub.add_parser("validate", help="Validate a bundled template or task template JSON on CPU")
    t.add_argument("template")
    g = sub.add_parser("validate-goal", help="Preflight a generated simulator goal JSON on CPU")
    g.add_argument("goal", type=Path)
    a = sub.add_parser("adapt-aiworker", help="Adapt emitted kitchen configs for AI Worker on CPU")
    a.add_argument("--kitchen", required=True, type=int)
    a.add_argument("--subs", default="", help="Comma-separated rotations; default: all on disk")
    a.add_argument("--goals-dir", type=Path, help="Rename authored goals to AI Worker task ids")
    a.add_argument("--repo-root", type=Path, default=None)
    rb = sub.add_parser("adapt-rby1", help="Adapt emitted kitchen configs for RB-Y1 on CPU")
    rb.add_argument("--kitchen", required=True, type=int)
    rb.add_argument("--subs", default="", help="Comma-separated rotations; default: all on disk")
    rb.add_argument("--goals-dir", type=Path, help="Rename authored goals to RB-Y1 task ids")
    rb.add_argument("--repo-root", type=Path, default=None)
    v = sub.add_parser("vqa", help="Expand captured SimVQA records into training Q/A examples on CPU")
    v.add_argument("input", type=Path)
    v.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run", help="Launch a checkout's Isaac Sim workflow with the current Python")
    r.add_argument("workflow", choices=WORKFLOWS)
    r.add_argument("--repo-root", type=Path, default=None)
    r.add_argument("--dry-run", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if rest and args.command != "run":
        parser.error("unrecognized arguments: " + " ".join(rest))
    try:
        if args.command == "doctor":
            report = doctor(root=args.repo_root, robot=args.robot)
            print(json.dumps(report, indent=2))
            ready = report["simulator_ready"] and (not args.robot or report["robot"]["ready"])
            return int(args.require_sim and not ready)
        if args.command == "scene":
            from .scenes import export_kitchen
            print(export_kitchen(args.output, layout=args.layout, seed=args.seed))
        elif args.command == "demo":
            from . import skills as _skills  # noqa: F401
            from .scenes import export_kitchen
            from .skill_contract import REGISTRY
            from .tasks import load_template
            from .task_template import to_json

            output = args.output_dir.resolve()
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging_name:
                staging = Path(staging_name)
                scene = staging / "kitchen.glb"
                task_path = staging / "task.json"
                manifest_path = staging / "manifest.json"
                export_kitchen(scene, layout=args.layout, seed=args.seed)
                task = load_template("bowl_to_drawer")
                task.name = "quickstart_bowl_to_drawer"
                task_path.write_text(to_json(task) + "\n", encoding="utf-8")
                manifest = {
                    "simvla": __version__, "layout": args.layout, "seed": args.seed,
                    "task": task.name, "steps": len(task.steps), "registered_skills": len(REGISTRY),
                    "artifacts": ["kitchen.glb", "task.json"],
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                staging.rename(output)
            print(json.dumps({**manifest, "output_dir": str(output)}, indent=2))
        elif args.command == "templates":
            from .tasks import list_templates
            print("\n".join(list_templates()))
        elif args.command == "init-task":
            from .tasks import load_template
            from .task_template import to_json
            task = load_template(args.source)
            task.name = args.name or args.output.stem
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(to_json(task) + "\n")
            print(json.dumps({"template": args.source, "name": task.name,
                              "output": str(args.output)}))
        elif args.command == "skills":
            from . import skills as _skills  # Register skill declarations without loading Isaac Sim.
            from .skill_contract import REGISTRY
            if args.skill_id is None:
                print("\n".join(sorted(REGISTRY)))
            else:
                spec = REGISTRY.get(args.skill_id)
                if spec is None:
                    raise ValueError(f"Unknown skill {args.skill_id!r}; run 'simvla skills' to list them")
                params = []
                for param in spec.params:
                    item = {"name": param.name, "type": type(param).__name__}
                    if hasattr(param, "options"):
                        item["options"] = param.options
                    if hasattr(param, "default"):
                        default = param.default
                        if isinstance(default, float) and not math.isfinite(default):
                            default = None
                        item["default"] = default
                    params.append(item)
                print(json.dumps({"id": spec.id, "label": spec.label,
                                  "actions": spec.actions, "params": params,
                                  "runtime": spec.is_runtime}, indent=2, allow_nan=False))
        elif args.command == "validate":
            from .tasks import load_template, template_warnings
            task = load_template(args.template)
            print(json.dumps({"name": task.name, "steps": len(task.steps),
                              "warnings": template_warnings(task), "valid": True}, indent=2))
        elif args.command == "validate-goal":
            from .goal_validate import validate_goal
            print(json.dumps(validate_goal(args.goal), indent=2))
        elif args.command == "adapt-aiworker":
            root = (args.repo_root or Path(os.environ.get("SIMVLA_REPO_ROOT", "."))).resolve()
            script = root / "scripts/simvla/aiworker_kitchen_cfg.py"
            if not script.is_file():
                raise ValueError("AI Worker adaptation requires a source checkout; pass --repo-root PATH")
            cmd = [sys.executable, str(script), "--kitchen", str(args.kitchen)]
            if args.subs:
                cmd.extend(("--subs", args.subs))
            if args.goals_dir:
                cmd.extend(("--goals-dir", str(args.goals_dir.resolve())))
            env = dict(os.environ, SIMVLA_REPO_ROOT=str(root))
            return subprocess.call(cmd, cwd=root, env=env)
        elif args.command == "adapt-rby1":
            root = (args.repo_root or Path(os.environ.get("SIMVLA_REPO_ROOT", "."))).resolve()
            script = root / "scripts/simvla/rby1_kitchen_cfg.py"
            if not script.is_file():
                raise ValueError("RB-Y1 adaptation requires a source checkout; pass --repo-root PATH")
            cmd = [sys.executable, str(script), "--kitchen", str(args.kitchen)]
            if args.subs:
                cmd.extend(("--subs", args.subs))
            if args.goals_dir:
                cmd.extend(("--goals-dir", str(args.goals_dir.resolve())))
            env = dict(os.environ, SIMVLA_REPO_ROOT=str(root))
            return subprocess.call(cmd, cwd=root, env=env)
        elif args.command == "vqa":
            from .vqa import convert
            count = convert(args.input, None, args.output, None)
            print(json.dumps({"examples": count, "output": str(args.output)}))
        elif args.command == "run":
            root = (args.repo_root or Path(os.environ.get("SIMVLA_REPO_ROOT", "."))).resolve()
            script = root / "scripts" / "simvla" / WORKFLOWS[args.workflow]
            if not script.is_file():
                raise ValueError("Simulator workflows require a source checkout; pass --repo-root PATH")
            cmd = [sys.executable, str(script), *(rest[1:] if rest[:1] == ["--"] else rest)]
            if args.dry_run:
                print(json.dumps({"cwd": str(root), "argv": cmd}, indent=2))
                return 0
            if not doctor()["simulator_ready"]:
                raise RuntimeError("Simulator prerequisites unavailable. Run simvla doctor and see docs/simulation.md")
            env = dict(os.environ, SIMVLA_REPO_ROOT=str(root))
            # Run the selected checkout's fork, even when another Isaac Lab is installed.
            source_paths = [str(root / "source" / name) for name in
                            ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")]
            inherited = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(source_paths + ([inherited] if inherited else []))
            if args.workflow == "compose":
                return subprocess.call(cmd, cwd=root, env=env)
            # Kit shutdown can terminate a failed process with status zero. Receive
            # the workflow result before teardown and reject missing completion.
            preserved_status = os.environ.get("SIMVLA_PRESERVE_STATUS_FILE")
            if preserved_status:
                status = Path(preserved_status).resolve()
                status.parent.mkdir(parents=True, exist_ok=True)
                status.unlink(missing_ok=True)
                env["SIMVLA_STATUS_FILE"] = str(status)
                code = subprocess.call(cmd, cwd=root, env=env)
                if code:
                    return code
                if not status.is_file():
                    raise RuntimeError("Simulator exited without a completion result; inspect its log")
                return int(json.loads(status.read_text())["exit_code"])
            with tempfile.TemporaryDirectory(prefix="simvla-status-") as directory:
                status = Path(directory) / "result.json"
                env["SIMVLA_STATUS_FILE"] = str(status)
                code = subprocess.call(cmd, cwd=root, env=env)
                if code:
                    return code
                if not status.is_file():
                    raise RuntimeError("Simulator exited without a completion result; inspect its log")
                return int(json.loads(status.read_text())["exit_code"])
    except (OSError, ValueError, RuntimeError, TaskTemplateError, SequenceError, SceneError, SpecError) as exc:
        print(f"simvla: {exc}", file=sys.stderr)
        return 1
    return 0
