#!/usr/bin/env python3
"""Write a machine-readable record tying release checks to a source revision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.rstrip("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(root: Path) -> dict:
    """Fingerprint finalized input bytes, not just the dataset's shape metadata.

    Use a closed, materialized dataset. Symlinks/special files are rejected so
    provenance cannot silently omit or follow data outside the selected tree.
    The content digest is relocation-independent; no file contents are emitted.
    """
    if root.is_symlink():
        raise ValueError(f"dataset root must not be a symlink: {root}")
    root = root.resolve()
    if not (root / "meta/info.json").is_file():
        raise ValueError(f"dataset is missing meta/info.json: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"dataset contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"dataset contains a special file: {path}")
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"dataset changed while hashing: {path}")
        records.append({"path": path.relative_to(root).as_posix(),
                        "bytes": after.st_size, "sha256": digest})
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root), "files": records,
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
            "scope": "Exact finalized input bytes, including videos; not proof of physical success or bitwise simulation determinism."}


RUNTIME_PACKAGES = {
    "simvla": "simvla", "torch": "torch", "numpy": "numpy",
    "curobo": "nvidia-curobo", "isaacsim": "isaacsim",
    "isaaclab": "isaaclab", "lerobot": "lerobot", "av": "av",
    "pyarrow": "pyarrow", "datasets": "datasets", "xatlas": "xatlas",
}
# Deliberately not a dump of the process environment: tokens and credentials
# must never end up in downloadable dataset provenance.
RUNTIME_SETTINGS = (
    "SIMVLA_GRIPPER_BALANCE_STEP_FRACTION",
    "SIMVLA_CUROBO_WORLD", "SIMVLA_CUROBO_TARGET_COLLISION",
    "SIMVLA_AIWORKER_GRASP_ROUTE", "SIMVLA_GRASP_CANDIDATE_INDEX",
    "SIMVLA_DIRECT_APPROACH_OFFSET_M", "SIMVLA_RIGHT_APPROACH_OFFSET_M",
    "SIMVLA_DIRECT_TRACK_POS_TOL", "SIMVLA_DIRECT_TRACK_ROT_TOL_DEG",
    "SIMVLA_DIRECT_TRACK_TIMEOUT", "SIMVLA_GRASP_GEOMETRY",
    "SIMVLA_RIGHT_TRACK_POS_TOL", "SIMVLA_RIGHT_TRACK_ROT_TOL_DEG",
    "SIMVLA_MUG_GRASP_HEIGHT", "SIMVLA_MAXFRAMES", "SIMVLA_EPISODE_STEPS",
    "SIMVLA_REVIEW_CAMERA", "SIMVLA_REVIEW_SECONDARY_CAMERA",
    "SIMVLA_REVIEW_EVERY",
    "SIMVLA_GRASP_POS_TOL", "SIMVLA_GRASP_ROT_TOL_DEG",
    "SIMVLA_POSTGRASP_LIFT", "SIMVLA_POSTGRASP_LIFT_LEFT",
    "SIMVLA_FIXED_OBJECT", "SIMVLA_PLANNER_ATTEMPTS",
    "SIMVLA_AIWORKER_URDF_PATH",
    "SIMVLA_AIWORKER_CONTACT_DEPTH_M",
    "SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S", "SIMVLA_AIWORKER_LIFT_M",
    "SIMVLA_AIWORKER_WRIST_STIFFNESS",
    "SIMVLA_RESET_POS_TOL_M", "SIMVLA_RESET_ROT_TOL_DEG",
    "SIMVLA_GRIPPER_PRELOAD_FRACTION", "SIMVLA_GRIPPER_FORCE_BALANCE",
    "SIMVLA_GRIPPER_FREEZE_AT_FORCE",
    "SIMVLA_TRAJOPT_SEEDS", "SIMVLA_GRAPH_SEEDS",
    "SIMVLA_FINETUNE_TRAJOPT",
    "SIMVLA_REQUIRE_CUROBO_PLAN",
    "SIMVLA_TRACK_LOADED_HOME",
    "SIMVLA_LOADED_HOME_TRACK_POS_TOL",
    "SIMVLA_LOADED_HOME_TRACK_ROT_TOL_DEG",
    "SIMVLA_TRACE_LIFT",
    "SIMVLA_PRINT_TERMS", "SIMVLA_SIMVQA", "SIMVLA_REPLAY_PASSES", "SIMVLA_REPLAY_BASE_TRACKING",
    "SIMVLA_REPLAY_JOINT_TARGETS", "SIMVLA_REPLAY_EPISODE_INDEX",
    "SIMVLA_REPLAY_ADAPTIVE_GRIPPER",
    "SIMVLA_REPLAY_INITIAL_JOINT_STATE",
    "SIMVLA_REPLAY_INITIAL_STEP",
    "SIMVLA_REPLAY_SETTLE_STEPS",
    "SIMVLA_RECORD_JOINT_SUBSTEPS", "SIMVLA_REPLAY_JOINT_SUBSTEPS",
    "SIMVLA_AIWORKER_PAYLOAD_COLLISION",
    "SIMVLA_SEED",
    "SIMVLA_NUM_DEMOS",
    "SIMVLA_LOADED_HOME_JOINT_TARGETS",
    "SIMVLA_TASK",
    "SIMVLA_CPU_THREADS",
    "SIMVLA_PHYSICS_SUBSTEPS",
    "SIMVLA_SINK_CAVITY_COLLISIONS",
    "SIMVLA_KITCHEN813_SINK_INTERIOR_GATE",
    "SIMVLA_PREGRASP_STANDOFF", "SIMVLA_PREGRASP_ABOVE",
    "SIMVLA_NAV_GOAL_DISTANCE", "SIMVLA_NAV_GOAL_YAW", "SIMVLA_NAV_FINAL_YAW",
    "SIMVLA_GOAL_NAV_XY_STD_M", "SIMVLA_GOAL_NAV_YAW_STD_RAD", "SIMVLA_GOAL_ARM_XYZ_STD_M",
    "SIMVLA_NAV_MIN_SPEED", "SIMVLA_NAV_DEFAULT_SPEED", "SIMVLA_NAV_SLOWDOWN_RADIUS",
    "SIMVLA_NAV_MIN_YAW_SPEED",
    "SIMVLA_NAV_YAW_RATE_MAX", "SIMVLA_NAV_YAW_SLEW", "SIMVLA_NAV_LIN_SLEW",
    "SIMVLA_GRIPPER_TARGET_FORCE_N", "SIMVLA_GRIPPER_MAX_FORCE_N",
    "SIMVLA_RBY1_GRIPPER_STIFFNESS",
    "SIMVLA_EXPORT_PYTHON",
)


def runtime_environment() -> dict:
    """Report import resolution without importing GPU/simulator modules."""
    packages = {}
    for module, distribution in RUNTIME_PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            spec = importlib.util.find_spec(module)
            origin = spec.origin if spec else None
            paths = list(spec.submodule_search_locations or ()) if spec else []
        except (ImportError, ValueError):
            origin, paths = None, []
        packages[module] = {"version": version, "origin": origin, "search_paths": paths}
    return {
        "executable": sys.executable,
        "packages": packages,
        "settings": {key: os.environ[key] for key in RUNTIME_SETTINGS if key in os.environ},
    }


def collect(root: Path, artifacts: Sequence[Path] = (), *, runtime: bool = False,
            datasets: Sequence[Path] = ()) -> dict:
    root = root.resolve()
    has_git = (root / ".git").exists()
    if not has_git and not ((root / "pyproject.toml").is_file()
                            and (root / "src/simvla").is_dir()):
        raise ValueError("repo root must contain Git metadata or an extracted SimVLA source archive")
    dirty = ([line for line in _git(root, "status", "--porcelain=v1").splitlines() if line]
             if has_git else [])
    # Identify tested dirty code without publishing its diff or any credentials.
    diff_hash = (hashlib.sha256(_git(root, "diff", "--binary", "HEAD").encode()).hexdigest()
                 if has_git else None)
    untracked_sources = []
    source_dirs = {"src", "scripts", "source", "configs", "examples", "tests", "docs", "env"}
    source_suffixes = {".py", ".json", ".yaml", ".yml", ".toml", ".sh", ".sbatch", ".md", ".env"}
    untracked_names = (_git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
                       if has_git else [])
    for name in sorted(filter(None, untracked_names)):
        relative = Path(name)
        path = root / relative
        if relative.parts[0] in source_dirs and relative.suffix in source_suffixes and path.is_file():
            untracked_sources.append({"path": name, "sha256": _sha256(path)})
    archive_sources = []
    if not has_git:
        # Archives have no trustworthy commit or clean/dirty state. Fingerprint
        # their source content instead, never borrowing a containing repo's HEAD.
        archive_suffixes = source_suffixes | {".txt", ".in", ".cff"}
        candidates = list(root.iterdir())
        for directory in sorted(source_dirs | {"apps"}):
            candidates.extend((root / directory).rglob("*"))
        for path in sorted(set(candidates)):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in archive_suffixes and "__pycache__" not in path.parts):
                archive_sources.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
    artifact_records = []
    for value in artifacts:
        path = value if value.is_absolute() else root / value
        if not path.is_file():
            raise ValueError(f"artifact is not a file: {value}")
        try:
            name = str(path.resolve().relative_to(root))
        except ValueError:
            name = str(path.resolve())
        artifact_records.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    report = {
        "schema_version": 1,
        "source": {
            "kind": "git" if has_git else "archive",
            "commit": _git(root, "rev-parse", "HEAD") if has_git else None,
            "tree": _git(root, "rev-parse", "HEAD^{tree}") if has_git else None,
            "commit_time": _git(root, "show", "-s", "--format=%cI", "HEAD") if has_git else None,
            "dirty": bool(dirty) if has_git else None,
            "changes": dirty,
            "tracked_diff_sha256": diff_hash,
            "untracked_source_files": untracked_sources,
            "submodules": _git(root, "submodule", "status", "--recursive").splitlines() if has_git else [],
            "archive_source_files": archive_sources,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "artifacts": artifact_records,
    }
    if runtime:
        report["environment"]["runtime"] = runtime_environment()
    if datasets:
        report["datasets"] = [dataset_fingerprint(path if path.is_absolute() else root / path)
                              for path in datasets]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--dataset", action="append", type=Path, default=[],
                        help="Hash every file of a closed, materialized LeRobot dataset (including videos)")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    parser.add_argument("--runtime", action="store_true",
                        help="Record package locations/versions and allowlisted collection settings")
    parser.add_argument(
        "--require-clean", action="store_true", help="Fail after reporting a dirty source tree"
    )
    args = parser.parse_args(argv)
    try:
        dataset_paths = [(path if path.is_absolute() else args.repo_root / path).absolute()
                         for path in args.dataset]
        if args.output and any(args.output.resolve().is_relative_to(path.resolve())
                               for path in dataset_paths):
            raise ValueError("evidence output must be outside the fingerprinted dataset")
        report = collect(args.repo_root, args.artifact, runtime=args.runtime, datasets=dataset_paths)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        if args.require_clean and report["source"]["dirty"] is not False:
            print("release evidence requires a verified clean Git source tree", file=sys.stderr)
            return 1
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release-evidence: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
