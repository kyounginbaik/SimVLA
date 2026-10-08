#!/usr/bin/env python3
"""Export successful staged episodes in a separate, non-Isaac Python environment.

Use the directory containing 000000/arrays.npz, not the run's top-level raw/
directory. Collection uses --skip_finalize. This script never uploads data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--goal", required=True, type=Path)
    parser.add_argument("--robot", required=True, choices=("anubis", "rby1", "aiworker"))
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    if args.fps < 1:
        parser.error("fps must be positive")
    if not args.stage.is_dir() or not list(args.stage.glob("*/arrays.npz")):
        parser.error("stage must contain at least one episode directory with arrays.npz")
    if not args.goal.is_file():
        parser.error("goal JSON does not exist")
    if args.output.exists():
        parser.error("output already exists; use a new path to preserve prior data")
    # Only the checkout's pure data utilities are used; never import/start Kit.
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts/simvla"),
                    str(root / "source/isaaclab")]
    from isaaclab.utils.datasets import TwoPhaseEpisodeWriter
    from validate_lerobot_collection import validate

    writer = TwoPhaseEpisodeWriter(str(args.stage), robot=args.robot)
    writer.finalize_lerobot(
        output_path=str(args.output), task_json=str(args.goal), fps=args.fps,
        export_all=True, export_subtasks=False, export_groups=None,
        repo_prefix="local", push=False, image_writer_processes=0,
        image_writer_threads=4,
    )
    result = validate(args.output / "all")
    print(f"Export verified: {result}. This verifies data integrity, not physical task success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
