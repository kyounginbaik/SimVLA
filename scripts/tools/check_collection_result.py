#!/usr/bin/env python3
"""Record collection/export outcomes, including empty or failed runs.

Passing export integrity is not proof of physical task success or an independent
reproduction. Those remain separate release checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_lerobot_collection import validate


def check(root: Path, collector_exit: int, exporter_exit: int = 0, expected_episodes: int = 1) -> dict:
    if type(expected_episodes) is not int or expected_episodes < 1:
        raise ValueError("expected_episodes must be a positive integer")
    if not root.is_dir():
        raise ValueError(f"collection root does not exist: {root}")
    staged = sorted((root / "raw").rglob("arrays.npz"))
    report = {
        "schema_version": 1,
        "collector_exit": collector_exit,
        "exporter_exit": exporter_exit,
        "expected_episodes": expected_episodes,
        "staged_episodes": len(staged),
        "validated_episodes": 0,
        "validated_frames": 0,
        "datasets": [],
        "errors": [],
        "scope": "Export integrity only; physical success and independent reproduction require separate evidence.",
    }
    if collector_exit:
        report["errors"].append(f"collector exited with status {collector_exit}")
    if exporter_exit:
        report["errors"].append(f"exporter exited with status {exporter_exit}")
    if not staged:
        report["errors"].append("no staged episodes (arrays.npz)")
    datasets = sorted(path.parent.parent for path in (root / "lerobot").rglob("meta/info.json")
                      if path.parent.parent.name == "all")
    if not datasets:
        report["errors"].append("no finalized all dataset")
    for dataset in datasets:
        entry = {"path": str(dataset.relative_to(root)), "valid": False}
        try:
            result = validate(dataset)
            entry.update(valid=True, counts=result)
            report["validated_episodes"] += result["episodes"]
            report["validated_frames"] += result["frames"]
        except Exception as exc:  # Preserve an actionable result even on a broken export/dependency.
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"].append(f"{entry['path']}: {entry['error']}")
        report["datasets"].append(entry)
    if staged and report["validated_episodes"] != len(staged):
        report["errors"].append("staged/finalized episode count mismatch")
    if report["validated_episodes"] < expected_episodes:
        report["errors"].append(
            f"incomplete collection: requested {expected_episodes} episodes, "
            f"validated {report['validated_episodes']}"
        )
    report["passed"] = not report["errors"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--collector-exit", type=int, default=0)
    parser.add_argument("--exporter-exit", type=int, default=0)
    parser.add_argument("--expected-episodes", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = check(args.run_root, args.collector_exit, args.exporter_exit, args.expected_episodes)
        output = args.run_root / "collection-result.json"
        # Never replace prior evidence. Choose a new run directory for a rerun.
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError) as exc:
        parser.exit(1, f"collection-result: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
