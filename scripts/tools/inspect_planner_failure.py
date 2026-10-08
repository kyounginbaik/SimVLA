"""Explain logged planner-start sphere overlaps without running the simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
from planner_diagnostics import overlapping_link_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--robot-config", required=True, type=Path)
    args = parser.parse_args()
    ignored = yaml.safe_load(args.robot_config.read_text())["robot_cfg"]["kinematics"]["self_collision_ignore"]
    count = 0
    with args.log.open() as stream:
        for line in stream:
            if not line.startswith("[planner-start] {"):
                continue
            payload = json.loads(line.removeprefix("[planner-start] "))
            print(json.dumps({"arm": payload["arm"], "step": payload["step"],
                              "status": payload["status"],
                              "joint_names": payload["joint_names"],
                              "joint_position": payload["joint_position"],
                              "sphere_overlaps": overlapping_link_pairs(payload, ignored)},
                             allow_nan=False))
            count += 1
    if not count:
        parser.error("no planner-start snapshots found; diagnostic runs log these on planning failure")


if __name__ == "__main__":
    main()
