"""Check all three robots' runtime contact points against public visible pad meshes.

Uses Isaac's USD Python libraries but does not boot Kit or require a GPU. Supply
SIMVLA_ASSETS_DIR and SIMVLA_RBY1M_DIR, or the equivalent command-line paths.
This checks geometry, not friction, grasp stability, or a complete episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/simvla"))
import check_aiworker_urdf_vs_usd  # noqa: E402,F401 - USD library bootstrap/re-exec
from build_aiworker_spheres import link_points  # noqa: E402
from collector_profile import jaw_pad_offset  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def check(robot: str, usd: Path) -> list[dict]:
    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        raise ValueError(f"could not open {usd}")
    if robot == "aiworker":
        specs = [(f"gripper_{side}_rh_p12_rn_{finger}", 1, finger == "l2")
                 for side in "lr" for finger in ("r2", "l2")]
    elif robot == "rby1":
        specs = [(f"ee_finger_{side}{finger}", 0, False)
                 for side in "lr" for finger in (1, 2)]
    else:
        specs = [(f"gripper{side}{finger}", 0, finger == "L")
                 for side in (1, 2) for finger in "LR"]
    cache = UsdGeom.XformCache()
    rows = []
    for name, axis, positive in specs:
        bodies = [p for p in stage.Traverse() if p.GetName() == name]
        if len(bodies) != 1:
            raise ValueError(f"expected exactly one {name}, found {len(bodies)}")
        points = link_points(stage, cache, str(bodies[0].GetPath()))
        if points is None or not len(points):
            raise ValueError(f"no visible pad geometry for {name}")
        extreme = points[:, axis].max() if positive else points[:, axis].min()
        inner = points[np.abs(points[:, axis] - extreme) <= .0005]
        lower, upper = inner.min(axis=0), inner.max(axis=0)
        centre = (lower + upper) / 2
        configured = np.asarray(jaw_pad_offset(name))
        error = float(np.linalg.norm(centre - configured))
        rows.append({"robot": robot, "body": name, "mesh_inner_bounds_m": [lower.tolist(), upper.tolist()],
                     "measured_center_m": centre.tolist(), "configured_center_m": configured.tolist(),
                     "error_m": error, "valid": error <= .0001})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=os.environ.get("SIMVLA_ASSETS_DIR"))
    parser.add_argument("--rby1m-dir", type=Path, default=os.environ.get("SIMVLA_RBY1M_DIR"))
    parser.add_argument("--robot", choices=("all", "aiworker", "rby1", "anubis"), default="all")
    parser.add_argument("--output", type=Path, help="New JSON output; existing files are refused")
    args = parser.parse_args(argv)
    if not args.assets_dir:
        parser.error("set SIMVLA_ASSETS_DIR or --assets-dir")
    rby1m = args.rby1m_dir or args.assets_dir.parent / "rby1m"
    paths = {"aiworker": args.assets_dir / "Robots/MM/aiworker/ffw_sg2.usd",
             "anubis": args.assets_dir / "Robots/anubis_simvla.usd",
             "rby1": rby1m / "models/rby1m/urdf/model/model_simvla_black_gripper.usd"}
    robots = tuple(paths) if args.robot == "all" else (args.robot,)
    rows = [row for robot in robots for row in check(robot, paths[robot])]
    report = {"schema_version": 1, "valid": all(row["valid"] for row in rows),
              "source_usd_sha256": {robot: hashlib.sha256(paths[robot].read_bytes()).hexdigest()
                                    for robot in robots},
              "collector_profile_sha256": hashlib.sha256(
                  (REPO / "scripts/simvla/collector_profile.py").read_bytes()).hexdigest(),
              "pads": rows}
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(rendered)
        print(f"{len(rows)} pad frames; valid={report['valid']}; {args.output}")
    else:
        print(rendered, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
