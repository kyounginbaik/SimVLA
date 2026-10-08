#!/usr/bin/env python3
"""Re-export a legacy stage's gripper channels from its exact recorded HDF5 commands.

Never edits the original stage. Images are hard-linked read-only inputs; corrected
arrays and a source-hash provenance record are written in a new stage directory.
This is data recovery, not a new physical collection or a replay success claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/isaaclab"))


def correct_arrays(arrays, raw_commands, joint_names, robot):
    result = {name: value.copy() for name, value in arrays.items()}
    frames = len(result["action__eef_pos"])
    if raw_commands.shape != (frames + 1, 17) or not np.isfinite(raw_commands).all():
        raise ValueError("raw command shape/values do not match this stage's initial-frame removal")
    commands = raw_commands[1:, 12:14]
    if not np.isin(commands, [-1., 1.]).all():
        raise ValueError("expected exact signed binary gripper commands")
    result["action__eef_pos"][:, [9, 19]] = np.where(commands < 0, .1, -1.6)
    if robot == "rby1":
        names, opened, closed = ("gripper_finger_l1", "gripper_finger_r1"), -.04, 0.
    elif robot == "aiworker":
        names, opened, closed = ("gripper_l_joint1", "gripper_r_joint1"), 0., 1.1002
    else:
        raise ValueError("recovery is restricted to the affected RBY1 and AI Worker schemas")
    values = result["joint_angles"][:, [joint_names.index(name) for name in names]]
    result["obs__eef_pose"][:, 18:20] = -1.6 + 1.7 * (values - opened) / (closed - opened)
    return result


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    import h5py
    from isaaclab.utils.datasets.robot_schemas import get_robot_schema
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robot", required=True, choices=("rby1", "aiworker"))
    args = parser.parse_args()
    episodes = sorted(args.stage.glob("[0-9][0-9][0-9][0-9][0-9][0-9]/arrays.npz"))
    if not episodes or args.output.exists():
        raise ValueError("requires nonempty stage and a new output directory")
    corrections = []
    with h5py.File(args.hdf5) as hdf:
        for path in episodes:
            meta = json.loads((path.parent / "meta.json").read_text())
            demo = hdf[f"data/demo_{meta['episode_id']}"]
            if not meta.get("success") or not bool(demo.attrs.get("success")):
                raise ValueError("only accepted physical episodes may be recovered")
            with np.load(path) as arrays:
                corrected = correct_arrays(dict(arrays), demo["actions"][:],
                                           get_robot_schema(args.robot)["joint_angles"], args.robot)
            corrections.append((path, corrected))
    args.output.mkdir()
    for path, corrected in corrections:
        target = args.output / path.parent.name
        target.mkdir()
        np.savez_compressed(target / "arrays.npz", **corrected)
        shutil.copy2(path.parent / "meta.json", target / "meta.json")
        for image in path.parent.glob("observation.images.*.npy"):
            os.link(image, target / image.name)
    provenance = {"schema_version": 1, "operation": "gripper_channel_recovery",
                  "robot": args.robot, "source_hdf5": str(args.hdf5),
                  "source_hdf5_sha256": digest(args.hdf5),
                  "source_arrays": [{"path": str(p), "sha256": digest(p)} for p, _ in corrections],
                  "recovery_script_sha256": digest(Path(__file__)),
                  "output_arrays": [{"path": str(args.output / p.parent.name / "arrays.npz"),
                                     "sha256": digest(args.output / p.parent.name / "arrays.npz")}
                                    for p, _ in corrections],
                  "episodes": len(corrections), "new_physical_collection": False}
    (args.output / "recovery-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
