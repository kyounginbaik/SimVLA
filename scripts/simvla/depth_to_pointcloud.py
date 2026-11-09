"""Turn replay depth captures (``--depth``) into coloured world-frame point clouds.

    python scripts/simvla/depth_to_pointcloud.py RUN/demo_replay/.../success1_wrist_right_depth.npz \
        --frames 0 600 1200 --out clouds/

Writes ``<npz stem>_<frame>.ply`` (binary little-endian, x y z float32 + r g b uint8). Colours come
from the matching ``<prefix>_<camera>.mp4`` next to the npz when it exists. Fisheye cameras are
refused (see rgbd_capture.py).
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rgbd_capture import unproject  # noqa: E402


def write_ply(path, points, colors):
    path = Path(path)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    record = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rows = np.empty(len(points), dtype=record)
    rows["x"], rows["y"], rows["z"] = np.asarray(points, dtype=np.float32).T
    rows["r"], rows["g"], rows["b"] = np.asarray(colors, dtype=np.uint8).T
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def rgb_frames(npz_path, frames):
    video = npz_path.with_name(npz_path.name.replace("_depth.npz", ".mp4"))
    if not video.is_file():
        return {}
    import imageio.v3 as iio
    wanted = set(frames)
    return {i: f[..., :3] for i, f in enumerate(iio.imiter(video)) if i in wanted}


def convert(npz_path, frames, out_dir, stride=1):
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    projection = str(data["projection"])
    count = len(data["depth_mm"])
    frames = list(range(count)) if frames is None else frames
    bad = [f for f in frames if not 0 <= f < count]
    if bad:
        raise SystemExit(f"{npz_path.name} has {count} frames; out of range: {bad}")
    colors = rgb_frames(npz_path, frames)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in frames:
        points, rows, cols = unproject(data["depth_mm"][f], data["intrinsics"], data["pos_w"][f],
                                       data["quat_w_ros"][f], projection, stride)
        rgb = colors[f][rows, cols] if f in colors else np.full((len(points), 3), 200, np.uint8)
        path = out_dir / f"{npz_path.stem}_{f:05d}.ply"
        write_ply(path, points, rgb)
        written.append((path, len(points)))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, nargs="+")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="Frame indices (default: all)")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth pixel in each direction")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for npz in args.npz:
        try:
            for path, n in convert(npz, args.frames, args.out, args.stride):
                print(f"{path} points={n}")
        except ValueError as error:
            print(f"skip {npz}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
