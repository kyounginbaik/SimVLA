"""Review videos from a replay run with ``--depth --seg_video`` (see rgbd_capture.py).

    # 4x3 grid: rows RGB / segmentation / depth / point cloud; columns head, left wrist, right wrist.
    # Also writes success1_<camera>_depth.mp4 (colour-coded depth) beside the inputs.
    python scripts/simvla/replay_videos.py tiled RUN/demo_replay/<robot>/<task>/<start>_<end> out.mp4

    # Head + wrist clouds fused, seen from behind and above the head camera as the robot moves;
    # left half RGB-coloured, right half segmentation-coloured.
    python scripts/simvla/replay_videos.py follow RUN/demo_replay/<robot>/<task>/<start>_<end> out.mp4

The head camera is a fisheye, so its clouds come from front_pinhole, the replay-only pinhole camera
on the same mount. Rendering is plain numpy point splatting with a z-buffer (no GPU, no Open3D).
"""
import argparse
from pathlib import Path
import sys

import imageio
import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rgbd_capture import FRONT_TWIN, quat_wxyz_to_matrix, unproject  # noqa: E402

TILE_W, TILE_H = 320, 240
BACKGROUND = 245
#: (column camera, camera whose depth makes the cloud, title)
COLUMNS = (("front", FRONT_TWIN, "Front (head)"), ("wrist_left", "wrist_left", "Left wrist"),
           ("wrist_right", "wrist_right", "Right wrist"))
#: Depth colour range in metres; wrist cameras work at tens of centimetres.
DEPTH_RANGE = {"front": (0.1, 2.0), "wrist_left": (0.05, 0.8), "wrist_right": (0.05, 0.8)}
#: Virtual viewpoint per tile: metres behind the camera, metres above it, focal length in px.
CLOUD_VIEW = {"front": (0.35, 0.20, 230.0), "wrist_left": (0.15, 0.08, 300.0),
              "wrist_right": (0.15, 0.08, 300.0)}


def turbo(x):
    """Polynomial approximation of the Turbo colormap; x in [0, 1] -> uint8 RGB."""
    x = np.clip(x, 0, 1)[..., None]
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.concatenate([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def depth_color(depth_mm, near, far):
    """Blue = near, red = far; pixels with no depth are black."""
    img = turbo((depth_mm.astype(np.float32) / 1000.0 - near) / (far - near))
    img[depth_mm == 0] = 0
    return img


def look_from(pos, quat_w_ros, back, up, ahead=1.0):
    """World->view rotation and eye for a viewpoint behind/above a camera, looking where it looks."""
    fwd = quat_wxyz_to_matrix(quat_w_ros)[:, 2]
    eye = np.asarray(pos) - back * fwd + np.array([0.0, 0.0, up])
    z = np.asarray(pos) + ahead * fwd - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return eye, np.stack([x, np.cross(z, x), z])


def splat(points, colors, eye, rotation, width, height, focal, dot=1):
    """Project world points; nearest point per pixel wins. colors: list of (N, 3) layers."""
    v = (np.asarray(points) - eye) @ rotation.T
    images = [np.full((height, width, 3), BACKGROUND, np.uint8) for _ in colors]
    front = v[:, 2] > 0.05
    v = v[front]
    colors = [c[front] for c in colors]
    u = (focal * v[:, 0] / v[:, 2] + width / 2).astype(int)
    w = (focal * v[:, 1] / v[:, 2] + height / 2).astype(int)
    zbuf = np.full((height, width), np.inf)
    for du in range(dot):
        for dv in range(dot):
            uu, ww = u + du, w + dv
            ok = (uu >= 0) & (uu < width) & (ww >= 0) & (ww < height)
            order = np.argsort(-v[ok, 2])  # far first, so nearer writes land last
            uu, ww, z = uu[ok][order], ww[ok][order], v[ok, 2][order]
            closer = z < zbuf[ww, uu]
            for img, c in zip(images, colors):
                img[ww[closer], uu[closer]] = c[ok][order][closer]
            zbuf[ww[closer], uu[closer]] = z[closer]
    return images


def camera_cloud(capture, t, rgb, max_range=None):
    depth_mm, k, pos_w, quat = capture
    pts, rows, cols = unproject(depth_mm[t], k, pos_w[t], quat[t])
    keep = np.ones(len(pts), bool) if max_range is None else np.linalg.norm(pts - pos_w[t], axis=1) < max_range
    return pts[keep], rgb[rows, cols][keep], rows[keep], cols[keep]


def load_capture(run, camera):
    d = np.load(run / f"success1_{camera}_depth.npz")
    if str(d["projection"]) != "pinhole":
        raise SystemExit(f"{camera} is {d['projection']}; clouds need a pinhole camera")
    return d["depth_mm"], d["intrinsics"], d["pos_w"], d["quat_w_ros"]


def frames(run, name):
    return iio.imiter(run / f"success1_{name}.mp4")


def label(tile, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(tile)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, int(draw.textlength(text)) + 8, 14], fill=(0, 0, 0))
    draw.text((4, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def fit(img):
    if img.shape[:2] == (TILE_H, TILE_W):
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((TILE_W, TILE_H), Image.BILINEAR))


def tiled(run, out, limit=None, fps=20):
    depth = {cam: np.load(run / f"success1_{cam}_depth.npz")["depth_mm"] for cam, _, _ in COLUMNS}
    clouds = {src: load_capture(run, src) for _, src, _ in COLUMNS}
    streams = {cam: (frames(run, cam), frames(run, f"{cam}_seg")) for cam, _, _ in COLUMNS}
    head_pinhole = frames(run, FRONT_TWIN)
    n = min(len(d) for d in depth.values()) if limit is None else limit
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    depth_videos = {} if limit is not None else {
        cam: imageio.get_writer(run / f"success1_{cam}_depth.mp4", fps=fps, codec="libx264", quality=8)
        for cam, _, _ in COLUMNS}
    for t in range(n):
        pin = next(head_pinhole)[..., :3]
        columns = []
        for cam, src, title in COLUMNS:
            rgb, seg = (next(s)[..., :3] for s in streams[cam])
            near, far = DEPTH_RANGE[cam]
            dcol = depth_color(depth[cam][t], near, far)
            if cam in depth_videos:
                depth_videos[cam].append_data(dcol)
            pts, col, _, _ = camera_cloud(clouds[src], t, pin if src == FRONT_TWIN else rgb, max_range=3.0)
            back, up, focal = CLOUD_VIEW[cam]
            eye, rot = look_from(clouds[src][2][t], clouds[src][3][t], back, up)
            (cloud,) = splat(pts, [col], eye, rot, TILE_W, TILE_H, focal)
            names = ("RGB", "Segmentation", f"Depth {near:g}-{far:g} m (blue=near)", "Point cloud")
            columns.append([label(tile, f"{title} | {name}")
                            for tile, name in zip((fit(rgb), fit(seg), fit(dcol), cloud), names)])
        writer.append_data(np.concatenate([np.concatenate(row, 1) for row in zip(*columns)], 0))
        if t % 200 == 0:
            print(f"frame {t}/{n}", flush=True)
    writer.close()
    for w in depth_videos.values():
        w.close()
    return n


def follow(run, out, limit=None, fps=20, back=0.45, up=0.30, width=640, height=480, focal=380.0):
    sources = (FRONT_TWIN, "wrist_right", "wrist_left")
    clouds = {src: load_capture(run, src) for src in sources}
    streams = {src: (frames(run, src), frames(run, f"{src}_seg")) for src in sources}
    head = clouds[FRONT_TWIN]
    n = min(len(c[0]) for c in clouds.values()) if limit is None else limit
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    for t in range(n):
        points, rgb_layer, seg_layer = [], [], []
        for src in sources:
            rgb, seg = (next(s)[..., :3] for s in streams[src])
            pts, col, rows, cols = camera_cloud(clouds[src], t, rgb)
            near_head = np.linalg.norm(pts - head[2][t], axis=1) < 4.0
            points.append(pts[near_head])
            rgb_layer.append(col[near_head])
            seg_layer.append(seg[rows, cols][near_head])
        eye, rot = look_from(head[2][t], head[3][t], back, up, ahead=1.2)
        images = splat(np.concatenate(points), [np.concatenate(rgb_layer), np.concatenate(seg_layer)],
                       eye, rot, width, height, focal, dot=2)
        writer.append_data(np.concatenate(images, 1))
        if t % 200 == 0:
            print(f"frame {t}/{n}", flush=True)
    writer.close()
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("tiled", "follow"))
    parser.add_argument("replay_dir", type=Path, help="Folder holding success1_*.mp4 and *_depth.npz")
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=int, default=None, help="Only the first N frames (preview)")
    args = parser.parse_args(argv)
    render = tiled if args.mode == "tiled" else follow
    n = render(args.replay_dir, args.out, args.frames)
    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
