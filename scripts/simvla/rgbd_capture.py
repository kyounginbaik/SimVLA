"""Depth capture for demo replay, and depth -> point cloud conversion.

Replay renders the robot cameras from a recorded episode. With ``--depth`` it also asks those
cameras for ``distance_to_image_plane`` and, for the successful pass, writes one
``<prefix>_<camera>_depth.npz`` per camera:

  depth_mm     (T, H, W) uint16  millimetres along the optical axis; 0 = no return / out of range
  intrinsics   (3, 3)    float32 pinhole K reported by Isaac Lab
  pos_w        (T, 3)    float32 camera position in the world frame (env origin included)
  quat_w_ros   (T, 4)    float32 camera orientation, wxyz, ROS convention (x right, y down, z fwd)
  projection   str       "pinhole" or "fisheye"

Frame t of the npz is frame t of the matching ``<prefix>_<camera>.mp4``. Only pinhole cameras
can be unprojected with K; the Anubis head camera is a Kannala-Brandt fisheye, so its depth is
saved but ``unproject`` refuses it rather than producing a wrongly-warped cloud.

Nothing here imports Isaac, so the conversion runs in any Python with numpy.
"""
from pathlib import Path

import numpy as np

DEPTH_CAMERAS = ("front", "wrist_left", "wrist_right")
DEPTH_ANNOTATOR = "distance_to_image_plane"
SEG_ANNOTATOR = "instance_id_segmentation_fast"
MAX_DEPTH_MM = np.iinfo(np.uint16).max
#: Replay-only pinhole camera at the fisheye head camera's mount, so the head view can be
#: unprojected. It renders but is not an observation and never enters the dataset.
FRONT_TWIN = "front_pinhole"


def add_pinhole_twin(scene_cfg, source="front", name=FRONT_TWIN, width=640, height=480,
                     hfov_deg=100.0, spawn_factory=None):
    """Add a pinhole copy of `source` (same parent, offset, update period, annotators).

    Call it AFTER any per-run camera noise so the twin sits exactly where `source` does.
    """
    import copy
    import math
    if spawn_factory is None:
        import isaaclab.sim as sim_utils
        spawn_factory = sim_utils.PinholeCameraCfg
    src = getattr(scene_cfg, source)
    twin = copy.deepcopy(src)
    twin.prim_path = f"{src.prim_path}_pinhole"
    twin.width, twin.height = width, height
    aperture = 20.955
    twin.spawn = spawn_factory(
        focal_length=aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0)),
        horizontal_aperture=aperture,
        clipping_range=(0.01, 1.0e5),
    )
    twin.data_types = list(dict.fromkeys(["rgb", *src.data_types]))
    setattr(scene_cfg, name, twin)
    return twin


def enable_segmentation(scene_cfg, cameras):
    """Colorized instance IDs (the annotator SimVQA uses) on each camera."""
    for name in cameras:
        camera = getattr(scene_cfg, name)
        camera.data_types = list(dict.fromkeys([*camera.data_types, SEG_ANNOTATOR]))
        camera.colorize_instance_id_segmentation = True


class FrameRecorder:
    """Per-step uint8 frames of one annotator for several cameras; writes one mp4 per camera."""

    def __init__(self, cameras, key, suffix):
        self.cameras, self.key, self.suffix = tuple(cameras), key, suffix
        self.reset()

    def reset(self):
        self.frames = {c: [] for c in self.cameras}

    def append(self, sensors):
        for name in self.cameras:
            x = sensors[name].data.output[self.key].detach().cpu().numpy()
            if x.dtype != np.uint8:
                x = (x * 255.0 if x.max() <= 1.0 else x).clip(0, 255).astype(np.uint8)
            self.frames[name].append(x[..., :3])

    def save(self, directory, prefix, env_index, fps) -> list[Path]:
        import imageio
        written = []
        for name in self.cameras:
            if self.frames[name]:
                path = Path(directory) / f"{prefix}_{name}{self.suffix}.mp4"
                imageio.mimwrite(path, [f[env_index] for f in self.frames[name]], fps=fps, codec="libx264")
                written.append(path)
        return written


def enable_depth(scene_cfg, cameras=DEPTH_CAMERAS):
    """Add the depth annotator to each camera config before the sensors are created."""
    for name in cameras:
        camera = getattr(scene_cfg, name)
        camera.data_types = list(dict.fromkeys([*camera.data_types, DEPTH_ANNOTATOR]))
        # Off by default in Isaac Lab: pos_w/quat_w_ros then stay at the spawn pose, so a camera
        # riding on the wrist or base would be unprojected from the wrong place every frame.
        camera.update_latest_camera_pose = True


def projection_of(camera_cfg) -> str:
    return "fisheye" if "fisheye" in type(camera_cfg.spawn).__name__.lower() else "pinhole"


def depth_to_mm(depth_m) -> np.ndarray:
    """Metres (any shape, float) -> uint16 millimetres; inf/nan/<=0/too far become 0."""
    depth = np.asarray(depth_m, dtype=np.float64)
    mm = np.rint(depth * 1000.0)
    valid = np.isfinite(mm) & (mm > 0) & (mm <= MAX_DEPTH_MM)
    return np.where(valid, mm, 0).astype(np.uint16)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def unproject(depth_mm, intrinsics, pos_w, quat_w_ros, projection="pinhole", stride=1):
    """One frame of depth -> (points_world (N, 3), pixel_rows (N,), pixel_cols (N,))."""
    if projection != "pinhole":
        raise ValueError(f"cannot unproject a {projection} camera with a pinhole K")
    depth = np.asarray(depth_mm)[::stride, ::stride].astype(np.float64) / 1000.0
    rows, cols = np.nonzero(depth > 0)
    z = depth[rows, cols]
    v = rows * stride
    u = cols * stride
    k = np.asarray(intrinsics, dtype=np.float64)
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    cam = np.stack([x, y, z], axis=1)
    world = cam @ quat_wxyz_to_matrix(quat_w_ros).T + np.asarray(pos_w, dtype=np.float64)
    return world, v, u


class DepthRecorder:
    """Accumulates per-step depth and camera poses for every env; saves one env's sequence."""

    def __init__(self, cameras=DEPTH_CAMERAS):
        self.cameras = tuple(cameras)
        self.reset()

    def reset(self):
        self.depth = {c: [] for c in self.cameras}
        self.pos = {c: [] for c in self.cameras}
        self.quat = {c: [] for c in self.cameras}
        self.intrinsics = {}

    def append(self, sensors):
        for name in self.cameras:
            data = sensors[name].data
            depth = data.output[DEPTH_ANNOTATOR].detach().cpu().numpy()
            if depth.ndim == 4:
                depth = depth[..., 0]
            self.depth[name].append(depth_to_mm(depth))
            self.pos[name].append(data.pos_w.detach().cpu().numpy().astype(np.float32))
            self.quat[name].append(data.quat_w_ros.detach().cpu().numpy().astype(np.float32))
            if name not in self.intrinsics:
                self.intrinsics[name] = data.intrinsic_matrices.detach().cpu().numpy().astype(np.float32)

    def save(self, directory, prefix, env_index, projections) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for name in self.cameras:
            if not self.depth[name]:
                continue
            path = directory / f"{prefix}_{name}_depth.npz"
            np.savez_compressed(
                path,
                depth_mm=np.stack([d[env_index] for d in self.depth[name]]),
                intrinsics=self.intrinsics[name][env_index],
                pos_w=np.stack([p[env_index] for p in self.pos[name]]),
                quat_w_ros=np.stack([q[env_index] for q in self.quat[name]]),
                projection=np.array(projections[name]),
            )
            written.append(path)
        return written
