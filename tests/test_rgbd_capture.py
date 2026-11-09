"""Depth capture helpers: encoding, unprojection geometry, and the npz -> PLY round trip."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/simvla"
sys.path.insert(0, str(SCRIPTS))
rgbd = importlib.import_module("rgbd_capture")
spec = importlib.util.spec_from_file_location("depth_to_pointcloud", SCRIPTS / "depth_to_pointcloud.py")
d2p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2p)

K = np.array([[100.0, 0, 16], [0, 100.0, 12], [0, 0, 1]])
IDENTITY = np.array([1.0, 0, 0, 0])


def test_depth_to_mm_marks_invalid_returns_as_zero():
    out = rgbd.depth_to_mm([1.2344, np.inf, np.nan, -1.0, 0.0, 70.0])
    assert out.dtype == np.uint16
    assert out.tolist() == [1234, 0, 0, 0, 0, 0]


def test_unproject_principal_point_lies_on_optical_axis_then_moves_with_pose():
    depth = np.zeros((24, 32), np.uint16)
    depth[12, 16] = 2000
    depth[12, 26] = 1000  # 10 px right of centre at 1 m -> x = 0.1 m
    pts, rows, cols = rgbd.unproject(depth, K, [0, 0, 0], IDENTITY)
    by_col = dict(zip(cols.tolist(), pts.tolist()))
    assert np.allclose(by_col[16], [0, 0, 2.0]) and np.allclose(by_col[26], [0.1, 0, 1.0])
    # Rotate the camera 90 deg about world z and move it: optical axis (z) stays world z.
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    pts, _, cols = rgbd.unproject(depth, K, [1, 2, 3], q)
    assert np.allclose(dict(zip(cols.tolist(), pts.tolist()))[26], [1.0, 2.1, 4.0])


def test_fisheye_is_refused():
    with pytest.raises(ValueError, match="fisheye"):
        rgbd.unproject(np.ones((2, 2), np.uint16), K, [0, 0, 0], IDENTITY, projection="fisheye")


def test_enable_depth_keeps_existing_types_once():
    cams = {n: SimpleNamespace(data_types=["rgb"]) for n in rgbd.DEPTH_CAMERAS}
    scene = SimpleNamespace(**cams)
    rgbd.enable_depth(scene)
    rgbd.enable_depth(scene)
    assert all(c.data_types == ["rgb", "distance_to_image_plane"] for c in cams.values())
    # Without this Isaac Lab reports the spawn pose forever, even for a camera on a moving wrist.
    assert all(c.update_latest_camera_pose is True for c in cams.values())


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def test_recorder_saves_selected_env_and_converter_writes_ply(tmp_path):
    def sensor(depth_m):
        return SimpleNamespace(data=SimpleNamespace(
            output={"distance_to_image_plane": _T(np.full((2, 24, 32, 1), depth_m))},
            pos_w=_T([[0, 0, 0], [5, 5, 5]]), quat_w_ros=_T([IDENTITY, IDENTITY]),
            intrinsic_matrices=_T([K, K])))
    rec = rgbd.DepthRecorder(cameras=("wrist_right",))
    rec.append({"wrist_right": sensor(1.0)})
    rec.append({"wrist_right": sensor(2.0)})
    (path,) = rec.save(tmp_path, "success1", 1, {"wrist_right": "pinhole"})
    data = np.load(path)
    assert data["depth_mm"].shape == (2, 24, 32) and data["depth_mm"][1, 0, 0] == 2000
    assert np.allclose(data["pos_w"][0], [5, 5, 5])
    (ply, n), = d2p.convert(path, [1], tmp_path / "pc", stride=4)
    assert n == 6 * 8
    raw = ply.read_bytes()
    assert raw.startswith(b"ply\nformat binary_little_endian") and b"element vertex 48\n" in raw
    body = np.frombuffer(raw.split(b"end_header\n", 1)[1], dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    assert np.allclose(body["xyz"][:, 2], 7.0)  # 2 m in front of a camera at z = 5


def test_pinhole_twin_copies_mount_and_annotators_with_wide_pinhole_lens():
    offset = SimpleNamespace(pos=(0.07, 0.0, 1.18), rot=(0.94, 0, 0.34, 0))
    front = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam", width=320, height=240,
                            data_types=["rgb"], offset=offset, spawn=SimpleNamespace(kind="fisheye"))
    scene = SimpleNamespace(front=front, wrist_left=SimpleNamespace(data_types=["rgb"]),
                            wrist_right=SimpleNamespace(data_types=["rgb"]))
    rgbd.enable_depth(scene)
    rgbd.enable_segmentation(scene, rgbd.DEPTH_CAMERAS)
    twin = rgbd.add_pinhole_twin(scene, spawn_factory=lambda **kw: SimpleNamespace(**kw))
    assert scene.front_pinhole is twin and twin.prim_path.endswith("head_cam_pinhole")
    assert twin.offset.pos == offset.pos and twin.offset is not offset  # copied, not shared
    assert twin.update_latest_camera_pose and twin.colorize_instance_id_segmentation
    assert twin.data_types == ["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"]
    hfov = 2 * np.degrees(np.arctan(twin.spawn.horizontal_aperture / 2 / twin.spawn.focal_length))
    assert np.isclose(hfov, 100.0) and (twin.width, twin.height) == (640, 480)
    assert front.spawn.kind == "fisheye" and front.width == 320  # source untouched


def test_frame_recorder_writes_selected_env_video(tmp_path):
    pytest.importorskip("imageio")  # simulator/export envs only
    frames = np.zeros((2, 8, 8, 4), np.uint8)
    frames[1, ..., 0] = 255
    sensors = {"front": SimpleNamespace(data=SimpleNamespace(output={"seg": _T(frames)}))}
    rec = rgbd.FrameRecorder(("front",), "seg", "_seg")
    for _ in range(3):
        rec.append(sensors)
    (path,) = rec.save(tmp_path, "success1", 1, fps=20)
    assert path.name == "success1_front_seg.mp4"
    import imageio.v3 as iio
    video = iio.imread(path)
    assert video.shape[0] == 3 and video[..., 0].mean() > 200 and video[..., 1].mean() < 60


def test_replay_video_splat_keeps_nearest_point_and_turbo_orders_near_blue():
    pytest.importorskip("imageio")  # simulator/export envs only
    spec = importlib.util.spec_from_file_location("replay_videos", SCRIPTS / "replay_videos.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    near, far = rv.turbo(np.array([0.15, 0.9]))
    assert near[2] > near[0] and far[0] > far[2]  # blue near, red far
    assert (rv.depth_color(np.array([[0]], np.uint16), 0.1, 1.0) == 0).all()
    # Two points on the optical axis: the nearer colour must win the pixel.
    eye, rot = np.zeros(3), np.eye(3)
    pts = np.array([[0, 0, 2.0], [0, 0, 1.0]])
    (img,) = rv.splat(pts, [np.array([[255, 0, 0], [0, 255, 0]], np.uint8)], eye, rot, 9, 9, 10.0)
    assert img[4, 4].tolist() == [0, 255, 0]
    # look_from: behind and above an upright camera looking along world +x, still looking forward.
    cam_quat = [0.5, -0.5, 0.5, -0.5]  # ROS camera axes: z along world +x
    eye, rot = rv.look_from([0, 0, 1.0], cam_quat, back=1.0, up=0.5)
    assert np.allclose(eye, [-1.0, 0, 1.5], atol=1e-6) and rot[2, 0] > 0.8
