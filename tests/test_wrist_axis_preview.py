import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "wrist_axis_preview", Path(__file__).resolve().parents[1] / "scripts/simvla/wrist_axis_preview.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("axis,index", [("x", 1), ("y", 2), ("z", 3)])
def test_quarter_turn(axis, index):
    expected = [2**-.5, 0., 0., 0.]
    expected[index] = 2**-.5
    assert module.local_quarter_turn((1, 0, 0, 0), axis) == pytest.approx(expected)


def test_previews_preserve_original_mounts():
    cam = lambda side: SimpleNamespace(prim_path=f"/Robot/{side}/cam",
                                       offset=SimpleNamespace(rot=(1, 0, 0, 0), pos=(1, 2, 3)))
    scene = SimpleNamespace(wrist_left=cam("left"), wrist_right=cam("right"))
    report = module.add_wrist_axis_previews(scene)
    assert len(report) == 6
    assert scene.wrist_left.offset.rot == (1, 0, 0, 0)
    assert scene.wrist_right.offset.rot == (1, 0, 0, 0)
    assert scene.wrist_left_z90.offset.pos == (1, 2, 3)
    assert scene.wrist_right_x90.prim_path == "/Robot/right/cam_x90"
