"""Robot-specific BODex-to-ee_link1 frame conversions must stay explicit."""

import math

import pytest

from simvla.skills import GRASP_TOOL_FRAME, grasp_tool_frame, preferred_grasp_roll


def _mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _same_rotation(actual, expected):
    # q and -q encode the same rotation; compare the absolute unit-quaternion dot product.
    dot = sum(a * b for a, b in zip(actual, expected))
    return abs(abs(dot) - 1.0) < 1e-6


def test_grasp_tool_frames_cover_all_supported_robots():
    assert set(GRASP_TOOL_FRAME) == {"anubis", "rby1", "aiworker"}
    for robot in GRASP_TOOL_FRAME:
        frame = grasp_tool_frame(robot)
        assert math.isclose(sum(x * x for x in frame["q_offset"]), 1.0, abs_tol=1e-6)
        assert isinstance(frame["y_down"], bool)


@pytest.mark.parametrize(
    ("robot", "relative"),
    [
        # ee_link1's approach axis is reversed on RB-Y1: rotate 180 degrees about tool X.
        ("rby1", (0.0, 1.0, 0.0, 0.0)),
        # AI Worker's jaw travel is local Y rather than local X: rotate -90 degrees about Z.
        ("aiworker", (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))),
    ],
)
def test_robot_grasp_offsets_are_relative_to_anubis(robot, relative):
    actual = _mul_wxyz(grasp_tool_frame("anubis")["q_offset"], relative)
    assert _same_rotation(actual, grasp_tool_frame(robot)["q_offset"])


def test_grasp_tool_frame_refuses_unmeasured_robot():
    with pytest.raises(KeyError, match="no ee_link1 grasp frame"):
        grasp_tool_frame("unknown")


def test_aiworker_horizontal_grasp_keeps_camera_up_despite_tiny_negative_jaw_tilt():
    # Native BoDex mug left candidate 97, composed with AI Worker's tool frame.
    q = (-0.5432310344, 0.4477219005, 0.5648809649, 0.4305283250)
    assert preferred_grasp_roll(q, "aiworker")
    flipped = _mul_wxyz(q, (0, 0, 0, 1))
    assert not preferred_grasp_roll(flipped, "aiworker")
    assert preferred_grasp_roll(tuple(-x for x in q), "aiworker")


@pytest.mark.parametrize("robot,expected", [("anubis", False), ("rby1", True)])
def test_historical_y_axis_rule_is_preserved(robot, expected):
    assert preferred_grasp_roll((math.sqrt(.5), math.sqrt(.5), 0, 0), robot) is expected


@pytest.mark.parametrize("q", [(0, 0, 0, 0), (1, float("nan"), 0, 0), (1, 0)])
def test_roll_selection_rejects_invalid_quaternions(q):
    with pytest.raises(ValueError):
        preferred_grasp_roll(q, "aiworker")
