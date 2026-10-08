"""door_open_trace against a stub context. No Isaac Sim, no GPU.

The module's job is arithmetic and file I/O over tensors an EvalContext already carries, so it is
testable on a login node in under a second -- which is the point. Every column in this CSV exists
because some previous session argued about its value from video instead of measuring it.
"""

import csv
import math
import os

import pytest
import torch

import door_open_trace as dot
from predicates_math import EvalContext

SPEC = {"all": [
    {"joint_pos": {"role": "refrigerator", "joint": "door_joint", "lo": math.radians(60)}},
    {"eef_home": {"radius": 0.12, "arm": "both"}},
    {"last_subtask": {}},
]}


def _ctx(num_envs=2, door_rad=0.0, gap=0.01, hand=(0.0, 0.0, 1.0), door_origin=(0.6, 0.0, 0.0)):
    return EvalContext(
        num_envs=num_envs,
        art_joint_pos={"refrigerator": torch.full((num_envs, 1), door_rad)},
        art_joint_names={"refrigerator": ["door_joint"]},
        art_body_pos_w={"refrigerator": torch.tensor([[list(door_origin)]] * num_envs)},
        art_body_names={"refrigerator": ["door"]},
        eef_pos_w={"left": torch.tensor([list(hand)] * num_envs),
                   "right": torch.tensor([list(hand)] * num_envs)},
        gripper_gap={"left": torch.full((num_envs,), gap),
                     "right": torch.full((num_envs,), gap)},
        base_pos_w=torch.zeros(num_envs, 3),
        goal_index=torch.zeros(num_envs, dtype=torch.long),
    )


def test_it_writes_nothing_unless_asked(tmp_path, monkeypatch):
    """Off by default and byte-identical when unset: this runs inside the TERMINATION manager,
    every step of every episode."""
    monkeypatch.delenv("SIMVLA_DOOR_TRACE", raising=False)
    dot.reset()
    dot.record(_ctx(), SPEC)
    assert list(tmp_path.iterdir()) == []


def test_one_row_per_env_per_call(tmp_path, monkeypatch):
    """PER ENV. SIMVLA_PRINT_TERMS prints env 0 only, and env 0's silence is exactly what hid the
    0.0-degree episode: three other envs were running at the same time and none was reported."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    dot.record(_ctx(num_envs=3), SPEC)
    dot.record(_ctx(num_envs=3), SPEC)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
    assert [r["env"] for r in rows] == ["0", "1", "2", "0", "1", "2"]
    assert [r["frame"] for r in rows] == ["0", "0", "0", "1", "1", "1"]


def test_the_door_angle_is_reported_in_degrees(tmp_path, monkeypatch):
    """joint_pos is RADIANS; every human-facing number about this door has been in degrees."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    dot.record(_ctx(door_rad=math.radians(87.4)), SPEC)
    assert float(list(csv.DictReader(out.open()))[0]["door_deg"]) == pytest.approx(87.4, abs=1e-3)


def test_a_real_zero_is_written_as_zero_not_blank(tmp_path, monkeypatch):
    """A blank means 'not measured'; 0.0 means 'measured, and it is zero'. The door sitting at
    exactly 0.0deg for thousands of steps is the symptom this whole file exists to explain -- if a
    genuine zero ever rendered blank, the trace would report 'not measured' for precisely the
    reading the diagnosis rests on, and the next task's scorer would silently agree."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    door_origin = (0.6, 0.0, 0.0)
    dot.record(_ctx(door_rad=0.0, hand=door_origin, door_origin=door_origin), SPEC)
    row = list(csv.DictReader(out.open()))[0]
    assert row["door_deg"] != ""
    assert float(row["door_deg"]) == pytest.approx(0.0)
    assert row["hand_l_to_door_m"] != ""
    assert float(row["hand_l_to_door_m"]) == pytest.approx(0.0)
    assert row["hand_r_to_door_m"] != ""
    assert float(row["hand_r_to_door_m"]) == pytest.approx(0.0)


def test_hand_to_door_origin_is_the_grasp_invariant(tmp_path, monkeypatch):
    """The door body's frame origin IS its hinge, so a hand closed on the handle stays a fixed
    distance from it for the whole swing. A CHANGE in this number is the grasp sliding off the
    bar -- the one failure that a door angle alone cannot distinguish from a base that never
    moved. _ctx() puts both hands at the same position, so this only pins that each column
    independently reports the shared distance -- see the test below for the case that actually
    tells the two hands apart."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    dot.record(_ctx(hand=(0.0, 0.0, 0.0), door_origin=(0.6, 0.0, 0.0)), SPEC)
    row = list(csv.DictReader(out.open()))[0]
    assert float(row["hand_l_to_door_m"]) == pytest.approx(0.6)
    assert float(row["hand_r_to_door_m"]) == pytest.approx(0.6)


def test_each_hand_reports_its_own_distance_never_a_minimum(tmp_path, monkeypatch):
    """The regression this whole fix exists for. door_open_trace used to report
    min(hand_l_to_door, hand_r_to_door) in one 'hand_to_door_m' column: on the diagonal-sweep
    task only the LEFT arm grasps the handle, but Anubis's idle RIGHT hand sits at its home pose
    close enough to the hinge (measured: 1.1499 m) that torch.minimum silently picked the IDLE
    hand's reading every frame, and a reader concluded the grasping hand was on the handle when
    it was never being measured at all. Here the left hand is far from the door (10.0 m -- clearly
    NOT grasping) and the right hand is near it (1.1499 m -- the measured idle-hand coincidence).
    A minimum over the two would report 1.1499 for both roles; the columns must instead show each
    hand's own truth, independently, with neither one clamped to the other."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    door_origin = (0.0, 0.0, 0.0)
    ctx = _ctx(door_origin=door_origin)
    ctx.eef_pos_w = {
        "left": torch.tensor([[10.0, 0.0, 0.0]] * ctx.num_envs),
        "right": torch.tensor([[1.1499, 0.0, 0.0]] * ctx.num_envs),
    }
    dot.record(ctx, SPEC)
    row = list(csv.DictReader(out.open()))[0]
    assert float(row["hand_l_to_door_m"]) == pytest.approx(10.0)
    assert float(row["hand_r_to_door_m"]) == pytest.approx(1.1499)
    # Neither column is a minimum of the two hands.
    assert float(row["hand_l_to_door_m"]) != pytest.approx(1.1499)
    assert float(row["hand_r_to_door_m"]) != pytest.approx(10.0)


def test_the_commanded_goal_appears_beside_what_the_base_did(tmp_path, monkeypatch):
    """Two sessions were spent arguing retreat direction from commanded values against a user
    reading actual footage. Both numbers, same row, ends that class of argument."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    dot.publish_goal(torch.tensor([0, 1]),
                     torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
                     torch.tensor([0.25, 0.75]))
    dot.record(_ctx(num_envs=2), SPEC)
    rows = list(csv.DictReader(out.open()))
    assert float(rows[0]["goal_x"]) == pytest.approx(1.5)
    assert float(rows[1]["goal_yaw"]) == pytest.approx(0.75)
    assert float(rows[0]["base_x"]) == pytest.approx(0.0)


def test_an_env_with_no_commanded_goal_yet_is_blank_not_zero(tmp_path, monkeypatch):
    """A blank reads as 'never commanded'. A 0.0 reads as 'commanded to the origin', and the
    scorer would silently treat the two the same."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    dot.record(_ctx(num_envs=1), SPEC)
    assert list(csv.DictReader(out.open()))[0]["goal_x"] == ""


def test_a_missing_door_joint_does_not_kill_the_run(tmp_path, monkeypatch):
    """This runs inside the termination manager. A trace that raises takes the whole episode with
    it, and the trace is diagnostic -- it must never be load-bearing."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    ctx = _ctx()
    ctx.art_joint_names = {"refrigerator": ["some_other_joint"]}
    dot.record(ctx, SPEC)            # must not raise
    assert list(csv.DictReader(out.open()))[0]["door_deg"] == ""


def test_the_trace_carries_a_contact_column_per_watched_body(tmp_path, monkeypatch):
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    ctx = _ctx(num_envs=2)                     # existing helper in this file
    ctx.door_contact = {"base": torch.tensor([0.0, 12.5]),
                        "grip_l": torch.tensor([3.1, 0.0]),
                        "grip_r": torch.tensor([0.0, 0.0])}
    dot.record(ctx, SPEC)
    rows = list(csv.DictReader(out.open()))
    assert rows[1]["touch_base_n"] == "12.5"
    assert rows[0]["touch_grip_l_n"] == "3.1"


def test_a_missing_contact_sensor_writes_blanks_not_zeros(tmp_path, monkeypatch):
    """A 0.0 reads as 'measured, no contact'. A blank reads as 'not measured'. Conflating them is
    how the truncated sweep was scored as complete."""
    out = tmp_path / "trace.csv"
    monkeypatch.setenv("SIMVLA_DOOR_TRACE", str(out))
    dot.reset()
    ctx = _ctx(num_envs=1)
    ctx.door_contact = {}
    dot.record(ctx, SPEC)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["touch_base_n"] == ""
