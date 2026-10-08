import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/validate_simvqa_capture.py"
spec = importlib.util.spec_from_file_location("capture_validation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def capture(tmp_path):
    metadata = {"subtasks": {"0": "Grasp mug"}, "targets": {"0": "mug0"},
                "grasp_steps": {"0": True}, "sampled_goal_indices": [0]}
    (tmp_path / "vqa_meta.json").write_text(json.dumps(metadata))
    row = {"id": "env_00_t0001", "frame_index": 0, "target_text": "Subtask: Grasp mug",
           "meta": {"goal_idx": 0, "obj_name": "mug0", "grasp_target": True,
                    "goal_state_gripper": [0, 0, 0, 1, 0, 0, 0]}, "images": {}, "seg_images": {}}
    for kind in ("images", "seg_images"):
        (tmp_path / kind).mkdir()
        for camera in ("front", "wrist_left", "wrist_right"):
            path = tmp_path / kind / (camera + ".png")
            Image.new("RGB", (4, 4), color="red").save(path)
            row[kind][camera] = str(path)
    return row


def test_complete_capture_validates(tmp_path):
    row = capture(tmp_path)
    (tmp_path / "vqa.jsonl").write_text(json.dumps(row) + "\n")
    result = module.validate(tmp_path)
    assert result["passed"] and result["images"] == 6 and result["records"] == 1


@pytest.mark.parametrize("failure", ["empty", "duplicate", "missing_image", "wrong_language",
                                     "wrong_target", "missing_grasp", "nonfinite", "coverage", "image_size"])
def test_incomplete_capture_fails_closed(tmp_path, failure):
    row = capture(tmp_path)
    if failure == "missing_image":
        (tmp_path / "images/front.png").unlink()
    elif failure == "image_size":
        Image.new("RGB", (8, 4)).save(tmp_path / "seg_images/front.png")
    elif failure == "wrong_language":
        row["target_text"] = "Subtask: Reset arm"
    elif failure == "wrong_target":
        row["meta"]["obj_name"] = "bowl0"
    elif failure == "missing_grasp":
        del row["meta"]["grasp_target"]
    elif failure == "nonfinite":
        row["meta"]["goal_state_gripper"][0] = float("nan")
    elif failure == "coverage":
        path = tmp_path / "vqa_meta.json"
        meta = json.loads(path.read_text())
        meta["sampled_goal_indices"] = [0, 1]
        path.write_text(json.dumps(meta))
    lines = [] if failure == "empty" else [json.dumps(row)] * (2 if failure == "duplicate" else 1)
    (tmp_path / "vqa.jsonl").write_text("\n".join(lines))
    result = module.validate(tmp_path)
    assert not result["passed"] and result["errors"]


def test_exponent_overflow_is_not_a_finite_json_number(tmp_path):
    row = capture(tmp_path)
    row["meta"]["goal_state_gripper"][0] = 123.456
    (tmp_path / "vqa.jsonl").write_text(json.dumps(row).replace("123.456", "1e999"))
    assert not module.validate(tmp_path)["passed"]
