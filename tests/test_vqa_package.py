"""CPU conversion behavior through the installed public command."""
import json
import subprocess
import sys


def test_vqa_conversion_and_overwrite_protection(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    (root / "vqa_meta.json").write_text(json.dumps({"subtasks": {"0": "Approach bowl", "1": "Grasp bowl"}}))
    record = {
        "id": "env_00_t0010", "task": "example", "prompt": "Pick up the bowl",
        "meta": {"goal_idx": 0, "obj_name": "bowl0"},
        "objects": [{"name": "bowl0", "present": True,
                     "bboxes": {"front": "<loc0100><loc0200><loc0300><loc0400>"},
                     "distance_base": [0.3, 0.1, 0.8]}],
    }
    (root / "vqa.jsonl").write_text(json.dumps(record) + "\n")
    output = tmp_path / "out/questions.jsonl"
    command = [sys.executable, "-m", "simvla", "vqa", str(root), "--output", str(output)]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    original = output.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    assert len({row["id"] for row in rows}) == len(rows)
    answers = {row["vqa_function"]: row["cotrain_answer"] for row in rows}
    assert answers["high_level_subtask"] == "Approach bowl"
    assert answers["next_subtask"] == "Grasp bowl"
    assert "<loc0100>" in answers["object_detection"]
    second = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert second.returncode != 0
    assert output.read_bytes() == original
