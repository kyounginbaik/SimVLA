import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "scripts/tools"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("collection_result_test", TOOLS / "check_collection_result.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def staged_dataset(root):
    raw = root / "raw/episode_0/arrays.npz"
    raw.parent.mkdir(parents=True)
    raw.touch()
    info = root / "lerobot/demo/all/meta/info.json"
    info.parent.mkdir(parents=True)
    info.write_text("{}")


def test_normal_exit_with_no_episodes_fails_and_records_evidence(module, tmp_path):
    assert module.main([str(tmp_path)]) == 1
    report = json.loads((tmp_path / "collection-result.json").read_text())
    assert report["passed"] is False
    assert report["collector_exit"] == 0
    assert report["validated_episodes"] == 0
    assert "no finalized all dataset" in report["errors"]


def test_corrupt_metadata_fails_closed(module, tmp_path):
    staged_dataset(tmp_path)
    report = module.check(tmp_path, 0)
    assert report["passed"] is False
    assert report["datasets"][0]["valid"] is False


@pytest.mark.parametrize("exit_code,passed", [(0, True), (137, False)])
def test_valid_export_never_erases_collector_failure(module, tmp_path, monkeypatch, exit_code, passed):
    staged_dataset(tmp_path)
    monkeypatch.setattr(module, "validate", lambda path: {"episodes": 1, "frames": 20, "video_keys": 5})
    report = module.check(tmp_path, exit_code)
    assert report["passed"] is passed
    assert report["validated_episodes"] == 1
    assert report["validated_frames"] == 20


def test_count_mismatch_is_rejected(module, tmp_path, monkeypatch):
    staged_dataset(tmp_path)
    monkeypatch.setattr(module, "validate", lambda path: {"episodes": 2, "frames": 20, "video_keys": 5})
    report = module.check(tmp_path, 0)
    assert report["passed"] is False
    assert "staged/finalized episode count mismatch" in report["errors"]


def test_export_failure_is_not_hidden_by_valid_partial_data(module, tmp_path, monkeypatch):
    staged_dataset(tmp_path)
    monkeypatch.setattr(module, "validate", lambda path: {"episodes": 1, "frames": 20, "video_keys": 5})
    report = module.check(tmp_path, 0, 2)
    assert not report["passed"]
    assert "exporter exited with status 2" in report["errors"]


def test_existing_evidence_is_not_replaced(module, tmp_path):
    path = tmp_path / "collection-result.json"
    path.write_text("keep prior evidence")
    with pytest.raises(SystemExit):
        module.main([str(tmp_path)])
    assert path.read_text() == "keep prior evidence"


def test_partial_bulk_collection_is_not_success(module, tmp_path, monkeypatch):
    staged_dataset(tmp_path)
    monkeypatch.setattr(module, "validate", lambda path: {"episodes": 1, "frames": 20})
    report = module.check(tmp_path, 0, expected_episodes=10)
    assert not report["passed"]
    assert report["expected_episodes"] == 10
    assert "incomplete collection: requested 10 episodes, validated 1" in report["errors"]


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_expected_count(module, tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        module.check(tmp_path, 0, expected_episodes=count)
