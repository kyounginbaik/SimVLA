import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/simvla/workflow_status.py"
spec = importlib.util.spec_from_file_location("workflow_status_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("process,recorded,expected", [
    (0, 0, 0), (0, 1, 1), (137, 0, 137), (0, True, 1), (0, "0", 1), (0, -1, 1),
])
def test_pre_shutdown_status_is_authoritative_on_zero_process_exit(tmp_path, process, recorded, expected):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"exit_code": recorded}))
    assert module.resolve_exit_code(process, path) == expected


def test_missing_or_corrupt_status_cannot_authorize_success(tmp_path):
    path = tmp_path / "status.json"
    assert module.resolve_exit_code(0, path) == 1
    path.write_text("partial")
    assert module.resolve_exit_code(0, path) == 1


def test_disk_quota_failure_still_closes_simulator(monkeypatch):
    events = []
    def quota_failure(code):
        raise OSError(122, "Disk quota exceeded")
    monkeypatch.setattr(module, "record_status", quota_failure)
    result = module.close_with_status(0, lambda: events.append("clear"), lambda: events.append("close"))
    assert result == 1
    assert events == ["clear", "close"]


def test_context_failure_still_closes_app(monkeypatch):
    events = []
    monkeypatch.setattr(module, "record_status", lambda code: None)
    def clear_failure():
        raise RuntimeError("clear failed")
    with pytest.raises(RuntimeError, match="clear failed"):
        module.close_with_status(0, clear_failure, lambda: events.append("close"))
    assert events == ["close"]
