import json
from pathlib import Path

from simvla import cli


def test_run_can_preserve_simulator_status_outside_temporary_directory(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    workflow = root / "scripts" / "simvla" / "smoke_test.py"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("# fake simulator workflow\n")
    status = tmp_path / "diagnostics" / "simulator-result.json"
    status.parent.mkdir()
    status.write_text('{"exit_code": 1}')

    monkeypatch.setenv("SIMVLA_PRESERVE_STATUS_FILE", str(status))
    monkeypatch.setattr(cli, "doctor", lambda: {"simulator_ready": True})

    def fake_call(command, *, cwd, env):
        assert command[-1] == "--headless"
        assert Path(cwd) == root
        assert env["SIMVLA_STATUS_FILE"] == str(status)
        Path(env["SIMVLA_STATUS_FILE"]).write_text(json.dumps({"exit_code": 0}))
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    assert cli.main([
        "run", "smoke", "--repo-root", str(root), "--", "--headless",
    ]) == 0
    assert json.loads(status.read_text()) == {"exit_code": 0}
