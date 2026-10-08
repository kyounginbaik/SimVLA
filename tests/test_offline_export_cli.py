from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/offline_finalize_lerobot.py"


def test_offline_export_help_does_not_import_simulator():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "never uploads" in result.stdout


def test_empty_stage_rejected_without_creating_output(tmp_path):
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--stage", str(tmp_path / "absent"),
        "--output", str(tmp_path / "output"), "--goal", str(tmp_path / "goal.json"),
        "--robot", "anubis"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "at least one episode" in result.stderr
    assert not (tmp_path / "output").exists()


def test_existing_output_rejected_before_imports(tmp_path):
    stage = tmp_path / "stage/000000"
    stage.mkdir(parents=True)
    (stage / "arrays.npz").touch()
    goal = tmp_path / "goal.json"
    goal.write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--stage", str(stage.parent),
        "--output", str(output), "--goal", str(goal), "--robot", "anubis"],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "output already exists" in result.stderr
    assert marker.read_text() == "keep"
