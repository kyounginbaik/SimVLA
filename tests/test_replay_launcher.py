import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_replay_uses_same_public_robot_assets_as_collection(tmp_path, robot):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SIMVLA_", "SLURM_"))}
    env.update(SIMVLA_REPO_ROOT=str(ROOT), SIMVLA_PYTHON=sys.executable,
               SIMVLA_DATASET=str(dataset), SIMVLA_GOALS_DIR=str(tmp_path / "goals"),
               SIMVLA_ASSETS_DIR=str(tmp_path / "assets"),
               SIMVLA_ROBOT_MODELS_DIR=str(tmp_path / "models"), ROBOT=robot,
               SIMVLA_COLLECTION_ROOT=str(tmp_path / "outputs"),
               TASK="test", SIMVLA_DRY_RUN="1")
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/replay_public_lerobot.sbatch")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"output={tmp_path}/outputs/simvla-replay-{robot}-local-" in result.stdout
    assert not (tmp_path / "outputs").exists()  # Dry run never writes output.
    if robot == "rby1":
        assert f"rby1m={tmp_path}/rby1m" in result.stdout
    if robot == "aiworker":
        assert f"aiworker_usd={tmp_path}/assets/Robots/MM/aiworker/ffw_sg2_simvla_grip.usd" in result.stdout
