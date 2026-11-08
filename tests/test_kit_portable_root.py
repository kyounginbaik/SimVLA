"""Slurm launchers keep Omniverse Kit's cache off shared storage.

Kit runs "portable" with its root inside the simulator venv, so every job on every node
shared one DerivedDataCache on Lustre. Two jobs starting together on different nodes left
Kit hung in Lustre sync I/O during isaacsim.core.simulation_manager startup. Each node now
gets its own root; same-node jobs share it safely via Kit's per-instance locks.
"""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/slurm/kit_root.sh"
LAUNCHERS = ["collect_public_lerobot.sbatch", "replay_public_lerobot.sbatch", "smoke_public.sbatch"]


def resolve(env_overrides, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIMVLA_")}
    env.update(env_overrides)
    script = f'source "{HELPER}" && printf "%s|%s" "$SIMVLA_KIT_ROOT_ARGS" "${{SIMVLA_KIT_ROOT:-}}"'
    return subprocess.run(["bash", "-c", script], env=env, cwd=tmp_path, capture_output=True, text=True)


def test_default_root_is_node_local_and_per_user(tmp_path):
    result = resolve({"TMPDIR": str(tmp_path), "USER": "alice"}, tmp_path)
    assert result.returncode == 0, result.stderr
    root = tmp_path / "simvla-kit-alice"
    assert result.stdout == f" --portable-root {root}|{root}"
    assert root.is_dir()


def test_explicit_root_and_opt_out(tmp_path):
    chosen = tmp_path / "kit"
    result = resolve({"SIMVLA_KIT_PORTABLE_ROOT": str(chosen)}, tmp_path)
    assert result.stdout == f" --portable-root {chosen}|{chosen}"
    result = resolve({"SIMVLA_KIT_PORTABLE_ROOT": "none"}, tmp_path)
    assert result.returncode == 0 and result.stdout == "|"


@pytest.mark.parametrize("bad", ["relative/kit", "/tmp/with space"])
def test_root_must_be_absolute_without_whitespace(tmp_path, bad):
    # kit_args is split on whitespace by AppLauncher, so a spaced path would be mangled.
    result = resolve({"SIMVLA_KIT_PORTABLE_ROOT": bad}, tmp_path)
    assert result.returncode == 2
    assert "SIMVLA_KIT_PORTABLE_ROOT" in result.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_every_launcher_passes_the_root_to_kit(launcher):
    text = (ROOT / "scripts/slurm" / launcher).read_text()
    assert "kit_root.sh" in text
    kit_lines = [line for line in text.splitlines() if "kit_args" in line and "threadCount" in line]
    assert kit_lines and all("$SIMVLA_KIT_ROOT_ARGS" in line for line in kit_lines), kit_lines


def test_collection_dry_run_reports_kit_root(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SIMVLA_", "SLURM_", "BODEX_", "OMNI_KIT_"))}
    for name in ("assets", "models", "bodex"):
        (tmp_path / name).mkdir()
    import sys
    env.update(SIMVLA_PYTHON=sys.executable, SIMVLA_ASSETS_DIR=str(tmp_path / "assets"),
               SIMVLA_ROBOT_MODELS_DIR=str(tmp_path / "models"), BODEX_OBJ_DIR=str(tmp_path / "bodex"),
               SIMVLA_COLLECTION_ROOT=str(tmp_path / "output"), SIMVLA_DRY_RUN="1", ROBOT="anubis",
               SIMVLA_KIT_PORTABLE_ROOT=str(tmp_path / "kit"))
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/collect_public_lerobot.sbatch")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"kit_root={tmp_path / 'kit'}\n" in result.stdout
