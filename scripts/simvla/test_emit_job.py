"""emit_job: build + submit the Slurm job behind the composer's 'Generate goals' button.

Pure-stdlib logic, so it is tested here on the login node without Omniverse, tkinter, or a real
sbatch. The script text is pinned so the button keeps running the VALIDATED task_emit invocation
(env exports + isaaclab.sh -p task_emit.py), and the submit path is exercised with a fake sbatch.
"""

import os
import subprocess
from pathlib import Path

import pytest

import emit_job


def _script(**overrides):
    kw = dict(
        template_path="scripts/simvla/templates/bowl_to_drawer.json",
        kitchens="813",
        out_dir="/home/u/simvla_out/composed_goals",
        repo_root="/home/u/proj/simvla",
        conda_base="/home/u/miniconda3",
        conda_env="env_isaaclab",
        hf_user="u",
    )
    kw.update(overrides)
    return emit_job.build_emit_sbatch(**kw)


def test_resolve_out_dir_prefers_simvla_goals_dir(monkeypatch):
    """The button writes into $SIMVLA_GOALS_DIR so a later flag-free simvla_gen finds the goals with no
    extra config; without it, a scratch dir under simvla_out (never the committed corpus)."""
    monkeypatch.setenv("SIMVLA_GOALS_DIR", "/scratch/goals")
    assert emit_job.resolve_out_dir() == "/scratch/goals"
    monkeypatch.delenv("SIMVLA_GOALS_DIR", raising=False)
    assert emit_job.resolve_out_dir().endswith("simvla_out/composed_goals")


def test_env_defaults_reads_the_live_environment(monkeypatch):
    """conda_base is CONDA_EXE's grandparent, so the job sources the same conda the generator runs in."""
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "env_isaaclab")
    monkeypatch.setenv("HF_USER", "alice")
    d = emit_job.env_defaults()
    assert d == {"conda_base": "/opt/conda", "conda_env": "env_isaaclab", "hf_user": "alice"}


def test_env_defaults_falls_back_when_unset(monkeypatch):
    """A missing HF_USER falls back to $USER, and a missing conda to ~/miniconda3 — no KeyError."""
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.delenv("HF_USER", raising=False)
    monkeypatch.setenv("USER", "bob")
    d = emit_job.env_defaults()
    assert d["conda_base"].endswith("miniconda3")
    assert d["conda_env"] == "env_isaaclab"
    assert d["hf_user"] == "bob"


def test_kitchen_is_on_disk_checks_the_first_rotation_usd(tmp_path):
    """The button's precondition guard: a kitchen with its -00 rotation USD reads as on-disk; one that
    was only previewed (no USD) does not, so the button says 'Accept & Generate first'."""
    (tmp_path / "kitchen_813_00.usd").write_text("x")
    assert emit_job.kitchen_is_on_disk(tmp_path, 813) is True
    assert emit_job.kitchen_is_on_disk(tmp_path, 42) is False


def test_the_sbatch_runs_the_validated_task_emit_invocation():
    """The script must call the exact validated path: isaaclab.sh -p task_emit.py with --template,
    --kitchens and --out — so the button and the proven CLI are the same command."""
    s = _script()
    assert "./isaaclab.sh -p scripts/simvla/task_emit.py" in s
    assert "--template scripts/simvla/templates/bowl_to_drawer.json" in s
    assert "--kitchens 813" in s
    assert "--out /home/u/simvla_out/composed_goals" in s


def test_the_sbatch_carries_the_env_task_emit_needs():
    """The env exports proven end-to-end: BODex objects, curobo on PYTHONPATH, and SIMVLA_GOALS_DIR set
    to the out dir so the job and a later simvla_gen agree on where the goals live."""
    s = _script()
    assert "#SBATCH --gres=gpu:1" in s
    assert "export BODEX_OBJ_DIR=BODex_obj" in s
    assert "third_party/curobo/src" in s
    assert "export SIMVLA_GOALS_DIR=/home/u/simvla_out/composed_goals" in s
    assert "conda activate env_isaaclab" in s
    assert "source /home/u/miniconda3/etc/profile.d/conda.sh" in s


def test_submit_returns_the_job_id_from_a_fake_sbatch(tmp_path, monkeypatch):
    """submit_emit_job writes the script and shells out to `sbatch --parsable`, returning the job id it
    prints. A fake sbatch on PATH stands in for the real scheduler."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "sbatch"
    fake.write_text("#!/bin/bash\necho '1234567;cluster'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")

    job_id = emit_job.submit_emit_job(_script(), script_dir=str(tmp_path / "job"))
    assert job_id == "1234567"                              # ';cluster' suffix stripped
    assert (tmp_path / "job" / "compose_emit.sbatch").exists()


def test_submit_raises_with_the_reason_when_sbatch_refuses(tmp_path, monkeypatch):
    """A non-zero sbatch is surfaced as a RuntimeError carrying its stderr, so the button shows the
    real reason instead of a silent no-op."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "sbatch"
    fake.write_text("#!/bin/bash\necho 'sbatch: error: bad partition' >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")

    with pytest.raises(RuntimeError) as exc:
        emit_job.submit_emit_job(_script(), script_dir=str(tmp_path / "job"))
    assert "bad partition" in str(exc.value)


def test_submit_raises_filenotfound_off_a_slurm_node(tmp_path, monkeypatch):
    """With no sbatch on PATH (not a Slurm node), submit raises FileNotFoundError — the button turns
    that into 'run task_emit directly on a GPU node' guidance rather than failing silently."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        emit_job.submit_emit_job(_script(), script_dir=str(tmp_path / "job"))


def test_importing_emit_job_boots_nothing_heavy():
    """It runs on a login node inside the GUI import path, so importing it must not pull Omniverse or
    torch — a subprocess check, like task_composer's, so a resident module can't hide a regression."""
    import subprocess as sp
    import sys
    import textwrap

    here = os.path.dirname(os.path.abspath(__file__))
    proc = sp.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import emit_job
            banned = ("omni", "isaaclab", "pxr", "torch", "tkinter")
            pulled = sorted(m for m in sys.modules if m.split(".")[0] in banned)
            assert not pulled, f"emit_job pulled {pulled}"
            print("clean")
        """)],
        cwd=here, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.split() == ["clean"]
