"""Build and submit the Slurm job that runs task_emit for a composed task.

This is the engine behind the composer's "Generate goals (this kitchen)" button. Generating goals is
a GPU batch — it boots Isaac Sim and loads each kitchen rotation's USD — so it does NOT run inside the
browser composer or block the generator GUI. The button hands the composed template to a Slurm job
(its own GPU allocation, no contention with the running generator) and reports the job id.

Pure stdlib on purpose: this must import and be tested on a login node, with no Omniverse and no
tkinter. The script it produces mirrors, line for line, the harness proven end-to-end (env exports +
`isaaclab.sh -p scripts/simvla/task_emit.py`), so the button runs exactly the validated path.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def resolve_out_dir() -> str:
    """Where the goal files go. $SIMVLA_GOALS_DIR when set (so a flag-free simvla_gen run picks them
    up with no extra config), else a scratch dir under simvla_out. task_emit itself refuses the real
    committed corpus, so this never overwrites it even if SIMVLA_GOALS_DIR points there."""
    env = os.environ.get("SIMVLA_GOALS_DIR")
    if env:
        return env
    return str(Path(os.path.expanduser("~")) / "simvla_out" / "composed_goals")


def kitchen_is_on_disk(kitchen_dir, num: int) -> bool:
    """True if kitchen <num>'s first rotation USD exists on disk — the precondition task_emit needs
    (it loads kitchen_<num>_<sub>.usd for each rotation). The 'Generate goals' button checks this so it
    can say 'Accept & Generate this kitchen first' rather than submitting a job that skips every
    rotation for a kitchen that was only ever previewed."""
    return Path(kitchen_dir, f"kitchen_{num:02d}_00.usd").exists()


def env_defaults() -> dict:
    """conda_base / conda_env / hf_user read from the LIVE environment, so the submitted job
    reproduces the environment the generator runs in rather than one user's hardcoded paths. CONDA_EXE
    is '<base>/bin/conda', so its grandparent is the conda base whose profile.d/conda.sh the job
    sources. Falls back to sensible defaults when a var is unset."""
    conda_exe = os.environ.get("CONDA_EXE", "")
    conda_base = (
        os.path.dirname(os.path.dirname(conda_exe)) if conda_exe
        else os.path.expanduser("~/miniconda3")
    )
    return {
        "conda_base": conda_base,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or "env_isaaclab",
        "hf_user": os.environ.get("HF_USER") or os.environ.get("USER") or "",
    }


def build_emit_sbatch(
    *,
    template_path: str,
    kitchens: str,
    out_dir: str,
    repo_root: str,
    conda_base: str,
    conda_env: str,
    hf_user: str,
    partition: str = "gpu",  # generic placeholder; override with your cluster's own GPU partition name
    bodex_obj_dir: str = "BODex_obj",
    log_dir: str | None = None,
    time_limit: str = "00:45:00",
    job_name: str = "compose_emit",
) -> str:
    """The sbatch script text that runs the validated task_emit for <kitchens> into <out_dir>.

    kitchens is task_emit's --kitchens string (e.g. '813' — which fans out to all 12 rotations). The
    env exports reproduce the environment the generator itself runs in, so the job is not tied to one
    user's hardcoded paths: conda_base/conda_env/hf_user are read from the live environment by the
    caller. SIMVLA_GOALS_DIR is set to out_dir so the job and a later simvla_gen agree on the location.
    """
    log_dir = log_dir or out_dir
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/{job_name}_%j.log

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
source {conda_base}/etc/profile.d/conda.sh
conda activate {conda_env}
cd {repo_root}
export BODEX_OBJ_DIR={bodex_obj_dir}
export PYTHONPATH="$PWD/third_party/curobo/src:$PYTHONPATH"
export HF_USER={hf_user}
export SIMVLA_LEROBOT_ROOT="$HOME/simvla_out/lerobot"
export SIMVLA_GOALS_DIR={out_dir}
export PYTHONUNBUFFERED=1
mkdir -p {out_dir}

echo "=== task_emit: template={template_path} kitchens={kitchens} -> {out_dir} ==="
./isaaclab.sh -p scripts/simvla/task_emit.py \\
  --template {template_path} \\
  --kitchens {kitchens} \\
  --out {out_dir}
echo "=== EXIT=$? ==="
"""


def submit_emit_job(script_text: str, *, script_dir: str | None = None) -> str:
    """Write the script to a file and `sbatch --parsable` it; return the Slurm job id (a string).

    Raises RuntimeError with sbatch's stderr if submission fails, and FileNotFoundError if sbatch is
    not on PATH (i.e. this is not a Slurm login node) — both surfaced to the button so the user sees a
    real reason rather than a silent no-op."""
    script_dir = script_dir or tempfile.mkdtemp(prefix="simvla_emit_job_")
    Path(script_dir).mkdir(parents=True, exist_ok=True)
    script_path = Path(script_dir) / "compose_emit.sbatch"
    script_path.write_text(script_text, encoding="utf-8")

    proc = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch refused the job (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    # --parsable prints just the job id (optionally 'jobid;cluster'); take the leading number.
    return proc.stdout.strip().split(";")[0].strip()
