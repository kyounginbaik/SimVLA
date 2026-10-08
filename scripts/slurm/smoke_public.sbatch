#!/usr/bin/env bash
# sbatch --array=0-2 scripts/slurm/smoke_public.sbatch
# Required: SIMVLA_PYTHON and the external asset variables in docs/installation.md.
# This checks startup/physics/cameras, NOT task success or dataset replay.
#SBATCH --job-name=simvla_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=simvla-public-smoke-%A_%a.log
set -euo pipefail
ulimit -c 0
REPO=${SIMVLA_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
PY=${SIMVLA_PYTHON:?Set SIMVLA_PYTHON to the simulator Python}
: "${SIMVLA_ASSETS_DIR:?Set SIMVLA_ASSETS_DIR}"
: "${SIMVLA_ROBOT_MODELS_DIR:?Set SIMVLA_ROBOT_MODELS_DIR}"
: "${BODEX_OBJ_DIR:?Set BODEX_OBJ_DIR}"
[[ ${OMNI_KIT_ACCEPT_EULA:-} == YES ]] || {
  echo 'Review and accept the NVIDIA EULA before setting OMNI_KIT_ACCEPT_EULA=YES' >&2
  exit 2
}
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) ROBOT=anubis; TASK=Isaac-Kitchen-v813-00 ;;
  1) ROBOT=rby1; TASK=Isaac-Kitchen-v813r-00 ;;
  2) ROBOT=aiworker; TASK=Isaac-Kitchen-v813a-00 ;;
  *) echo 'Array indices must be 0 (anubis), 1 (rby1), or 2 (aiworker)' >&2; exit 2 ;;
esac
export PYTHONPATH="$REPO/src:$REPO/scripts/simvla:$REPO/source/isaaclab_tasks:$REPO/source/isaaclab_assets:$REPO/source/isaaclab:$REPO/source/isaaclab_rl"
unset PYTHONHOME
export PXR_WORK_THREAD_LIMIT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$PXR_WORK_THREAD_LIMIT
export OPENBLAS_NUM_THREADS=$PXR_WORK_THREAD_LIMIT
OUTPUT=${SIMVLA_SMOKE_ROOT:-$REPO/outputs}/smoke-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}-$ROBOT
cd "$REPO"
"$PY" scripts/simvla/smoke_test.py --headless --robot "$ROBOT" --task "$TASK" \
  --steps "${SIMVLA_SMOKE_STEPS:-100}" --seed "${SIMVLA_SEED:-0}" \
  --goal_file "$REPO/examples/goals/$TASK.json" --output_dir "$OUTPUT" \
  "--kit_args=--/plugins/carb.tasking.plugin/threadCount=$PXR_WORK_THREAD_LIMIT"
