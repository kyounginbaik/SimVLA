#!/usr/bin/env bash
# Replay an already validated local all/ dataset in a fresh output directory.
# Required: SIMVLA_REPO_ROOT, SIMVLA_PYTHON, SIMVLA_DATASET, SIMVLA_GOALS_DIR,
# SIMVLA_ASSETS_DIR, SIMVLA_ROBOT_MODELS_DIR, ROBOT, TASK.
# Set the same physics/contact settings used by collection.
#SBATCH --job-name=simvla_replay
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=simvla-public-replay-%j.log
#SBATCH --error=simvla-public-replay-%j.log
set -euo pipefail
ulimit -c 0
REPO=${SIMVLA_REPO_ROOT:?Set SIMVLA_REPO_ROOT}
PY=${SIMVLA_PYTHON:?Set SIMVLA_PYTHON}
DATASET=${SIMVLA_DATASET:?Set SIMVLA_DATASET to the finalized all directory}
[[ -f "$DATASET/meta/info.json" ]] || { echo "Missing dataset metadata" >&2; exit 2; }
: "${SIMVLA_GOALS_DIR:?Set SIMVLA_GOALS_DIR to the collected goal directory}"
: "${SIMVLA_ASSETS_DIR:?Set SIMVLA_ASSETS_DIR}"
: "${SIMVLA_ROBOT_MODELS_DIR:?Set SIMVLA_ROBOT_MODELS_DIR}"
: "${ROBOT:?Set ROBOT}" "${TASK:?Set TASK}"
RUN_ID=${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}
RUN_ROOT="${SIMVLA_COLLECTION_ROOT:-$REPO/outputs}/simvla-replay-${ROBOT}-${RUN_ID}"
case "$ROBOT" in
  anubis) ;;
  rby1)
    export SIMVLA_RBY1M_DIR="${SIMVLA_RBY1M_DIR:-$(dirname "$SIMVLA_ASSETS_DIR")/rby1m}"
    ;;
  aiworker)
    export SIMVLA_AIWORKER_USD_PATH="${SIMVLA_AIWORKER_USD_PATH:-$SIMVLA_ASSETS_DIR/Robots/MM/aiworker/ffw_sg2_simvla_grip.usd}"
    ;;
  *) echo "unsupported ROBOT: $ROBOT" >&2; exit 2 ;;
esac
if [[ ${SIMVLA_DRY_RUN:-0} == 1 ]]; then
  printf 'robot=%s\ndataset=%s\nrby1m=%s\naiworker_usd=%s\noutput=%s\n' \
    "$ROBOT" "$DATASET" "${SIMVLA_RBY1M_DIR:-}" "${SIMVLA_AIWORKER_USD_PATH:-}" "$RUN_ROOT"
  exit 0
fi
[[ ${OMNI_KIT_ACCEPT_EULA:-} == YES ]] || { echo "Read and accept the Omniverse EULA first" >&2; exit 2; }
export SIMVLA_CPU_THREADS=${SIMVLA_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
[[ "$SIMVLA_CPU_THREADS" =~ ^[1-9][0-9]{0,2}$ ]] && (( SIMVLA_CPU_THREADS <= 256 )) || exit 2
export PXR_WORK_THREAD_LIMIT=$SIMVLA_CPU_THREADS OMP_NUM_THREADS=$SIMVLA_CPU_THREADS OPENBLAS_NUM_THREADS=$SIMVLA_CPU_THREADS
export PYTHONPATH="$REPO/src:$REPO/scripts/simvla:$REPO/source/isaaclab_tasks:$REPO/source/isaaclab_assets:$REPO/source/isaaclab:$REPO/source/isaaclab_rl:$REPO/third_party/curobo/src"
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
export SIMVLA_STATUS_FILE="$RUN_ROOT/replay-status.json"
"$PY" "$REPO/scripts/tools/release_evidence.py" --repo-root "$REPO" --runtime \
  --dataset "$DATASET" \
  --artifact "$DATASET/meta/info.json" --artifact "$SIMVLA_GOALS_DIR/$TASK.json" \
  --artifact "$REPO/scripts/simvla/simvla_replay.py" \
  --artifact "$REPO/scripts/simvla/replay_targets.py" \
  --artifact "$REPO/scripts/simvla/replay_initial_state.py" \
  --artifact "$REPO/scripts/simvla/replay_initial_step.py" \
  --artifact "$REPO/scripts/simvla/replay_substeps.py" \
  --artifact "$REPO/scripts/simvla/sink_collision.py" \
  --artifact "$REPO/source/isaaclab/isaaclab/managers/action_manager.py" \
  --artifact "$REPO/source/isaaclab_assets/isaaclab_assets/robots/aiworker_collisions.py" \
  --artifact "$REPO/scripts/simvla/calibration/$ROBOT.json" \
  --artifact "$REPO/source/isaaclab/isaaclab/simvla/simvqa.py" \
  --artifact "$REPO/src/simvla/vqa.py" \
  --artifact "$REPO/scripts/tools/validate_simvqa_capture.py" \
  --output "$RUN_ROOT/source-evidence.json"
cd "$RUN_ROOT"
ARGS=()
if [[ -n ${SIMVLA_REPLAY_EPISODE_INDEX:-} ]]; then
  [[ "$SIMVLA_REPLAY_EPISODE_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "SIMVLA_REPLAY_EPISODE_INDEX must be a nonnegative integer" >&2; exit 2;
  }
  ARGS+=(--episode_index "$SIMVLA_REPLAY_EPISODE_INDEX")
fi
if [[ ${SIMVLA_SIMVQA:-0} == 1 ]]; then
  ARGS+=(--simvqa --vqa_output_dir "$RUN_ROOT/simvqa")
fi
REPLAY_EXIT=0
"$PY" "$REPO/scripts/simvla/simvla_replay.py" --task "$TASK" --robot "$ROBOT" \
  --dataset_file "$DATASET" --local --num_envs 1 --seed "${SIMVLA_SEED:-0}" \
  --max_passes "${SIMVLA_REPLAY_PASSES:-1}" --headless --enable_cameras \
  "${ARGS[@]}" \
  --kit_args="--/plugins/carb.tasking.plugin/threadCount=$SIMVLA_CPU_THREADS" || REPLAY_EXIT=$?
"$PY" - "$REPLAY_EXIT" "$SIMVLA_STATUS_FILE" <<'PYCODE'
import sys
from workflow_status import resolve_exit_code
raise SystemExit(resolve_exit_code(int(sys.argv[1]), sys.argv[2]))
PYCODE
if [[ ${SIMVLA_SIMVQA:-0} == 1 ]]; then
  "$PY" "$REPO/scripts/tools/validate_simvqa_capture.py" "$RUN_ROOT/simvqa" \
    --output "$RUN_ROOT/simvqa-result.json"
  "$PY" -m simvla vqa "$RUN_ROOT/simvqa" --output "$RUN_ROOT/vqa-qa.jsonl"
fi
