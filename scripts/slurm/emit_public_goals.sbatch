#!/usr/bin/env bash
# Emit v2 task goals from the pinned public kitchen on a GPU node.
# Example: sbatch --partition=YOUR_GPU_PARTITION --export=ALL,SIMVLA_GOAL_OUTPUT_ROOT=/path/to/my-goals \
#   scripts/slurm/emit_public_goals.sbatch
#SBATCH --job-name=simvla_goals
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=simvla-public-goals-%j.log
#SBATCH --error=simvla-public-goals-%j.log

set -euo pipefail

REPO=${SIMVLA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PY=${SIMVLA_PYTHON:-python}
PY=$(command -v "$PY") || { echo "Set SIMVLA_PYTHON to the simulator Python executable" >&2; exit 2; }
OVERLAY=${SIMVLA_COMPAT_OVERLAY:-}
ASSETS=${SIMVLA_ASSETS_DIR:?Set SIMVLA_ASSETS_DIR; see docs/installation.md}
MODELS=${SIMVLA_ROBOT_MODELS_DIR:?Set SIMVLA_ROBOT_MODELS_DIR; see docs/installation.md}
BODEX=${BODEX_OBJ_DIR:?Set BODEX_OBJ_DIR; see docs/installation.md}
TEMPLATE=${SIMVLA_TASK_TEMPLATE:-$REPO/src/simvla/templates/bowl_to_drawer.json}
OUTPUT=${SIMVLA_GOAL_OUTPUT_ROOT:?set an output directory outside the source goal corpus}
KITCHEN=${SIMVLA_KITCHEN_NUMBER:-813}
SUB=${SIMVLA_KITCHEN_SUB:-0}

for required_dir in "$REPO/src/simvla" "$ASSETS" "$MODELS" "$BODEX"; do
  [[ -d "$required_dir" ]] || { echo "Missing directory: $required_dir" >&2; exit 2; }
done
[[ -f "$TEMPLATE" ]] || { echo "Missing template: $TEMPLATE" >&2; exit 2; }
if [[ -n "$OVERLAY" ]]; then
  [[ -d "$OVERLAY/lib/python3.10/site-packages" ]] || {
    echo "Invalid SIMVLA_COMPAT_OVERLAY: expected lib/python3.10/site-packages" >&2; exit 2;
  }
fi
case ${SIMVLA_ROBOT:-anubis} in
  anubis|rby1|aiworker) ;;
  *) echo "SIMVLA_ROBOT must be anubis, rby1, or aiworker" >&2; exit 2 ;;
esac
if [[ ${SIMVLA_DRY_RUN:-0} == 1 ]]; then
  printf 'robot=%s\nrepo=%s\ntemplate=%s\noutput=%s\n' "${SIMVLA_ROBOT:-anubis}" "$REPO" "$TEMPLATE" "$OUTPUT"
  exit 0
fi
[[ ${OMNI_KIT_ACCEPT_EULA:-} == YES ]] || {
  echo "Read the NVIDIA Omniverse EULA and, if you accept it, export OMNI_KIT_ACCEPT_EULA=YES" >&2
  exit 2
}
export SIMVLA_REPO_ROOT="$REPO"
export SIMVLA_ASSETS_DIR="$ASSETS"
export SIMVLA_ROBOT_MODELS_DIR="$MODELS"
export SIMVLA_RBY1M_DIR="${SIMVLA_RBY1M_DIR:-$(dirname "$ASSETS")/rby1m}"
export BODEX_OBJ_DIR="$BODEX"
if [[ ${SIMVLA_ROBOT:-anubis} == rby1 ]]; then
  [[ -d "$SIMVLA_RBY1M_DIR" ]] || {
    echo "Missing RB-Y1 model directory: $SIMVLA_RBY1M_DIR" >&2; exit 2;
  }
fi
export PYTHONPATH="$REPO/src:${OVERLAY:+$OVERLAY/lib/python3.10/site-packages:}$REPO/scripts/simvla:$REPO/source/isaaclab_tasks:$REPO/source/isaaclab_assets:$REPO/source/isaaclab:$REPO/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$OVERLAY" ]]; then
  export LD_LIBRARY_PATH="$OVERLAY/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[run] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} template=$TEMPLATE kitchen=$KITCHEN sub=$SUB output=$OUTPUT"
sha256sum "$TEMPLATE" "$REPO/scripts/simvla/task_emit.py"
cd "$REPO"
if [[ ${SIMVLA_ROBOT:-anubis} == aiworker && -z ${SIMVLA_AIWORKER_URDF_PATH:-} ]]; then
  export SIMVLA_AIWORKER_URDF_PATH="$OUTPUT/.planning/ffw_sg2_follower-${SLURM_JOB_ID:-local-$$}.urdf"
  "$PY" scripts/tools/build_aiworker_urdf.py \
    --source "$MODELS/ai_worker_min/ffw_description/urdf/ffw_bg2_rev5_follower/ffw_bg2_follower.urdf" \
    --output "$SIMVLA_AIWORKER_URDF_PATH"
  sha256sum "$SIMVLA_AIWORKER_URDF_PATH"
fi
"$PY" scripts/simvla/task_emit.py --template "$TEMPLATE" \
  --kitchens "$KITCHEN" --subs "$SUB" --out "$OUTPUT"
# Geometry authoring uses the selected robot, but task_emit writes the common
# Anubis-shaped config and unsuffixed goal name. Finish the embodiment adapter
# before handing a generated task to collection.
case ${SIMVLA_ROBOT:-anubis} in
  rby1|aiworker)
    "$PY" "scripts/simvla/${SIMVLA_ROBOT}_kitchen_cfg.py" \
      --kitchen "$KITCHEN" --subs "$SUB" --goals-dir "$OUTPUT"
    ;;
esac
