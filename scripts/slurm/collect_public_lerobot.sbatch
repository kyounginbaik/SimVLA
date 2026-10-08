#!/usr/bin/env bash
# One public-bundle LeRobot collection trial. Submit once per robot with
# `sbatch --export=ALL,ROBOT=anubis scripts/slurm/collect_public_lerobot.sbatch`.
# The goal's requested sub-grasp count is reduced to one because this trial
# collects one demonstration; retaining the emitter's default of seven wastes
# GPU time exploring extra accepted grasps before completing the task.
#SBATCH --job-name=simvla_lerobot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=simvla-public-lerobot-%x-%j.log
#SBATCH --error=simvla-public-lerobot-%x-%j.log

set -euo pipefail
# Native simulator teardown can abort after a Python exception. Keep its traceback
# and videos without filling the working filesystem with multi-gigabyte core files.
if [[ ${SIMVLA_ALLOW_CORE_DUMPS:-0} != 1 ]]; then
  ulimit -c 0
fi

REPO=${SIMVLA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PY=${SIMVLA_PYTHON:-python}
PY=$(command -v "$PY") || { echo "Set SIMVLA_PYTHON to the simulator Python executable" >&2; exit 2; }
EXPORT_PY=${SIMVLA_EXPORT_PYTHON:-}
if [[ -n "$EXPORT_PY" ]]; then
  EXPORT_PY=$(command -v "$EXPORT_PY") || { echo "SIMVLA_EXPORT_PYTHON is not executable" >&2; exit 2; }
fi
OVERLAY=${SIMVLA_COMPAT_OVERLAY:-}
ASSETS=${SIMVLA_ASSETS_DIR:?Set SIMVLA_ASSETS_DIR; see docs/installation.md}
MODELS=${SIMVLA_ROBOT_MODELS_DIR:?Set SIMVLA_ROBOT_MODELS_DIR; see docs/installation.md}
BODEX=${BODEX_OBJ_DIR:?Set BODEX_OBJ_DIR; see docs/installation.md}
RUN_ID=${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}
RUN_ROOT=${SIMVLA_COLLECTION_ROOT:-$REPO/outputs}/simvla-public-lerobot-${ROBOT:?set ROBOT to anubis, rby1, or aiworker}-${RUN_ID}
GOALS="$RUN_ROOT/goals"
export SIMVLA_SEED="${SIMVLA_SEED:-0}"
export SIMVLA_NUM_DEMOS="${SIMVLA_NUM_DEMOS:-1}"
if [[ ! "$SIMVLA_NUM_DEMOS" =~ ^[1-9][0-9]{0,5}$ ]]; then
  echo "SIMVLA_NUM_DEMOS must be an integer between 1 and 999999" >&2
  exit 2
fi
export SIMVLA_CPU_THREADS="${SIMVLA_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
if [[ ! "$SIMVLA_CPU_THREADS" =~ ^[1-9][0-9]{0,2}$ ]] || (( SIMVLA_CPU_THREADS > 256 )); then
  echo "SIMVLA_CPU_THREADS must be an integer between 1 and 256" >&2
  exit 2
fi
# Kit otherwise sees every host CPU, even inside a much smaller Slurm allocation.
# Excess worker contention can stall its first renderer update for minutes.
export PXR_WORK_THREAD_LIMIT="$SIMVLA_CPU_THREADS"
export OMP_NUM_THREADS="$SIMVLA_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$SIMVLA_CPU_THREADS"
if [[ ! "$SIMVLA_SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || (( SIMVLA_SEED > 4294967295 )); then
  echo "SIMVLA_SEED must be an integer between 0 and 4294967295" >&2
  exit 2
fi

case "$ROBOT" in
  anubis)
    TASK=Isaac-Kitchen-v813-00
    SOURCE_GOAL="$REPO/examples/goals/Isaac-Kitchen-v813-00.json"
    DEFAULT_ARM=right
    # A bowl candidate parked 19 mm from the target while the 12 mm wrist gate
    # kept jogging. The subsequent physical grasp gate remains active.
    export SIMVLA_GRASP_POS_TOL="${SIMVLA_GRASP_POS_TOL:-0.02}"
    ;;
  rby1)
    TASK=Isaac-Kitchen-v813r-00
    SOURCE_GOAL="$REPO/examples/goals/Isaac-Kitchen-v813r-00.json"
    DEFAULT_ARM=right
    export SIMVLA_GRASP_GEOMETRY=cylindrical_mug
    export SIMVLA_RBY1M_DIR="${SIMVLA_RBY1M_DIR:-$(dirname "$ASSETS")/rby1m}"
    ;;
  aiworker)
    TASK=Isaac-Kitchen-v813a-00
    SOURCE_GOAL="$REPO/examples/goals/Isaac-Kitchen-v813a-00.json"
    DEFAULT_ARM=left
    export SIMVLA_AIWORKER_USD_PATH="$ASSETS/Robots/MM/aiworker/ffw_sg2_simvla_grip.usd"
    export SIMVLA_POSTGRASP_LIFT=0.20
    export SIMVLA_POSTGRASP_LIFT_LEFT=1
    export SIMVLA_RESET_ROT_TOL_DEG="${SIMVLA_RESET_ROT_TOL_DEG:-30}"
    ;;
  *)
    echo "unsupported ROBOT='$ROBOT' (expected anubis, rby1, or aiworker)" >&2
    exit 2
    ;;
esac

# Keep collector quota, selected grasp candidate, and goal metadata aligned
# when a task uses the non-default arm for an embodiment.
COLLECT_ARM=${SIMVLA_COLLECT_ARM:-$DEFAULT_ARM}
case "$COLLECT_ARM" in
  right) ARGS=(--sub_good_goal_count_r 1 --sub_good_goal_count_l 0) ;;
  left)
    ARGS=(--sub_good_goal_count_r 0 --sub_good_goal_count_l 1)
    export SIMVLA_POSTGRASP_LIFT_LEFT=1
    ;;
  *) echo "SIMVLA_COLLECT_ARM must be left or right" >&2; exit 2 ;;
esac

if [[ -n ${SIMVLA_TASK:-} ]]; then
  [[ -n ${SIMVLA_SOURCE_GOAL:-} ]] || {
    echo "SIMVLA_TASK requires an explicit SIMVLA_SOURCE_GOAL" >&2; exit 2;
  }
  TASK=$SIMVLA_TASK
fi
SOURCE_GOAL=${SIMVLA_SOURCE_GOAL:-$SOURCE_GOAL}
ARGS+=("--kit_args=--/plugins/carb.tasking.plugin/threadCount=$SIMVLA_CPU_THREADS")
if [[ -n "$EXPORT_PY" ]]; then
  ARGS+=(--skip_finalize)
fi
if [[ ${SIMVLA_FIXED_OBJECT:-0} == 1 ]]; then
  ARGS+=(--obj_fix_init)
fi
for required_dir in "$REPO/src/simvla" "$ASSETS" "$MODELS" "$BODEX"; do
  [[ -d "$required_dir" ]] || { echo "Missing directory: $required_dir" >&2; exit 2; }
done
[[ -f "$SOURCE_GOAL" ]] || { echo "Missing goal: $SOURCE_GOAL" >&2; exit 2; }
PYTHONPATH="$REPO/src:$REPO/scripts/simvla" "$PY" - "$SOURCE_GOAL" "$TASK" "$ROBOT" "$REPO" "$COLLECT_ARM" <<'PYCODE'
import json
import re
import sys
from pathlib import Path
import skills  # register the same canonical declarations used by the collector
from executor_dispatch import load_script
task, robot, repo, arm = sys.argv[2:]
match = re.fullmatch(r"Isaac-Kitchen-v([0-9]+)([ar]?)-([0-9]{2})", task)
suffix = {"anubis": "", "rby1": "r", "aiworker": "a"}[robot]
if not match or match[2] != suffix:
    raise SystemExit(f"Task {task!r} does not match robot {robot}")
goal = json.loads(Path(sys.argv[1]).read_text())
if (goal.get("kitchen_num"), goal.get("kitchen_sub_num")) != (int(match[1]), int(match[3])):
    raise SystemExit("Task and source goal kitchen/rotation metadata disagree")
object_key = "obj_name_l" if arm == "left" else "obj_name"
cfg = Path(repo) / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen" / f"kitchen_{match[1]}{suffix}_{match[3]}.py"
if not cfg.is_file():
    raise SystemExit(f"Missing generated task configuration: {cfg}; emit/register the task first")
steps, _, _ = load_script(sys.argv[1])
expected_action = "A_l" if arm == "left" else "A_r"
if goal.get("version") == 2 and any(step.skill == "arm.grasp" for step in steps):
    if goal.get("run_config", {}).get(object_key) in (None, "", "none"):
        raise SystemExit(f"Goal has no target object for selected {arm} grasp arm")
    if not any(step.skill == "arm.grasp" and step.action == expected_action for step in steps):
        raise SystemExit(f"Goal has no arm.grasp step on selected {arm} arm")
print(f"[preflight] executor accepted {len(steps)} goal steps")
PYCODE
if [[ -n "$OVERLAY" ]]; then
  [[ -d "$OVERLAY/lib/python3.10/site-packages" ]] || {
    echo "Invalid SIMVLA_COMPAT_OVERLAY: expected lib/python3.10/site-packages" >&2; exit 2;
  }
fi
if [[ ${SIMVLA_DRY_RUN:-0} == 1 ]]; then
  printf 'num_demos=%s\n' "$SIMVLA_NUM_DEMOS"
  printf 'robot=%s\ntask=%s\npython=%s\nrepo=%s\ngoal=%s\noutput=%s\noverlay=%s\nseed=%s\nexport_python=%s\ncpu_threads=%s\n' \
    "$ROBOT" "$TASK" "$PY" "$REPO" "$SOURCE_GOAL" "$RUN_ROOT" "${OVERLAY:-none}" "$SIMVLA_SEED" "${EXPORT_PY:-inline}" "$SIMVLA_CPU_THREADS"
  exit 0
fi
[[ ${OMNI_KIT_ACCEPT_EULA:-} == YES ]] || {
  echo "Read the NVIDIA Omniverse EULA and, if you accept it, export OMNI_KIT_ACCEPT_EULA=YES" >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing existing run directory: $RUN_ROOT" >&2; exit 2; }
# Filesystem free-space totals do not account for per-user Lustre quotas. Verify
# an actual durable write before booting CUDA/Kit. This is not a capacity reservation.
"$PY" - "$RUN_ROOT" <<'PYCODE'
import os
from pathlib import Path
import sys
import tempfile
parent = Path(sys.argv[1]).parent
parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryFile(dir=parent) as stream:
    stream.write(bytes(65536))
    stream.flush()
    os.fsync(stream.fileno())
print(f"[storage] durable write probe passed in {parent}")
PYCODE
export SIMVLA_REPO_ROOT="$REPO"
export SIMVLA_SCRIPTS="$REPO/scripts/simvla"
export SIMVLA_ASSETS_DIR="$ASSETS"
export SIMVLA_ROBOT_MODELS_DIR="$MODELS"
export BODEX_OBJ_DIR="$BODEX"
export SIMVLA_GOALS_DIR="$GOALS"
export SIMVLA_LEROBOT_ROOT="$RUN_ROOT/lerobot"
export SIMVLA_EPISODE_STEPS="${SIMVLA_EPISODE_STEPS:-3000}"
export SIMVLA_MAXFRAMES="${SIMVLA_MAXFRAMES:-8000}"
export SIMVLA_POSTGRASP_LIFT="${SIMVLA_POSTGRASP_LIFT:-0.20}"
export SIMVLA_PLANNER_ATTEMPTS="${SIMVLA_PLANNER_ATTEMPTS:-2}"
export SIMVLA_TRAJOPT_SEEDS="${SIMVLA_TRAJOPT_SEEDS:-4}"
export SIMVLA_GRAPH_SEEDS="${SIMVLA_GRAPH_SEEDS:-4}"
export SIMVLA_FINETUNE_TRAJOPT="${SIMVLA_FINETUNE_TRAJOPT:-1}"
export SIMVLA_CUROBO_WORLD="${SIMVLA_CUROBO_WORLD:-live}"
export PYTHONPATH="$REPO/src:${OVERLAY:+$OVERLAY/lib/python3.10/site-packages:}$REPO/scripts/simvla:$REPO/source/isaaclab_tasks:$REPO/source/isaaclab_assets:$REPO/source/isaaclab:$REPO/source/isaaclab_rl:$REPO/third_party/curobo/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$OVERLAY" ]]; then
  export LD_LIBRARY_PATH="$OVERLAY/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Catch unusable allocations before Kit can hang while creating its GPU context.
# This checks CUDA only; a successful result does not certify Vulkan/rendering.
"$PY" - <<'PYCODE'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: CUDA is unavailable in the selected Python environment")
try:
    probe = torch.ones(1, device="cuda")
    probe.add_(1)
    torch.cuda.synchronize()
    assert probe.item() == 2
except Exception as exc:
    raise SystemExit(f"GPU preflight failed: {type(exc).__name__}: {exc}") from exc
print(f"[gpu] {torch.cuda.get_device_name()} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
PYCODE

mkdir -p "$GOALS" "$RUN_ROOT/raw" "$SIMVLA_LEROBOT_ROOT"
MODEL_EVIDENCE=()
if [[ "$ROBOT" == aiworker ]]; then
  # The pinned archive's planning URDF predates the corrected jaw rotations.
  # Regenerate outside the immutable asset tree and use it for this run only.
  if [[ -z ${SIMVLA_AIWORKER_URDF_PATH:-} ]]; then
    export SIMVLA_AIWORKER_URDF_PATH="$RUN_ROOT/planning/ffw_sg2_follower.urdf"
    "$PY" "$REPO/scripts/tools/build_aiworker_urdf.py" \
      --source "$MODELS/ai_worker_min/ffw_description/urdf/ffw_bg2_rev5_follower/ffw_bg2_follower.urdf" \
      --output "$SIMVLA_AIWORKER_URDF_PATH"
  fi
  MODEL_EVIDENCE=(--artifact "$SIMVLA_AIWORKER_URDF_PATH")
fi
"$PY" - "$SOURCE_GOAL" "$GOALS/$TASK.json" "$ROBOT" "$COLLECT_ARM" "${SIMVLA_GRASP_CANDIDATE_INDEX:-}" <<'PYCODE'
import json
import subprocess
import sys
from pathlib import Path

src, dst, robot, arm, candidate = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
goal = json.loads(src.read_text())
if candidate:
    from select_goal_grasp import select_candidate
    goal = select_candidate(goal, arm, int(candidate))
    print(f"[goal] pinned authored BoDex {arm} candidate {candidate}", flush=True)
cfg = goal.setdefault("run_config", {})
cfg["sub_good_goal_count_r"] = int(arm == "right")
cfg["sub_good_goal_count_l"] = int(arm == "left")
dst.write_text(json.dumps(goal, indent=2, allow_nan=True) + "\n")
print(f"[goal] copied {src} -> {dst}; run_config={cfg}", flush=True)
PYCODE

echo "[run] robot=$ROBOT arm=$COLLECT_ARM task=$TASK host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} output=$RUN_ROOT" | tee "$RUN_ROOT/run.txt"
echo "[grasp] route=${SIMVLA_AIWORKER_GRASP_ROUTE:-direct} candidate=${SIMVLA_GRASP_CANDIDATE_INDEX:-all} world=${SIMVLA_CUROBO_WORLD:-collector-default} pause_fraction=0.5 pause_steps=51" | tee -a "$RUN_ROOT/run.txt"
echo "[grasp] axial_offset_m=${SIMVLA_DIRECT_APPROACH_OFFSET_M:-0} jaw_height_m=${SIMVLA_MUG_GRASP_HEIGHT:-collector-default}" | tee -a "$RUN_ROOT/run.txt"
echo "[mug-control] pregrasp_standoff_m=${SIMVLA_PREGRASP_STANDOFF:-collector-default} live_z_track=${SIMVLA_MUG_LIVE_Z_TRACK:-0} open_target_tol_m=${SIMVLA_GRIPPER_OPEN_TARGET_TOL_M:-off}" | tee -a "$RUN_ROOT/run.txt"
echo "[release] retreat_m=${SIMVLA_POSTRELEASE_RETREAT_M:-0} retreat_lift_m=${SIMVLA_POSTRELEASE_RETREAT_LIFT_M:-collector-default}" | tee -a "$RUN_ROOT/run.txt"
{
  printf '[settings] goal=%s grasp_pos_tol=%s grasp_rot_tol_deg=%s pose_pos_tol=%s pose_rot_tol_deg=%s reset_rot_tol_deg=%s gripper_preload=%s force_balance=%s postgrasp_lift=%s fixed_object=%s\n' \
    "$SOURCE_GOAL" "${SIMVLA_GRASP_POS_TOL:-default}" \
    "${SIMVLA_GRASP_ROT_TOL_DEG:-default}" \
    "${SIMVLA_POSE_POS_TOL:-default}" \
    "${SIMVLA_POSE_ROT_TOL_DEG:-default}" \
    "${SIMVLA_RESET_ROT_TOL_DEG:-default}" \
    "${SIMVLA_GRIPPER_PRELOAD_FRACTION:-0.40}" \
    "${SIMVLA_GRIPPER_FORCE_BALANCE:-0}" "$SIMVLA_POSTGRASP_LIFT" \
    "${SIMVLA_FIXED_OBJECT:-0}"
  printf '[release-profile] retreat_m=%s retreat_lift_m=%s final_home=%s\n' \
    "${SIMVLA_POSTRELEASE_RETREAT_M:-0}" \
    "${SIMVLA_POSTRELEASE_RETREAT_LIFT_M:-0}" \
    "${SIMVLA_POSTRELEASE_FINAL_HOME:-0}"
  if [[ -e "$REPO/.git" ]]; then
    printf '[source] commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  else
    printf '[source] archive (content hashes in source-evidence.json)\n'
  fi
  sha256sum "$SOURCE_GOAL" "$REPO/scripts/simvla/simvla_gen.py" \
    "$REPO/scripts/simvla/select_goal_grasp.py" \
    "$REPO/scripts/simvla/collector_profile.py" \
    "$REPO/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/contact_gripper.py" \
    "$REPO/source/isaaclab_assets/isaaclab_assets/robots/aiworker.py" \
    "$REPO/source/isaaclab_assets/isaaclab_assets/robots/rby1.py" \
    "$REPO/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/kitchen_813a_00.py" \
    "$REPO/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/kitchen_813r_00.py"
} >> "$RUN_ROOT/run.txt"
cd "$REPO"
ENV_CFG_STEM=${TASK#Isaac-Kitchen-v}
ENV_CFG_STEM=${ENV_CFG_STEM//-/_}
MODEL_EVIDENCE+=(--artifact "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/kitchen_${ENV_CFG_STEM}.py"
  --artifact scripts/simvla/collector_profile.py --artifact scripts/simvla/collision_mesh.py
  --artifact scripts/simvla/carried_collision.py
  --artifact source/isaaclab/isaaclab/managers/action_manager.py
  --artifact scripts/simvla/goal_bank_retry.py
  --artifact source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/registration.py
  --artifact source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/__init__.py
  --artifact "source/isaaclab_assets/isaaclab_assets/robots/${ROBOT}.py"
  --artifact source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py
  --artifact source/isaaclab/isaaclab/utils/datasets/episode_data.py
  --artifact source/isaaclab/isaaclab/envs/mdp/recorders/recorders.py
  --artifact source/isaaclab/isaaclab/envs/mdp/recorders/recorders_cfg.py
  --artifact source/isaaclab_assets/isaaclab_assets/robots/mobile_base.py
  --artifact source/isaaclab_assets/isaaclab_assets/robots/aiworker_collisions.py
  --artifact source/isaaclab_assets/isaaclab_assets/robots/anubis_wheels.py
  --artifact scripts/tools/check_collection_result.py
  --artifact scripts/tools/validate_lerobot_collection.py
  --artifact scripts/simvla/workflow_status.py --artifact scripts/slurm/collect_public_lerobot.sbatch
  --artifact scripts/simvla/sink_collision.py
  --artifact scripts/simvla/carried_collision.py
  --artifact src/simvla/gripper.py
  --artifact source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/contact_gripper.py)
if [[ "$ROBOT" == aiworker || "$ROBOT" == rby1 ]]; then
  for arm in left right; do
    MODEL_EVIDENCE+=(--artifact "configs/curobo/robot/${ROBOT}_${arm}_arm.yml"
      --artifact "configs/curobo/robot/spheres/${ROBOT}_spheres_${arm}.yml")
  done
else
  MODEL_EVIDENCE+=(--artifact "$ASSETS/curobo/robot/anubis_left_arm.yml"
    --artifact "$ASSETS/curobo/robot/anubis_right_arm.yml")
fi
"$PY" scripts/tools/release_evidence.py --repo-root "$REPO" --runtime \
  --artifact "$GOALS/$TASK.json" --artifact scripts/simvla/simvla_gen.py \
  "${MODEL_EVIDENCE[@]}" \
  --output "$RUN_ROOT/source-evidence.json"
if [[ -n ${SIMVLA_REVIEW_CAMERA:-} ]]; then
  ARGS+=(--dump_frames "$RUN_ROOT/review_frames" --dump_camera "$SIMVLA_REVIEW_CAMERA" \
    --dump_every "${SIMVLA_REVIEW_EVERY:-1}")
  echo "[review] camera=$SIMVLA_REVIEW_CAMERA every=${SIMVLA_REVIEW_EVERY:-1} frames=$RUN_ROOT/review_frames" | tee -a "$RUN_ROOT/run.txt"
fi
SIM_EXIT=0
export SIMVLA_STATUS_FILE="$RUN_ROOT/collector-status.json"
"$PY" scripts/simvla/simvla_gen.py \
  --task "$TASK" --robot "$ROBOT" --num_envs 1 --num_demos "$SIMVLA_NUM_DEMOS" \
  --good_goal_count 1 "${ARGS[@]}" --action_chunk_size 50 --seed "$SIMVLA_SEED" \
  --run_id "public_${ROBOT}_${RUN_ID}" --output_root "$RUN_ROOT/raw" \
  --enable_cameras --headless || SIM_EXIT=$?
SIM_EXIT=$("$PY" - "$SIM_EXIT" "$SIMVLA_STATUS_FILE" <<'PYCODE'
import sys
from workflow_status import resolve_exit_code
print(resolve_exit_code(int(sys.argv[1]), sys.argv[2]))
PYCODE
)
echo "[collector] exit_status=$SIM_EXIT" | tee -a "$RUN_ROOT/run.txt"

# Review captures are diagnostic videos, not successful demonstrations. Keep them even when
# the collection postcheck below correctly reports zero accepted LeRobot episodes.
if [[ -n ${SIMVLA_REVIEW_CAMERA:-} ]]; then
  FRAME_DIR="$RUN_ROOT/review_frames"
  FFMPEG=$(command -v "${SIMVLA_FFMPEG:-ffmpeg}" || true)
  if [[ -f "$FRAME_DIR/frame_000000.png" && -x "$FFMPEG" ]]; then
    "$FFMPEG" -hide_banner -loglevel error -y \
      -framerate "20/${SIMVLA_REVIEW_EVERY:-1}" -i "$FRAME_DIR/frame_%06d.png" \
      -c:v libx264 -crf 24 -pix_fmt yuv420p "$RUN_ROOT/review-${SIMVLA_REVIEW_CAMERA}.mp4"
    echo "[review] video=$RUN_ROOT/review-${SIMVLA_REVIEW_CAMERA}.mp4" | tee -a "$RUN_ROOT/run.txt"
    SECONDARY_CAMERA=${SIMVLA_REVIEW_SECONDARY_CAMERA:-}
    if [[ -n "$SECONDARY_CAMERA" && -f "$FRAME_DIR/secondary-$SECONDARY_CAMERA/frame_000000.png" ]]; then
      "$FFMPEG" -hide_banner -loglevel error -y \
        -framerate "20/${SIMVLA_REVIEW_EVERY:-1}" \
        -i "$FRAME_DIR/secondary-$SECONDARY_CAMERA/frame_%06d.png" \
        -c:v libx264 -crf 24 -pix_fmt yuv420p "$RUN_ROOT/review-$SECONDARY_CAMERA.mp4"
      echo "[review] secondary=$RUN_ROOT/review-$SECONDARY_CAMERA.mp4" | tee -a "$RUN_ROOT/run.txt"
    fi
  else
    echo "[review] no frame sequence or ffmpeg unavailable; see $FRAME_DIR" | tee -a "$RUN_ROOT/run.txt"
  fi
fi

# The simulator can terminate normally after its frame budget with zero accepted episodes.
# Its recorder HDF5 may also remain a 96-byte header even when a staged episode exists.
# Record negative outcomes too, so a campaign cannot silently count only its successes.
# A Slurm timeout/SIGKILL can prevent this tail from running: missing evidence is not success.
CHECK_EXIT=0
EXPORT_EXIT=0
CHECK_PY="$PY"
if [[ -n "$EXPORT_PY" ]]; then
  CHECK_PY="$EXPORT_PY"
  # Do not leak the simulator compatibility overlay into the clean export process.
  export PYTHONPATH="$REPO/src:$REPO/scripts/simvla:$REPO/source/isaaclab"
  unset LD_LIBRARY_PATH
  "$EXPORT_PY" scripts/tools/release_evidence.py --repo-root "$REPO" --runtime \
    --artifact env/exporter-pip.lock.txt --artifact scripts/tools/offline_finalize_lerobot.py \
    --artifact source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py \
    --output "$RUN_ROOT/export-evidence.json"
  if [[ "$SIM_EXIT" -eq 0 ]]; then
    "$EXPORT_PY" scripts/tools/offline_finalize_lerobot.py \
      --stage "$RUN_ROOT/raw/public_${ROBOT}_${RUN_ID}/$TASK" \
      --output "$SIMVLA_LEROBOT_ROOT/$TASK" --goal "$GOALS/$TASK.json" --robot "$ROBOT" \
      || EXPORT_EXIT=$?
  fi
fi
"$CHECK_PY" scripts/tools/check_collection_result.py "$RUN_ROOT" \
  --expected-episodes "$SIMVLA_NUM_DEMOS" \
  --collector-exit "$SIM_EXIT" --exporter-exit "$EXPORT_EXIT" || CHECK_EXIT=$?
if [[ "$SIM_EXIT" -ne 0 ]]; then
  echo "Collector failed; diagnostic frames/videos retained in $RUN_ROOT" >&2
  exit "$SIM_EXIT"
fi
if [[ "$CHECK_EXIT" -eq 0 ]]; then
  "$CHECK_PY" scripts/tools/release_evidence.py --repo-root "$REPO" \
    --dataset "$SIMVLA_LEROBOT_ROOT/$TASK/all" \
    --output "$RUN_ROOT/dataset-evidence.json"
fi
exit "$CHECK_EXIT"
