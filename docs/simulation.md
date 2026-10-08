# Simulator workflows

This guide runs from a kitchen scene to robot demonstrations and SimVQA data. Run commands from the repository root in the [Isaac Sim environment](installation.md). Robot models, kitchen assets, BODex grasps, recordings, and checkpoints are external downloads; they are not in this Git repository.

## Runtime and assets

Follow [installation](installation.md) to set up Isaac Sim 4.5, the pinned cuRobo submodule and patch, the [public robot/kitchen bundle](installation.md#external-asset-acquisition-status), and the separate LeRobot export environment. Accept Isaac Sim's NVIDIA license before a headless or Slurm run. Set these paths to your extracted assets:

```bash
export SIMVLA_REPO_ROOT="$PWD"
export SIMVLA_ASSETS_DIR=/path/to/isaaclab_assets/data
export SIMVLA_ROBOT_MODELS_DIR=/path/to/robot-models
export SIMVLA_RBY1M_DIR=/path/to/rby1m
export BODEX_OBJ_DIR=/path/to/BODex_obj
simvla doctor --require-sim
simvla doctor --robot anubis --repo-root .
simvla doctor --robot rby1 --repo-root .
simvla doctor --robot aiworker --repo-root .
```

`SIMVLA_ASSETS_DIR` contains kitchen and robot USDs; `SIMVLA_ROBOT_MODELS_DIR` contains URDFs and meshes; `SIMVLA_RBY1M_DIR` points to the prepared RB-Y1 model; `BODEX_OBJ_DIR` contains object meshes and grasp banks. The BODex subset is a separate, revision-pinned download:

```bash
python -m pip install huggingface_hub
python scripts/tools/download_bodex.py \
  --allow-patterns use_data.tar.gz graspdata_final.tar.gz \
  --revision 3d0326ecc8f6074f3cb1796a249acebda564cab7
```

Initialize cuRobo with `git submodule update --init --recursive`, then apply its [local patch](installation.md#isaac-sim-research-environment) before building it. `doctor` checks files and dependencies; it does not prove a robot can complete a task.

### Three-robot kitchen smoke

These short runs check that each robot, physics, and cameras load. The bundled goal is used only for its spawn band in the RB-Y1 and AI Worker smokes; it is **not** a collection goal for those robots.

```bash
simvla run smoke --repo-root . -- \
  --task Isaac-Kitchen-v813-00 --robot anubis --steps 10 --headless \
  --goal_file examples/goals/Isaac-Kitchen-v813-00.json
simvla run smoke --repo-root . -- \
  --task Isaac-Kitchen-v813r-00 --robot rby1 --steps 10 --headless \
  --goal_file examples/goals/Isaac-Kitchen-v813-00.json
simvla run smoke --repo-root . -- \
  --task Isaac-Kitchen-v813a-00 --robot aiworker --steps 10 --headless \
  --goal_file examples/goals/Isaac-Kitchen-v813-00.json
```

For a tested collection → export → replay → SimVQA recipe for each robot, use the exact goals and profiles in the [reproducibility map](reproducibility.md). A smoke pass is not a successful demonstration.

## 1. Kitchen Scene Generation

Generate a kitchen, task template, and twelve rotated variants with an unused ID and a writable output directory:

```bash
export HF_USER="<your-hf-username>"
simvla run generate --repo-root . -- \
  --kitchen-id 99000 --seed 0 --layout l_shaped \
  --mesh "$BODEX_OBJ_DIR/use_data/core_mug_39361b14ba19303ee42cfae782879837/mesh/simplified.obj" \
  --output outputs/generated
```

The command refuses existing IDs or output directories. Another mesh needs matching BODex grasps. The example mug is wider than Anubis's nominal gripper opening, so scene generation alone does not establish a feasible grasp. Generated MDL materials may need an initial NVIDIA material-server download and shader compilation. Keep the kitchen ID, seed, inputs, and dependency versions to reproduce the authored choices; byte-identical USD and GPU physics are not guaranteed. The browser composer (`simvla run compose --repo-root .`) is experimental.

## 2. Goal Generation

Bind a task template to the generated kitchen. Omit `--subs` for all variants; `--subs 0` makes one quick authoring check.

```bash
export SIMVLA_ROBOT=anubis  # or rby1 or aiworker; set before goal emission
simvla run goals --repo-root . -- \
  --template outputs/generated/template.json --kitchens 99000 \
  --subs 0 --out outputs/authored-goals
export SIMVLA_GOALS_DIR="$PWD/outputs/authored-goals"
simvla validate-goal "$SIMVLA_GOALS_DIR/Isaac-Kitchen-v99000-00.json"
simvla run smoke --repo-root . -- \
  --task Isaac-Kitchen-v99000-00 --seed 0 --headless \
  --goal_file "$SIMVLA_GOALS_DIR/Isaac-Kitchen-v99000-00.json"
```

For RB-Y1 or AI Worker, adapt the emitted kitchen configs with `simvla adapt-rby1` or `simvla adapt-aiworker`; their task IDs become `v99000r-00` or `v99000a-00`. See [task authoring](task-authoring.md) for skills, templates, and the robot-specific emission order. Use `SIMVLA_ENV_CFG_OUTPUT_DIR` for scratch configs if you do not want generation to write into the checkout. Validate the robot-specific goal and run its smoke before collection.

## 3. Hybrid System Identification

Fit candidate stiffness and damping values to a robot-compatible Parquet episode:

```bash
simvla run identify --repo-root . -- \
  --robot anubis --control eef --num_envs 64 \
  --data_file /path/to/recording.parquet --episode 0 \
  --termination_t 839 --max_done 1 --max_steps 2000 \
  --json outputs/sysid.json --headless
```

The recording needs `episode_index`, `action`, `observation.state`, and `qpos`. Set `termination_t` to its episode length. The JSON contains candidates, not a ranked calibration.

## 4. SimAction Generation

For a first Anubis run using the bundled kitchen-813 task:

```bash
unset SIMVLA_GOALS_DIR
SIMVLA_POSTGRASP_LIFT=0.20 SIMVLA_EPISODE_STEPS=3000 \
simvla run collect --repo-root . -- \
  --task Isaac-Kitchen-v813-00 --robot anubis \
  --num_envs 1 --num_demos 1 --good_goal_count 1 --action_chunk_size 50 \
  --seed 0 --run_id quickstart --output_root outputs/collection \
  --enable_cameras --headless
```

Raw files go to `outputs/collection`; accepted LeRobot episodes go under `outputs/lerobot` (or `SIMVLA_LEROBOT_ROOT` if set). Use a new `run_id` per attempt. Check the collection result and accepted episode count: a clean process exit, grasp attempt, or video does not by itself establish task success. The named [Anubis, RB-Y1, and AI Worker reference recipes](reproducibility.md) include the required robot-specific goals, control profiles, export, and replay checks. Use those before scaling to a new task or seed.

For a new RB-Y1 task, `simvla init-task --from mug_to_sink --name my_mug_to_sink --output outputs/rby1/mug_to_sink.json` creates an editable skill sequence. Validate it, emit goals with `SIMVLA_ROBOT=rby1`, then run `simvla adapt-rby1 --repo-root . --kitchen 99000 --goals-dir /path/to/goals`. For AI Worker, use `src/simvla/templates/mug_to_sink_left.json`, emit with `SIMVLA_ROBOT=aiworker`, and run `simvla adapt-aiworker --repo-root . --kitchen 99000 --subs 0 --goals-dir /path/to/goals`. These are authoring paths, **not** claims that an arbitrary generated kitchen can be collected successfully.

AI Worker physical grasping also needs the local collision overlay and matching USD path:

```bash
./isaaclab.sh -p scripts/tools/patch_aiworker_gripper_collisions.py
export SIMVLA_AIWORKER_USD_PATH="$SIMVLA_ASSETS_DIR/Robots/MM/aiworker/ffw_sg2_simvla_grip.usd"
```

The overlay leaves the downloaded USD unchanged. Use the exact AI Worker profile in the [reproducibility map](reproducibility.md) for the validated physics-substep settings. Older sink goals checked proximity rather than basin containment. Use the repaired-basin goals and predicates when claiming in-basin placement. The [validation record](validation.md) has failure videos, diagnostics, and the limits of each reference.

## 5. SimVQA Generation

Replay an accepted LeRobot episode with the matching goal, robot profile, and initial scene state. This basic Anubis example captures segmentation and converts it to questions:

```bash
simvla run replay --repo-root . -- \
  --task Isaac-Kitchen-v813-00 --robot anubis --num_envs 4 --seed 0 \
  --dataset_file "outputs/lerobot/Put bowl inside drawer./Isaac-Kitchen-v813-00/all" \
  --local --max_passes 3 --simvqa \
  --vqa_output_dir outputs/vqa/Isaac-Kitchen-v813-00 \
  --enable_cameras --headless
simvla vqa outputs/vqa/Isaac-Kitchen-v813-00 --output outputs/questions.jsonl
```

Use the collection's saved goal and initial-object metadata for randomized scenes. Replay must pass the physical task predicate before treating VQA captures as validated. `--seed` pins sampling and initialization, not bitwise GPU physics. For a CPU-only converter example, run `simvla vqa examples/vqa --output outputs/questions.jsonl`.

## 6–7. Training and SimDeploy

The paper uses SimAction and SimVQA pre-training followed by SimAction and SimDeploy post-training. Training code, deployment adapters, and public checkpoints are not included.

## 8. Evaluate in IsaacLab

Smoke a held-out kitchen before evaluation:

```bash
simvla run smoke --repo-root . -- \
  --task Isaac-Kitchen-v99001-00 --seed 0 --headless
```

For zero-shot evaluation, choose a kitchen number excluded from training and report multiple variants and seeds. `simvla run evaluate` can replay a compatible recorded-action episode to check action application, rendering, and result writing; that is not a policy benchmark. Policy evaluation requires a compatible policy server, checkpoint, and embodiment-matched normalization. See [evaluation and evidence](validation.md).

## Slurm

One GPU was enough for the documented smokes. Replace the partition name and use the same launch prefix for other workflows:

```bash
srun --partition=YOUR_GPU_PARTITION --gres=gpu:1 --cpus-per-task=16 --mem=32G \
  python -m simvla run smoke --repo-root "$SIMVLA_REPO_ROOT" -- --headless \
  --kit_args "--/plugins/carb.tasking.plugin/threadCount=8 --/persistent/physics/numThreads=8"
```

`simvla run WORKFLOW --repo-root . --dry-run -- ...` prints a launch command without starting Isaac Sim.
