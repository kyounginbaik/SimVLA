# Installation

SimVLA has an independently usable CPU package and a research simulator path. Start with the CPU package unless you need USD generation, collection, replay, or evaluation.

| Path | Use | Requirements | First verified result |
| --- | --- | --- | --- |
| CPU package | Tasks, skills, VQA conversion, procedural GLB scenes | Python 3.10–3.12 | `outputs/quickstart/{kitchen.glb,task.json,manifest.json}` |
| Isaac Sim overlay | USD generation, collection, replay, evaluation | Tested simulator stack plus external assets | Finite joints and camera images from `simvla run smoke` |

## CPU package

Requires Linux or macOS and Python 3.10–3.12.

```bash
git clone https://github.com/kyounginbaik/SimVLA.git
cd SimVLA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[scenes]'
simvla doctor
simvla demo --output-dir outputs/quickstart
```

The demo should create:

```text
outputs/quickstart/
├── kitchen.glb
├── manifest.json
└── task.json
```

Run the portable checks with:

```bash
python -m pip install -e '.[scenes,dev]'
python -m pytest tests
```

Use `python -m pip install -e .` if you only need task and VQA tools.

## Isaac Sim research environment

For new export setups, use the [separate CPU exporter](../env/README.md#separate-offline-exporter-python-310-linux-x86_64)
and set `SIMVLA_EXPORT_PYTHON` in the public collection launcher. This keeps
LeRobot/Rerun's NumPy requirements out of the simulator process. Pair it with the
[raw-only simulator recipe](../env/README.md#raw-simulator-compatibility-recipe),
which passes dependency checks, CPU regressions and three-robot GPU camera/physics
smokes. Fresh public-runtime collection, replay and SimVQA also pass for Anubis
and RB-Y1. AI Worker also has a bounded physics-12 loaded-home collection, LeRobot
export, first-pass replay, and SimVQA pass for the documented kitchen-813 seed. This
is a reference recipe, not evidence of broad multi-seed robustness. See [validation](validation.md).

For historical investigation only, the original research environment used Linux,
Python 3.10.19, Isaac Sim 4.5.0, Torch 2.5.1+cu124, NumPy 2.2.6 and LeRobot 0.4.2.
An intermediate Torch 2.6.0/NumPy 1.26.4 compatibility overlay still had a
LeRobot/Rerun dependency conflict. Do not use either as the new-install recipe:
the separate simulator/export setup above replaces them and passes `pip check`.

`env/simulator-conda.yml` and `env/simulator-pip.txt` are byte-for-byte snapshots of the research checkout's environment files at source commit `db771603`. They provide exact package pins for investigation, but are not yet a verified one-command clean install. The pip file explicitly bypasses dependency resolution. See the [environment snapshot notes](../env/README.md) for observed build prerequisites and use a separate environment when trying these files.

In your configured simulator environment:

```bash
python -m pip install --no-deps .
export SIMVLA_REPO_ROOT="$PWD"
export SIMVLA_ASSETS_DIR=/path/to/isaaclab_assets/data
export SIMVLA_ROBOT_MODELS_DIR=/path/to/robot-models
export BODEX_OBJ_DIR=/path/to/BODex_obj
export SIMVLA_LEROBOT_ROOT="$PWD/outputs/lerobot"
simvla doctor --require-sim
simvla run smoke --repo-root . -- --headless
```

Before the first headless or Slurm run, each user must accept NVIDIA's Omniverse/Isaac Sim
license. Start Isaac Sim once from an interactive session and answer its prompt, or, after
reviewing and accepting the license, set `OMNI_KIT_ACCEPT_EULA=YES` in that user's job
environment. Batch jobs cannot answer the prompt. SimVLA does not set this variable for you.

The first renderer startup on a fresh node can take several minutes to compile
shaders. In the 2026-10-06 RTX 3090 checks, the first app update took about four
minutes before environment initialization continued. The detailed Kit log prints
`Waiting for RtPso async group async compilation` during this stage. Allow startup
time in Slurm limits; this message alone is not a task-planning failure.

GPU users working from a Git clone should initialize the pinned dependencies:

```bash
git submodule update --init --recursive
cd third_party/curobo
git apply ../curobo.simvla.patch
cd ../..
```

The repository release archive has no Git metadata or submodule contents. From an extracted
archive, acquire the same pinned revisions directly:

```bash
git clone https://github.com/NVlabs/curobo third_party/curobo
git -C third_party/curobo checkout 0a50de1ba72db304195d59d9d0b1ed269696047f
git -C third_party/curobo apply ../curobo.simvla.patch
git clone https://github.com/Physical-Intelligence/openpi third_party/openpi
git -C third_party/openpi checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
```

Download the tested public object subset with:

```bash
python -m pip install huggingface_hub
export BODEX_OBJ_DIR=/path/to/BODex_obj
python scripts/tools/download_bodex.py \
  --allow-patterns use_data.tar.gz graspdata_final.tar.gz \
  --revision 3d0326ecc8f6074f3cb1796a249acebda564cab7
test -d "$BODEX_OBJ_DIR/use_data" && test -d "$BODEX_OBJ_DIR/graspdata_final"
python scripts/tools/build_grasp_manifest.py --bodex-root "$BODEX_OBJ_DIR"
```

Robot models, complete kitchen assets, real recordings, and policy checkpoints are external. Continue with [simulator workflows](simulation.md) for exact commands and [asset provenance](asset-provenance.md) for sources and hashes.

### External asset acquisition status

Large assets stay outside this repository. Set the paths above to your own writable asset tree;
do not copy a research asset directory into a public fork. The currently available routes are:

| Input | How to obtain it | Public standalone reproduction |
| --- | --- | --- |
| BODex object meshes and grasps | Run the revision-pinned `download_bodex.py` command above; the helper verifies archive hashes. | Yes, for the tested subset. |
| Anubis and kitchen 813 | Download the [SimVLA asset bundle](https://huggingface.co/datasets/kyounginbaik/SimVLA-assets), verify SHA-256 `b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8`, and extract it outside this repository. | Yes, exact tested inputs are hash-locked. |
| AI Worker FFW-SG2 | Included in the [SimVLA asset bundle](https://huggingface.co/datasets/kyounginbaik/SimVLA-assets). The separate official [ROBOTIS `cyclo_lab` repository](https://github.com/ROBOTIS-GIT/cyclo_lab) is a different Isaac Sim 5.1 / Isaac Lab 2.2+ asset revision. | Yes, exact tested input is hash-locked; keep upstream terms. |
| RB-Y1 | Included in the [SimVLA asset bundle](https://huggingface.co/datasets/kyounginbaik/SimVLA-assets) as the prepared combined model. The separate [Rainbow Robotics simulator repository](https://github.com/RainbowRobotics/rby1-sim-isaac) provides different Isaac Sim 5.1 assets. | Yes, exact prepared model and its USD layers are hash-locked. |

See [asset provenance](asset-provenance.md) for exact identity checks and component-level terms.
The public bundle is external to the source repository; project-maintainer permission to publish
project-controlled assets does not supersede upstream third-party terms.

The external archive is generated deterministically from the verified stage. Maintainers can
rebuild it with the pinned asset tools:

```bash
.asset-tools/bin/python scripts/tools/package_asset_bundle.py \
  --stage outputs/repro-assets-portable \
  --output /path/to/share/simvla-assets.tar.zst
```

The published archive SHA-256 is
`b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8` (633,215,795 bytes).
Download the archive and verify it before extracting. Set the environment paths to its extracted
directories as shown; keep the asset tree external to the Git checkout:

```bash
curl -L https://huggingface.co/datasets/kyounginbaik/SimVLA-assets/resolve/e7ea01d56d00b70afd05a45bca21460cf7389589/simvla-assets.tar.zst -o /tmp/simvla-assets.tar.zst
echo 'b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8  /tmp/simvla-assets.tar.zst' | sha256sum -c -
mkdir -p external/simvla-assets
tar --zstd -xf /tmp/simvla-assets.tar.zst -C external/simvla-assets
python scripts/tools/verify_asset_bundle.py external/simvla-assets
export SIMVLA_REPO_ROOT="$PWD"
export SIMVLA_ASSETS_DIR="$PWD/external/simvla-assets/assets"
export SIMVLA_ROBOT_MODELS_DIR="$PWD/external/simvla-assets/models"
export SIMVLA_RBY1M_DIR="$PWD/external/simvla-assets/rby1m"
```

The per-file manifest inside the archive records source revisions and hashes. Review the asset
terms in the dataset card and upstream notices before use. Some tasks that use object grasps still
require the separately downloadable BODex subset noted above.

For AI Worker, regenerate the small **planning** URDF outside the verified bundle.
The archive's older generated model omitted the gripper-base rotation: its wrist
and finger centres matched, but its finger collision shapes were rotated 180°.
The simulation USD is unchanged. The public goal/collection launchers do this
automatically; for direct CLI commands, run:

```bash
export SIMVLA_AIWORKER_URDF_PATH="$PWD/outputs/planning/ffw_sg2_follower.urdf"
python scripts/tools/build_aiworker_urdf.py \
  --source "$SIMVLA_ROBOT_MODELS_DIR/ai_worker_min/ffw_description/urdf/ffw_bg2_rev5_follower/ffw_bg2_follower.urdf" \
  --output "$SIMVLA_AIWORKER_URDF_PATH"
```

The corrected output SHA-256 is
`796dc31d6d3d7e607e7ba5662cbea742906268393a7bb9758e6051d79ac21fe5`.
Both goal clearance and collection honor this override. In the simulator Python,
`scripts/tools/check_aiworker_urdf_vs_usd.py --urdf "$SIMVLA_AIWORKER_URDF_PATH"`
with `--usd` pointing to the AI Worker USD checks the wrist and all ten hand frames.
This is a geometry consistency check, not an episode-success claim. New research
stages use the corrected generated-model hash; they will not recreate the byte-identical
older published archive above. Keep that archive immutable and apply this local overlay.

### Reproduce the three-robot kitchen setup from a research checkout

If you have the original SimVLA research checkout and the prepared RB-Y1 model tree, stage a
verified, writable asset layout. The command checks fourteen pinned source files in
[`research-asset-lock.json`](research-asset-lock.json), builds the AI Worker planning URDF, and
writes an environment file. It links large files instead of copying the 3.5 GB kitchen corpus. The
RB-Y1 USD bundle includes a generated layer with an absolute workstation reference; install the
small pinned USD helper and normalize it to a relative reference during staging:

```bash
python -m venv .asset-tools
.asset-tools/bin/python -m pip install --require-hashes -r env/asset-tools.txt
python scripts/tools/stage_research_assets.py \
  --source-root /path/to/research-simvla \
  --rby1-root /path/to/rby1m \
  --usd-python .asset-tools/bin/python \
  --output outputs/repro-assets-portable
source outputs/repro-assets-portable/env.sh
simvla doctor --robot anubis --repo-root .
simvla doctor --robot rby1 --repo-root .
simvla doctor --robot aiworker --repo-root .
```

Re-running the stage command verifies the existing layout. The output is ignored by Git. Its
links require the two source trees to remain at their original locations. For object placement
and collection, download the pinned BODex subset below and pass `--bodex-root` when staging;
the directory must contain both `use_data/` and `graspdata_final/`. See the
[three-robot smoke recipe](simulation.md#three-robot-kitchen-smoke) for simulator verification.

The research assets are not included in the public source repository; the hash-locked robot and
kitchen bundle is publicly downloadable from Hugging Face. A dependency-consistent clean Isaac Sim
installation and fresh GPU end-to-end collection remain outstanding before a new user can reproduce
the entire GPU pipeline from public downloads alone.

### Fetch upstream robot model sources (not the tested SimVLA asset bundle)

These commands acquire immutable upstream revisions for inspection or for a separately validated
Isaac Sim 5.1 workflow. They do not create the kitchen 813 scene, convert assets for Isaac Sim 4.5,
or reproduce the local hash-locked models above. Review each upstream repository's license and
asset terms before redistribution.

```bash
mkdir -p external
git clone https://github.com/ROBOTIS-GIT/cyclo_lab external/cyclo_lab
git -C external/cyclo_lab checkout f4c0470a5e0af54a18327cf96967e8716d64dbc0
sha256sum external/cyclo_lab/source/cyclo_lab/data/robots/FFW/FFW_SG2.usd
# Expected: a058ecf3266714440bef9ff26d7fe46a1ef60f768f0d5dee754b252b05688689

git clone https://github.com/RainbowRobotics/rby1-sim-isaac external/rby1-sim-isaac
git -C external/rby1-sim-isaac checkout 2417a2b2c83bc80b3ad605ab14f4d508d90089a9
sha256sum external/rby1-sim-isaac/assets/model_v_1_2_m.usd
# Expected: ab0890c4b0bbe0086c331fe1f8da79c9f3c31dea54b7f0c2c95c756bd65b2b85
```

The RB-Y1 repository documents Isaac Sim 5.1.0, while the ROBOTIS repository requires Isaac Lab
2.2 or newer. This project's verified environment is Isaac Sim 4.5.0. Treat these upstream sources
as useful starting points, not drop-in replacements. A public URL for the separately built
owner-authorized asset bundle is still needed for fully standalone GPU reproduction.

## Troubleshooting

- Run `simvla doctor` to inspect the active installation.
- Use `simvla doctor --require-sim` for simulator checks.
- Isaac Sim 4.5 declares `numpy<2`, but the observed research environment uses NumPy 2.2.6. Treat that environment as a compatibility exception, not a recommended clean-install recipe.
- Run commands from the repository root and use absolute asset paths.
- Check [validation](validation.md) before comparing research results.
