# Simulator and export environments

Use separate environments for raw simulation and LeRobot export. The research
snapshot below is a bootstrap input, not a dependency-consistent environment.

Budget storage separately for environments, download caches, extracted assets,
raw frames and exported videos. The tested simulator environment alone occupies
about 24 GB; this is not a total installation-space estimate. On shared clusters,
check your user quota as well as `df` (for example `lfs quota -u "$USER" /lustre`).
The launcher's small durable-write probe detects an already exhausted quota,
but cannot guarantee space for the rest of a collection run.

## Raw simulator compatibility recipe

### Direct public-package installation

Use an empty Python 3.10 environment on Linux x86_64, with a supported NVIDIA
driver and CUDA toolkit/compiler. This route does not copy the research environment:

```bash
python3.10 -m venv .venv-sim
unset PYTHONPATH PYTHONHOME
.venv-sim/bin/python -m pip install -r env/simulator-raw.in
git submodule update --init third_party/curobo
git -C third_party/curobo apply --check ../curobo.simvla.patch
git -C third_party/curobo apply ../curobo.simvla.patch
# Set CUDA_HOME to your CUDA toolkit; 8.6 is the RTX 3090 architecture.
CUDA_HOME=/path/to/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=1 \
  .venv-sim/bin/python -m pip install --no-deps --no-build-isolation -e third_party/curobo
.venv-sim/bin/python -m pip install --no-deps --no-build-isolation \
  -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks \
  -e source/isaaclab_rl -e .
.venv-sim/bin/python -m pip check
```

For a source archive, use the pinned cuRobo clone commands in
[installation](../docs/installation.md) instead of the submodule command; apply
the patch only once. Do not reuse a native cuRobo binary compiled against a
different Torch version. Do not install the separate exporter into this venv.

On 2026-10-06 this public-only install and native cuRobo build passed, including
the final editable-package dependency check (job 2423262): CPython 3.10.20,
Torch 2.5.1+cu124, NumPy 1.26.4, CUDA toolkit 12.8.93 and GCC 11.4.
The exact new installation then passed 100-step physics/RGB checks for all three
robots (2424366), with another clean dependency check afterward. See
[the evidence record](../docs/evidence/clean-public-runtime-smoke-2026-10-06.json).
The first startup exposed optional GUI/VR imports, now deferred until those
features are requested. Fresh-runtime collection, physical replay and SimVQA also
passed for Anubis (2426755 → 2429275) and RBY1 (2424756 → 2426057).
AI Worker also has a bounded fresh-runtime extended-carry pass (2436772 → 2437094),
using physics-rate hybrid motor/contact replay and 0.9 seconds extra terminal settling.
Its separate physics-12 loaded-home/repaired-basin collection, replay and SimVQA
reference (2437904 → 2437908) passes for the documented seed, with zero extra
terminal hold steps. Neither result is action-only replay or broad reliability
evidence. See [current validation](../docs/validation.md).

After downloading assets, setting their documented environment variables, and
personally accepting the NVIDIA EULA, run all three camera/physics checks on Slurm:

```bash
export SIMVLA_REPO_ROOT="$PWD"
export SIMVLA_PYTHON="$PWD/.venv-sim/bin/python"
sbatch --array=0-2 scripts/slurm/smoke_public.sbatch
```

Add your cluster's partition/resource options. The source and environment must
be available on the compute node. These checks do not collect demonstrations or
certify task success; follow the separate collection/replay validation steps.

### Historical snapshot bootstrap

After the snapshot bootstrap commands below, keep its paired Torch 2.5.1 and
torchvision 0.20.1 CUDA 12.4 packages and apply:

```bash
/path/to/new-env/bin/python -m pip uninstall -y \
  beaker-py decord rl-games lerobot rerun-sdk torchaudio
/path/to/new-env/bin/python -m pip install -r env/simulator-gpu-overrides.txt
```

Then apply the cuRobo patch and install the editable simulator packages using the
commands below, **without the inline-export Torch upgrade**. Build cuRobo against
the final Torch version. Install the separate exporter and set
`SIMVLA_EXPORT_PYTHON` when collecting. Direct collector invocations must use
`--skip_finalize`; inline export intentionally requires LeRobot.

On 2026-10-06 an isolated recreation had a clean `pip check`, imported Torch
2.5.1+cu124 / torchvision 0.20.1+cu124 / NumPy 1.26.4, resolved cuRobo from this
checkout, and passed 426 selected portable/source regression tests (one skip).
Neither LeRobot nor Rerun was installed. The same recreation then passed 30-step
camera/physics smokes for all three robots on an RTX 3090 (job 2398448), with a
clean dependency check on the compute node and no compatibility overlay.
[Recorded evidence](../docs/evidence/raw-runtime-smoke-2026-10-06.json) covers this
installation check, not complete collection. Subsequent kitchen-813 references
passed collection/export/physical replay/SimVQA for all three robots. Anubis and AI Worker
use recorded joint targets and base-pose feedback during replay; follow the
exact profiles in the [current reproducibility map](../docs/reproducibility.md).

The editable `isaaclab`, `isaaclab_tasks`, and `isaaclab_rl` package metadata now
pins Torch 2.5.1 (and torchvision 0.20.1 where required), matching this raw runtime.
Earlier metadata still requested Torch 2.6.0 and could silently change a fresh install.
`simulator-raw.in` and `simulator-tested-constraints.txt` are a direct public-package
installation candidate under validation; do not treat them as a completed GPU
reproduction until their clean-install and physical workflow gates pass. Constraints
pin versions, not artifact hashes, and do not include local editable packages.

For direct smoke invocations on an eight-CPU allocation, also set
`PXR_WORK_THREAD_LIMIT=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` and pass
`--kit_args="--/plugins/carb.tasking.plugin/threadCount=8"`. Use your allocation's
CPU count. The public collection launcher sets these automatically; otherwise
Kit may create a pool sized to the entire host rather than the Slurm allocation.

## Historical snapshot and inline-export overlay

These two files are byte-identical copies of `environment.yml` and `requirements.txt` in
the local SimVLA research checkout at commit `db77160323852abf59527f17efdf02b767181a5f`.
Their SHA-256 values are `96d8131cee692628141df80fe64f8573112d6b2b160c03a6be9746774950c32d`
and `4a13797bbd1241b857630a236fa20c41a87c2404b05ca094603f85cbd0a69dcd`,
respectively.

They capture the author's installed packages, not a dependency-consistent lock. In particular,
the snapshot pins NumPy 2.2.6 while Isaac Sim 4.5 declares `numpy<2`, and its pip file says to
install with `--no-deps`. Do not replace an existing working environment with this snapshot.

An isolated recreation can start with:

```bash
conda env create -f env/simulator-conda.yml -p /path/to/new-env
/path/to/new-env/bin/python -m pip install 'setuptools==80.10.2' \
  'poetry-core==1.9.1' 'numpy==2.2.6' 'Cython==3.2.3'
/path/to/new-env/bin/python -m pip install --no-deps --no-build-isolation \
  'toppra==0.6.3'
/path/to/new-env/bin/python -m pip install --no-deps --no-build-isolation \
  -r env/simulator-pip.txt
```

The first two commands supply legacy source-build prerequisites and prebuild `toppra`; they are
bootstrap steps, not evidence that the final simulator environment is compatible. See
[installation](../docs/installation.md#isaac-sim-research-environment) and
[validation](../docs/validation.md) for the latest clean-install status and runtime checks.

cuRobo's native extension needs an explicit CUDA architecture when built on a login node without a
visible GPU. Apply the compatibility overrides below before building it, so the native extension
is compiled against the final Torch version. Set `TORCH_CUDA_ARCH_LIST=8.6` for the RTX 3090
validation machine; use your GPU's architecture rather than copying `8.6` blindly.

The NVIDIA Omniverse EULA must be accepted by the user before a noninteractive GPU job; the
validation jobs used `OMNI_KIT_ACCEPT_EULA=YES` after confirming acceptance in the existing
simulator installation. The frozen snapshot above is an exact capture of the research checkout,
not the supported compatibility set. After it is installed, apply the GPU overrides and the
PyTorch CUDA 12.4 triplet:

```bash
/path/to/new-env/bin/python -m pip install -r env/simulator-gpu-overrides.txt
/path/to/new-env/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
/path/to/new-env/bin/python -m pip uninstall -y beaker-py decord rl-games
/path/to/new-env/bin/python -m pip install --no-deps --no-build-isolation \
  -e source/isaaclab -e source/isaaclab_rl -e source/isaaclab_tasks
```

The overrides align NumPy and protobuf with Isaac Lab/Isaac Sim, pair Torch with the LeRobot
version in the snapshot, match `boto3`/`botocore`/`s3transfer` to Isaac Kit's bundled Botocore
1.34.68, pin a NumPy-1-compatible OpenCV wheel, and remove optional research packages whose
constraints conflict with the simulator stack. Install the pinned cuRobo patch and native extension
as above. The PyTorch version pairing follows its
[official CUDA 12.4 install matrix](https://docs.pytorch.org/get-started/previous-versions/).

Then, from the repository root, apply the pinned cuRobo patch, build its native extension, and
install the editable simulator packages:

```bash
git submodule update --init --recursive
git -C third_party/curobo apply --check ../curobo.simvla.patch
git -C third_party/curobo apply ../curobo.simvla.patch
TORCH_CUDA_ARCH_LIST=8.6 /path/to/new-env/bin/python -m pip install \
  --no-deps --no-build-isolation -e third_party/curobo
/path/to/new-env/bin/python -m pip install --no-deps --no-build-isolation \
  -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks \
  -e source/isaaclab_rl -e .
/path/to/new-env/bin/python -m pip check
/path/to/new-env/bin/python -m pytest tests -q
```

The original fresh clone had a failing `pip check` before these overrides. It reported:

- Isaac Sim 4.5, Isaac Lab 0.39, `dex-retargeting`, and `cmeel-boost` require NumPy below 2
  or in the 1.26 series; the observed environment has NumPy 2.2.6.
- Isaac Lab RL and tasks require protobuf below 5; the observed environment has protobuf 6.33.6.
- LeRobot 0.4.2 requests torchvision 0.21–0.22; the observed environment has torchvision 0.20.1.
- Isaac Sim and LeRobot each report missing `boto3` and `cmake`, respectively.
- Other optional-package findings include `readme-renderer`/`docutils`, `rl-games`/`wandb`,
  and `torchaudio`/`torch` version mismatches; `decord` is unsupported on this platform.

Those findings describe the original clone before overrides. On the corrected overlay, `pip check`
now reports one upstream metadata conflict: LeRobot 0.4.2 requires `rerun-sdk>=0.24`, and even
Rerun 0.24.1 declares `numpy>=2` in its [published package metadata](https://pypi.org/pypi/rerun-sdk/0.24.1/json),
while Isaac Sim 4.5 requires NumPy below 2. SimVLA's collection/export path does not import Rerun,
and `rerun` imports successfully with NumPy 1.26.4, but this is not a clean
dependency graph; do not hide or waive that finding. A first GPU smoke exposed and fixed a separate
Botocore shadowing issue; the all-robot retry is recorded in [validation](../docs/validation.md).
For inline export, continue describing this exact compatibility overlay and its
runtime evidence rather than claiming a conflict-free simulator install.

## Separate offline exporter (Python 3.10, Linux x86_64)

Raw collection with `--skip_finalize` no longer imports LeRobot or Rerun. Export
can run in a separate CPU environment with NumPy 2, without installing Isaac Sim:

```bash
python3.10 -m venv .venv-export
.venv-export/bin/python -m pip install -r env/exporter-pip.lock.txt
.venv-export/bin/python -m pip check
```

The version-pinned export environment uses hash-pinned official CPU Torch wheels.
It is separate from the historical simulator snapshots above; **do not install
its requirements into Isaac Sim's environment**. The remaining packages are
version-pinned, not artifact-hash-locked. Offline export tests cover all three
robot schemas using synthetic episodes, real Parquet and decoded videos. They
are data-integrity checks, not physical collection success.

For the public collection launcher, set
`SIMVLA_EXPORT_PYTHON=/absolute/path/to/.venv-export/bin/python`. It stages data
in the simulator, strips the compatibility-overlay import/library paths before
starting the exporter, records separate export provenance, then validates the
result. For already staged data:

```bash
.venv-export/bin/python scripts/tools/offline_finalize_lerobot.py \
  --stage /path/containing/000000/ \
  --goal /path/to/the/exact/collection-goal.json \
  --robot anubis --output /path/to/new/export
```

Here `--stage` is the parent of numbered episode directories, not `000000` itself.
On Slurm, the exporter environment must be visible on the compute node (for
example on shared storage); a login-node-only `/tmp` environment is insufficient.
Use `rby1` or `aiworker` for their respective recordings. Existing outputs are
refused; nothing is uploaded. The historical initial-frame removal remains in
place, but joint angles now use the same frame slice as images, state and actions.
Unequal stream lengths fail before a staged episode is written.
Re-exporting old staged arrays cannot repair an already recorded joint-angle
offset; regenerate those arrays from the original recordings or collect new data.

This separation removes the need to resolve incompatible NumPy requirements in
one process. Dependency consistency and isolated simulator smokes are verified
above. The two-process launcher also passed the documented Anubis and RB-Y1
physical chains and AI Worker's explicitly qualified hybrid-and-settling reference.
