---
license: other
task_categories:
  - robotics
tags:
  - isaac-sim
  - mobile-manipulation
  - simvla
  - robot-assets
---

# SimVLA simulator asset bundle

This dataset repository distributes the external simulator assets needed by the tested SimVLA
three-robot kitchen setup. It is intentionally separate from the source repository and is not a
dataset of demonstrations, trained weights, or policy checkpoints.

## Integrity and provenance

Download `simvla-assets.tar.zst` and verify this exact SHA-256 before extraction:

```text
b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8
```

The archive is 633,215,795 bytes and expands to 4,472,738,159 bytes across 21,228 files. Its
`BUNDLE-MANIFEST.json` records source revisions and SHA-256/size/mode for every archived file.
The inputs are pinned to SimVLA source revision
`db77160323852abf59527f17efdf02b767181a5f` and RB-Y1 source revision
`a3274bb2cba307cc114108479eac6d55fc9557cf`. RB-Y1's generated root USD has been normalized to a
relative reference so the archive does not depend on its original workstation path.

## Download and configure

From a SimVLA source checkout, download the archive, verify it, and extract it outside the
checkout:

```bash
curl -L https://huggingface.co/datasets/kyounginbaik/SimVLA-assets/resolve/e7ea01d56d00b70afd05a45bca21460cf7389589/simvla-assets.tar.zst \
  -o /tmp/simvla-assets.tar.zst
echo 'b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8  /tmp/simvla-assets.tar.zst' \
  | sha256sum -c -
mkdir -p external/simvla-assets
tar --zstd -xf /tmp/simvla-assets.tar.zst -C external/simvla-assets
python scripts/tools/verify_asset_bundle.py external/simvla-assets
export SIMVLA_REPO_ROOT="$PWD"
export SIMVLA_ASSETS_DIR="$PWD/external/simvla-assets/assets"
export SIMVLA_ROBOT_MODELS_DIR="$PWD/external/simvla-assets/models"
export SIMVLA_RBY1M_DIR="$PWD/external/simvla-assets/rby1m"
```

See the SimVLA [`docs/installation.md`](https://github.com/kyounginbaik/SimVLA/blob/main/docs/installation.md)
and [`docs/asset-provenance.md`](https://github.com/kyounginbaik/SimVLA/blob/main/docs/asset-provenance.md)
for simulator setup, asset terms, and validation boundaries. The separately downloadable BODex
mesh/grasp subset remains required for placement/collection tasks that use object grasps.

## Asset terms

The SimVLA maintainer has authorized publication of project-controlled assets in this bundle.
That authorization does not replace third-party licenses or vendor terms. In particular, the
Rainbow Robotics SDK material retains Apache-2.0 terms; NVIDIA Omniverse/Isaac Sim content remains
subject to NVIDIA's terms. Review `ASSET-README.md`, notices within the archive, and upstream terms
before using or redistributing individual assets. This repository uses `license: other` because a
single open-source license does not cover every binary asset here. A compatible Isaac Sim
installation and any required account/license acceptance are the user's responsibility.
