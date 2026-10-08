# Repository structure

Start with the [reproducibility map](reproducibility.md) when choosing between the portable CPU package and asset-dependent simulator workflows.

| Directory | Purpose |
| --- | --- |
| `src/simvla/` | Installable CPU library, one-command demo, task validation, VQA conversion, bundled task templates |
| `examples/goals/` | Executable goals for the two retained kitchen tasks |
| `examples/vqa/` | One actual simulator capture with RGB/segmentation images for CPU conversion |
| `scripts/simvla/` | Supported workflow entry points and their authoring/runtime helpers |
| `scripts/simvla/calibration/` | Robot layouts and system-identification mappings |
| `source/isaaclab*/` | Modified Isaac Lab runtime, assets definitions, two example kitchen configs, real-to-sim task |
| `third_party/` | Pinned dependency references and cuRobo patch |
| `apps/` | Isaac Sim application configurations |
| `tests/` | Portable package tests; additional research tests live beside simulator scripts |
| `docs/` | Setup, tested capabilities, and provenance |
| `outputs/` | Ignored local outputs; created by examples |

The wheel intentionally excludes Isaac Lab, simulator scripts, model assets, recordings, and checkpoints. Use a source checkout for GPU workflows. The source archive includes the workflow code but not third-party submodule contents.

Generated task corpora, experiment campaigns, deployment prototypes, private data, and machine-specific dependency freezes are excluded from this release. Existing copyright headers and third-party notices are retained.
