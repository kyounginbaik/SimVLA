# Release reference comparison

This comparison evaluates repository usability and reproducibility, not research quality or task
difficulty. It was checked on 2026-09-30 against these immutable revisions:

- [SimToolReal `313d5ae`](https://github.com/tylerlum/simtoolreal/tree/313d5aea1f507c6cfe097b672b62945d7b0bbff5)
- [FlashSAC `87edc90`](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab)
- SimVLA `ab092676dd48ea5c05af7b5dc8a40879359892dd`, before this document was added

## Patterns worth keeping

| Reference | Strong onboarding and reproducibility patterns | Evidence |
| --- | --- | --- |
| SimToolReal | A checkpoint downloader leads directly to an interactive demo; one command evaluates all 24 benchmark combinations. Separate guides cover the recommended Isaac Sim path, the legacy paper path, task creation, data acquisition, and deployment. Its simulator guide pins tested versions and gives standalone smoke commands. | [README](https://github.com/tylerlum/simtoolreal/blob/313d5aea1f507c6cfe097b672b62945d7b0bbff5/README.md), [Isaac Sim installation](https://github.com/tylerlum/simtoolreal/blob/313d5aea1f507c6cfe097b672b62945d7b0bbff5/docs/isaacsim_installation.md), [DexToolBench guide](https://github.com/tylerlum/simtoolreal/blob/313d5aea1f507c6cfe097b672b62945d7b0bbff5/docs/dextoolbench.md), [deployment guide](https://github.com/tylerlum/simtoolreal/blob/313d5aea1f507c6cfe097b672b62945d7b0bbff5/docs/deployment.md) |
| FlashSAC | A lockfile-backed `uv sync` setup leads to a default training command. Hardware/Python combinations, optional simulator extras, and incompatible extras are explicit. Per-backend launch scripts, checkpoint/resume instructions, policy visualization, video recording, and committed result CSVs make the experiment path easy to inspect. CI repeats the locked install and static checks. | [README](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/README.md), [dependency model](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/pyproject.toml), [training entry point](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/train.py), [CI workflow](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/.github/workflows/test.yaml), [published results](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab/results) |

## SimVLA's current position

SimVLA already provides several release features at the same level or beyond these references:

- The CPU path installs independently and produces a kitchen, task, and manifest with one command;
  it does not require Isaac Sim for task authoring, skill inspection, VQA conversion, or GLB scene
  generation. See [installation](installation.md).
- Package CI spans Python 3.10–3.12, builds the wheel and source archive, checks both distributions,
  reinstalls the wheel, reruns tests, and records artifact provenance. The cuRobo patch is checked
  separately. See [release validation](validation.md) and [release instructions](releasing.md).
- Task and skill contracts are machine-validated before simulator startup, with an executable
  extension example and explicit limits on what CPU validation proves. See [task authoring](task-authoring.md).
- Dataset revisions, row counts, schemas, and license gaps are recorded rather than inferred from
  mutable dataset viewers. See [datasets](datasets.md).
- The evaluation protocol separates paper results, public reproductions, and integration smokes;
  it requires attempted-trial denominators, fixed splits and seeds, hashes, failure retention, and
  per-task outcomes. See [evaluation protocol](evaluation-protocol.md).
- The [reproducibility map](reproducibility.md) and [release audit](release-audit.md) connect each
  workflow to commands, evidence, and known limits instead of treating repository presence as proof.

## Gaps to close

These are the remaining release gaps, ordered by their effect on an unaffiliated user:

1. **Publish a complete reproduction bundle.** Provide rights-cleared robot and kitchen assets, a
   compatible checkpoint and normalization state, the training configuration and data manifest,
   and expected evaluation output. SimToolReal currently offers the clearest checkpoint-to-demo-to-
   benchmark path; SimVLA cannot yet reproduce a paper result from public materials alone.
2. **Verify a clean GPU installation.** Replace the inherited simulator environment with a pinned,
   fresh-machine recipe. Record the dependency solution, `pip check`, smoke output, GPU/driver
   matrix, and known incompatible combinations. FlashSAC's lockfile and explicit extra conflicts are
   a useful model.
3. **Make one public end-to-end path the default.** A new user should be able to acquire one task's
   assets, run one bounded collection, convert its VQA data, and evaluate a supplied policy by
   copying a short sequence of commands. Publish the expected files, runtime, and a reference video.
4. **Run simulator checks in hosted automation.** Keep the current CPU and distribution gates, then
   add a version-pinned simulator smoke on suitable GPU infrastructure. A local cluster job is useful
   evidence but cannot protect the public branch.
5. **Publish and verify the release externally.** Push the release commit and signed or annotated tag,
   observe hosted CI, publish checksummed artifacts and sanitized validation logs, then test anonymous
   downloads from a clean account and machine.
6. **Commission an independent reproduction.** Ask someone without access to the research asset tree
   to follow the public instructions. Preserve their failures and resulting fixes as release evidence.

The first three items determine whether SimVLA is a runnable research release. The remaining items
turn that runnable path into durable public evidence.
