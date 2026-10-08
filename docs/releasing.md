# Release artifacts

Version `1.0.0` targets the initial open-source release. The public source is on `main`; a versioned GitHub release and tag have not yet been published. The portable CPU package is independently installable. Simulator source requires the documented runtime and external assets. Bounded kitchen-813, seed-0 collection, export, physical replay, and SimVQA references passed for all three robots, using the documented robot-specific replay modes. AI Worker has a distinct physics-12 loaded-home and repaired-basin pass with no extra terminal settling; the older extended-carry reference uses 0.9 seconds of settling. Neither establishes multi-seed or arbitrary-task reliability. Trained-policy inference is not included. See the [current reproducibility map](reproducibility.md).

Build from a clean, committed checkout. Confirm `git status --porcelain` prints nothing; `git archive HEAD` cannot include uncommitted files. Start with an empty `dist/` directory so version globs cannot pick up old artifacts. Run the portable checks and package build:

```bash
python -m pip install '.[scenes,dev]'
python -m pytest tests
python -m pytest scripts/simvla/test_skill_contract.py
python -m pytest scripts/simvla/test_select_goal_grasp.py
python -m pytest scripts/simvla/test_aiworker_kitchen_cfg.py
python -m pytest scripts/simvla/test_aiworker_home.py
python -m pytest scripts/simvla/test_retry_diagnostics.py
python -m pytest scripts/simvla/test_hdf5_compat.py
python -m pytest scripts/simvla/test_rby1_kitchen_cfg.py
python -m build
python -m twine check dist/simvla-1.0.0-py3-none-any.whl \
  dist/simvla-1.0.0.tar.gz
python scripts/tools/check_distribution.py \
  --wheel dist/simvla-1.0.0-py3-none-any.whl \
  --sdist dist/simvla-1.0.0.tar.gz
python scripts/tools/release_evidence.py --require-clean \
  --output dist/release-evidence.json \
  --artifact dist/simvla-1.0.0-py3-none-any.whl \
  --artifact dist/simvla-1.0.0.tar.gz
```

Before advertising **reproducible three-robot collection**, additionally require
an independent clean-environment run for each robot: hash-verified external
assets, a complete physically successful episode, continuous front-camera video,
and a passing `validate_lerobot_collection.py` report. Retain seeds, exact goals,
settings, and failure counts. A camera smoke, successful sub-grasp, or empty
dataset container cannot satisfy this gate. Local reference evidence is recorded;
an unaffiliated external reproduction is not yet available. Consult
[current validation](validation.md#follow-up-checks-2026-10-06).

Before describing a sink episode as placement **inside** the basin, also close
the [sink-cavity follow-up](validation.md#sink-cavity-follow-up). Legacy proximity
predicates can accept a mug resting on the countertop; old positive replay
records do not validate the repaired cavity or its below-rim predicate.

The evidence JSON records the exact commit and tree, commit time, dirty-tree state,
submodule revisions, Python/platform information, and artifact sizes and SHA-256 digests. The
hosted package matrix uploads one such record per Python version. Preserve those files with the
release checks; a record with `source.dirty: true` is diagnostic only and must not accompany a
release.

As a required release gate, run the selected source-workflow regression suite in the configured simulator Python (it imports Torch and is not part of the CPU venv):

```bash
python -m pytest scripts/simvla/test_task_emit.py scripts/simvla/test_systemid.py \
  scripts/simvla/test_simvqa.py scripts/simvla/test_simvla_paths.py \
  scripts/simvla/test_kitchen_wizard.py -q
```

Run that command with the simulator environment activated. Keep its output with the release records. The fresh public runtime passes its dependency check, all three robots have explicitly scoped references, and browser setup/step transitions pass a real Chromium smoke. AI Worker's loaded-home route has a bounded physics-12 pass for its named seed; the complete GPU-backed browser workflow and broader task reliability remain separate validation gaps.

After the source commit exists, create the full source archive from that exact commit and checksum all three artifacts:

```bash
git archive --format=tar.gz --prefix=simvla-1.0.0/ \
  --output=dist/simvla-1.0.0-repository.tar.gz HEAD
sha256sum dist/simvla-1.0.0-py3-none-any.whl \
  dist/simvla-1.0.0.tar.gz \
  dist/simvla-1.0.0-repository.tar.gz > dist/SHA256SUMS
sha256sum -c dist/SHA256SUMS
tar -tzf dist/simvla-1.0.0-repository.tar.gz | head
```

Install the wheel in a separate environment and verify the installed CLI from outside the checkout:

```bash
python -m venv /tmp/simvla-wheel-check
/tmp/simvla-wheel-check/bin/python -m pip install --no-deps dist/simvla-1.0.0-py3-none-any.whl
cd /tmp
/tmp/simvla-wheel-check/bin/simvla --version
/tmp/simvla-wheel-check/bin/simvla validate bowl_to_drawer
/tmp/simvla-wheel-check/bin/simvla skills nav.to_prim
```

The small VQA capture is in the repository and Python sdist, not the wheel; run that conversion example from a checkout or extracted sdist.

Distribution files:

- `simvla-1.0.0-py3-none-any.whl`: CPU package and simulator launcher.
- `simvla-1.0.0.tar.gz`: Python source distribution.
- `simvla-1.0.0-repository.tar.gz`: full source checkout, excluding external assets and submodule contents.
- `SHA256SUMS`: checksums for these artifacts.
- `release-evidence.json`: machine-readable source revision, environment, and artifact hashes.

The Python source distribution includes the standalone CPU-package tests, not
repository-only tests that require the excluded simulator, scripts or CI files.
After installing `.[scenes,dev]` from the extracted archive, `python -m pytest
tests` must work from that archive directory. CI checks this outside the checkout.
The full regression suite remains in the repository archive.

The release repository retains source provenance, copyright notices, and the provisional citation from the project page. The release-candidate source is public and its [hosted package checks](https://github.com/kyounginbaik/SimVLA/actions/runs/37549361897) passed; publishing a versioned GitHub release or PyPI package is a separate action. The curated BODex object subset has a tested public download. The deterministic robot/kitchen archive is publicly hosted in the [SimVLA-assets dataset](https://huggingface.co/datasets/kyounginbaik/SimVLA-assets), with SHA-256 `b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8`. The dependency-consistent public simulator passes three-robot smokes and bounded physical chains with a separate exporter. AI Worker's loaded-home and repaired-basin result is limited to the documented physics-12 seed; do not extend the qualified reference claims beyond that workflow. Update `CITATION.cff` and the README when the paper receives a permanent identifier or final publication details.

Pushing an annotated `v*` tag starts `.github/workflows/release.yml`. It repeats the portable,
skill-contract, AI Worker, RB-Y1, package-scope, checksum, and provenance checks before creating the
GitHub release. The workflow uses GitHub's scoped token with `contents: write`; protect release
tags so only maintainers can trigger it. PyPI publication remains a separate maintainer action.

## GitHub launch settings

After publishing the source and before announcing it, configure the repository surface that is
not stored in Git:

- Description: `Zero-shot sim-to-real VLA data generation for mobile manipulation`.
- Website: `https://kyounginbaik.github.io/simvla/`.
- Topics: `robotics`, `vision-language-action`, `mobile-manipulation`, `sim-to-real`,
  `isaac-sim`, `isaac-lab`, `embodied-ai`.
- Use the project overview as the social preview and verify it remains readable at card size.
- Enable Issues and the reproduction issue form; enable Discussions only if maintainers will
  answer Q&A there.
- Enable private vulnerability reporting and test the route named in `SECURITY.md` while signed
  in with a non-owner account.
- Protect `main` and `v*` tags, require the package and cuRobo-patch jobs, and prevent tag reuse.
- Confirm the About license is detected as BSD-3-Clause, the citation button reads
  `CITATION.cff`, and the release contains all files listed above.
- Update the project page with the exact release link and direct BODex dataset link. Replace the
  provisional paper link and citation only when an arXiv identifier or DOI is actually public.

Finally, repeat the README quickstart from an anonymous archive download. Open the generated GLB,
inspect the task JSON, and attach the resulting hosted CI/release URLs to the release record.
