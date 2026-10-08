# Changelog

## 1.0.0 (release draft)

- Initial release candidate of the portable CPU package and simulator source interfaces.
- Adds a validated `mug_to_sink` authoring template and explicit RB-Y1 asset diagnostics.
- Provides CPU scene generation, task and skill authoring, goal validation, and SimVQA conversion.
- Documents eight simulator and research workflows with tested scope and external requirements.
- Adds visual workflow examples, installation guidance, reproducibility records, and verified release artifacts.
- Adds fresh public-runtime Anubis and RB-Y1 collection/replay/SimVQA reference evidence, separate from three-robot startup smokes.
- Restores recorded object momentum during replay and stops failed passes at automatic-reset boundaries.
- Preserves continuous failure video from front and both wrist cameras at the configured control rate.
- Corrects AI Worker's USD-derived home predicate; its loaded-home carry and fresh replay remain experimental rather than validated release features.
- Adds real-browser setup-flow coverage for all three robot selections.

## 0.1.0a2

- Organized the release around eight documented workflows, with training and SimDeploy explicitly documentation-only.
- Added headless kitchen generation, goal emission, bounded system identification, and CPU SimVQA conversion entry points.
- Bundled two executable goal examples and a real SimVQA capture; simplified local SimAction collection.
- Fixed base-frame comparison in system identification, segmentation capture, logical mesh path handling, and smoke-test robot initialization.
- Added launcher completion reporting to detect failures masked by Isaac Sim shutdown.
- Added CPU task initialization, skill inspection, goal preflight, and AI Worker asset diagnostics.
- Added a CPU-tested AI Worker kitchen adapter that rewrites emitted environment configs, task registrations, and goal names as one workflow.
- Unified the portable and simulator skill registries and added a runnable skill-extension example.
- Pinned and verified public BODex mesh/grasp acquisition; hardened tar extraction against unsafe entries.
- Tested collection, replay, VQA, full-recording system identification, and recorded-action evaluation on RTX 3090 GPUs.
- Removed generated task corpora, stale machine dependency freezes, and unrelated experiment/deployment scripts from the release tree.
- Added a workflow-to-evidence reproducibility map and a structured issue form for sharing comparable reproductions and extensions.
- Added a one-command CPU demo plus a results-led README with direct teaser and reproducibility links.

## 0.1.0a1

- Initial installable CPU package: five kitchen layouts, six task templates, validation API, CLI, tests, and CI.

See the dated validation records for current clean-install and public-asset evidence.
Trained-policy evaluation remains outside the verified scope.
