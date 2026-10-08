# Contributing

Install with `python -m pip install -e '.[scenes,dev]'` and run `python -m pytest tests`.

For dataset/export changes, also install `.[data,dev]` and run
`python -m pytest tests/test_lerobot_integrity.py`. These tests create real small
Parquet/video fixtures; passing schema checks alone does not prove a physical task
succeeded. Simulator changes need both successful and failed-attempt evidence,
including the exact robot, goal, asset hashes, settings, and continuous review video.

Reusable APIs live in `src/simvla`. Keep imports free of simulator startup and account authentication. Scene generation dependencies belong in the `scenes` extra. Include a small reproducible example with bug reports, along with `simvla doctor` output.

The `scripts` and `source` directories retain the research simulator implementation. Changes to the packaged scene/task modules and their runtime counterparts must be checked together until that migration is complete. The portable placement registry is shared by the packaged validator and scene builder.

Skill IDs and template field names are authoring interfaces; generated goal files persist skill names. Keep existing IDs and field meanings stable within a release. Do not persist the numeric values returned by `SKILL_ID()`: adding a skill can renumber them. If a change needs to rename a skill, change a parameter's meaning, or alter the goal format, document the migration, update its consumers, and test an old template or goal against the new reader before release. Skill declarations live in `src/simvla/skills.py`; the modules under `scripts/simvla/` are compatibility imports for standalone simulator entry points. Public interfaces follow semantic versioning from 1.0.0. Preserve compatibility within a major version; document migrations and reserve breaking changes for the next major release.

For simulator skill changes, run `python -m pytest tests/test_skill_registry_parity.py scripts/simvla/test_skill_contract.py` in addition to the portable tests. These contract checks run without Isaac Sim; a new runtime skill also needs a one-kitchen simulator smoke test before its behavior can be claimed.

Do not commit datasets, generated scenes, robot binaries, checkpoints, run logs, credentials, or machine-specific configuration. Store test outputs outside the repository or under `outputs/`.

## Share a reproduction or extension

Reproductions, new tasks and skills, robot integrations, and measured results are useful even when an experiment fails. Open a **Reproduction or extension report** and follow the [result-reporting checklist](docs/reproducibility.md#reporting-a-comparable-result). Include the exact version, task and goal, asset identities, environment, protocol, per-trial outcomes, and shareable artifacts. State deviations and limitations explicitly so others can compare the result without guessing.
