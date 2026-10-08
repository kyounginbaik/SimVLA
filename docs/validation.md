# Release validation

Results below are integration evidence unless a row explicitly says otherwise. Use the
[evaluation protocol](evaluation-protocol.md) for policy benchmarks and public reproductions.

Package version: **1.0.0**, Linux. The CPU package and simulator collector have
different validation boundaries; a version number is not an end-to-end collection guarantee.

## Follow-up checks (2026-10-06)

Historical bounded kitchen-813, seed-0 collection/export/physical-replay/SimVQA
references passed for all three robots. A new AI Worker loaded-home and repaired-basin
chain now passes with the explicit physics-12 profile below; the earlier extended-carry
reference still uses 18 extra terminal hold steps. These are local reproductions with
the documented control modes—not arbitrary-task reliability or external independent
replication.
Failures and follow-up checks remain visible below:

- Latest recording-boundary checks: Anubis **2437663 → 2437741** passed
  1,418 exported frames plus the explicitly recorded first control tick, with
  no terminal settling; 26 SimVQA records, 156 images and 273 QA examples.
  AI Worker **2437660** exported 1,954 frames, but hybrid replay **2437740**
  failed to lift/deliver the mug. Its successful older reference does not
  establish this new chain. RB-Y1 nominal batch **2437355** exported three
  episodes, but only **one of three** action-only replays passed: one failed
  the interior-region check and one left the mug near the pickup counter.
  See [the recording-boundary evidence](evidence/recording-boundary-followup-2026-10-06.json).
  New loaded-home and placement-route experiments are not release passes while
  their collection/replay checks remain incomplete.
- The revised RB-Y1 turn-then-straight approach subsequently passed collection
  **2437751**: three accepted episodes, **3,873 frames** and three 20-FPS cameras.
  Independent replays **2437810/11/12 all passed on their first pass**, with
  18 SimVQA records and 108 checked images each (188/184/185 QA examples).
  Replay uses Cartesian/binary actions, recorded initial joint state and the
  one-tick prelude; no joint-motor override, base feedback or terminal settling.
  All three terminal mug roots satisfy the unchanged repaired-basin gate.
  This validates the new nominal kitchen-813/seed-0 recipe, not other scenes or
  the earlier failed batch. See [batch evidence](evidence/rby1-basin-batch-2026-10-06.json)
  and `configs/collection/rby1-kitchen813-basin-approach.env`.
- AI Worker collection **2437904 → replay 2437908** passed for one episode with
  3,592 exported rows, one recorded prelude tick, and three synchronized 20 FPS
  videos. The new 12-substep profile preserved the original gripper balance rate
  while refining physics integration. Replay passed on its first pass, completed
  the loaded-home, aisle, basin-release and final arm-home checks, and used zero
  extra terminal hold steps. SimVQA passed with 20 records, 120 images and 205
  Q/A examples. See the [full evidence record](evidence/aiworker-loaded-home-physics12-2026-10-07.json).
  The review video and dataset remain local under ignored `outputs/`; they are not
  bundled in the source release. This is one
  kitchen-813/seed-0 profile, not a multi-seed reliability benchmark.
- Latest focused portable `tests/` run in Python 3.10: **648 passed, 4 skipped**.
  The complete release workflow source gate passed **816 tests, 4 skipped** on Python
  3.10 and **757 tests, 27 skipped** on both Python 3.11 and 3.12. Installed final
  wheels on Python 3.11/3.12 passed `pip check`, CPU demo, and template validation.
  Wheel/sdist build, Twine metadata, archive-scope, and checksum checks passed; the
  extracted final source archive passed **189 bundled tests** outside the checkout.
  The separate real Chromium run passed setup validation, robot selection and live
  step transitions. This does not validate the GPU director, 3D preview or generated-
  scene physics. Earlier offline-export/controller (333 passed, 2 USD skips) and
  sink/USD (11 passed) results remain separate targeted checks. CPU checks do not
  substitute for physical simulator validation.
- Approved RBY1 camera-07 mounts now have fresh end-to-end evidence:
  collection **2420742**, 1,126 frames and three 20-FPS cameras, followed by
  successful first-pass physical replay **2421471**, 16 SimVQA records,
  96 checked images and 164 QA examples. See
  [the evidence record](evidence/rby1-camera07-validation-2026-10-06.json).
  Seed 1 also passed export; the seed-2 trial exposed delivery failures.
  Do not report a three-seed reliability pass until that campaign is complete.
- The AI Worker head housing collided with its grandparent torso collision hull.
  Controlled 20-Hz excitation measured peak head joint change of 0.918 degrees
  per frame at the original gains; filtering only `head_link2` against
  `arm_base_link` reduced this to 0.0061 degrees (2421972).
  Other self-collisions remain enabled and downloaded assets are unchanged.
  The spawn-time fix is implemented; collection/video validation is still running.
- Loaded-home trial **2421043** retained the mug but stalled at a Cartesian
  waypoint and correctly rejected the episode. A joint-target execution path
  is experimental (`SIMVLA_LOADED_HOME_JOINT_TARGETS=1`); it controls only the
  seven left-arm joints and preserves gripper control. It is not yet a validated
  replacement for the earlier extended-carry reference.
- Fresh public dependency installation completed in an empty venv and passed
  `pip check` with Python 3.10.20, Torch 2.5.1+cu124 and NumPy 1.26.4.
  Pinned public cuRobo compilation and editable-package installation also passed
  (2423262, CUDA toolkit 12.8.93/GCC 11.4). Fresh GPU startup exposed unnecessary
  GUI/VR dependencies, now imported only when requested. The corrected installation
  passed 100-step finite-physics/RGB checks for all three robots (2424366), and
  `pip check` passed again afterward. See [clean-runtime evidence](evidence/clean-public-runtime-smoke-2026-10-06.json).
  RBY1 then passed collection **2424756** (1,155 frames) and first-pass physical
  replay **2426057**, with 16 SimVQA records, 96 checked images and 164 QA examples.
  Replay exposed a missing `datasets` dependency; the recipe now includes it under
  the tested constraints and `pip check` passed again. See
  [fresh public RBY1 evidence](evidence/clean-public-rby1-2026-10-06.json).
  Anubis also passed fresh collection **2426755** (1,212 frames) and first-pass
  physical replay **2429275**, producing 26 SimVQA records, 156 checked images
  and 275 QA examples. The preceding replay failed because recorded object reset
  velocities were not restored; restoration now has regression coverage. See
  [fresh public Anubis evidence](evidence/clean-public-anubis-2026-10-06.json).
  AI Worker fresh collection **2426866** exported 1,905 frames, but its replay
  **2428380** failed. That fresh-runtime chain is not yet validated.
  Follow-up **2429831** now terminates correctly at frame 1,698 when the mug
  falls below 0.3 m, instead of continuing against a reset scene. Experimental
  adaptive-gripper replay keeps live contact feedback while replaying arm joint
  targets; this is not the same mode as full joint-motor replay.
  Adaptive-gripper replay **2430773** retained the mug through transport but
  failed the corrected terminal home check: the recorded episode ends with the
  left hand about 9 cm from home. No successful replay or SimVQA export is claimed.
  Fresh corrected-home collection **2432349** then passed: 1,914 accepted frames,
  three 20-FPS cameras, mug 7.1 cm from the sink target and left-hand home error
  4.9 cm against the corrected 5-cm limit. This is extended-carry collection,
  not a loaded-home transport pass. Its adaptive (**2434391**), full motor
  (**2435309**) and initial-joint-state-restored (**2435668**) replays all failed
  to retain the mug. Collection/export success does not close this replay gate.
  See [current AI Worker evidence](evidence/aiworker-current-chain-2026-10-06.json).
  Physics-rate recording follow-up **2436772** exported 1,896 accepted frames
  with six motor-target samples per 20-Hz frame. Full-rate (**2436899**), 20-Hz
  control (**2436900**) and initial-joint-restored (**2436966**) replays failed;
  the last lost the mug during transport at frame 1,468. The recording upgrade
  alone does not establish reproducibility. Hybrid replay **2436970** retained
  the mug through transport and reached the legacy proximity target, but did not
  pass the full terminal gate before its 1,896 recorded frames ended. This remains
  a failed replay, not a SimVQA success. Its three failure videos are retained in
  `outputs/clean-public-aiworker-2436772/replay-hybrid-failure` locally.
  Follow-up **2437094** used the same hybrid mode with an explicit 100-step
  settling budget and passed at frame 1,914: **18 extra steps / 0.9 seconds**.
  SimVQA validation passed with 16 records, 96 images and 164 Q/A examples.
  Repeat replay **2437352** passed at the same frame 1,914 with the same 18
  settling ticks and the same capture counts. This repeats the saved episode,
  not a new collection or a multi-seed reliability measurement.
  This bounded positive preserves the legacy proximity predicate; it is not
  basin containment or the requested loaded-home route. The original-timing
  failure remains a failure, and the default settling budget remains zero.
  A SimVQA target-map shadowing bug
  found during the first replay starts was fixed and regression-tested.
- AI Worker's old home predicate used the wrong reference by about 15 cm.
  The corrected reference is derived from the public robot USD at each supported
  lift setting, with a 5-cm terminal tolerance instead of 18 cm. All three static
  FK checks pass; see [home FK evidence](evidence/aiworker-home-fk-2026-10-06.json).
  Earlier physical references do not validate this new terminal gate.
  Experimental symmetric wrist stiffness 1,000 allowed a loaded home return
  (**2425828**), but the following base turn lost the mug. This is not an
  end-to-end success or a promoted default.
  A slower-yaw trial **2429832** lost the mug at frame 2,056 near the end of the
  loaded home motion, before validating its navigation change. Lower yaw speed
  alone is therefore not a demonstrated fix.
  Persistent joint-target holding through pauses/navigation (**2431676**) also
  failed: the mug contacted the counter on the low home path and was lost.
  The opt-in planning-only carried-object collision proxy retained the mug
  through actual home and the first turn/aisle move (**2432749**), but subsequent
  base translation lost it at frame 3,041 against the counter edge. The proxy
  improves arm planning; it does not make base navigation payload-aware, nor
  weld, teleport or constrain the simulated mug. A raised-clearance follow-up
  **2435195** retained the mug through home, clearance lift and sink transport,
  but released beside the basin because the old relative placement offset was
  calibrated for extended carry. It failed the unchanged physical target gate.
  The absolute-basin follow-up remains experimental.
  A subsequent repeat (**2436773**) lost the mug on the low home path. Code
  inspection found that cuRobo's locked-joint refresh replaced dynamically
  attached payload spheres. Attachment now happens after that refresh, with a
  runtime equality check against the planner's live spheres and a regression
  test. Earlier enabled-proxy trials do not establish payload-aware planning.
  The live equality check passed in **2436851**. Its first attempt lost the mug
  at home; the second retained it through home/aisle but hit the 4,800-frame run
  limit before full delivery. It exported no successful episode.
- Replay now stops a pass on failed automatic reset, preventing remaining actions
  and VQA samples from crossing into a replacement scene. Failure videos include
  front and both wrists at the configured control FPS. Post-reset diagnostics are
  explicitly labeled as non-terminal state.
- Anubis batch **2423508** completed all three requested episodes: 3,947 frames,
  three 20-FPS cameras, and passing export validation. All three episode IDs then
  passed first-pass physical replay on the fresh public runtime (**2426756**,
  **2426927**, **2426928**), each yielding 26 SimVQA records, 156 checked images
  and 273 QA examples. See [batch evidence](evidence/anubis-batch-replay-2026-10-06.json).
  This is one recipe, not a multi-scene or multi-seed reliability estimate.
- Fresh public headless goal authoring passed for all three robot IDs, including
  config adaptation for RBY1 and AI Worker. The launcher now performs that required
  adaptation automatically. A sparse Anubis OmniPBR shader also exposed a missing
  `project_uvw` input; a stage-local fix passed a 100-step GPU smoke without logged
  errors (**2427922**). See [authoring/material evidence](evidence/public-authoring-and-materials-2026-10-06.json).
- Fresh public headless kitchen generation **2434441** created kitchen 99106,
  all 12 rotated USDs and a task template. Its predecessor exposed missing
  `xatlas`; the clean recipe now installs the pinned dependency. Two isolated
  seeded runs (**2435249**, kitchen 99107, seed 42) matched material selections,
  mesh hash, recorded mug geometry measurements and the template, with 12
  rotations each. This is not bitwise USD identity or manipulation success.
  Goal authoring plus 100-step physics/front/both-wrist camera checks then passed
  for all three robots on kitchen 99106 rotation 00 (**2434881**). The first
  new-material render took about 14 minutes; subsequent robot smokes were faster.
  See [generation evidence](evidence/public-kitchen-generation-2026-10-06.json).
- Additional kitchen-00 Anubis trial **2424697** exhausted its bounded budget
  without an accepted episode (grasp IK failures). RBY1 experimental absolute
  placement **2424617** exported one episode but replay **2425960** failed physical
  success. Neither trial extends the supported reference claim.
- Fresh goal generation produced kitchens 00, 01 and 02. Collection then exposed
  missing registration for emitted configurations; lazy generated-task discovery
  now has regression coverage. Kitchen 435 rotation 03 was rejected for insufficient
  navigation clearance. Generation alone is not a physical collection pass.

- First wrist-preview revision (2026-10-06): both AI Worker cameras use
  camera-local Z=-90 degrees relative to the original mount (the Z=+90 preview
  flipped upside down). That earlier RBY1 right revision used local Z=180 degrees; its left was unchanged.
  Mount positions and viewing directions are unchanged. Adapters and checked-in
  kitchen-813 configs agree; 30 focused tests and both 30-step GPU camera smokes
  passed (2416127). Earlier collection/replay videos retain their original framing;
  these smoke checks are not fresh end-to-end collection evidence.
  Subsequent feedback approved AI Worker unchanged and requested above-gripper
  RBY1 mounts instead, then selected candidate **07** for both wrists. The adapter
  and checked-in kitchen now use 10 cm upper-side offset, 16 cm setback, and 20°
  downward pitch, with the original right roll and an upright mirrored left view.
  GPU preview 2416970 measured both mounts about 10.36 cm above their wrists at
  the render pose. These fixed mounts rotate with the wrists. Earlier videos
  retain their old camera geometry; they are not pixel-equivalent recordings.
- Requested AI Worker home-return trial **2415397** retained the native grasp and
  midpoint pause, lifted the mug about 0.165 m, and planned the actual left-arm
  home reset in joint space. The mug dropped back onto the counter during that
  reset; by frame 1,400 both finger loads were zero. The arm reached its reset
  target, but loaded-home transport failed. The diagnostic was stopped after
  confirming this and its front/CCTV videos were retained. No successful episode
  or replay is claimed for `--return-home`.

- AI Worker **2408283** completed native-BoDex mug-to-sink collection on the
  first episode: 1,915 frames, three synchronized cameras, and both scene
  snapshots. The goal uses a 15 cm forward carry extension, wall-clearance
  parking at yaw -0.50 rad, and a short non-cross-body placement. Failed-plan
  fallback was disabled; grasp, carry, placement, and reset obtained cuRobo plans.
  Export validation passed. Replay **2409434** exposed missing dataset-frame
  inverse offsets in the AI Worker calibration. Those now exactly invert the
  shared exporter (not a hardware calibration), with all-three-robot round-trip
  regression tests. Replay **2409654** failed because its start-position code
  confused `base_link` with the articulation root, shifting AI Worker sideways
  by about 13.5 cm. Diagnostic **2410550** confirmed that offset and was stopped.
  Replay now restores the recorded root pose/velocity from `scene_states.json`;
  AI Worker fails early if that snapshot is absent instead of guessing an offset.
  Physical replay **2410711** still missed the grasp after smaller heading drift.
  A reset-event variant (**2411416**) also missed the grasp. State-referenced base
  tracking (**2411543**) grasped and lifted, then lost the mug during carry as its
  adaptive finger targets diverged. Recorded joint-motor-target replay
  (**2412231**) passed on its first pass at tick 1,915, with both the mug-in-sink
  and arm-home predicates satisfied. SimVQA validation passed (16 records,
  96 images, 161 Q/A examples). This requires both recorded joint targets and
  base-pose velocity feedback; the earlier failed modes are not success evidence.
  [Exact evidence and limits](evidence/aiworker-reference-replay-2026-10-06.json).
- Fresh RBY1 **2405318** completed collection with the corrected exporter and
  retry-snapshot lifecycle: 1,127 frames, three 20 FPS cameras, and both initial
  and terminal scene snapshots. Independent replay **2406630** reproduced physical
  success on its first pass at tick 1,127; SimVQA validation passed with 16 records,
  96 images, and 158 Q/A examples. This artifact needs no legacy-data recovery.
  Root-snapshot replay **2411154** also passed on its first pass with action-only
  base velocity, 16 SimVQA records, 96 images, and 157 Q/A examples.
- Fresh Anubis **2408837** exhausted 6,500 frames without an accepted episode.
  A partial-success grasp bank repeatedly reused a lifted bowl grasp that failed
  the full task. The manager now retires matching partial-bank entries after two
  physical full-task failures; fully successful goals and ordinary planner/timeouts
  are protected. CPU regression tests cover that retirement branch; the successful
  subsequent collection did not exercise it, so GPU branch coverage is not claimed.
- Longer Anubis collection **2411130** passed: 1,352 frames, validated export,
  and both initial/terminal scene snapshots. It used a 10,000-frame overall
  budget and completed before frame 4,600. Cartesian replay **2412491** failed.
  Recorded joint-target replay with bounded base feedback **2412901** passed on
  its first pass at tick 1,352, including SimVQA validation (26 records, 156 images,
  275 Q/A examples). This is the current Anubis profile. Separate collection
  **2411668** also passed (1,180 frames, both snapshots), but its replay **2412902**
  failed the terminal predicate. Both outcomes are retained; no success-rate claim
  follows from these selected references.
  [Fresh reference evidence](evidence/anubis-fresh-reference-2026-10-06.json).
- AI Worker **2405388** retained the mug through the final sink-side turn using
  wall-clearance parking, then failed placement with a self-colliding planner
  start. Its preceding carry had used the legacy Cartesian fallback after an
  IK failure. `SIMVLA_REQUIRE_CUROBO_PLAN=1` now permits strict profiles to reject
  failed arm plans instead of taking that fallback; the experimental AI Worker
  profile enables it. Planner-failure snapshots record ordered joints and sphere
  geometry without disabling collision checks. No accepted episode is claimed.
- AI Worker **2407108**, with a 10 cm carry extension and failed-plan fallback
  disabled, retained the mug throughout transport. Placement then failed IK;
  the recorded start had no non-ignored sphere overlaps. The target crossed the
  left arm's lateral workspace. An angled sink-parking recipe with a 15 cm carry
  extension is under test. These remain rejected diagnostic trials.
- RBY1 **2402869** completed mug-to-sink collection and passed export validation:
  one 1,127-frame episode and three synchronized videos. It used the short-reach
  sink diagnostic (parking advance .10 m, forward reach .30 m), six physics
  substeps, gripper stiffness 1000, and force feedback at 20 N (maximum 30 N).
  A subsequent replay (**2404119**) exposed an export-semantic defect: the
  negative-opening finger joint became a positive policy command, so replay
  closed both hands. The original export is **not release-quality data**.
  Export now records binary open/close commands independently of motor polarity
  or contact-holding targets, and RBY1 observations use named finger joints.
  `recover_gripper_export.py` creates a separate corrected stage using the exact
  original HDF5 commands; originals are never overwritten. The stricter export
  validator rejects these invalid gripper values. Corrected-export replay
  **2405317** passed physical success on its first pass at tick 1,127 and SimVQA
  validation (16 records, 96 images, 157 Q/A examples). Fresh collection remains
  a separate check. [Recovery and replay evidence](evidence/rby1-corrected-replay-2026-10-06.json).
- Anubis **2402141**, on a second GPU node, collected another accepted episode
  (1,212 frames, three validated videos) with the metadata-enriched reference
  goal. Replay **2403786** reproduced success on its first pass at tick 1,212
  and passed SimVQA capture validation (26 records, 156 images, 275 generated
  Q/A examples). Its terminal scene snapshot is recorded, but its manual retry path
  omitted the full initial snapshot. Initial object poses are present. That
  recorder-lifecycle gap is now fixed for future collection runs.
- AI Worker trials **2401238** (slow navigation) and **2402139** (slower actual
  finger actuators) retained the mug through initial rotation and transport,
  but failed during the final sink-side turn. The gripper-speed setting had
  incorrectly controlled the lift actuator; it now controls the fingers, with
  the default speed unchanged. A sink-clearance diagnostic is under test.
- The expanded portable/source test selection passed **740 tests, one skipped**,
  including replay asset resolution, gripper export semantics, lateral placement
  dispatch, and missing-initial-snapshot tests. SimVQA
  capture validation is now a separate fail-closed replay-launcher gate.
  The isolated offline-export CI selection passed 197 tests. Wheel/sdist builds,
  Twine metadata checks, distribution-content checks, and `git diff --check` passed.
- Corrected Anubis collection **2400046** exported one valid 1,418-frame episode
  with three synchronized 20 FPS videos and nonconstant joint targets (16 of 19
  axes; base position targets are unused). Replay **2400601**, from a node-local
  source copy, reproduced task success on its first pass and saved three videos
  plus initial/terminal physical state. The terminal bowl position was
  `(1.11752, -0.18735, 0.67891)` m, within this drawer's asset bounds; drawer
  displacement was 0.02979 m. This is one scene/seed, not all-robot validation.
  Its VQA capture exposed a separate per-arm target-label bug; that capture is
  not semantic VQA release evidence. [Corrected collection evidence](evidence/anubis-corrected-collection-2026-10-06.json).
- RBY1 **2400030** passed the previously blocked base turn and carried the mug
  to the sink, but missed the basin during placement. A revised parking/place
  diagnostic is under test. Trial **2400053** was interrupted by a shared-source
  filesystem transport error; it is not counted as a motion-planning failure.
- AI Worker **2400052** retained the lifted mug with native BoDex orientation,
  but lost it during base rotation. Freezing finger targets on reaching the
  requested force in **2400083** instead failed during lift; that optional
  experiment remains disabled by default. No complete episode is claimed.
- Anubis trial 2399882 completed the physical task and exported one accepted
  episode (1,412 frames, three synchronized cameras). Its drawer diagnostic used
  a 17 cm lateral / 5 cm backward parking adjustment, swapped parallel-jaw
  orientation, and six physics substeps per 20 Hz control tick. **This is not yet
  release-quality reproduction evidence:** an audit found that `action.joint`
  aliased a mutable simulator buffer, making all saved joint-target frames equal
  to the final target. `EpisodeData.add` now snapshots tensors; recollection and
  replay are required. Cartesian actions and measured joint angles in that trial
  varied normally. Its Cartesian replay (2400028) succeeded on the first pass.
  The old artifact is retained, not relabeled as fixed.
  [Dated evidence](evidence/anubis-physical-replay-2026-10-06.json).
- RBY1 trial 2399905 stalled at base yaw -0.0001858 rad, exactly its USD's
  +359.98935 degree virtual-yaw hard stop modulo a full turn. Raising the minimum
  turn command did not resolve it. The RBY1/Anubis spawn helper now removes limits
  on this virtual base joint only, in memory; physical arm limits and downloaded
  asset bytes are unchanged. Physical follow-up validation is in progress.
- AI Worker trial 2399837 retained two lifts with whole-finger feedback, but
  failed subsequent transport/placement. Carry-phase object/EEF/contact traces
  now distinguish a successful lift from a successful delivery. No complete
  AI Worker demonstration is claimed from that run.

- Clean-runtime AI Worker trial 2399789 measured a 167 mm object lift with
  proximal finger loads of 33.9/45.9 N, despite one unloaded distal pad.
  The old distal-only controller drove that already-loaded finger fully closed.
  Controller and acceptance instrumentation now sum the object-filtered loads
  over both segments of each adaptive finger; bilateral contact and observed
  object lift are still required. Missing/non-finite filtered measurements cannot
  fall back to counter contact. [Diagnostic evidence](evidence/aiworker-finger-contact-2026-10-06.json)
  records the rejected run; it is not retroactively counted as a demonstration.
- A right-arm approach could plan to an explicitly configured 8 cm standoff,
  then use a hard-coded 18 cm standoff during final correction. Both now use
  the same validated setting. Follow-up physical trials remain necessary.

- A left-arm correction incorrectly held wrist rotation whenever a mug-only stage
  counter was zero, including drawer reaches. Anubis job 2397827 now corrects that
  rotation (96.5° down to 27–34°), but the drawer pose remains unreachable in these
  attempts. Bowl lifts of 166.4 and 171.8 mm did not complete the task.
- Joint-angle export used frame 0 while the other staged streams started at frame 1.
  All streams now use the same slice; unequal lengths and existing outputs are
  rejected. The exporter no longer deletes previous datasets.
- The collector imported LeRobot even with `--skip_finalize`. Imports are now lazy,
  with a standalone offline exporter and optional `SIMVLA_EXPORT_PYTHON` launcher
  path. A fresh CPU export environment passed dependency checks and real synthetic
  Parquet/video export tests for Anubis, RB-Y1 and AI Worker. These are not robot
  success episodes or a clean simulator-install result.

Lift telemetry in RB-Y1 job 2397815 measured the mug at **0.72628 kg**, with default
object friction 0.5/0.5. Bilateral forces around 6 N were insufficient in this run;
one jaw reached its closed-position limit before contact was lost. The frame-budget
run correctly produced zero accepted episodes. AI Worker job 2397820 briefly raised
the mug about 119 mm, then dropped it before completion; this is also failure, not
retention. No task-success threshold was relaxed. Subsequent grip-force and
contact-height trials must be evaluated separately.

AI Worker lower-contact-height trial 2397906 raised the mug 166.3 mm, but ended
with only one loaded pad. The unchanged bilateral retention gate rejected it;
zero episodes were exported. RB-Y1 higher-force trial 2397936 instead failed
approach-displacement checks (80.9 and 54.3 mm), before testing its grip. Enabling
target collision in 2397978 caused repeated IK failures. None is a success.
The later candidate-9 RB-Y1 trial 2398032 also rejected nine plans at IK, ended
its 2,200-frame budget with zero episodes, and preserved a 110-second front video
at 20 FPS. Its machine-readable collection result correctly reports failure.

Anubis waypoint trial 2398031 ended with no episode after grasp-tracking stalls.
Focused candidate-10 trial 2398065 then lifted the bowl 157.5 mm and tested the
12 cm lateral base waypoint. The drawer pre-grasp improved to 64.9 mm / 11.2°
error but still failed its unchanged reach gate. Its 2,600-frame budget produced
zero episodes and an uncut 130-second front video. The diagnostic waypoint is
not promoted as a validated replacement task. Candidate selection now supports
Anubis's legacy goal format as well as both other robots' dictionary format.

The fresh raw-only simulator environment now passes `pip check` without LeRobot
or Rerun, with Torch 2.5.1+cu124 / torchvision 0.20.1+cu124 / NumPy 1.26.4 and
cuRobo resolving to this checkout. It passed **426 tests / one skip** across
`tests/` and the episode-writer, HDF5 compatibility, collector-profile, grasp
selection and two adapted kitchen-config suites. After bounding CPU workers to the eight allocated
cores, **job 2398448 passed 30-step camera/physics smokes for all three robots**
in that isolated runtime, without a compatibility overlay. All three saved
workflow statuses and the Slurm exit code were zero. Anubis had four RGB cameras,
RB-Y1 and AI Worker five each; AI Worker's lift-home error was 13.3 mm.
[Machine-readable evidence](evidence/raw-runtime-smoke-2026-10-06.json) records the
scope, versions and log hash. The preceding unrestricted worker pool stalled
inside Kit's initial renderer update and was intentionally stopped; it was not a
passing run. CUDA allocation and this checkout's cuRobo MotionGen import also
passed on the same isolated runtime. Full physical collection remains unverified.

The portable environment passed **312 tests / three skips**. Its installed wheel
also produced the kitchen/task/manifest demo and 12 example VQA records. A fresh
installation from `env/exporter-pip.lock.txt` passed **seven integration tests**,
including all three synthetic robot exports and frame-coded joint/video alignment.
The wheel and source distribution build, metadata and content checks passed.
Subsequently, the public [release-candidate package run](https://github.com/kyounginbaik/SimVLA/actions/runs/37549361897)
passed all eight jobs on code commit `ab48ae9`, including synthetic export for
Anubis, RB-Y1 and AI Worker. This hosted CPU/offline result is distinct from the
Slurm GPU physical-collection references and is not an external replication.

The launcher now records unsuccessful collection results, explicitly seeds runs,
checks goal dispatch before booting Kit, probes actual storage writes, and reads
the saved workflow status even when Kit returns process status zero after failure.
A disk-quota failure in 2397988 exposed shutdown handling: status-write errors now
still close the simulator. That run's partial video is not valid evidence. The
quota incident was resolved by moving an agent-created temporary environment to
local scratch; no existing dataset or asset was deleted.

## Sink cavity follow-up

The kitchen-813 sink references above use `obj_near_prim` at a door-derived
anchor. That is proximity, not containment: a mug resting on the countertop
can satisfy it. Do not describe those references as verified basin placement.

An isolated drop test (2436896/2436905, not a robot demonstration) found:

| Collision configuration | Settled mug-root height | Below basin rim? |
| --- | ---: | --- |
| Authored asset | 0.9011 m | No |
| Decompose only sink and countertop meshes | 0.9011 m | No |
| Remove only redundant sink Xform colliders | 0.9011 m | No |
| Both changes | 0.7468 m | Yes |

The rim is at 0.8656 m. `scripts/simvla/sink_collision.py` implements both
changes in the live stage, leaving downloaded assets untouched. Physical mesh
and box colliders, articulation/body APIs and materials are retained. Enable
`SIMVLA_SINK_CAVITY_COLLISIONS=1` identically for collection and replay.
This is opt-in pending new complete robot collection/replay checks.

For kitchen **813 rotation 00 only**, the separate experimental
`SIMVLA_KITCHEN813_SINK_INTERIOR_GATE=1` replaces the legacy proximity leaf with
an interior XY disk and mug-root height in `(0.73, 0.83)` m, retaining the home
and last-subtask requirements. It rejects countertop/floor placement. This
checks the object root, not full mesh containment or long-term stability.
`scripts/tools/make_sink_cavity_diagnostic.py` authors an absolute release target
and records both required settings; collection/replay refuse that diagnostic
if either setting is missing. Do not apply this measured region to other scenes.

[Probe evidence and hashes](evidence/sink-cavity-probe-2026-10-06.json).
The probe alone is not robot collection or independent replay; the robot-specific
follow-ups are below. Earlier positive sink records remain historical proximity
tests and are not silently reclassified.

Robot follow-up **2436960** retained the mug during transport but missed the
absolute placement pose by 0.2016 m; it was interrupted after that failure.
With the reference parking advance restored, **2437048** released the mug below
the rim at `[1.7409, -2.3245, 0.7725]` m. It still missed the unchanged 0.12-m
interior disk, so it was rejected; no accepted episode was exported. The local
front-camera review is `outputs/rby1-sink-cavity-2437048/review-front.mp4`.
An adjusted release target is a new experiment, not a relabeling of this failure.

That adjusted RB-Y1 route **2437107 → 2437144** passed collection, action-only
physical replay and SimVQA: 1,210 frames, three 20-FPS cameras, 16 records,
96 images and 162 Q/A examples. Replay used no extra settling and finished with
the mug root at `[1.789, -2.226, 0.7725]` m, inside the unchanged interior gate.
See [the bounded basin reference](evidence/rby1-basin-reference-2026-10-06.json)
and [exact goal/profile commands](reproducibility.md#repaired-basin-experiments-kitchen-81300-only).

Repeated-collection follow-up **2437222** requested three episodes but reached
10,000 frames with zero accepted exports. Target perturbations could park the
base too far from the fixed placement target; retries also released below the
rim but outside the interior gate. Increasing only the release target X to
1.88 m in **2437245** remained unreachable and was deliberately interrupted.
Neither run validates bulk collection. A separate nominal-target recipe
disables the documented goal perturbations and advances parking by 15 cm;
it is being checked as a separate experiment, not a robustness benchmark.

AI Worker's actual loaded-home route **2436991** passed collection and exported
3,526 frames with all three cameras and physics-rate motor sidecars. It retained
the mug through home, the aisle detour and clearance raise, then released below
the rim and passed the 5-cm final home gate. The first attempt lost the mug during
home; the second succeeded. Independent replay **2437146** lost the mug during
loaded home. The control **2437202**, with initial joint-state restoration
disabled, retained it through home but dropped it during the first aisle turn
and failed at frame 2,254. Neither replay validates this loaded-home chain.
Front and both wrist failure videos are retained in
`outputs/aiworker-loaded-basin-2436991/replay-initial-state-failure` and
`outputs/aiworker-loaded-basin-2436991/replay-no-initial-state-failure`.

## Historical release gate (2026-10-05)

At this earlier checkpoint, the CPU quickstart was usable but complete collection
on all three robots was not validated. The table below is retained as dated trial
history; use the 2026-10-06 follow-up checks above for the current bounded evidence.

| Check | Current evidence |
| --- | --- |
| Fresh CPU installation | Python 3.10 virtual environment, 267 passed / 2 skipped, clean `pip check`; kitchen/task demo and 12 example VQA records |
| Source regression checks | 899 passed / 8 skipped across the portable tests and selected source workflow, grasp, clearance, home, contact-controller, and collector-profile tests in the simulator environment |
| Dataset integrity | Real tiny Parquet/video fixtures exercise timestamps, finite actions/states, episode/frame indices, FPS, dimensions, and decoded frame counts |
| Public launchers | Explicit external asset paths, optional compatibility overlay, local or Slurm execution, no-output dry run, explicit EULA acceptance, existing-output refusal |
| Anubis | Historical public-bundle job 2369539 exported 1,082 frames. New job 2395797 physically lifted the bowl 166.9 mm, then failed to finish the drawer reach; zero accepted episodes |
| RB-Y1 | Job 2395845 reached bilateral mug contact but lost it during lift; the new failed-lift reset executed. No complete episode verified |
| AI Worker | Job 2395804 executed native BoDex orientation and midpoint pause, reached bilateral contact, then lost the mug during lift. No complete episode verified |

The AI Worker geometry audit found that authoring read stale external collision
spheres while collection read checked-in ones. They now use the same checked-in
geometry. A conservative grid covers the wrist bounding box without the previous
oversized sphere's counter intrusion. The wrist-camera-up roll rule uses tool X,
not the horizontal jaw-travel Y axis; no forced 90-degree approach rotation is added.
These fixes do not themselves establish successful physical grasping.

A subsequent hand-frame audit found a 180° gripper-base rotation missing from the
generated AI Worker planning URDF. All ten hand-frame positions **and rotations**
now match the public simulation USD (position residual approximately 10 nanometres,
orientation residual 0.000007°). The earlier wrist-only check could not detect this.
The generator, hash lock, installation overlay, and regression tests have been
updated. Physical pad centres are also distinguished from linkage origins; the
19.64 mm axial offset matters during horizontal approaches. Experimental target-
collision trials 2395546 and 2395612 failed IK and were intentionally stopped after
repeated failures, preserving front/CCTV videos. They do not validate the subsequent
pad-centre and hand-sphere corrections.

Right-arm playback trial 2395614 on Anubis exposed tracking stalls (about 12–32 mm
residual in the failed attempts) and exported no episode. Stalled plans now fail
without a remaining-path Cartesian fallback. RB-Y1 trial 2395559, with target
collision enabled, rejected grasp candidates at IK and exported no episode.

AI Worker trial 2395660 then planned with the mug included as an obstacle, executed
the midpoint pause, and reached bilateral pad contact without the earlier approach
shove (9.21 N / 5.39 N at the pre-lift check). The subsequent lift lost both pad
contacts and raised the mug only 25.6 mm, so the candidate was correctly rejected.
The retry exposed a separate crash: four collector paths assigned a Python list to
cuRobo's tensor `retract_config`, breaking the next locked-joint model refresh.
Those assignments were removed; measured starts already enter through `JointState`.
An import-free regression guards all four paths. This trial produced no accepted
episode; synchronized front and overhead failure videos were preserved.

The [all-robot pad-frame audit](evidence/gripper-pad-frames-2026-10-05.json)
checks all twelve physical pad centres against the public USD meshes. RB-Y1's
30.5 mm and Anubis's 108.8 mm axial body-origin offsets are now accounted for.
This is a geometry check, not a successful grasp result. RB-Y1 baseline trial
2395713 pushed the mug 73.4 mm; corrected-pad trial 2395796 also rejected a
64.4 mm displacement. AI Worker trial 2395710 reached bilateral force on a retry
but again lost the object during lift (25.7 mm rise), despite experimental force
balancing. No completed episode is established by these observations.

The later RB-Y1 attempt in 2395796 reached centered bilateral contact
(4.91 N / 3.30 N, pad midpoint 4.6 mm from the object axis). The 2,600-frame budget
ended before its post-lift check, so retention is **unknown**, not successful.
Focused candidate-8 trial 2395845 subsequently reached bilateral contact
(3.43 N / 3.21 N) but left the mug on the counter during lift, losing both pad
forces. Its log confirms the new `post_lift_object_not_retained` reset executes
before another grasp attempt; this validates failure handling, not collection.

AI Worker trial 2395804 tested the explicit joint-name force mapping. It passed
pre-lift contact (9.71 N / 5.37 N) but lost both contacts during lift; the mug ended
0.7 mm below its pre-lift height and the jaw gap collapsed to 13.0 mm. It did not
solve object retention. This run started before the immediate failed-lift reset
change, and therefore cannot validate that later change.

The force-balance controller now uses explicit sensor-to-joint-name mappings
instead of assuming contiguous halves of the articulation joint list; CPU tests
exercise interleaved joints and inactive environments. Failed physical lifts now
reset the attempt before empty-handed downstream motion. These last runtime
changes require a fresh GPU acceptance run; passing CPU guards is not sufficient.
A CUDA preflight also rejects unusable allocations before starting Kit (job
2395734 was stopped after CUDA-context initialization failed on its node).

Fixed-object Anubis trial 2395797 reached a bowl grasp with 7.8 mm / 0.6° wrist
error and retained a 24.4 mm jaw gap while lifting the bowl 166.9 mm. The next
left-arm drawer reach failed IK and remained about 96° off orientation during its
bounded correction. The 2,000-frame run exported no accepted episode. Its front
video includes the successful sub-grasp and the subsequent incomplete task.

Review frames are simulation-step indexed and encoded at control FPS divided by
the frame-dump interval (normally 20 FPS), not at rendering wall-clock speed.
The collector retains diagnostics even on simulator failure. A nonempty file or
well-formed video does not substitute for physical success and full-task predicates.

Collection provenance records the goal and collector hashes, source revision and
dirty status, package import origins/versions, and allowlisted run settings. This
also makes stale editable imports visible. That earlier GPU environment resolved
cuRobo through the research checkout; it was not a clean-room installation proof.
The later fresh public-runtime checks at the top of this page instead use a pinned
public cuRobo clone and new native build. Hosted CI and a clean release commit
remain required before publication.

## Broader source checks and external furniture fixtures

The selected release source-workflow suite was also rerun in the fresh public
simulator Python on 2026-10-06: **356 passed, 16 skipped**. This covered
`test_task_emit`, `test_systemid`, `test_simvqa`, `test_simvla_paths`, and
`test_kitchen_wizard`. The skips require an installed chair library (including two
distinct chair variants); they are not successful chair-library validation.
With the existing external chair and table libraries explicitly configured, the
same five-file suite subsequently passed **372 tests with no skips**. This checks
library-aware workflow behavior; those optional curated libraries are not part
of the three-robot asset bundle.

The broader CPU-compatible source sweep initially reported 1,578 passes, 12 skips,
and five failures. Three failures were stale tests (a retired cup category and
two references to an unshipped research exporter); two required a missing external
room-shell mesh. The stale tests now exercise the shipped registry/exporter, and
the restored mesh makes both real-geometry checks pass. None of these failures
was hidden with a new skip or looser assertion.
The complete CPU-compatible sweep was subsequently rerun, including kitchen
build: **1,632 passed, 16 skipped**, in 15 minutes 20 seconds. The command below
excludes the two simulator/USD startup modules, so this is not all GPU behavior.
The optional browser and exporter tests have separate environments/gates above.
Upstream `yourdfpy` NumPy deprecation warnings remain in the tested pinned stack.
The repository's deprecated scene-concatenation calls were replaced with
`Scene.to_geometry()` (also present in the minimum supported Trimesh 4.6.0).
The affected wizard/gallery/preview selection then passed **290 tests, one skip**.

Furniture tests deliberately require real external meshes. Twenty fixtures
(about 56.5 MB total, excluding Objaverse's index) are pinned by UID and SHA-256 in
[the fixture lock](evidence/furniture-test-fixtures.json). They are not bundled or
relicensed. Respect each source object's licence; the fixture tool downloads only
when explicitly requested and never overwrites a checksum mismatch.

With the documented simulator environment (including Objaverse 0.1.7):

```bash
export OBJAVERSE_PATH=/path/to/external/objaverse-test-cache
python scripts/tools/furniture_test_fixtures.py --cache-root "$OBJAVERSE_PATH" --download
export PYTHONPATH="$PWD/src:$PWD/scripts/simvla:$PWD/source/isaaclab${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest scripts/simvla --import-mode=importlib \
  --ignore=scripts/simvla/test_kitchen_usd_load.py \
  --ignore=scripts/simvla/test_door_geometry.py -q
```

Omit `--download` for offline verification. The two excluded modules require USD
or simulator startup and are separate integration checks. This sweep exercises
real procedural geometry and can take about 16 minutes on one CPU core; it does
not validate the browser visually, GPU task completion, or all simulator modules.

## Earlier public-bundle trial history

In the following failed public-bundle trials, collection was **not validated end to end**.
Anubis public job 2393577 physically lifted a bowl by 0.1521 m
and banked the grasp, then repeatedly missed a left-arm drawer reach by about 98°; its HDF5
remained a 96-byte empty header. RB-Y1 job 2393662 and AI Worker job 2393634 established
bilateral mug-pad contact but lost both contacts during a 0.2 m lift; their mugs rose less than
1 cm. The public mug mesh is about 8.6 cm tall, so the jaw-placement gate was corrected to
reject rim-height approaches; a separate guard now rejects approaches that displace the mug
over 3 cm. A rotated AI Worker grasp in job 2393811 retained bilateral pad contact and lifted
the mug 0.143 m, but the following arm reset stayed about 84° off its commanded orientation;
it did not produce a complete episode. A separate trial is testing that reset gate without
weakening the physical grasp checks. The checked-in RB-Y1/AI Worker goals and contact-hold
actions pass focused simulator-environment tests (307 passed on 2026-10-05), and the portable
suite passes 213 tests. Follow-up jobs are experimental until they produce nonempty accepted
episodes and pass video/row-count validation; neither a grasp bank nor an initialized HDF5 is
a LeRobot dataset.

The pinned public bundle was also used to re-emit kitchen-813 v2 goals on a GPU node:
RB-Y1 job 2394192 and AI Worker job 2394195 each wrote a goal whose contents match the
checked-in example except for its trailing newline. This verifies goal authoring and asset
path setup, not physical episode collection. A new compositional Anubis goal emitted in job
2394187 and lifted the bowl in collection job 2394189, but its first dynamic handle-pregrasp
plan returned IK_FAIL. A bounded left-arm fallback in job 2394194 reached the handle position
but remained about 98° off orientation. A diagnostic bar-handle goal reduced that to about
93°, and a 90° wrist-roll variant in job 2394255 reached pregrasp at 34°. That trial ended
in a retry termination before a verified drawer open; several other trials missed the
right-arm carry reset by roughly 30 cm. The unvalidated wrist-roll and bar-handle diagnostic
files were removed from the public examples. No accepted Anubis episode was exported by
these trials (the separate historical job 2369539 is documented below).

## CPU package

A wheel was installed into a clean Python 3.12 virtual environment, independently of the research dependencies. On 2026-10-03, the portable suite passed 208 tests (one skipped) and the collector-profile suite passed 58 tests, including the combined AI Worker geometry/pad-load gate; four new CPU checks also pin the Anubis/RB-Y1/AI Worker grasp-frame conversions. The combined portable, collector-profile, and robot-frame regression run passes 270 tests (one skipped). The broader selected source-workflow suite previously passed 735 tests (one skipped). The installed CPU wheel reports `1.0.0` and its own `pip check` passes. A freshly built wheel also passed CLI checks for skill inspection, template and goal validation, doctor, task initialization, and a goal-command dry run from outside the checkout. The package tests cover task validation, role binding, scene specifications, CLI behavior, installed resources, geometry export, VQA conversion, overwrite protection, skill registry parity, simulator result propagation, release evidence, and distribution scope/version checks.

All five layouts were generated outside the checkout and reloaded with trimesh; each has nonempty geometry and finite bounds. `kitchens.png` shows those outputs. The bundled real SimVQA capture converts into 12 records without a GPU or account.

The GitHub Actions matrix covers Python 3.10–3.12, wheel installation, package tests, and distribution checks. Hosted CI has not run until this repository is pushed.

`cffconvert 2.0.0` validates `CITATION.cff` against CFF 1.2 and renders its two authors,
title, and project URL to BibTeX. `detect-secrets 1.5.0` scanned every tracked and staged
source file. Its 307 high-entropy findings were hex/base64 values in hashes, revision IDs,
geometry fixtures, and asset identifiers; it reported no credential-pattern findings.
Re-run this scan on the final clean commit.

The portable regression run passes 208 tests (1 skipped), the focused collector-profile suite passes 58, and the robot-specific grasp-frame regression has four passing cases; together they pass 270 tests (1 skipped). The expanded selected source-workflow command below passed 753 tests (1 skipped; 754 collected) on 2026-10-03. It covers goal emission and strict JSON output, both-arm runtime dispatch, system identification transforms, VQA conversion, path resolution, object-reset pose export/restore, research-asset staging, robot configs, the browser composer, browser-server logic, and phased pregrasp behavior. Separate collection is needed because three legacy source test modules share basenames with portable tests. These are selected suites, not every Isaac-only source test; browser UI behavior is also not certified end to end.

Historical AI Worker collection attempt (Slurm job 2377772, seed 3) failed three sampled left-arm mug approaches and was stopped. The first missed its reach gate after 600 corrections at 0.3610 m wrist error / 11.7° orientation error; the next two reached 0.0880 m / 7.7° and 0.1223 m / 8.8° but still missed the 0.04 m / 5° gate. Their jaws ended 0.0648 m and 0.1059 m off-axis, respectively. The HDF5 output stayed an empty 96-byte container; this is not a demonstration. This predates the later bounded collection/replay references documented above.

Re-run this same selected gate from the repository root with the simulator Python environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts/simvla:$PWD/source/isaaclab${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests \
  scripts/simvla/test_skill_contract.py \
  scripts/simvla/test_aiworker_kitchen_cfg.py \
  scripts/simvla/test_aiworker_home.py \
  scripts/simvla/test_retry_diagnostics.py \
  scripts/simvla/test_hdf5_compat.py \
  scripts/simvla/test_rby1_kitchen_cfg.py \
  scripts/simvla/test_task_emit.py \
  scripts/simvla/test_goal_format.py \
  scripts/simvla/test_executor_dispatch.py \
  scripts/simvla/test_collector_profile.py \
  scripts/simvla/test_episode_writer_object_pose.py \
  scripts/simvla/test_replay_initial_state.py \
  scripts/simvla/test_systemid.py \
  scripts/simvla/test_simvqa.py \
  scripts/simvla/test_task_composer.py \
  scripts/simvla/test_grasp_clearance.py \
  scripts/simvla/test_grasp_frame.py \
  scripts/simvla/test_select_goal_grasp.py \
  scripts/simvla/test_simvla_paths.py -q
```

## Simulator tests

Tests ran through Slurm on RTX 3090 GPUs, using an installed wheel, the cleaned source checkout, existing external assets, and an overlay on the research environment. Separately, a fresh Conda environment in `/tmp` installed the pinned research snapshot after prebuilding `toppra` and supplying `setuptools<81`, `poetry-core`, NumPy, and Cython. The patched cuRobo extension built with `TORCH_CUDA_ARCH_LIST=8.6`; editable Isaac Lab/SimVLA packages installed; and 197 portable tests passed. `pip check` still reports the inherited version conflicts, so the package installation alone does not certify compatibility.

The fresh environment was cloned to shared scratch for Slurm. Job 2368349 passed `doctor
--require-sim` on an RTX 3090. The 10-step kitchen 813 smokes passed for Anubis (2368351),
RB-Y1 (2368424), and AI Worker (2368425). Anubis rendered three non-constant cameras; RB-Y1
and AI Worker rendered four each. AI Worker's final lift error was 0.0138 m from its −0.30 m
target. These establish a clean-install reset/rendering path for all three robots, not complete
episodes or a dependency-consistent environment. Cold first-run RTX shader compilation took
several minutes on each node.

The local asset staging tool now verifies a pinned research source and a separate prepared RB-Y1
model, builds the AI Worker planning URDF, and emits a repeatable environment file. Staged paths
passed 10-step kitchen 813 GPU smoke checks on 2026-10-01 for Anubis (job 2366699), RB-Y1
(2366717), and AI Worker (2366731). The RB-Y1 and AI Worker checks each rendered four
non-constant cameras; Anubis rendered three. These are integration checks, not complete
demonstrations or a clean installation test. See [installation](installation.md#reproduce-the-three-robot-kitchen-setup-from-a-research-checkout).

On 2026-10-02, the public asset archive was downloaded anonymously, SHA-256 checked, extracted,
and verified against its per-file manifest. Job 2374604 then passed ten-step kitchen 813 smokes
for Anubis, RB-Y1, and AI Worker using those downloaded assets. The first public-path RB-Y1
goal-emission attempts exposed a sphere-file path mismatch (jobs 2374917 and 2375175); after the
portable asset-root resolver was added, job 2375633 wrote a validated goal for kitchen 813
rotation 0. Its adapted task passed a fresh ten-step GPU smoke with five non-constant cameras
(job 2375645). Complete new RB-Y1 episodes remain unverified.

The public kitchen USD's baked maintainer path exposed a BODex relocation defect. Grasp lookup now
uses `BODEX_OBJ_DIR`; a focused Isaac Sim utility regression passed on RTX 3090 (job 2375316), and
the re-emitted Anubis 813 task config passed a fresh ten-step smoke (job 2375386). This fixes asset
lookup portability for the packaged kitchen and supports public-bundle goal emission.

Public AI Worker authoring also completed for kitchen 813 rotation 0 on the pinned bundle: the
right-arm goal retained 13 of 19 authored grasps, while the explicit left-arm template retained 10
of 40 after collision-clearance filtering. The right-arm adapted goal passed a ten-step GPU smoke
with five non-constant cameras (job 2375696); its lift-home error was 13.3 mm. Its collector then
repeatedly failed the self-collision start-state gate, so the left-arm goal is now the documented
public recipe and its fresh smoke passed. The left-arm collector passed its pre-lift mug grasp
check and reached two reset targets without an optional extra lift, but one attempt ended
`fail_but_done`; see the bounded trial details below. No successful public AI Worker episode has
been exported. A bounded public RB-Y1 collection initialized both planners but failed its first
reach attempt at 0.594 m / 40.3°. The corrected left-arm trial was stopped after 27 minutes with
repeated large misses (0.3799–0.8117 m positional error and 23.0–104.8° orientation error); it
produced no episode. Neither robot has a new validated public-path episode.

On 2026-10-02, the public Hugging Face bundle was downloaded without authentication at immutable
asset revision `e7ea01d56d00b70afd05a45bca21460cf7389589`. Its archive SHA-256 matched
`b4661b95c99f0212c8770e12e01099f7ac7dbe8b2565b7eb9e88f2b7f2c28fb8`; Zstandard integrity and all
21,228 per-file manifest hashes passed after extraction. All three `simvla doctor --robot ...`
checks reported their required files ready from that extracted tree. Slurm job 2374604 then ran
10-step kitchen 813 smoke tasks for Anubis, RB-Y1, and AI Worker using those extracted assets;
all completed successfully, rendering three, four, and four non-constant cameras, respectively.
This verifies the public acquisition-to-reset/render path on RTX 3090, not successful collection
for RB-Y1/AI Worker or bitwise-identical physics.

### Current public collection gate (2026-10-03)

The current release gate is intentionally stricter than reset/render smoke tests: each robot must
export at least one task-success episode with finite, varying action/state data and synchronized
camera streams from the documented public asset bundle. Focused candidate-0 tests on the mug
robots did not produce accepted episodes. The Anubis gap-gate run (job 2380019) reached its
45-minute Slurm time limit without exporting an episode; its final candidate failed jaw/object
alignment (0.179 m XY, +0.153 m jaw height, 8.2 mm jaw gap). There is not yet a new three-robot
collection claim. Redundant baseline jobs were stopped after their logs were captured.

Anubis continues to reach and lift the bowl in some candidates, but the current full task then
fails on a later left-arm `arm.pose` plan (`MotionGenStatus.IK_FAIL`); one nearby candidate was
incorrectly counted as a grasp despite a collapsed 0.3 mm jaw gap and a dropped bowl. The
collector now defers reusable pose-bank promotion until after the commanded lift, and requires
the object to rise by at least 50 mm while the jaws retain a gap of at least 15 mm. Focused unit
tests cover the measured dropped-object and retained-object cases. This is a stricter bank
criterion, not proof of successful full-task collection:
the new public trial also banked candidates with 30.8–54 mm jaw gaps, then lost the bowl during
lift (one fell below the kitchen floor). No Anubis public full episode has been exported by this
trial.

For the public mug task, the left jaw target was adjusted to half the configured mug height above
the object root and augmentation seeds are deterministic. This did not fix robot reachability.
RB-Y1's selected candidate-0 trial exhausted 600 corrections at 0.216 m / 32.7 degrees from the
target with the jaw midpoint 0.188 m from the mug; moving the orientation waypoint from 0.18 m to
0.35 m worsened the final error to 0.357 m / 31.9 degrees. AI Worker's candidate-0 trials showed
the mug being displaced 86 mm and tilted 26 degrees with the near waypoint, and tipped 91 degrees
with the far waypoint; neither reached a valid grasp. The previous 600-step attempts also failed.
All candidate-0 output HDF5 files remain empty 96-byte containers. These runs used public assets
and robot-specific kitchen configs, but exported no accepted episodes. The remaining blocker is
robot-specific grasp reachability/orientation and subsequent task completion, not public asset
acquisition. Do not describe collection as reproducible on all three robots until new accepted
episodes pass the checks above.

On 2026-10-03, AI Worker candidate 0 was rerun after aligning the authoring templates and public
task's object friction/restitution with the high-friction, low-bounce contact setup (job 2380766,
A6000). The corrected above-object pregrasp reached a 4.8 mm / 1.5-degree cuRobo endpoint residual,
the mug remained upright during approach, and the pre-lift proximity gate passed (17.5 mm jaw
midpoint error, +77.3 mm jaw height). The mandatory carry gate then rejected it: after the 0.20 m
arm lift the mug had fallen 6.3 mm and the jaws had collapsed to 12.6 mm. No episode was exported
(the HDF5 is still the empty 96-byte header). Lower restitution/friction settings alone therefore
do not establish retention; gripper/object contact geometry or the selected grasp remains the
AI Worker blocker. These candidate changes must be revalidated in a fresh run. This is a failure
trace, not evidence of reproducible collection.

A four-environment follow-up (job 2380794) varied the robot's initial base placement under seed 0.
Environments 0, 2, and 3 reached the same pre-lift pose window (jaw XY 7.6–14.7 mm, jaw height
79–81 mm) but then closed to 12.5–12.6 mm jaw gaps while the mug rose only 3–6 mm; all three
failed the post-lift retention check. Environment 1 failed the pre-lift XY check (29.6 mm). No
episode was exported. This rules out the single environment-0 initial base placement as the sole
cause; the measured failure is common across the sampled starts. The GPU batch was stopped after
these gates failed, and its outputs are not demonstrations.

An AI Worker control trial raised `SIMVLA_GRIPPER_CLOSE_MAX_TICKS` from 80 to 200 (job 2381028).
It again passed only the pre-lift pose check (14.1 mm jaw XY, +79.6 mm height, 61.4 mm jaw gap);
after lift the mug fell 3.9 mm and the gap closed to 12.5 mm. The larger close allowance did not
improve retention. Its output remained empty, and the job was stopped after that failed gate.

The RB-Y1 author-time object park is now tightened from a -0.05 m to -0.10 m profile adjustment.
This is based on its remaining 0.216 m reach miss and the measured 0.345 m forward footprint;
the arithmetic leaves an estimated 0.065 m nominal front clearance. Existing goal JSON files do
not inherit this source change: regenerate the task/goal, run the clearance check and a fresh
GPU smoke, then remeasure arm reach before attempting collection. This setting is unvalidated
and does not change the public collection gate above.

The fresh RB-Y1 re-emission did pass its 20-step GPU smoke with five non-constant cameras. Its
nav-clearance pass moved the mug park to 0.490 m from the object with a 0.039 m measured cabinet
clearance and a straight route. Collection then failed the first two pre-lift checks: jaw XY was
66.9 mm and 75.3 mm, with the fingers already collapsed to 7.2–7.3 mm. The verbose trace also
showed 52.7-degree orientation error on the fallback jog. This exposed that RB-Y1 mugs defaulted
to raw authored wrist targets because the measured jaw-frame cylindrical-mug correction was
AI-Worker-only. With the correction explicitly enabled (job 2381245), the first candidate's far
waypoint had near-zero FK error, but its final reach still missed by 96.6 mm / 7.5 degrees and its
jaw midpoint remained 90.1 mm from the mug. A four-environment diagnostic (job 2381307) sampled
four candidates in parallel with the same correction. None reached the grasp gate: position
misses ranged from 0.114 to 0.363 m and orientation misses from 9.8 to 27.1 degrees; one start
tipped the mug 91 degrees before contact. No episode was exported. The generic
AI Worker jaw-frame override therefore does not solve RB-Y1's pose/control mismatch; do not make
it the RB-Y1 default. The remaining RBY blocker is reaching a stable, collision-clearing grasp
pose from its measured base park, not task registration or camera setup.

Anubis job 2381340 used the public 813 bowl-to-drawer goal and reproduced one physically retained
grasp: after a 0.20 m lift, the bowl rose 0.1523 m and retained a 16.4 mm jaw gap. This passed the
post-lift pose-bank check, but not the full task. The next authored left-arm `arm.pose` target
returned `MotionGenStatus.IK_FAIL`; the new bounded Cartesian fallback plateaued first at 74.2 mm
/ 95.7 degrees and, on another retry, 20.6 mm / 97.7 degrees before resetting. Another candidate
failed retention (0.4 mm object rise). The HDF5 stayed at its empty 96-byte header and the bounded
batch was stopped after repeated failures. Thus the fallback is exercised but the target remains
unreachable in this task; no public Anubis episode was exported. The public full-collection gate
remains open for all three robots.

The compatibility-overlay reproduction also passed a clean-environment, public-asset smoke for
all three robots on 2026-10-02 (Anubis job 2376110; RB-Y1 and AI Worker job 2376124). Each reset
and rendered five non-constant cameras. The overlay matches Kit's bundled Botocore and resolves
the simulator's NumPy/protobuf constraints, but `pip check` still reports the upstream
LeRobot 0.4.2 → Rerun SDK → NumPy 2 requirement against Isaac Sim 4.5's NumPy <2 requirement.
Runtime imports and smoke pass with NumPy 1.26.4; the dependency metadata conflict is disclosed,
not silently suppressed.

Public left-arm kitchen-813 collector trials use freshly emitted goals with a pre-lift grasp
checkpoint. AI Worker job 2376127 reached a valid mug grasp (11.5 mm EEF goal error), passed the
new object-to-EEF check (80.7 mm under the 150 mm bound), and advanced the left subtask 1/1. With
the optional extra 0.20 m lift disabled, one attempt reached both authored arm.reset targets but
ended `fail_but_done` rather than task success. A trace-enabled baseline (job 2376165) showed
`obj_near_prim`, `eef_home`, and `last_subtask` all false at termination; the final reset planner
reported `INVALID_START_STATE_JOINT_LIMITS`. The left-arm planner start state now receives the
same joint-limit clamp as the right-arm path. A clean-overlay GPU rerun (job 2376208) confirmed
both left reset plans succeeded and advanced `last_subtask`, but task success still failed with
`obj_near_prim` and `eef_home` false. The distance-instrumented run (job 2376221) measured that
the mug remained about 2.03 m from the sink and the settled home pose was 0.155–0.156 m from the
configured 0.12 m threshold; no demonstration was exported. The home tolerance was subsequently
widened to 0.18 m, but a complete public AI Worker episode is still unverified.
A later attempt passed the same grasp check again but failed its next reach. The
20-minute bounded no-extra-lift run exported no demonstration (its HDF5 is an empty 96-byte
container) and was stopped after repeated retries; an earlier extra-lift run repeatedly failed its
first reset. RB-Y1 job 2376125 initialized both
planners but had not passed its strict 12 mm / 5-degree reach gate. Neither run has exported a
complete public-path episode, so all-robot collection is not yet validated. The corrected RB-Y1
trial's logged position/orientation errors included 0.3799 m / 49.3°, 0.4272 m / 51.7°,
0.5048 m / 104.8°, 0.3906 m / 58.9°, and 0.8117 m / 23.0° from target. Its HDF5 output remained
an empty 96-byte container; these are not marginal tolerance misses.

On 2026-10-03, a public AI Worker left-arm rerun with a 20 mm pregrasp transition reached a
0.412 m lateral starting error but missed the grasp after 300 corrections (0.258 m EEF error,
0.073 m jaw-axis error, +0.239 m jaw-height error; job 2376932). A deterministic seed-1 rerun
with the new opt-in `SIMVLA_REACH_RETRIES=600` reduced lateral error from 0.178 m to about
0.020 m, then plateaued; after 600 corrections it still missed by 0.088 m, with 0.057 m jaw-axis
and +0.066 m jaw-height errors (job 2376935). Both wrote only empty HDF5 containers and were
stopped before an invalid demonstration could be reported as data. Seed 2 began 0.969 m laterally
from the standoff and was stopped during bounded correction (job 2377107). A staged seed-3 run
then exposed another collision: after 600 corrections its EEF was 0.205 m from target, the jaw
axis 0.181 m from the mug, and the jaw 0.086 m too high (job 2377502). A measured-orientation
approach still displaced the mug while rotating at the 0.18 m standoff (job 2377515). Moving to a
0.35 m waypoint before local-IK rotation preserved the mug better, but local IK plateaued at
11.7° orientation error after 600 corrections; jaw error was 0.057 m XY and +0.356 m height (job
2377769). The next change asks cuRobo to plan the large orientation change at that distant
waypoint, then advances to the 0.18 m standoff and final grasp. Focused tests passed; the planner
variant did not produce a public episode in the subsequent GPU trial (below).

The subsequent public-bundle trials on 2026-10-03 also failed to produce episodes. Anubis job
2379734 attempted the public bowl-to-drawer sequence; its right-arm close approach plateaued
0.0208–0.0210 m short after 600 corrections (2.6–2.7°), and its HDF5 remained a 96-byte empty
container. AI Worker job 2379731 missed the left mug grasp by 0.3102 m / 110.8° after 600
corrections, with the jaw midpoint 0.2398 m off-axis. RB-Y1 job 2379505's left-arm jog plateaued
0.2138 m / 24.2° from the grasp goal, with jaws 0.1972 m off-axis. The latter two runs likewise
contained only empty 96-byte HDF5 containers. All three runs were stopped; none is collection
evidence. These results show that reproducible full public collection has not yet been
demonstrated on any of the three current public paths.

Controlled band-0 follow-ups on the public bundle narrowed two failure modes. Anubis job 2379765
missed one bowl grasp by 0.8 mm (12.8 mm vs its 12 mm gate); job 2379990 with a 15 mm gate did
pass one reach and the loose proximity check, but the jaw gap was only 7.8 mm. During lift the
bowl fell (`lifted_by=-23.5 mm`, 58.9° tilt) and the following arm plan failed. This proves that
EEF proximity alone admitted an empty close into the “good goal” bank. The pre-lift check now
rejects jaw gaps below 15 mm; this evidence-based gate has CPU coverage and is being revalidated
on GPU. Anubis has not yet exported a full episode from the public bundle.

The band-0 AI Worker job 2379991 reached its mug approach but displaced the mug 145 mm and tipped
it 18.9° before missing the reach gate by 97.8 mm / 7.4°; no episode was exported. RB-Y1 job
2379992's first left grasp missed by 232.9 mm / 55.4° after 600 corrections; later attempts are
still being evaluated. A separate AI Worker/RB-Y1 rerun now targets the measured half-height
above the mug root on the left arm, matching the existing calibrated right-arm target; complete
episodes remain unverified pending those GPU runs.

A follow-up source audit found one independent reproducibility defect: collection seeded Python,
NumPy, Torch, Isaac Lab, and the camera randomizers, but passed `seed=None` to lighting
randomization. Camera and lighting seeds are now derived deterministically and independently from
the CLI seed, logged at startup, and covered by a regression test. The selected CPU/source gate
passed afterward (722 passed, 1 skipped); this source-level fix does not change the failed GPU
collection evidence above.

A public Anubis seed-1 attempt (job 2377180) remained 0.182 m from its wrist target with 34.1°
orientation error at correction step 561 and was stopped; its HDF5 container had no episode. The
public RB-Y1 left-arm seed-1 attempt (job 2377187) remained 0.381 m / 128.8° from the target by
step 81 and was stopped before a demonstration could be exported. These runs reinforce that the
public asset-to-smoke path works, but do not certify collection for any robot.

A one-environment Anubis collector trial with the staged assets (job 2366831) initialized the
kitchen and both cuRobo motion generators. Its first two sampled grasp approaches reset at the
strict right-arm reach gate (position errors 0.0535 m and 0.1899 m against a 0.012 m limit). It
was stopped after 6–7 minutes without exporting a demonstration. This is evidence that reset,
rendering, and planner initialization work, **not** that complete collection is reproducible.
After restricting cylindrical-mug corrections to AI Worker mug grasps, job 2369539 completed one
Anubis bowl-to-drawer episode from the freshly installed environment and staged assets. It used
`SIMVLA_POSTGRASP_LIFT=0.20` and `SIMVLA_EPISODE_STEPS=3000`; the default 1,500-step horizon had
timed out during the lift in job 2368950. The export contains one HDF5 demonstration, 1,082
LeRobot rows with finite and varying 23-D actions and states, and three camera videos. The
collector logged bowl rises of 0.1219 m and 0.1496 m in the successful attempt. This validates
the local Anubis collection path, not public asset availability or RB-Y1/AI Worker completion.
One-pass replay of this randomized-object episode (job 2370846) did not reproduce task success
and exported no SimVQA records. The LeRobot dataset stores the robot start pose, not the object's;
the replay drew a fresh object offset. The collector now records object reset poses in a local
LeRobot sidecar and replay restores them when present, but that new path still needs a GPU
reproduction. For older datasets, fixed-object collection and replay are the deterministic
pairing; this task's fixed-object collection trial did not itself complete an episode.
Job 2371057 subsequently reached the full Anubis task success predicate with a bounded gripper
close and post-grasp lift, but its new sidecar export crashed because Isaac Lab supplied
`EpisodeData.env_id=None`. The exporter now associates episodes with the reset/success env masks;
this fix is unit-tested but still awaits a completed GPU export.
The 30-minute follow-up collector (job 2371266) used that fix, authored bowl grasps, and a
strict left-hand handle reach gate. It exported no episode before the Slurm time limit: several
attempts lifted and carried the bowl, then stopped 0.064–0.073 m from the drawer-handle grasp
goal after 300 correction steps (0.012 m required); others lost the bowl earlier. Do not use
this run as evidence that sidecar replay or stable multi-skill collection works.
The seeded, fixed-object follow-up (job 2371791; 45-minute limit) also exported no episode. It
reached two bowl-lift checks of 0.156 m and 0.119 m, but both attempts failed the task's object
height predicate; other candidates failed the 0.012 m / 5-degree grasp gate. Two left-handle
attempts stopped about 0.071 m from the goal after 300 correction steps. The run used seed 0,
the zero-radian per-joint IK deadzone, and the safer initial base pose; preserve its log under
`outputs/collect-safe-drawer-anubis-2371791.log` when retaining local validation artifacts.
The earlier expanded 630-test portable-plus-source gate also passed on this checkout; the current
735-test result is at the top of this report. The browser-server tests need localhost socket access.

The asset stage now verifies both source revisions and fourteen input hashes, regenerates the
AI Worker planning URDF, and converts the prepared RB-Y1 root USD's workstation-specific absolute
reference into a portable relative one. A fresh 21,252-link stage and its idempotent recheck passed
on 2026-10-02. A relocation check loaded all five RB-Y1 USD layers from a different directory;
five focused staging/normalization tests passed. That initial relocation check used an RTX 4090
allocation that failed CUDA/PhysX initialization before a frame, but the later RTX 3090 public
asset and compatibility-overlay smoke checks listed above passed for all three robots.

| Workflow | Observed result |
| --- | --- |
| Physics / existing task cameras | Falling cube and 20-step Anubis checks passed; finite joints and front/wrist RGB images |
| Kitchen generation | Public CLI generated twelve rotated USDs, task configs, template, and metadata (2349398) |
| Goal generation | Public CLI emitted twelve goal files; generated-task smoke passed all four cameras and finite joints with seed 0 (2349398) |
| AI Worker config | A transformed `Isaac-Kitchen-v1215a-00` reset on one RTX 3090 and exposed a 28-joint robot state plus 240×320 RGB/RGBA tensors for front, both wrists, and CCTV (2351146) |
| AI Worker lift home | `Isaac-Kitchen-v1967amug-00` stepped 60 frames with five nonconstant cameras. The live lift target remained within 2.77 mm at `0.0`, 13.11 mm at `-0.30`, and 13.09 mm at `-0.35` (2352292, 2352309). Historical reach probes favored `-0.30`: both arms solved while retaining 4.8 cm more low-link clearance than `-0.35`. |
| AI Worker symmetric home | Pose option #1 is now the `-0.30` m default and its exact mirrored arm joint values are pinned by a CPU test. Job 2366178 rendered those joints from six distinct camera angles with a visibility-only neutral material override. Earlier reset and collection evidence (2357114, 2357115) used a different mirrored home; a fresh reset/collection run for option #1 has not been recorded. |
| AI Worker collection | Two bounded 16-environment trials used the `-0.30` m home on `Isaac-Kitchen-v1967amug-00`. Job 2352314 spent part of 30 minutes building a right-pose bank, then logged 33 lift checks: 18 reached at least 0.10 m and the best reached 0.1851 m. Job 2352485 loaded that bank with the documented subgoal overrides, entered full collection immediately, and logged 33 checks: 22 reached at least 0.10 m and the best reached 0.1937 m. Both cuRobo arms initialized, but neither job recorded a full mug-to-sink episode. |
| AI Worker completed dataset | Historical research job 2318419_1 recorded five successful left-arm mug-to-sink episodes for `Isaac-Kitchen-v1967amuglh-00`. The retained LeRobot dataset contains 8,137 frames across episodes of 1,471, 1,899, 1,530, 1,602, and 1,635 frames. Its front and two wrist AV1 videos each contain 8,137 frames at 20 fps and 320×240. This proves the research collector can complete the task with the external runtime and assets; it does not validate a clean public installation. |
| AI Worker reproduction | Job 2356722 reproduced one successful left-arm mug-to-sink episode with the historical collector's complete runtime settings: live cuRobo world, left post-grasp lift, and 0.05-radian final-yaw tolerance. The fresh LeRobot dataset has 1,593 rows with finite, nonconstant 23-D actions and states, plus three synchronized 1,593-frame AV1 videos at 20 fps and 320×240. The historical collector's HDF5 remained empty, so HDF5 replay is not claimed. |
| AI Worker symmetric-home reproduction | Job 2357115 repeated the full mug-to-sink workflow after mirroring the default arms. It recorded one successful episode with 1,375 finite rows and three synchronized 1,375-frame AV1 videos at 20 fps and 320×240. All 23 state dimensions and 13 of 23 action dimensions varied. The first and final front frames show both grippers at matching lower-left and lower-right positions. This used the historical collector and external assets; the [sanitized evidence manifest](evidence/aiworker-symmetric-home-2026-09-30.json) records the command, bounds, and hashes. It is historical evidence, not validation of the current public collector. |
| AI Worker retry diagnosis | Job 2355741 loaded the warmed pose bank and classified two task retries before a deliberate early stop. Both were persistent base/furniture-plan overlap; neither was object height below 0.3 m nor robot-base height below -0.1 m. One affected episode had lifted the mug 0.1770 m before it fell back to the 0.8102 m worktop. |
| RB-Y1 smoke | `Isaac-Kitchen-v1202r-00` reset and stepped 20 frames on one RTX 3090 at 120 Hz physics; front, both wrists, and CCTV returned nonconstant 240×320 RGB images (2351813) |
| RB-Y1 collection | A 15-minute, one-environment `simvla run collect` trial initialized both cuRobo arms and executed grasp reaches. One retry reached the grasp goal at 1.6 mm / 0.4°, but the task's retry termination then fired; two other reaches exhausted the corrective jog. No demonstration was recorded (2351823). |
| System identification | Full 839-frame real episode processed; four candidate gain sets over 19 joints (job 2348723) |
| SimAction | Bundled-goal short command exported one 1,003-frame episode and subtask datasets (2348808); earlier run exported 1,112 frames |
| SimVQA | Successful replay captured 24 records, 72 RGB and 72 segmentation images; expanded into 259 Q/A examples (2348601). A second run produced 253 (2348749) |
| Replay | Successful task replay observed; randomized initial conditions also produced a failed trial |
| Recorded-action evaluation | 20 steps, camera videos and result file; success rate 0.0 for this short integration check (2348414) |

System identification now compares recorded and simulated end-effector positions in the same robot-base frame. Before this fix, base rotation caused valid candidates to be rejected. Accepted gains still need held-out validation; they are not a measured policy or calibration-quality result.

Release checks also found missing VQA segmentation sensors, a symlink-sensitive BODex lookup, and a smoke test that could place the robot inside furniture. The capture enables the required sensors, generation preserves the logical mesh path, and camera tests initialize from authored free-space bands. The CLI checks a result file written before Isaac Sim teardown because Kit can otherwise return zero after a failed workflow.

For `Isaac-Kitchen-v1202r-00`, the retry termination is the union of mug height below 0.3 m and robot-base height below -0.1 m. Job 2351823 logged only the union, so it cannot identify which condition fired. The collector now labels this event `retry_termination` instead of the misleading `OOB`.

## Limits

The detailed findings below include dated historical trials. They do not mean
that AI Worker has no complete public-runtime reference today: collection 2437904,
replay 2437908, and SimVQA now pass together for the documented physics-12
kitchen-813 profile. The new result is bounded to that recipe and seed; unrelated
historical failures and broader unverified scope still apply.

- A fresh compatibility overlay passed five-camera smokes for all three robots against the complete hash-locked, publicly downloadable robot/kitchen asset bundle. Its `pip check` still reports the upstream LeRobot/Rerun/NumPy metadata conflict; do not claim a fully conflict-free GPU dependency graph. The public BODex mesh and grasp archives at revision `3d0326ecc8f6074f3cb1796a249acebda564cab7` were anonymously downloaded, hash-checked, and inspected. See [setup](simulation.md#runtime-and-assets).
- Historical AI Worker collection failure (job 2376221): clean-overlay collection passed left-hand pre-lift proximity twice but retries did not move the mug to its sink predicate and wrote no episode. Later, this was addressed for the physics-12 kitchen-813 profile: collection 2437904, replay 2437908, export, and SimVQA passed together. This remains a bounded seed-specific result, not broad robustness evidence.
- Historical RB-Y1 collection failure (job 2376125): planners initialized but reach errors were 0.38–0.81 m and no episode was written. Later RB-Y1 batch collection/replay passed (see dated evidence above); preserve this failure as diagnostic history, not current status.
- The remaining job-by-job AI Worker and RB-Y1 diagnostics below are historical investigations. Their local conclusions describe those trials and predate the bounded 2026-10-06/07 robot references; they are retained as failure analysis, not current collection status.
- The public AI Worker model used by `AIWORKER_CFG` is a stripped kinematic USD: its configured finger roots have neither visible meshes nor collision shapes. A reproducible overlay tool now references the four distal meshes from the companion public `Robots/FFW_SG2.usd` and authors convex-hull colliders without changing either downloaded source. Its v2 overlay also binds a hash-locked 1.5/1.2 static/dynamic-friction jaw material; both public USDs remain unchanged. Synthetic USD tests pass, and public five-camera smoke jobs 2381623 and 2382424 loaded the collision and sensor-enabled task configs. Collection job 2381809 reached stationary jaw-to-mug contact twice (the mug shifted 0 m and tilted 0.4°), but its goal file supplied `sub_grasp_idx_l=4` even though the closing command is at index 2. The collector therefore never ran the grasp/lift gate, timed out, and wrote only a 96-byte HDF5 header. Follow-up job 2381911 reproduced the same stale-index behavior and was stopped after diagnosis. A CPU preflight and runtime guard now reject active grasp-check indices unless they name that arm's closing gripper step. With the corrected index and per-pad contact sensors, job 2382430 measured about 19 N on one pad and 2.4 N on the other before lift; both readings reached zero during lift, while the mug rose at most 4 mm. Follow-ups with the v2 friction overlay (2382605), a 200-tick close (2382618), and a slower 0.5 rad/s close (2382729) again lost both contacts during lift; in 2382729 the mug fell 8.4 mm after 17.6 N / 2.4 N pre-lift contact. Lowering the grasp target by 1 cm (2382704) caused roughly 9 mm of lateral mug slip and missed the pre-lift XY gate. In job 2382742, a 15 mm pre-close jaw-axis check achieved 14.6–14.9 mm alignment, but closure shifted the mug so the pre-lift error became 29.4–29.6 mm with 17.5 N / 2.1–2.3 N pad loading; both candidates were rejected before lift, and only a 96-byte HDF5 header was written. Follow-up job 2382781 combined a 10 mm centering gate with a 0.5 rad/s close; its final approach pushed the open hand into the mug, displacing it 38 mm and tipping it 15.5° before closure, and wrote only a 96-byte header. Both stricter centering gates have been reverted. Signed-offset diagnostics in job 2382787 showed the relative jaw-midpoint-to-mug vector change from (-12.1, -15.5) mm before close to (-6.7, +23.5) mm just before lift; pads loaded at 17.6 N / 2.4 N, then unloaded completely while the mug fell 8.4 mm. Job 2382795's absolute coordinates refine this: from close to pre-lift the jaw midpoint moved (+3.4, +9.6, -17.6) mm, while the mug root moved (-1.8, -27.5, +7.7) mm. Thus the relative change includes substantial mug motion during closure, not only linkage motion. Pad forces were 18.7 N / 2.5 N before lift, then zero while the mug fell 7.8 mm; the retry failed at reset/reach, and only a 96-byte HDF5 header was written. No complete public AI Worker episode has been validated; the collision/friction overlay and close tuning do not fix retention.
- Follow-up mug-to-sink reach trials on the hash-verified public bundle (jobs 2376827 RB-Y1 and 2376828 AI Worker) confirmed these are not small acceptance-threshold misses. RB-Y1's right arm converged its orientation to 0° but remained 0.106 m from the authored wrist target; another attempt stalled around 0.05–0.11 m and 38–52°. AI Worker's left arm remained 0.196–0.203 m and 27–35° from target after 300 corrective steps; its measured jaw midpoint was 0.174–0.178 m off the mug axis. Both bounded trials were stopped without exporting an episode. The current `mug_to_sink` target/controller pairing therefore is not reproducible for RB-Y1 or AI Worker, despite successful public-path smoke tests.
- Fresh RB-Y1 run 2382934 used the current v813r kitchen adapter, the latest public left-arm goal artifact (`near-goals3`), and the extracted hash-locked RB-Y1/public kitchen assets. Both cuRobo arms initialized, but the first left-arm pre-lift check failed at 61.0 mm jaw-axis error with a 7.2 mm jaw gap; no pad sensor is configured for this embodiment. The run was stopped after the diagnostic and wrote only a 96-byte HDF5 header. The latest authored goal therefore still does not complete a stable RB-Y1 grasp.
- Follow-up RB-Y1 multi-candidate run 2382972 used that goal's native seven-candidate grasp bank, seed 0, the public asset bundle, and default authored geometry. In the bounded 17-minute run, five candidates reached the pre-lift check and all failed: jaw-axis error was 47.0–69.1 mm, jaw-midpoint height was 89.3–117.9 mm above the mug root, and jaw gaps were 6.5–20.5 mm. Another reach plateaued at 113.4 mm / 69.3° after 300 corrections, and a later candidate returned cuRobo `IK_FAIL`. The process was stopped after these repeated failures; its HDF5 remained an empty 96-byte container. This does not establish a complete seven-candidate sweep or an RB-Y1 episode, but confirms the current public grasp bank/tool-frame/controller combination is not collection-ready.
- Follow-up public right-arm RB-Y1 run 2383110 used the v813r adapter, hash-locked public assets, and a freshly emitted mug-to-sink goal. The arm initialized and reached several sampled goals within 3.4–20.1 mm / 0.2–4.2°, but five pre-lift checks failed: logged jaw-axis error was 63.9–93.0 mm, jaw-midpoint height was 45.2–118.5 mm above the mug root, and jaw gaps were 6.5–60.3 mm. Two other attempts failed reach (one cuRobo `IK_FAIL`, one 323.5 mm / 41.4° plateau); the 17-minute bounded run was stopped and exported only the empty 96-byte HDF5 header. Since the current check compares jaw-body origins and the rigid-object root rather than calibrated pad-contact frames, this is evidence of failure, not enough evidence to tune another numeric tolerance safely. RB-Y1 public collection remains unverified.
- The RB-Y1 adapter now emits filtered contact sensors for both fingers of both hands, and the collector uses bilateral mug-pad force for pre-lift acceptance and post-lift retention. Where bilateral sensors exist, their physical measurement replaces root-relative jaw-axis/height checks; missing expected sensors fail closed. CPU adapter/profile tests and generated-config compilation pass. This hardware-path change awaits a GPU integration run and is not collection evidence.
- Follow-up RB-Y1 job 2382942 applied the documented `SIMVLA_GRASP_GEOMETRY=cylindrical_mug` jaw-frame correction to that same public goal. The distant cuRobo waypoint solved, but final approach hit a collision plateau: after 300 jog steps the end-effector remained 0.2613 m / 21.6° from the goal, and the jaw midpoint was 0.237 m off the mug axis. It wrote only a 96-byte HDF5 header. The optional cross-robot correction is not a validated RB-Y1 collection fix.
- AI Worker effort-limit diagnostic job 2382863 used a known-good RTX 3090 node, the corrected left close-step index, signed contact sensors, 0.5 rad/s close, and a temporary 0.2 N·m effort ceiling. It reached the mug, but at pre-lift both jaw-pad sensors read 0 N; the jaw gap remained 0.1019 m before lift and no object lift/contact was registered. The candidate effort cap was removed from the public configuration; default actuation remains unchanged. This is another failed bounded collector trial, not an exported episode.
- That trace also found the AI Worker pre-lift predicate could pass an empty 101.9 mm gap using proximity alone. When both AI Worker mug-pad sensors are configured, the collector now requires at least 1 N on each pad before lift and again after lift; the generic geometry check remains for tasks without those sensors. Force-gate regression tests pass. Follow-up GPU job 2382880 reached closure with 18.46/2.40 N on the pads but failed the preceding 25 mm jaw-axis gate at 26.2 mm; it was stopped with only a 96-byte HDF5 header, so the new force gate/post-lift check still needs a GPU trace that reaches them.
- Robot USD/URDFs, BODex inputs, real recordings, and checkpoints are external inputs. The BODex mesh/grasp pair used by the authoring example is publicly obtainable, but its mug is not physically validated with Anubis.
- Training and SimDeploy integrations are documentation-only. Trained-policy evaluation needs a compatible checkpoint and normalization statistics; recorded-action tests do not substitute for it.
- Generated scenes and emitted goals are not guarantees of feasible or successful manipulation. Fixed-object collection and the bounded RB-Y1 collection trial both failed to record a complete episode.
- Replicator material-randomization errors appear in runtime logs despite successful rendering and collection. Visual randomization is not certified.
- The browser composer is experimental: browser-server unit tests pass, but its UI has not been validated end to end on this release. AI Worker task creation, reset, joints, cameras, reach, grasp, post-grasp lift, and a five-episode research collection are verified. That complete AI Worker run used the historical research collector and external assets; it does not establish current public-bundle collection. Arbitrary embodiments and real-robot execution remain unverified. Jobs 2352314 and 2352485 logged only the external `v1967amug-00` retry union; the collector now separates its object-low and robot-fell leaves and safely infers the single stateful base-overlap leaf without advancing its frame counter.
- The AI Worker collection jobs ran the public generator and bundled planner files against external research assets. The external `v1967amug-00` task also required its historical composed-predicate implementation, which is absent from this source release; this compatibility overlay does not establish a clean public installation.

## Distribution scope

The wheel contains the portable library, CLI, eight templates, and license notices. The repository additionally contains simulator source and small examples. Large assets, recordings, checkpoints, logs, machine settings, and original research history are excluded from the clean release repository. Public release artifacts include wheel, Python sdist, and a separate full-source archive.
