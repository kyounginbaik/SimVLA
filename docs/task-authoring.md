# Authoring tasks and skills

SimVLA has two related JSON formats. **Task templates** are the editable source of a task: roles, ordered skills, scene objects, and completion conditions. **Goal files** are generated for a specific kitchen after those roles are bound and planning has run. Edit templates and regenerate goals; a `*.reloadable.json` goal file is authoring metadata, not the portable task template.

## Combine existing skills into a task

The bundled templates are listed by `simvla templates`. Create a copy with a new task name and validate your edits on a CPU:

```bash
simvla init-task --from bowl_to_drawer --name my_bowl_task \
  --output outputs/my-tasks/my-task.json
simvla skills
simvla skills nav.to_prim
simvla validate outputs/my-tasks/my-task.json
```

For example, make a bowl-to-drawer variant with a larger navigation safety margin. The first step uses `nav.to_prim`; `simvla skills nav.to_prim` shows its `safety` parameter and default. After the copy above, run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/my-tasks/my-task.json")
task = json.loads(path.read_text())
task["name"] = "bowl_to_drawer_cautious"
task["steps"][0]["params"]["safety"] = 0.18
path.write_text(json.dumps(task, indent=2) + "\n")
PY
simvla validate outputs/my-tasks/my-task.json
```

This changes one skill call while retaining the other navigation, grasp, drawer, placement, and success steps. For a different task, edit `roles`, `scene`, `steps`, and `success` together; validation catches structural errors before goal generation.

A template has these fields:

| Field | Meaning |
| --- | --- |
| `name`, `language` | Task identifier and the language label saved with data. Avoid path separators and leading/trailing spaces in `language`. |
| `roles` | Named scene bindings. Match a rigid `object_type`, an `articulation_with` feature or feature list, or the `handle_of` another role. |
| `steps` | Ordered `{ "skill", "action", "params", "language" }` calls. A `prim_path` can reference a role such as `@target`. |
| `success` | Required physical completion condition, composed with `all`, `any`, or `not` and the registered predicates. |
| `retry` | Optional condition for abandoning an attempt. |
| `scene` | Objects to place. An empty list defers to the kitchen's existing scene. |
| `subtask_groups` | Optional groups of zero-based step indices for dataset export. |

The bundled `bowl_to_drawer` starts with navigation to `@target`, grasps with the right arm, opens the drawer with the left, places the bowl, and checks a composed success condition. Refer to its [complete JSON](../src/simvla/templates/bowl_to_drawer.json) for a valid example. Another starting point is [open_dishwasher](../src/simvla/templates/open_dishwasher.json). `init-task` refuses to overwrite an existing file. The validator rejects unknown skills, invalid action channels, broken role references, and missing success conditions. It does not prove that an action is reachable or that a generated scene can be solved.

Python callers can inspect the same task and skill registry without starting Isaac Sim:

```python
from simvla.tasks import load_template
from simvla.skill_contract import REGISTRY

task = load_template("bowl_to_drawer")
print(task.name, len(task.steps))
for skill_id, spec in sorted(REGISTRY.items()):
    print(skill_id, spec.actions, [param.name for param in spec.params])
```

`load_template` registers the built-in skills. Registry names and parameter declarations are the source of truth; use them when composing steps instead of copying numeric skill IDs in a generated goal.

## Generate goals for a kitchen

Goal generation needs the simulator stack, a generated or registered kitchen, and its assets. First validate your template on a CPU. Then, in the configured simulator environment and from the checkout:

```bash
simvla run goals --repo-root . -- \
  --template outputs/my-tasks/my-task.json \
  --kitchens 99000 \
  --out outputs/my-goals
```

The kitchen number must exist; `--kitchens` can take a comma-separated list. The command writes task-specific goal files to a new output directory and refuses the bundled corpus directory. Preflight each emitted file with `simvla validate-goal outputs/my-goals/Isaac-Kitchen-v99000-00.json` before collection. Version 2 goals receive skill/action checks; unversioned legacy goals receive structural checks only because their skill identities live in historical numeric payloads. [Scene generation and the full runtime setup](simulation.md) explain how to create kitchen `99000`, check its cameras, and point collection at the emitted goals. Generated goals depend on object placements and robot-specific planning; validate them in simulation before treating them as training data.

## Add a skill API

A skill is a Python registration in [`src/simvla/skills.py`](../src/simvla/skills.py), declared with `@skill(id=..., actions=(...), label=..., params=[...])`. Each skill provides exactly one of `plan()` (resolved while authoring a goal) or `resolve()` (resolved at runtime). The declarations use `PrimPath`, `Choice`, `Float`, and `Bool` parameter types from [`skill_contract.py`](../src/simvla/skill_contract.py). Skill names are persisted in goal files; sorted numeric IDs are local to a process and must not be written as an API.

The portable and simulator entry points import the same canonical registry in `src/simvla/skills.py`; the script-side module is a compatibility shim. Add a declaration once in the canonical module and run both portable and simulator contract tests. The executor can still need a separate dispatch branch for a new runtime behavior.

For an existing action channel, register the skill in the canonical module, then add a template step with its `skill` ID, an allowed `action`, and matching `params`. Skills that need USD geometry or the grasp chooser declare `plan = _authored("your.skill")` and register their planning body in [`simvla_data_generator.py`](../scripts/simvla/simvla_data_generator.py). A runtime skill needs a `resolve()` implementation, parameter encoding in [`executor_dispatch.py`](../scripts/simvla/executor_dispatch.py), and a matching dispatch branch in [`simvla_gen.py`](../scripts/simvla/simvla_gen.py); registration alone is insufficient, even on an existing action channel. Add simulator-side contract and dispatch tests. A new action channel also needs executor support.

Before using a new skill in a GPU job, run the CPU package tests and the simulator-side skill contract tests. Then emit a goal for one kitchen, inspect its planned values, and smoke-test the scene. Planning may depend on a robot's tool frame and grasp assets, so a skill tested on one embodiment is not automatically valid on another.

### Worked runtime skill: `arm.pause`

[`arm.pause`](../src/simvla/skills.py) is the smallest complete runtime skill already shipped. It uses an existing right/left/both arm action channel, takes no parameters, and resolves to the hand's current pose. Its declaration in the canonical registry is:

```python
@skill(id="arm.pause", actions=("A_r", "A_l", "A_b"), label="Pause", params=[])
class ArmPause:
    def resolve(self, ctx, envs, params, eef_idx=None):
        pos_w = ctx.robot.data.body_pos_w[:, eef_idx][envs]
        quat = ctx.robot.data.body_quat_w[:, eef_idx][envs]
        return pos_w - ctx.env_origins[envs], quat
```

The executor looks up `SID_PAUSE = skill_ids["arm.pause"]` and, for the selected arm, calls `resolve_skill("arm.pause", ...)` before writing position to payload slots `:3` and quaternion to `3:7` and marking the step resolved. See the actual right-arm branch in [`simvla_gen.py`](../scripts/simvla/simvla_gen.py). A task template can invoke it as `{ "skill": "arm.pause", "action": "A_r", "params": {}, "language": "Hold position" }` when the hand should stay at its current pose.

To add a different runtime arm skill, use that path as a checklist: add its declaration and `resolve()` to the canonical registry; add its name lookup, unresolved mask, resolver call, payload write, and resolved flag in the corresponding arm branch; add a contract test and a task-template validation test. A skill with a new payload shape must also update [`executor_dispatch.py`](../scripts/simvla/executor_dispatch.py). These CPU checks exercise the shipped example and the registry contract:

```bash
simvla skills arm.pause
python -m pytest tests/test_skill_registry_parity.py scripts/simvla/test_skill_contract.py \
  tests/test_task_runconfig.py -q
```

The code path is complete in the source; validating a newly written resolver's physical behavior still requires a bounded simulator run.

### Runnable new-skill example: `gripper.open`

[`examples/skills/gripper_open.py`](../examples/skills/gripper_open.py) declares a distinct planned skill for the existing `G_r`/`G_l` channels. Its `plan()` returns `False`, which those channels already execute as an open command. The example stays isolated from the production registry so the public skill list is unchanged. [`test_skill_extension_example.py`](../tests/test_skill_extension_example.py) imports that class, builds and validates a one-step template with a physical `gripper_open` success condition, writes the planned result in the version 2 goal shape, and runs the CPU goal preflight.

```bash
python -m pytest tests/test_skill_extension_example.py -q
```

To make this a shipped skill, move the class into `src/simvla/skills.py`, add the task and goal checks to the package tests, and inspect any state tracking in `task_validate.py` and `task_runconfig.py` that depends on `gripper.set`. The generic gripper executor already handles the `False` planned payload on `G_r`/`G_l`; a skill using a different channel or payload still needs an explicit runtime dispatch update and a GPU smoke test. This example establishes contract, template, and goal behavior on a CPU; it does not claim a real robot run.

## Using AI Worker

There are two AI Worker names in the source: `aiworker` selects the FFW_SG2 kitchen robot in [`aiworker.py`](../source/isaaclab_assets/isaaclab_assets/robots/aiworker.py), kitchen previews, and the demonstration collector; `ai_worker_bg2` selects the separate BG2 real-to-sim/system-identification configuration in [`aiworker_BG2.py`](../source/isaaclab_assets/isaaclab_assets/robots/aiworker_BG2.py). They are not interchangeable CLI values.

For an AI Worker kitchen experiment, provide the matching external FFW_SG2 USD and source BG2 URDF, then build the cuRobo planning URDF into your robot model tree:

```bash
python scripts/tools/build_aiworker_urdf.py \
  --source "$SIMVLA_ROBOT_MODELS_DIR/ai_worker_min/ffw_description/urdf/ffw_bg2_rev5_follower/ffw_bg2_follower.urdf" \
  --output "$SIMVLA_ROBOT_MODELS_DIR/ai_worker_min/ffw_description/urdf/ffw_sg2_follower/ffw_sg2_follower.urdf"
```

The output is a planning model; the simulator continues to use the FFW_SG2 USD and its revolute gripper. Compare the generated planning frame with the simulator USD before collection:

```bash
python scripts/tools/check_aiworker_urdf_vs_usd.py \
  --urdf "$SIMVLA_ROBOT_MODELS_DIR/ai_worker_min/ffw_description/urdf/ffw_sg2_follower/ffw_sg2_follower.urdf" \
  --usd "$SIMVLA_ASSETS_DIR/Robots/MM/aiworker/ffw_sg2.usd"
```

The repository includes the two cuRobo configs and measured collision spheres. `simvla doctor --robot aiworker --repo-root .` lists every required file. Generate a kitchen as described in the [simulator guide](simulation.md#1-kitchen-scene-generation), then author and adapt it in this order:

```bash
export SIMVLA_ROBOT=aiworker
simvla run goals --repo-root . -- \
  --template src/simvla/templates/mug_to_sink_left.json --kitchens 99000 \
  --out outputs/aiworker-goals
simvla adapt-aiworker --repo-root . --kitchen 99000 \
  --goals-dir outputs/aiworker-goals
export SIMVLA_GOALS_DIR="$PWD/outputs/aiworker-goals"
simvla run smoke --repo-root . -- \
  --task Isaac-Kitchen-v99000a-00 --robot aiworker --seed 0 --headless
```

Setting `SIMVLA_ROBOT` before emission makes grasp planning use the AI Worker tool frame. `adapt-aiworker` converts every emitted environment config with checked substitutions for the robot, seven-joint arms, revolute grippers, camera mounts, frame offsets, contact sensors, and physics rate. It registers `Isaac-Kitchen-v99000a-NN` task IDs and renames their goals to the same IDs; it refuses stale source-template anchors and conflicting goal files. Run a one-environment smoke using `Isaac-Kitchen-v99000a-00`, then collect with `--robot aiworker` only after checking physics, cameras, reachability, and gripper behavior. The [simulator setup](simulation.md#runtime-and-assets) lists the external asset roots. `doctor` checks file presence, installed packages, and reports whether the SimVLA cuRobo patch is present; neither it nor the transformer tests proves manipulation success.

The default `SIMVLA_AIWORKER_LIFT_M=-0.30` home preserves the original hand height while lowering the shoulder for counter work. Set it to `0` to restore the original top-lift C2 pose. Only `0`, `-0.30`, and `-0.35` have matching measured arm homes; other values are rejected. The collector copies the live reset state into cuRobo's retract and locked-lift settings before planning. Robot models remain external, and a new counter height or object can still change reachability and grasp success.
