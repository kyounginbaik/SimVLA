"""Validate the single physical control tick omitted from exported observations."""
import json
import math
from pathlib import Path


def load_initial_step(dataset_root, episode_index, *, action_dim, joint_names, decimation):
    root = Path(dataset_root)
    states = json.loads((root / "meta/scene_states.json").read_text())
    if states.get("schema_version") != 1 or states.get("frame") != "environment":
        raise ValueError("Unsupported initial-step scene state schema")
    episodes = states.get("episodes")
    if (type(episode_index) is not int or not isinstance(episodes, list)
            or not 0 <= episode_index < len(episodes)):
        raise ValueError("Initial-step episode index is unavailable")
    entry = episodes[episode_index]
    if not isinstance(entry, dict):
        raise ValueError("Invalid initial-step episode entry")
    step = entry.get("initial_step")
    if (not entry.get("initial") or not isinstance(step, dict)
            or step.get("schema_version") != 1
            or step.get("initial_state_phase") != "before_first_action"):
        raise ValueError("Dataset has no verified pre-action initial step; recollect with the current recorder")

    def vector(value, width, name):
        if (not isinstance(value, list) or len(value) != width
                or not all(type(v) in (int, float) and math.isfinite(v) for v in value)):
            raise ValueError(f"Invalid initial-step {name}")
        return value

    info = json.loads((root / "meta/info.json").read_text())
    feature = info.get("features", {}).get("action.joint", {})
    names = feature.get("names", {})
    if isinstance(names, dict):
        names = names.get("action.joint")
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != len(names) or len(set(joint_names)) != len(joint_names)
            or set(names) != set(joint_names) or feature.get("shape") != [len(names)]):
        raise ValueError("Initial-step motor joint names do not match the robot")
    order = [names.index(n) for n in joint_names]
    motors = vector(step.get("motor_targets"), len(names), "motor targets")
    goal = step.get("subtask_index")
    if type(goal) is not int or goal < 0:
        raise ValueError("Invalid initial-step subtask index")
    result = dict(raw_action=vector(step.get("raw_action"), action_dim, "raw action"),
                  subtask_index=goal, motor_targets=[motors[i] for i in order])
    samples = step.get("motor_substeps")
    if samples is not None:
        if not isinstance(samples, list) or len(samples) != decimation:
            raise ValueError("Invalid initial-step motor substep count")
        result["motor_substeps"] = [
            [vector(sample, len(names), "motor substep")[i] for i in order]
            for sample in samples]
    return result
