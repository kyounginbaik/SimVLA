"""Strict loading of optional physics-rate motor commands beside LeRobot data."""
import json
from pathlib import Path

import numpy as np


def load_joint_substeps(root, episode_index, live_names, *, frames, decimation, control_fps):
    meta = Path(root).resolve() / "meta"
    manifest = json.loads((meta / "joint_substeps.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("control_fps") != control_fps:
        raise ValueError("Motor substeps require matching schema and control FPS")
    names = manifest.get("joint_names")
    if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or len(set(live_names)) != len(live_names)
            or set(names) != set(live_names)):
        raise ValueError("Motor substeps must name every live joint exactly once")
    episodes = manifest.get("episodes")
    if (type(episode_index) is not int or not isinstance(episodes, list)
            or not 0 <= episode_index < len(episodes) or not isinstance(episodes[episode_index], dict)):
        raise ValueError("No motor substeps for requested episode")
    entry = episodes[episode_index]
    if entry.get("frames") != frames or entry.get("substeps") != decimation:
        raise ValueError("Motor substeps must match episode length and physics decimation")
    file = entry.get("file")
    if not isinstance(file, str):
        raise ValueError("Motor substep file must be a relative metadata path")
    path = (meta / file).resolve()
    if Path(file).is_absolute() or not path.is_relative_to(meta):
        raise ValueError("Motor substep file must stay inside dataset metadata")
    array = np.load(path, allow_pickle=False)
    if (array.shape != (frames, decimation, len(names)) or array.dtype.kind != "f"
            or not np.isfinite(array).all()):
        raise ValueError("Invalid motor substep array")
    return array[:, :, [names.index(name) for name in live_names]]
