"""Discover generated task names without importing simulator configurations."""
import re


def register_kitchens(gym, directory, package):
    classes = {"": "AnubisKitchenEnvCfg", "r": "RBY1KitchenEnvCfg", "a": "AIWorkerKitchenEnvCfg"}
    for path in sorted(directory.glob("kitchen_*.py")):
        match = re.fullmatch(r"kitchen_([0-9]+)([ar]?)_([0-9]{2})\.py", path.name)
        if not match:
            continue
        number, robot, rotation = match.groups()
        task = f"Isaac-Kitchen-v{number}{robot}-{rotation}"
        if task not in gym.registry:
            gym.register(
                id=task,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": f"{package}.{path.stem}:{classes[robot]}"},
            )
