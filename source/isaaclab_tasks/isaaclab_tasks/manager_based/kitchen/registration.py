"""Discover generated task names without importing simulator configurations."""
from pathlib import Path
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


def register_output_dir(gym, package_path, package, environ):
    """Register configs that goal emission wrote to SIMVLA_ENV_CFG_OUTPUT_DIR.

    Appending the directory to the package ``__path__`` lets the entry point
    ``<package>.kitchen_N_SS`` import from there (relative ``.mdp`` imports still
    resolve inside the package), so generated configs never have to be written
    into the source checkout.
    """
    raw = environ.get("SIMVLA_ENV_CFG_OUTPUT_DIR")
    if not raw:
        return
    directory = Path(raw).expanduser().resolve()
    if not directory.is_dir() or any(Path(p).resolve() == directory for p in package_path):
        return
    package_path.append(str(directory))
    register_kitchens(gym, directory, package)
