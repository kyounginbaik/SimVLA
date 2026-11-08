"""Register bundled and newly authored kitchen configurations lazily."""
import os
from pathlib import Path

import gymnasium as gym

from .registration import register_kitchens, register_output_dir

register_kitchens(gym, Path(__file__).parent, __name__)
register_output_dir(gym, __path__, __name__, os.environ)

gym.register(
    id="Isaac-Kitchen-v1407r-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1407r_00:RBY1KitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1407a-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1407a_00:AIWorkerKitchenEnvCfg",
    },
)
