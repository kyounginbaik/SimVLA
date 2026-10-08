import gymnasium as gym

gym.register(
    id="Real2Sim-v1-anubis",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.real2sim_env_cfg_anubis:Real2SimEnvCfg",
    }
)
gym.register(
    id="Real2Sim-v1-ai_worker_bg2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.real2sim_env_cfg_ai_worker_bg2:Real2SimEnvCfg",
    }
)

# Mode-suffixed ids for scripts/simvla/systemid.py, which resolves
# Real2Sim-v1-<robot>-<control>. anubis' cfg uses a DifferentialIK action term (eef);
# ai_worker_bg2's uses a JointPosition action term (joint) — so each robot's existing
# cfg is aliased to the control mode it actually implements.
gym.register(
    id="Real2Sim-v1-anubis-eef",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.real2sim_env_cfg_anubis:Real2SimEnvCfg",
    }
)
gym.register(
    id="Real2Sim-v1-ai_worker_bg2-joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.real2sim_env_cfg_ai_worker_bg2:Real2SimEnvCfg",
    }
)

gym.register(
    id="Real2Sim-v1-rby1-joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.real2sim_env_cfg_rby1:Real2SimEnvCfg",
    },
)
