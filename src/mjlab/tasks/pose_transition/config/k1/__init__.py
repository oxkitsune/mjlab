import gymnasium as gym

ENV_CFG_ENTRY = "mjlab.tasks.pose_transition.config.k1.env_cfg:BoosterK1PoseTransitionEnvCfg"
RL_CFG_ENTRY = "mjlab.tasks.pose_transition.config.k1.rl_cfg:BoosterK1PoseTransitionPPORunnerCfg"

GYM_ID = "Mjlab-PoseTransition-Booster-K1"

gym.register(
  id=GYM_ID,
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ENV_CFG_ENTRY,
    "rl_cfg_entry_point": RL_CFG_ENTRY,
  },
)
