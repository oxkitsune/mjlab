from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class BoosterK1PoseTransitionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  policy: RslRlPpoActorCriticCfg = field(
    default_factory=lambda: RslRlPpoActorCriticCfg(
      init_noise_std=0.5,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(256, 128, 64),
      critic_hidden_dims=(256, 128, 64),
      activation="elu",
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )
  experiment_name: str = "k1_pose_transition"
  save_interval: int = 50
  num_steps_per_env: int = 24
  max_iterations: int = 20_000
