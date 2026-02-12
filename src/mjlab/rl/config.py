"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Config for a single neural network model (Actor or Critic)."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the network."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize the observations. Default is False."""
  init_noise_std: float = 1.0
  """The initial noise standard deviation."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """The type of noise standard deviation."""
  stochastic: bool = False
  """Whether the model output is stochastic."""
  cnn_cfg: dict[str, Any] | None = None
  """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
  class_name: str = "MLPModel"
  """Model class name resolved by RSL-RL (MLPModel or CNNModel)."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  share_cnn_encoders: bool = False
  """Share CNN encoders between actor and critic."""
  class_name: str = "PPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under
  ``logs/rsl_rl/{experiment_name}/``."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  actor: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(stochastic=True))
  """The actor configuration."""
  critic: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(stochastic=False))
  """The critic configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""


@dataclass
class RslRlFastSacAlgorithmCfg:
  """Configuration for the FastSAC algorithm."""

  class_name: str = "mjlab.rl.fast_sac.fast_sac:FastSAC"
  """Fully qualified class name resolved by rsl-rl's resolve_callable."""

  # Learning rates
  critic_lr: float = 3e-4
  """Critic learning rate."""
  actor_lr: float = 3e-4
  """Actor learning rate."""
  alpha_lr: float = 3e-4
  """Entropy temperature learning rate."""

  # SAC hyper-parameters
  gamma: float = 0.97
  """Discount factor."""
  tau: float = 0.125
  """Soft target update coefficient."""
  batch_size: int = 8192
  """Global batch size (split across envs)."""
  learning_starts: int = 10
  """Number of env steps before training begins."""
  policy_frequency: int = 4
  """Actor update frequency relative to critic updates."""
  num_updates: int = 8
  """Number of gradient updates per env step (UTD ratio)."""
  target_entropy_ratio: float = 0.0
  """Target entropy as a ratio of -n_act."""
  alpha_init: float = 0.001
  """Initial entropy temperature."""
  use_autotune: bool = True
  """Whether to auto-tune the entropy temperature."""

  # Network architecture
  actor_hidden_dim: int = 512
  """Actor hidden layer base width (halving: 512->256->128)."""
  critic_hidden_dim: int = 768
  """Critic hidden layer base width (halving: 768->384->192)."""
  num_atoms: int = 101
  """Number of atoms for the distributional critic (C51)."""
  v_min: float = -20.0
  """Minimum support value for C51."""
  v_max: float = 20.0
  """Maximum support value for C51."""
  num_q_networks: int = 2
  """Number of Q-networks in the critic ensemble."""
  use_layer_norm: bool = True
  """Whether to use LayerNorm in actor and critic."""
  use_tanh: bool = True
  """Whether to use tanh squashing on actor output."""
  log_std_min: float = -5.0
  """Minimum log standard deviation for the actor."""
  log_std_max: float = 0.0
  """Maximum log standard deviation for the actor."""

  # Replay buffer
  buffer_size: int = 1024
  """Per-env replay buffer capacity."""
  num_steps: int = 1
  """Number of n-step returns."""

  # Observation normalization
  obs_normalization: bool = True
  """Whether to use empirical observation normalization."""

  # Optimization
  max_grad_norm: float = 0.0
  """Maximum gradient norm (0 = no clipping)."""
  weight_decay: float = 0.001
  """AdamW weight decay."""

  # Performance
  compile: bool = True
  """Whether to use torch.compile for update functions."""
  amp: bool = True
  """Whether to use automatic mixed precision."""
  amp_dtype: str = "bf16"
  """AMP dtype: 'bf16' or 'fp16'."""


@dataclass
class RslRlFastSacRunnerCfg(RslRlBaseRunnerCfg):
  """Runner configuration for FastSAC.

  FastSAC is off-policy, so we use num_steps_per_env=1 and let the
  algorithm handle replay internally.
  """

  class_name: str = "FastSACRunner"
  """
  Runner class name. 
  """
  num_steps_per_env: int = 1
  """Must be 1 for off-policy SAC."""
  algorithm: RslRlFastSacAlgorithmCfg = field(default_factory=RslRlFastSacAlgorithmCfg)
  """The FastSAC algorithm configuration."""
