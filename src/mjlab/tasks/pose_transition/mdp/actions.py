from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


class PoseBlendAction(ActionTerm):
  """Maps a single scalar phase action into joint position targets."""

  cfg: PoseBlendActionCfg

  def __init__(self, cfg: PoseBlendActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    # Resolve actuators/joints controlled by this action.
    actuator_ids, self._actuator_names = self._asset.find_actuators(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    joint_ids, self._joint_names = self._asset.find_joints(
      self._actuator_names, preserve_order=cfg.preserve_order
    )

    self._actuator_ids = torch.tensor(actuator_ids, device=self.device, dtype=torch.long)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    if cfg.start_keyframe is None or cfg.end_keyframe is None:
      raise ValueError("PoseBlendAction requires both start and end keyframes to be set.")

    default_joint_pos = self._asset.data.default_joint_pos
    assert default_joint_pos is not None
    base = default_joint_pos[0, self._joint_ids].clone()

    start_targets = self._build_keyframe_tensor(
      keyframe=cfg.start_keyframe,
      base=base,
      joint_names=self._joint_names,
    )
    end_targets = self._build_keyframe_tensor(
      keyframe=cfg.end_keyframe,
      base=base,
      joint_names=self._joint_names,
    )
    self._start_targets = start_targets.unsqueeze(0)
    self._delta_targets = (end_targets - start_targets).unsqueeze(0)

    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    init_phase = torch.scalar_tensor(cfg.initial_phase, device=self.device).clamp(
      cfg.min_phase, cfg.max_phase
    )
    self._phase = init_phase.repeat(self.num_envs, 1)
    self._targets = torch.zeros(
      self.num_envs, len(self._actuator_ids), device=self.device
    )

    self._scale = cfg.scale
    self._offset = cfg.offset
    self._min_phase = cfg.min_phase
    self._max_phase = cfg.max_phase

    self._update_targets()

  # Properties.

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def phase(self) -> torch.Tensor:
    return self._phase

  # Methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape[1] != 1:
      raise ValueError(
        f"PoseBlendAction expects shape (N, 1), received {actions.shape}."
      )
    self._raw_actions[:] = actions
    phase = actions * self._scale + self._offset
    self._phase[:] = phase.clamp(self._min_phase, self._max_phase)
    self._update_targets()

  def apply_actions(self) -> None:
    self._asset.write_joint_position_target_to_sim(
      self._targets, self._actuator_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    init = torch.scalar_tensor(self.cfg.initial_phase, device=self.device).clamp(
      self._min_phase, self._max_phase
    )
    self._phase[env_ids] = init
    self._update_targets(env_ids)

  # Helpers.

  def _update_targets(self, env_ids: torch.Tensor | slice | None = None) -> None:
    phase = self._phase.clamp(self._min_phase, self._max_phase)
    blended = self._start_targets + phase * self._delta_targets
    if env_ids is None:
      self._targets[:] = blended
    else:
      self._targets[env_ids] = blended[env_ids]

  def _build_keyframe_tensor(
    self,
    keyframe: EntityCfg.InitialStateCfg,
    base: torch.Tensor,
    joint_names: Sequence[str],
  ) -> torch.Tensor:
    values = base.clone()
    for pattern, value in keyframe.joint_pos.items():
      regex = re.compile(pattern)
      matched = [idx for idx, name in enumerate(joint_names) if regex.fullmatch(name)]
      if not matched:
        continue
      values[matched] = value
    return values.to(self.device)


@dataclass(kw_only=True)
class PoseBlendActionCfg(ActionTermCfg):
  class_type: type[ActionTerm] = field(default=PoseBlendAction, init=False)
  asset_name: str = "robot"
  actuator_names: Sequence[str] = (".*",)
  start_keyframe: EntityCfg.InitialStateCfg | None = None
  end_keyframe: EntityCfg.InitialStateCfg | None = None
  scale: float = 0.5
  offset: float = 0.5
  min_phase: float = 0.0
  max_phase: float = 1.0
  initial_phase: float = 0.0
  preserve_order: bool = False


class PhaseVelocityAction(ActionTerm):
  """Maps a single scalar velocity action into phase, controlling transition rate."""

  cfg: PhaseVelocityActionCfg

  def __init__(self, cfg: PhaseVelocityActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    # Resolve actuators/joints controlled by this action.
    actuator_ids, self._actuator_names = self._asset.find_actuators(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    joint_ids, self._joint_names = self._asset.find_joints(
      self._actuator_names, preserve_order=cfg.preserve_order
    )

    self._actuator_ids = torch.tensor(actuator_ids, device=self.device, dtype=torch.long)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    if cfg.start_keyframe is None or cfg.end_keyframe is None:
      raise ValueError("PhaseVelocityAction requires both start and end keyframes to be set.")

    default_joint_pos = self._asset.data.default_joint_pos
    assert default_joint_pos is not None
    base = default_joint_pos[0, self._joint_ids].clone()

    start_targets = self._build_keyframe_tensor(
      keyframe=cfg.start_keyframe,
      base=base,
      joint_names=self._joint_names,
    )
    end_targets = self._build_keyframe_tensor(
      keyframe=cfg.end_keyframe,
      base=base,
      joint_names=self._joint_names,
    )
    self._start_targets = start_targets.unsqueeze(0)
    self._delta_targets = (end_targets - start_targets).unsqueeze(0)

    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    init_phase = torch.scalar_tensor(cfg.initial_phase, device=self.device).clamp(
      cfg.min_phase, cfg.max_phase
    )
    self._phase = init_phase.repeat(self.num_envs, 1)
    self._prev_phase = self._phase.clone()
    self._phase_velocity = torch.zeros(self.num_envs, 1, device=self.device)
    self._targets = torch.zeros(
      self.num_envs, len(self._actuator_ids), device=self.device
    )

    self._velocity_scale = cfg.velocity_scale
    self._min_phase = cfg.min_phase
    self._max_phase = cfg.max_phase
    self._damping_margin = cfg.damping_margin
    self._damping_strength = cfg.damping_strength

    self._update_targets()

  # Properties.

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def phase(self) -> torch.Tensor:
    return self._phase

  @property
  def phase_velocity(self) -> torch.Tensor:
    return self._phase_velocity

  @property
  def phase_acceleration(self) -> torch.Tensor:
    """Compute phase acceleration from change in phase velocity."""
    dt = self._env.step_dt
    return (self._phase - self._prev_phase - self._phase_velocity * dt) / (dt * dt + 1e-8)

  # Methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape[1] != 1:
      raise ValueError(
        f"PhaseVelocityAction expects shape (N, 1), received {actions.shape}."
      )
    self._raw_actions[:] = actions

    # Action directly controls phase velocity
    self._phase_velocity[:] = actions * self._velocity_scale

    # Integrate velocity to get phase
    dt = self._env.step_dt
    self._prev_phase[:] = self._phase
    new_phase = self._phase + self._phase_velocity * dt

    # Apply damping near boundaries to prevent overshooting
    # Damping reduces velocity when approaching limits
    lower_dist = new_phase - self._min_phase
    upper_dist = self._max_phase - new_phase

    damping = torch.ones_like(new_phase)
    if self._damping_margin > 0.0:
      lower_damping = torch.sigmoid((lower_dist - self._damping_margin) * self._damping_strength)
      upper_damping = torch.sigmoid((upper_dist - self._damping_margin) * self._damping_strength)
      damping = torch.minimum(lower_damping, upper_damping)

    self._phase[:] = (self._prev_phase + self._phase_velocity * dt * damping).clamp(
      self._min_phase, self._max_phase
    )

    self._update_targets()

  def apply_actions(self) -> None:
    self._asset.write_joint_position_target_to_sim(
      self._targets, self._actuator_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    init = torch.scalar_tensor(self.cfg.initial_phase, device=self.device).clamp(
      self._min_phase, self._max_phase
    )
    self._phase[env_ids] = init
    self._prev_phase[env_ids] = init
    self._phase_velocity[env_ids] = 0.0
    self._update_targets(env_ids)

  # Helpers.

  def _update_targets(self, env_ids: torch.Tensor | slice | None = None) -> None:
    phase = self._phase.clamp(self._min_phase, self._max_phase)
    blended = self._start_targets + phase * self._delta_targets
    if env_ids is None:
      self._targets[:] = blended
    else:
      self._targets[env_ids] = blended[env_ids]

  def _build_keyframe_tensor(
    self,
    keyframe: EntityCfg.InitialStateCfg,
    base: torch.Tensor,
    joint_names: Sequence[str],
  ) -> torch.Tensor:
    values = base.clone()
    for pattern, value in keyframe.joint_pos.items():
      regex = re.compile(pattern)
      matched = [idx for idx, name in enumerate(joint_names) if regex.fullmatch(name)]
      if not matched:
        continue
      values[matched] = value
    return values.to(self.device)


@dataclass(kw_only=True)
class PhaseVelocityActionCfg(ActionTermCfg):
  class_type: type[ActionTerm] = field(default=PhaseVelocityAction, init=False)
  asset_name: str = "robot"
  actuator_names: Sequence[str] = (".*",)
  start_keyframe: EntityCfg.InitialStateCfg | None = None
  end_keyframe: EntityCfg.InitialStateCfg | None = None
  velocity_scale: float = 1.0
  """Scale factor for action to velocity conversion (phase_units/sec)."""
  min_phase: float = 0.0
  max_phase: float = 1.0
  initial_phase: float = 0.0
  damping_margin: float = 0.1
  """Distance from boundaries where damping starts."""
  damping_strength: float = 10.0
  """Strength of boundary damping (higher = more aggressive)."""
  preserve_order: bool = False


class AutoPhaseTransitionAction(ActionTerm):
  """Automatic smooth phase transitions - policy optionally outputs residuals.

  Phase automatically tracks the command using a smooth low-pass filter.
  Policy has no direct control over phase. Policy can output:
  - Nothing (action_dim=0): Pure trajectory tracking
  - Joint residuals: Fine-tuning on top of keyframe blend
  """

  cfg: AutoPhaseTransitionActionCfg

  def __init__(self, cfg: AutoPhaseTransitionActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    # Resolve actuators/joints controlled by this action.
    actuator_ids, self._actuator_names = self._asset.find_actuators(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    joint_ids, self._joint_names = self._asset.find_joints(
      self._actuator_names, preserve_order=cfg.preserve_order
    )

    self._actuator_ids = torch.tensor(actuator_ids, device=self.device, dtype=torch.long)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    if cfg.start_keyframe is None or cfg.end_keyframe is None:
      raise ValueError("AutoPhaseTransitionAction requires both start and end keyframes.")

    default_joint_pos = self._asset.data.default_joint_pos
    assert default_joint_pos is not None
    base = default_joint_pos[0, self._joint_ids].clone()

    start_targets = self._build_keyframe_tensor(
      keyframe=cfg.start_keyframe,
      base=base,
      joint_names=self._joint_names,
    )
    end_targets = self._build_keyframe_tensor(
      keyframe=cfg.end_keyframe,
      base=base,
      joint_names=self._joint_names,
    )
    self._start_targets = start_targets.unsqueeze(0)
    self._delta_targets = (end_targets - start_targets).unsqueeze(0)

    # Phase state (automatically controlled)
    init_phase = torch.scalar_tensor(cfg.initial_phase, device=self.device).clamp(0.0, 1.0)
    self._phase = init_phase.repeat(self.num_envs, 1)
    self._phase_target = self._phase.clone()

    # For smooth transitions
    self._phase_smoothing = cfg.phase_smoothing

    # Action space (residuals or none)
    if cfg.use_residuals:
      self._raw_actions = torch.zeros(self.num_envs, len(self._actuator_ids), device=self.device)
    else:
      self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    self._residual_scale = cfg.residual_scale
    self._use_residuals = cfg.use_residuals
    self._targets = torch.zeros(self.num_envs, len(self._actuator_ids), device=self.device)

    self._update_targets()

  # Properties.

  @property
  def action_dim(self) -> int:
    return len(self._actuator_ids) if self._use_residuals else 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def phase(self) -> torch.Tensor:
    return self._phase

  @property
  def phase_target(self) -> torch.Tensor:
    """The target phase from the command."""
    return self._phase_target

  # Methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    if self._use_residuals:
      if actions.shape[1] != len(self._actuator_ids):
        raise ValueError(
          f"AutoPhaseTransitionAction expects shape (N, {len(self._actuator_ids)}), "
          f"received {actions.shape}."
        )
      self._raw_actions[:] = actions
    else:
      if actions.shape[1] != 0:
        raise ValueError(
          f"AutoPhaseTransitionAction expects no actions (N, 0), received {actions.shape}."
        )

    # Update phase based on command (happens every step)
    self._update_phase_from_command()
    self._update_targets()

  def apply_actions(self) -> None:
    self._asset.write_joint_position_target_to_sim(self._targets, self._actuator_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    self._raw_actions[env_ids] = 0.0

    # Reset phase to match command if we have access to it
    # Otherwise initialize to configured initial phase
    init = torch.scalar_tensor(self.cfg.initial_phase, device=self.device).clamp(0.0, 1.0)
    self._phase[env_ids] = init
    self._phase_target[env_ids] = init

    self._update_targets(env_ids)

  # Helpers.

  def _update_phase_from_command(self) -> None:
    """Update phase to smoothly track command using low-pass filter."""
    # Get command from environment
    command_manager = self._env.command_manager
    command = command_manager.get_command("pose")  # Assumes command is named "pose"

    if command is None:
      return

    # Command is the target phase (0.0 or 1.0)
    self._phase_target[:] = torch.clamp(command[:, 0:1], 0.0, 1.0)

    # Smooth phase towards target using exponential smoothing
    # Higher smoothing = slower transitions (more filtering)
    dt = self._env.step_dt
    alpha = 1.0 - torch.exp(torch.tensor(-dt / self._phase_smoothing, device=self.device))
    self._phase[:] = self._phase + alpha * (self._phase_target - self._phase)
    self._phase[:] = torch.clamp(self._phase, 0.0, 1.0)

  def _update_targets(self, env_ids: torch.Tensor | slice | None = None) -> None:
    phase = self._phase.clamp(0.0, 1.0)

    # Blend keyframes based on current phase
    blended = self._start_targets + phase * self._delta_targets

    # Add residuals if enabled
    if self._use_residuals:
      blended = blended + self._raw_actions * self._residual_scale

    if env_ids is None:
      self._targets[:] = blended
    else:
      self._targets[env_ids] = blended[env_ids]

  def _build_keyframe_tensor(
    self,
    keyframe: EntityCfg.InitialStateCfg,
    base: torch.Tensor,
    joint_names: Sequence[str],
  ) -> torch.Tensor:
    values = base.clone()
    for pattern, value in keyframe.joint_pos.items():
      regex = re.compile(pattern)
      matched = [idx for idx, name in enumerate(joint_names) if regex.fullmatch(name)]
      if not matched:
        continue
      values[matched] = value
    return values.to(self.device)


@dataclass(kw_only=True)
class AutoPhaseTransitionActionCfg(ActionTermCfg):
  class_type: type[ActionTerm] = field(default=AutoPhaseTransitionAction, init=False)
  asset_name: str = "robot"
  actuator_names: Sequence[str] = (".*",)
  start_keyframe: EntityCfg.InitialStateCfg | None = None
  end_keyframe: EntityCfg.InitialStateCfg | None = None
  initial_phase: float = 0.0
  phase_smoothing: float = 0.3
  """Time constant for phase smoothing (seconds). Higher = slower transitions."""
  use_residuals: bool = False
  """If True, policy outputs joint residuals. If False, action_dim=0 (pure tracking)."""
  residual_scale: float = 0.1
  """Scale factor for residual actions (radians)."""
  preserve_order: bool = False
