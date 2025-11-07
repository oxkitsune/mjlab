from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class PoseTransitionCommand(CommandTerm):
  """Binary command that toggles between two keyframe targets."""

  cfg: PoseTransitionCommandCfg

  def __init__(self, cfg: PoseTransitionCommandCfg, env):  # type: ignore[override]
    super().__init__(cfg=cfg, env=env)
    init = torch.full((self.num_envs, 1), cfg.initial_state, device=self.device)
    self._command = init.clamp(0.0, 1.0)
    self._command_target = self._command.clone()
    self.metrics["switches"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    # Track how smooth the command transitions have been over the episode.
    max_time = self.cfg.resampling_time_range[1]
    normalized = torch.clamp_min(self.time_left / max_time, 0.0)
    self.metrics["switches"] += normalized

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    counters = self.command_counter[env_ids]
    first_mask = counters == 0

    # Keep the initial command for freshly reset environments.
    if torch.any(first_mask):
      keep_ids = env_ids[first_mask]
      self._command_target[keep_ids, 0] = self._command[keep_ids, 0]

    active_ids = env_ids[~first_mask]
    if len(active_ids) == 0:
      return

    prev_target = self._command_target[active_ids, 0].clone()

    if self.cfg.toggle_on_resample:
      new_target = 1.0 - prev_target
    else:
      new_target = torch.randint(
        low=0,
        high=2,
        size=(len(active_ids),),
        dtype=torch.float32,
        device=self.device,
      )
      new_target = new_target.clamp(0.0, 1.0)

    self._command_target[active_ids, 0] = new_target
    self.metrics["switches"][active_ids] += torch.abs(new_target - prev_target)

  def _update_command(self) -> None:
    smoothing = max(0.0, min(1.0, float(self.cfg.resample_smoothing)))
    if smoothing <= 0.0:
      self._command.copy_(self._command_target)
    else:
      self._command += smoothing * (self._command_target - self._command)
      self._command.clamp_(0.0, 1.0)

  def set_state(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
    """Directly set the command and target for specific environments."""
    if values.ndim == 1:
      values = values.unsqueeze(-1)
    values = values.clamp(0.0, 1.0)
    self._command[env_ids, 0] = values[:, 0]
    self._command_target[env_ids, 0] = values[:, 0]


@dataclass(kw_only=True)
class PoseTransitionCommandCfg(CommandTermCfg):
  class_type: type[CommandTerm] = field(default=PoseTransitionCommand, init=False)
  initial_state: float = 0.0
  toggle_on_resample: bool = True
  resample_smoothing: float = 0.05
