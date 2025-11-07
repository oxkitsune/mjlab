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
    self.metrics["switches"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    # Normalize switch counts by resampling interval to keep the metric bounded.
    max_time = self.cfg.resampling_time_range[1]
    self.metrics["switches"] += self.time_left / max_time

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    prev = self._command[env_ids, 0].clone()
    if self.cfg.toggle_on_resample:
      self._command[env_ids, 0] = 1.0 - prev
    else:
      self._command[env_ids, 0] = torch.randint(
        low=0, high=2, size=(len(env_ids),), dtype=torch.float32, device=self.device
      )
    switched = (self._command[env_ids, 0] != prev).float()
    self.metrics["switches"][env_ids] += switched

  def _update_command(self) -> None:
    # Command is piecewise constant between resamples.
    pass


@dataclass(kw_only=True)
class PoseTransitionCommandCfg(CommandTermCfg):
  class_type: type[CommandTerm] = field(default=PoseTransitionCommand, init=False)
  initial_state: float = 0.0
  toggle_on_resample: bool = True
