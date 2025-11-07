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
