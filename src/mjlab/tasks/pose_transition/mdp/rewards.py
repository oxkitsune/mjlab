from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pose_transition.mdp.actions import PoseBlendAction

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class TrackPoseKeyframeReward:
  """Dense reward that encourages matching a commanded keyframe pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.asset_cfg: SceneEntityCfg = params["asset_cfg"]
    self.command_name: str = params["command_name"]
    self.std: float = params.get("std", 0.25)

    start_keyframe: EntityCfg.InitialStateCfg | None = params.get("start_keyframe")
    end_keyframe: EntityCfg.InitialStateCfg | None = params.get("end_keyframe")
    if start_keyframe is None or end_keyframe is None:
      raise ValueError("TrackPoseKeyframeReward requires start/end keyframes.")

    asset: Entity = env.scene[self.asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(
      self.asset_cfg.joint_names, preserve_order=True
    )
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    base = default_joint_pos[0, self._joint_ids]

    self._start = self._build_keyframe_tensor(
      start_keyframe, base, joint_names, env.device
    ).unsqueeze(0)
    self._end = self._build_keyframe_tensor(
      end_keyframe, base, joint_names, env.device
    ).unsqueeze(0)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
  ) -> torch.Tensor:
    del asset_cfg, command_name, std  # Configured in __init__.
    asset: Entity = env.scene[self.asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, self._joint_ids]
    command = env.command_manager.get_command(self.command_name)
    assert command is not None
    weights = command.clamp(0.0, 1.0)
    target_pose = self._start + weights * (self._end - self._start)
    error = torch.mean(torch.square(joint_pos - target_pose), dim=1)
    return torch.exp(-error / (self.std**2))

  def _build_keyframe_tensor(
    self,
    keyframe: EntityCfg.InitialStateCfg,
    base: torch.Tensor,
    joint_names: Sequence[str],
    device: torch.device,
  ) -> torch.Tensor:
    values = base.clone()
    for pattern, value in keyframe.joint_pos.items():
      regex = re.compile(pattern)
      matched = [idx for idx, name in enumerate(joint_names) if regex.fullmatch(name)]
      if not matched:
        continue
      values[matched] = value
    return values.to(device)


def phase_command_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward alignment between commanded state and scalar phase action."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  phase_term = env.action_manager.get_term("phase")
  assert isinstance(phase_term, PoseBlendAction)
  phase = phase_term.phase.squeeze(-1)
  target = command[:, 0]
  error = torch.square(phase - target)
  return torch.exp(-error / (std**2))
