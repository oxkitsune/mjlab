from __future__ import annotations

import re
from typing import Sequence, Tuple

import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def build_keyframe_tensor(
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


def resolve_keyframe_poses(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
  start_keyframe: EntityCfg.InitialStateCfg,
  end_keyframe: EntityCfg.InitialStateCfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]]:
  assert asset_cfg.joint_names is not None, (
    "Keyframe pose resolution requires joint_names to be specified in asset_cfg."
  )
  joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  device = default_joint_pos.device

  joint_ids_tensor = torch.tensor(joint_ids, device=device, dtype=torch.long)
  base = default_joint_pos[0, joint_ids_tensor]

  start_pose = build_keyframe_tensor(start_keyframe, base, joint_names, device)
  end_pose = build_keyframe_tensor(end_keyframe, base, joint_names, device)

  return start_pose, end_pose, base, joint_ids_tensor, joint_names
