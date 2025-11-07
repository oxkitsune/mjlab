from __future__ import annotations

import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs.manager_based_env import ManagerBasedEnv
from mjlab.envs.mdp.events import _DEFAULT_ASSET_CFG
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pose_transition.mdp.commands import PoseTransitionCommand
from mjlab.tasks.pose_transition.mdp.interpolation import resolve_keyframe_poses


def _broadcast_keyframes(
  start_pose: torch.Tensor, end_pose: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
  start = start_pose.unsqueeze(0).expand(count, -1)
  end = end_pose.unsqueeze(0).expand(count, -1)
  return start, end


def _ensure_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  return env_ids


def _update_command(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  weights: torch.Tensor,
  command_name: str,
) -> None:
  command_manager = getattr(env, "command_manager", None)
  if command_manager is None:
    return
  term = command_manager.get_term(command_name)
  if not isinstance(term, PoseTransitionCommand):
    return
  term.set_state(env_ids, weights.squeeze(-1))


def reset_joints_to_transition_pose(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  start_keyframe: EntityCfg.InitialStateCfg,
  end_keyframe: EntityCfg.InitialStateCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str | None = None,
  sync_command: bool = False,
) -> None:
  env_ids = _ensure_env_ids(env, env_ids)

  asset: Entity = env.scene[asset_cfg.name]
  (
    start_pose,
    end_pose,
    _base,
    joint_ids_tensor,
    _,
  ) = resolve_keyframe_poses(asset, asset_cfg, start_keyframe, end_keyframe)

  count = len(env_ids)
  start_pose_b, end_pose_b = _broadcast_keyframes(start_pose, end_pose, count)
  weights = torch.rand((count, 1), device=env.device)
  joint_pos = start_pose_b + weights * (end_pose_b - start_pose_b)
  joint_vel = torch.zeros_like(joint_pos)

  asset.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    env_ids=env_ids,
    joint_ids=joint_ids_tensor,
  )

  if sync_command and command_name is not None:
    _update_command(env, env_ids, weights, command_name)
