from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp.rewards import _DEFAULT_ASSET_CFG
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.tasks.pose_transition.mdp.actions import PoseBlendAction, PhaseVelocityAction
from mjlab.tasks.pose_transition.mdp.interpolation import resolve_keyframe_poses
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class TrackPoseKeyframeReward:
  """Dense reward that encourages matching a commanded keyframe pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.asset_cfg: SceneEntityCfg = params["asset_cfg"]
    self.command_name: str = params["command_name"]
    self.std: float = params.get("std", 0.25)
    self.root_height_weight: float = params.get("root_height_weight", 0.0)
    self.root_height_std: float = params.get("root_height_std", 0.05)
    self.root_orientation_weight: float = params.get("root_orientation_weight", 0.0)
    self.root_orientation_std: float = params.get("root_orientation_std", 0.35)
    completion_margin = params.get("completion_margin")
    if completion_margin is None:
      self._completion_margin = None
    else:
      completion_margin = float(completion_margin)
      if completion_margin <= 0.0:
        self._completion_margin = None
      elif completion_margin > 0.5:
        raise ValueError("completion_margin must be in the interval (0, 0.5].")
      else:
        self._completion_margin = completion_margin

    start_keyframe: EntityCfg.InitialStateCfg | None = params.get("start_keyframe")
    end_keyframe: EntityCfg.InitialStateCfg | None = params.get("end_keyframe")
    if start_keyframe is None or end_keyframe is None:
      raise ValueError("TrackPoseKeyframeReward requires start/end keyframes.")

    asset: Entity = env.scene[self.asset_cfg.name]
    (
      start_pose,
      end_pose,
      base,
      joint_ids_tensor,
      _joint_names,
    ) = resolve_keyframe_poses(asset, self.asset_cfg, start_keyframe, end_keyframe)
    del _joint_names
    self._joint_ids = joint_ids_tensor
    self._start = start_pose.unsqueeze(0)
    self._end = end_pose.unsqueeze(0)

    dtype = base.dtype
    device = base.device
    self._start_height = torch.tensor(start_keyframe.pos[2], device=device, dtype=dtype)
    self._end_height = torch.tensor(end_keyframe.pos[2], device=device, dtype=dtype)
    self._height_delta = self._end_height - self._start_height

    self._start_quat = F.normalize(
      torch.tensor(start_keyframe.rot, device=device, dtype=dtype), dim=0
    )
    self._end_quat = F.normalize(
      torch.tensor(end_keyframe.rot, device=device, dtype=dtype), dim=0
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    start_keyframe: EntityCfg.InitialStateCfg,
    end_keyframe: EntityCfg.InitialStateCfg,
    **unused_params: Any,
  ) -> torch.Tensor:
    del (
      asset_cfg,
      command_name,
      std,
      start_keyframe,
      end_keyframe,
    )  # Configured in __init__.
    del unused_params

    asset: Entity = env.scene[self.asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, self._joint_ids]
    command = env.command_manager.get_command(self.command_name)
    assert command is not None

    weights = command.clamp(0.0, 1.0)
    weight_scalar = weights.squeeze(-1)

    target_pose = self._start + weights * (self._end - self._start)
    joint_error = torch.mean(torch.square(joint_pos - target_pose), dim=1)
    joint_reward = torch.exp(-joint_error / (self.std**2))

    reward = joint_reward
    weight_sum = torch.ones_like(joint_reward)

    if self.root_height_weight > 0.0:
      root_height = asset.data.root_link_pose_w[:, 2]
      target_height = self._start_height + weight_scalar * self._height_delta
      height_error = root_height - target_height
      height_reward = torch.exp(-torch.square(height_error) / (self.root_height_std**2))
      reward = reward + self.root_height_weight * height_reward
      weight_sum = weight_sum + self.root_height_weight

    if self.root_orientation_weight > 0.0:
      root_quat = asset.data.root_link_pose_w[:, 3:7]
      target_quat = self._blend_root_quat(weight_scalar)
      orientation_error = torch.square(quat_error_magnitude(root_quat, target_quat))
      orientation_reward = torch.exp(
        -orientation_error / (self.root_orientation_std**2)
      )
      reward = reward + self.root_orientation_weight * orientation_reward
      weight_sum = weight_sum + self.root_orientation_weight

    reward = reward / weight_sum

    if self._completion_margin is not None:
      margin = self._completion_margin
      near_start = weight_scalar <= margin
      near_end = (1.0 - weight_scalar) <= margin
      evaluate_mask = near_start | near_end
      reward = torch.where(evaluate_mask, reward, torch.zeros_like(reward))

    return reward

  def _blend_root_quat(self, weights: torch.Tensor) -> torch.Tensor:
    """Spherically interpolate between start/end root orientations."""
    t = weights.unsqueeze(-1)
    start = self._start_quat.unsqueeze(0).expand(t.shape[0], -1)
    end = self._end_quat.unsqueeze(0).expand(t.shape[0], -1)

    dot = torch.sum(start * end, dim=-1, keepdim=True)
    end_adjusted = torch.where(dot < 0.0, -end, end)
    dot = torch.sum(start * end_adjusted, dim=-1, keepdim=True)

    linear = start + t * (end_adjusted - start)

    dot_clamped = torch.clamp(dot, -1.0, 1.0)
    theta_0 = torch.acos(dot_clamped)
    sin_theta_0 = torch.sin(theta_0)
    sin_theta_0 = torch.where(
      sin_theta_0.abs() < 1e-6,
      torch.full_like(sin_theta_0, 1.0),
      sin_theta_0,
    )

    coeff_start = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    coeff_end = torch.sin(t * theta_0) / sin_theta_0
    slerp = coeff_start * start + coeff_end * end_adjusted

    use_linear = dot.abs() > 0.9995
    result = torch.where(use_linear, linear, slerp)
    return F.normalize(result, dim=-1)


def phase_command_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  completion_margin: float | None = None,
) -> torch.Tensor:
  """Reward alignment between commanded state and scalar phase action."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = torch.clamp(command[:, 0], 0.0, 1.0)

  phase_term = env.action_manager.get_term("phase")
  assert isinstance(phase_term, PoseBlendAction)
  phase = phase_term.phase.squeeze(-1)
  error = torch.square(phase - target)
  reward = torch.exp(-error / (std**2))

  if completion_margin is not None:
    margin = float(completion_margin)
    if margin <= 0.0 or margin > 0.5:
      raise ValueError("completion_margin must be in the interval (0, 0.5].")
    near_start = target <= margin
    near_end = (1.0 - target) <= margin
    mask = near_start | near_end
    reward = torch.where(mask, reward, torch.zeros_like(reward))

  return reward


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def phase_acceleration_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty for high phase acceleration (encourages smooth transitions).

  Only works with PhaseVelocityAction which tracks phase velocity.
  """
  phase_term = env.action_manager.get_term("phase")
  if not isinstance(phase_term, PhaseVelocityAction):
    return torch.zeros(env.num_envs, device=env.device)

  acceleration = phase_term.phase_acceleration.squeeze(-1)
  return torch.square(acceleration)


def phase_velocity_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
) -> torch.Tensor:
  """Reward for phase velocity aligned with direction to target.

  Encourages moving toward the commanded target (0 or 1).
  """
  phase_term = env.action_manager.get_term("phase")
  if not isinstance(phase_term, PhaseVelocityAction):
    return torch.zeros(env.num_envs, device=env.device)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = torch.clamp(command[:, 0], 0.0, 1.0)

  phase = phase_term.phase.squeeze(-1)
  phase_velocity = phase_term.phase_velocity.squeeze(-1)

  # Compute desired direction: positive if target > phase, negative otherwise
  error = target - phase
  desired_direction = torch.sign(error)

  # Reward when velocity aligns with desired direction
  alignment = phase_velocity * desired_direction
  # Use smooth reward function
  reward = torch.exp(-torch.square(alignment - torch.abs(error)) / (std**2))

  return reward


def dense_pose_tracking(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  std: float,
  start_keyframe: EntityCfg.InitialStateCfg,
  end_keyframe: EntityCfg.InitialStateCfg,
  root_height_weight: float = 0.0,
  root_height_std: float = 0.05,
  root_orientation_weight: float = 0.0,
  root_orientation_std: float = 0.35,
) -> torch.Tensor:
  """Dense pose tracking reward without completion margin.

  This is a wrapper that creates a TrackPoseKeyframeReward with completion_margin=None,
  providing reward signal throughout the entire transition.
  """
  # Create reward instance if not cached
  if not hasattr(dense_pose_tracking, "_reward_instance"):
    cfg = RewardTermCfg(
      func=dense_pose_tracking,
      weight=1.0,
      params={
        "asset_cfg": asset_cfg,
        "command_name": command_name,
        "std": std,
        "start_keyframe": start_keyframe,
        "end_keyframe": end_keyframe,
        "root_height_weight": root_height_weight,
        "root_height_std": root_height_std,
        "root_orientation_weight": root_orientation_weight,
        "root_orientation_std": root_orientation_std,
        "completion_margin": None,  # No margin = dense reward
      },
    )
    dense_pose_tracking._reward_instance = TrackPoseKeyframeReward(cfg, env)  # type: ignore[attr-defined]

  return dense_pose_tracking._reward_instance(  # type: ignore[attr-defined]
    env=env,
    asset_cfg=asset_cfg,
    command_name=command_name,
    std=std,
    start_keyframe=start_keyframe,
    end_keyframe=end_keyframe,
  )


def dense_phase_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.75,
) -> torch.Tensor:
  """Dense phase alignment reward without completion margin.

  Provides reward throughout transition, not just at endpoints.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = torch.clamp(command[:, 0], 0.0, 1.0)

  # Works with both action types
  phase_term = env.action_manager.get_term("phase")
  if isinstance(phase_term, PhaseVelocityAction):
    phase = phase_term.phase.squeeze(-1)
  elif isinstance(phase_term, PoseBlendAction):
    phase = phase_term.phase.squeeze(-1)
  else:
    return torch.zeros(env.num_envs, device=env.device)

  error = torch.square(phase - target)
  reward = torch.exp(-error / (std**2))

  return reward
