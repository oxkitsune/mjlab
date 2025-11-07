"""Observation functions specific to pose transition tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.pose_transition.mdp.actions import PhaseVelocityAction, PoseBlendAction

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def phase_value(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Returns the current phase value from the phase action term.

  Works with both PoseBlendAction and PhaseVelocityAction.
  """
  phase_term = env.action_manager.get_term("phase")
  if isinstance(phase_term, (PhaseVelocityAction, PoseBlendAction)):
    return phase_term.phase
  return torch.zeros(env.num_envs, 1, device=env.device)


def phase_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Returns the current phase velocity from PhaseVelocityAction.

  Only works with PhaseVelocityAction. Returns zeros for other action types.
  """
  phase_term = env.action_manager.get_term("phase")
  if isinstance(phase_term, PhaseVelocityAction):
    return phase_term.phase_velocity
  return torch.zeros(env.num_envs, 1, device=env.device)
