"""Example showing how to integrate PoseTransitionCurriculum with training.

This example demonstrates the integration pattern for the curriculum-based
velocity training approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.pose_transition.mdp import PoseTransitionCurriculum

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CurriculumWrapper:
  """Wrapper that integrates curriculum into training loop."""

  def __init__(self, env: ManagerBasedRlEnv, command_name: str = "pose"):
    """Initialize curriculum for pose transition training.

    Args:
      env: The pose transition environment.
      command_name: Name of the pose transition command (default: "pose").
    """
    self.env = env
    self.curriculum = PoseTransitionCurriculum(env, command_name=command_name)

  def compute_episode_success(self) -> torch.Tensor:
    """Compute success for each environment based on task criteria.

    Success is defined as:
    1. Episode completed without termination (no falls/collisions)
    2. Final pose close to commanded target (within threshold)

    Returns:
      Boolean tensor of shape (num_envs,) indicating success per environment.
    """
    # Get the pose command to know the target
    command = self.env.command_manager.get_command("pose")
    target_phase = command[:, 0] if command is not None else torch.zeros(
      self.env.num_envs, device=self.env.device
    )

    # Get current phase from action
    phase_term = self.env.action_manager.get_term("phase")
    current_phase = phase_term.phase.squeeze(-1) if phase_term else torch.zeros(
      self.env.num_envs, device=self.env.device
    )

    # Success if phase is close to target (within 0.2 margin)
    phase_error = torch.abs(current_phase - target_phase)
    phase_close = phase_error < 0.2

    # Success if didn't terminate early
    # (episode_length_buf tracks how many steps have elapsed)
    max_steps = int(self.env.cfg.episode_length_s / self.env.step_dt)
    completed_episode = self.env.episode_length_buf >= max_steps - 1

    # Overall success: close to target OR completed full episode
    success = phase_close | completed_episode

    return success

  def on_episode_reset(self, env_ids: torch.Tensor) -> None:
    """Called when episodes reset. Records results and updates curriculum.

    Args:
      env_ids: Tensor of environment IDs being reset.
    """
    if len(env_ids) == 0:
      return

    # Compute success for the environments that are resetting
    all_success = self.compute_episode_success()
    success = all_success[env_ids]

    # Update curriculum with results
    self.curriculum.record_episode_result(env_ids, success)

  def get_metrics(self) -> dict[str, float]:
    """Get curriculum metrics for logging.

    Returns:
      Dictionary of curriculum metrics.
    """
    return self.curriculum.get_stage_progress()


# ============================================================================
# Integration Example with Training Loop
# ============================================================================


def example_training_loop():
  """Example showing how to integrate curriculum in a training loop."""
  from mjlab.tasks.pose_transition.config.k1.env_cfg_velocity import (
    BoosterK1VelocityPoseTransitionEnvCfg,
  )

  # 1. Create environment
  env_cfg = BoosterK1VelocityPoseTransitionEnvCfg()
  env_cfg.scene.num_envs = 4096
  # env = create_env(env_cfg)  # Your env creation function

  # 2. Initialize curriculum wrapper
  # curriculum_wrapper = CurriculumWrapper(env)

  # 3. Training loop (pseudocode)
  # for iteration in range(max_iterations):
  #   # Collect rollout
  #   for step in range(num_steps):
  #     actions = policy.act(obs)
  #     obs, rewards, dones, infos = env.step(actions)
  #
  #     # Check for resets and update curriculum
  #     reset_env_ids = torch.where(dones)[0]
  #     if len(reset_env_ids) > 0:
  #       curriculum_wrapper.on_episode_reset(reset_env_ids)
  #
  #   # Update policy
  #   policy.update(...)
  #
  #   # Log curriculum metrics
  #   if iteration % log_interval == 0:
  #     metrics = curriculum_wrapper.get_metrics()
  #     logger.log(metrics)
  #     print(f"Stage {metrics['curriculum/stage']:.0f}, "
  #           f"Success Rate: {metrics['curriculum/stage_success_rate']:.2%}")

  print("Example training loop structure shown in comments above.")


# ============================================================================
# Alternative: Direct Integration (No Wrapper)
# ============================================================================


def example_direct_integration():
  """Alternative example using curriculum directly without wrapper."""
  from mjlab.tasks.pose_transition.config.k1.env_cfg_velocity import (
    BoosterK1VelocityPoseTransitionEnvCfg,
  )

  # Create environment
  env_cfg = BoosterK1VelocityPoseTransitionEnvCfg()
  # env = create_env(env_cfg)

  # Create curriculum directly
  # curriculum = PoseTransitionCurriculum(env, command_name="pose")

  # In training loop when episodes complete:
  # reset_ids = torch.where(dones)[0]
  # if len(reset_ids) > 0:
  #   # Define success criteria (example: completed without early termination)
  #   success = torch.ones(len(reset_ids), dtype=torch.bool, device=env.device)
  #   curriculum.record_episode_result(reset_ids, success)
  #
  #   # Get updated command resampling time
  #   time_range = curriculum.get_resampling_time_range()

  print("Direct integration example shown in comments above.")


if __name__ == "__main__":
  print("Curriculum Integration Examples")
  print("=" * 70)
  print()
  print("This file demonstrates two ways to integrate the curriculum:")
  print("1. Using CurriculumWrapper class (recommended)")
  print("2. Direct integration in training loop")
  print()
  print("Key steps:")
  print("  - Initialize curriculum with environment")
  print("  - Compute episode success based on task criteria")
  print("  - Call record_episode_result() when episodes reset")
  print("  - Log curriculum metrics for monitoring")
  print()
  example_training_loop()
  print()
  example_direct_integration()
