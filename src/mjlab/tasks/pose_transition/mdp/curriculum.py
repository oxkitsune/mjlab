from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class CurriculumStage:
  """Configuration for a single curriculum stage."""

  name: str
  """Name/description of this stage."""
  resampling_time_range: tuple[float, float]
  """Min/max time between command changes (seconds)."""
  enable_command_changes: bool = True
  """Whether commands change at all (False = static pose holding)."""
  success_threshold: float = 0.8
  """Fraction of episodes that must succeed to advance to next stage."""
  min_episodes: int = 100
  """Minimum episodes in this stage before allowing progression."""


class PoseTransitionCurriculum:
  """Curriculum manager for pose transition training.

  Progressively increases difficulty from static poses to fast transitions.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    stages: list[CurriculumStage] | None = None,
    command_name: str = "pose_transition",
  ):
    self.env = env
    self.command_name = command_name
    self.device = env.device

    # Default curriculum if none provided
    if stages is None:
      stages = self._default_curriculum()

    self.stages = stages
    self.current_stage_idx = 0

    # Tracking metrics per environment
    self.episode_count = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self.episode_success = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)

    # Global stage tracking
    self.stage_episode_count = 0
    self.stage_success_count = 0

    print(f"[Curriculum] Initialized with {len(self.stages)} stages")
    self._print_current_stage()

  @staticmethod
  def _default_curriculum() -> list[CurriculumStage]:
    """Default 4-stage curriculum from static to fast transitions."""
    return [
      CurriculumStage(
        name="Stage 1: Static Pose Holding",
        resampling_time_range=(100.0, 100.0),  # Effectively infinite
        enable_command_changes=False,
        success_threshold=0.75,
        min_episodes=200,
      ),
      CurriculumStage(
        name="Stage 2: Slow Transitions",
        resampling_time_range=(3.0, 5.0),
        enable_command_changes=True,
        success_threshold=0.80,
        min_episodes=400,
      ),
      CurriculumStage(
        name="Stage 3: Medium Transitions",
        resampling_time_range=(1.5, 3.0),
        enable_command_changes=True,
        success_threshold=0.80,
        min_episodes=400,
      ),
      CurriculumStage(
        name="Stage 4: Target Speed",
        resampling_time_range=(0.6, 1.1),
        enable_command_changes=True,
        success_threshold=0.85,
        min_episodes=1000,
      ),
    ]

  @property
  def current_stage(self) -> CurriculumStage:
    """Get current curriculum stage."""
    return self.stages[self.current_stage_idx]

  @property
  def is_final_stage(self) -> bool:
    """Check if we're on the final stage."""
    return self.current_stage_idx >= len(self.stages) - 1

  def get_resampling_time_range(self) -> tuple[float, float]:
    """Get current resampling time range for command manager."""
    return self.current_stage.resampling_time_range

  def should_change_commands(self) -> bool:
    """Check if commands should change in current stage."""
    return self.current_stage.enable_command_changes

  def record_episode_result(self, env_ids: torch.Tensor, success: torch.Tensor) -> None:
    """Record success/failure for completed episodes.

    Args:
      env_ids: Tensor of environment IDs that completed episodes.
      success: Boolean tensor indicating success for each env_id.
    """
    if len(env_ids) == 0:
      return

    self.episode_count[env_ids] += 1
    self.episode_success[env_ids] = success

    # Update stage-level statistics
    self.stage_episode_count += len(env_ids)
    self.stage_success_count += success.sum().item()

    # Check if we should advance to next stage
    if self._should_advance_stage():
      self._advance_stage()

  def _should_advance_stage(self) -> bool:
    """Check if conditions are met to advance to next stage."""
    if self.is_final_stage:
      return False

    stage = self.current_stage

    # Need minimum number of episodes
    if self.stage_episode_count < stage.min_episodes:
      return False

    # Check success rate
    if self.stage_episode_count == 0:
      return False

    success_rate = self.stage_success_count / self.stage_episode_count
    return success_rate >= stage.success_threshold

  def _advance_stage(self) -> None:
    """Advance to the next curriculum stage."""
    if self.is_final_stage:
      return

    old_stage_idx = self.current_stage_idx
    self.current_stage_idx += 1

    # Reset stage tracking
    self.stage_episode_count = 0
    self.stage_success_count = 0

    print(
      f"\n[Curriculum] ADVANCING: {self.stages[old_stage_idx].name} -> "
      f"{self.current_stage.name}"
    )
    self._print_current_stage()

    # Update command manager with new resampling times
    self._update_command_manager()

  def _update_command_manager(self) -> None:
    """Update command manager with current stage parameters."""
    command = self.env.command_manager.get_command(self.command_name)
    if command is None:
      return

    # Update resampling time range
    time_range = self.get_resampling_time_range()
    command.cfg.resampling_time_range = time_range

    print(f"[Curriculum] Updated command resampling range: {time_range}")

  def _print_current_stage(self) -> None:
    """Print information about current stage."""
    stage = self.current_stage
    print(f"[Curriculum] Current: {stage.name}")
    print(f"  - Resampling time: {stage.resampling_time_range}")
    print(f"  - Command changes: {stage.enable_command_changes}")
    print(f"  - Success threshold: {stage.success_threshold:.1%}")
    print(f"  - Min episodes: {stage.min_episodes}")

  def get_stage_progress(self) -> dict[str, float]:
    """Get current stage progress metrics for logging."""
    stage = self.current_stage
    success_rate = (
      self.stage_success_count / self.stage_episode_count
      if self.stage_episode_count > 0
      else 0.0
    )

    return {
      "curriculum/stage": float(self.current_stage_idx),
      "curriculum/stage_episodes": float(self.stage_episode_count),
      "curriculum/stage_success_rate": success_rate,
      "curriculum/episodes_to_advance": max(
        0, stage.min_episodes - self.stage_episode_count
      ),
    }
