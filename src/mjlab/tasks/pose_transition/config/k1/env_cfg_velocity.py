"""K1 Pose Transition config using PhaseVelocityAction with curriculum learning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mjlab.asset_zoo.robots.booster_k1.k1_constants import (
  CRAWL_KEYFRAME,
  HOME_KEYFRAME,
  K1_ROBOT_CFG,
)
from mjlab.envs import mdp
from mjlab.managers.manager_term_config import (
  ObservationTermCfg as ObsTerm,
  RewardTermCfg as RewardTerm,
  term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pose_transition import mdp as pose_mdp
from mjlab.tasks.pose_transition.pose_transition_env_cfg import PoseTransitionEnvCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


@dataclass
class BoosterK1VelocityPoseTransitionEnvCfg(PoseTransitionEnvCfg):
  """K1 config using velocity-based phase control with curriculum."""

  def __post_init__(self):
    # Switch to PhaseVelocityAction
    self.actions.phase = term(
      pose_mdp.PhaseVelocityActionCfg,
      asset_name="robot",
      actuator_names=(".*",),
      start_keyframe=CRAWL_KEYFRAME,
      end_keyframe=HOME_KEYFRAME,
      velocity_scale=2.0,  # Max velocity = 2.0 phase_units/sec
      initial_phase=0.0,
      damping_margin=0.15,
      damping_strength=8.0,
    )

    super().__post_init__()

    self.scene.entities = {"robot": replace(K1_ROBOT_CFG)}

    geom_names = [
      "left_hand_collision",
      "right_hand_collision",
      "left_foot_collision",
      "right_foot_collision",
    ]

    # Sensors reused from the velocity task to monitor contacts.
    feet_ground_cfg = ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_foot_link|right_foot_link|left_hand_link|right_hand_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
      name="nonfoot_ground_touch",
      primary=ContactMatch(
        mode="geom",
        entity="robot",
        pattern=r".*_collision\d*$",
        exclude=(
          "left_hand_collision",
          "right_hand_collision",
          "left_foot_collision",
          "right_foot_collision",
          "Right_Shank_collision",
          "Left_Shank_collision",
          "Right_Hip_Yaw_collision",
          "Left_Hip_Yaw_collision",
          "right_hand_link_collision",
          "left_hand_link_collision",
          "Right_Arm_3_collision",
          "Left_Arm_3_collision",
        ),
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    self.scene.sensors = (feet_ground_cfg, self_collision_cfg, nonfoot_ground_cfg)

    # Reset ranges: keep root near origin and facing forward.
    self.events.reset_base.params["pose_range"] = {
      "x": (-0.2, 0.2),
      "y": (-0.2, 0.2),
      "yaw": (-1.67, -1.47),
    }
    self.events.reset_base.params["velocity_range"] = {
      "lin_vel": (-0.05, 0.05),
      "ang_vel": (-0.05, 0.05),
    }
    self.events.reset_robot_joints.func = pose_mdp.reset_joints_to_transition_pose
    self.events.reset_robot_joints.params = {
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "start_keyframe": CRAWL_KEYFRAME,
      "end_keyframe": HOME_KEYFRAME,
      "command_name": "pose",
      "sync_command": True,
    }

    # Initial command resampling (will be overridden by curriculum)
    self.commands.pose.resampling_time_range = (3.0, 5.0)
    self.commands.pose.resample_smoothing = 0.8
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names
    self.events.push_robot = None

    # === NEW REWARD STRUCTURE ===
    # Use dense rewards without completion margins
    self.rewards.pose_tracking = term(
      RewardTerm,
      func=pose_mdp.dense_pose_tracking,
      weight=8.0,  # Increased from 6.0 since this is the main reward
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "command_name": "pose",
        "start_keyframe": CRAWL_KEYFRAME,
        "end_keyframe": HOME_KEYFRAME,
        "std": 0.6,  # Slightly tighter than before
        "root_height_weight": 1.5,
        "root_height_std": 0.1,
        "root_orientation_weight": 5.0,
        "root_orientation_std": 0.6,
      },
    )

    self.rewards.phase_alignment = term(
      RewardTerm,
      func=pose_mdp.dense_phase_alignment,
      weight=3.0,  # Increased from 2.0
      params={"command_name": "pose", "std": 0.5},
    )

    # Add phase smoothness reward
    self.rewards.phase_smoothness = term(
      RewardTerm,
      func=pose_mdp.phase_acceleration_penalty,
      weight=-0.1,  # Penalize jerky phase changes
      params={},
    )

    # Add phase velocity alignment reward
    self.rewards.phase_velocity = term(
      RewardTerm,
      func=pose_mdp.phase_velocity_alignment,
      weight=1.0,  # Reward moving toward target
      params={"command_name": "pose", "std": 0.5},
    )

    # Keep existing penalties
    self.rewards.upright.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=["Trunk"]
    )
    self.rewards.body_ang_vel.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=["Trunk"]
    )
    self.rewards.body_ang_vel.weight = -0.02  # Reduced penalty

    # Reduce action rate penalty since velocity control needs more freedom
    self.rewards.action_rate_l2.weight = -0.02

    self.terminations.fell_over = None

    if self.terminations.illegal_contact is not None:
      self.terminations.illegal_contact.params = {"sensor_name": "nonfoot_ground_touch"}

    self.viewer.body_name = "Trunk"

    # Add phase velocity to observations
    self.observations.policy.phase_velocity = term(
      ObsTerm,
      func=pose_mdp.phase_velocity,
      params={},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    self.observations.critic.phase_velocity = term(
      ObsTerm,
      func=pose_mdp.phase_velocity,
      params={},
    )
