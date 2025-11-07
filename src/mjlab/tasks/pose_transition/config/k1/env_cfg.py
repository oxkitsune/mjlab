from __future__ import annotations

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.booster_k1.k1_constants import (
  CRAWL_KEYFRAME,
  HOME_KEYFRAME,
  K1_ROBOT_CFG,
)
from mjlab.managers.manager_term_config import (
  RewardTermCfg as RewardTerm,
  term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pose_transition import mdp as pose_mdp
from mjlab.tasks.pose_transition.pose_transition_env_cfg import PoseTransitionEnvCfg


@dataclass
class BoosterK1PoseTransitionEnvCfg(PoseTransitionEnvCfg):
  def __post_init__(self):
    # Replace action with AutoPhaseTransitionAction (no term() wrapper)
    self.actions.phase = pose_mdp.AutoPhaseTransitionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      start_keyframe=CRAWL_KEYFRAME,
      end_keyframe=HOME_KEYFRAME,
      initial_phase=0.0,
      phase_smoothing=0.4,  # 0.4s time constant for smooth transitions
      use_residuals=False,  # Pure tracking, action_dim=0
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
    # Start with moderate command frequency
    self.commands.pose.resampling_time_range = (1.0, 2.0)
    self.commands.pose.resample_smoothing = 0.8
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names
    self.events.push_robot = None

    # DENSE REWARD STRUCTURE (no completion margins)
    # Use dense pose tracking throughout the transition
    self.rewards.pose_tracking = term(
      RewardTerm,
      func=pose_mdp.dense_pose_tracking,
      weight=10.0,  # Primary reward
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "command_name": "pose",
        "start_keyframe": CRAWL_KEYFRAME,
        "end_keyframe": HOME_KEYFRAME,
        "std": 0.5,
        "root_height_weight": 2.0,
        "root_height_std": 0.08,
        "root_orientation_weight": 6.0,
        "root_orientation_std": 0.5,
      },
    )

    # Dense phase alignment (no completion margin)
    self.rewards.phase_alignment = term(
      RewardTerm,
      func=pose_mdp.dense_phase_alignment,
      weight=2.0,
      params={"command_name": "pose", "std": 0.6},
    )

    self.rewards.upright.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=["Trunk"]
    )
    self.rewards.upright.weight = 0.5

    self.rewards.body_ang_vel.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=["Trunk"]
    )
    self.rewards.body_ang_vel.weight = -0.02

    # No action rate penalty since action_dim=0
    self.rewards.action_rate_l2.weight = 0.0

    # Strong penalties for safety violations
    self.rewards.joint_pos_limits.weight = -1.0
    self.rewards.self_collisions.weight = -2.0

    self.terminations.fell_over = None
    # # Terminate based on crawl-friendly orientation limits.
    # if self.terminations.fell_over is not None:
    #   self.terminations.fell_over.func = bad_orientation_crawl
    #   self.terminations.fell_over.params = {
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "limit_angle": 1.2,
    #   }

    if self.terminations.illegal_contact is not None:
      self.terminations.illegal_contact.params = {"sensor_name": "nonfoot_ground_touch"}

    self.viewer.body_name = "Trunk"
