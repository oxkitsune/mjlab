from __future__ import annotations

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.booster_k1.k1_constants import (
  CRAWL_KEYFRAME,
  HOME_KEYFRAME,
  K1_ROBOT_CFG,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pose_transition import mdp as pose_mdp
from mjlab.tasks.pose_transition.pose_transition_env_cfg import PoseTransitionEnvCfg


@dataclass
class BoosterK1PoseTransitionEnvCfg(PoseTransitionEnvCfg):
  def __post_init__(self):
    # Provide keyframes before base validation.
    self.actions.phase.start_keyframe = CRAWL_KEYFRAME
    self.actions.phase.end_keyframe = HOME_KEYFRAME

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
    self.commands.pose.resampling_time_range = (0.6, 1.1)
    self.commands.pose.resample_smoothing = 1.0
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names
    self.events.push_robot = None

    # Reward shaping specific to K1.
    self.rewards.pose_tracking.params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=[".*"]
    )
    self.rewards.pose_tracking.params["start_keyframe"] = CRAWL_KEYFRAME
    self.rewards.pose_tracking.params["end_keyframe"] = HOME_KEYFRAME
    self.rewards.pose_tracking.params["std"] = 0.7
    self.rewards.pose_tracking.params["root_height_weight"] = 1.5
    self.rewards.pose_tracking.params["root_height_std"] = 0.1
    self.rewards.pose_tracking.params["root_orientation_weight"] = 5.0
    self.rewards.pose_tracking.params["root_orientation_std"] = 0.6
    # Only evaluate pose tracking reward when the command is near either keyframe.
    self.rewards.pose_tracking.params["completion_margin"] = 0.35
    self.rewards.pose_tracking.weight = 6.0
    self.rewards.phase_alignment.params["completion_margin"] = 0.35
    self.rewards.phase_alignment.weight = 2.0

    self.rewards.upright.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=["Trunk"]
    )
    self.rewards.body_ang_vel.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=["Trunk"]
    )

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
