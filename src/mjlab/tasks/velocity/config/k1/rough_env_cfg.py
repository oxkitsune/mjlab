from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.booster_k1.k1_constants import (
  K1_ROBOT_CFG,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class BoosterK1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(K1_ROBOT_CFG)}

    # Constants.
    site_names = ["left_foot", "right_foot"]
    geom_names = ["left_foot_collision", "right_foot_collision"]
    target_foot_height = 0.15

    # Sensors.
    feet_ground_cfg = ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_foot_link|right_foot_link)$",
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
    self.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    # Events.
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    # Rewards.
    self.rewards.upright.params["asset_cfg"].body_names = ["Trunk"]
    # Tight control when stationary: maintain stable default pose.
    self.rewards.pose.params["std_standing"] = {".*": 0.05}
    # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
    self.rewards.pose.params["std_walking"] = {
      # Head
      r".*Head.*": 0.1,
      # Lower body.
      r".*Hip_Pitch.*": 0.3,
      r".*Hip_Roll.*": 0.15,
      r".*Hip_Yaw.*": 0.15,
      r".*Knee.*": 0.35,
      r".*Ankle_Pitch.*": 0.25,
      r".*Ankle_Roll.*": 0.1,
      # Arms.
      r".*Shoulder_Pitch.*": 0.15,
      r".*Shoulder_Roll.*": 0.15,
      r".*Elbow.*": 0.15,
    }
    # Maximum freedom for dynamic motion.
    self.rewards.pose.params["std_running"] = {
      # Head
      r".*Head.*": 0.1,
      # Lower body.
      r".*Hip_Pitch.*": 0.5,
      r".*Hip_Roll.*": 0.2,
      r".*Hip_Yaw.*": 0.2,
      r".*Knee.*": 0.6,
      r".*Ankle_Pitch.*": 0.35,
      r".*Ankle_Roll.*": 0.15,
      # Arms.
      r".*Shoulder_Pitch.*": 0.5,
      r".*Shoulder_Roll.*": 0.2,
      r".*Elbow.*": 0.35,
    }
    self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
    self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["target_height"] = target_foot_height
    self.rewards.foot_clearance.params["target_height"] = target_foot_height
    self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["Trunk"]

    # Observations.
    self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

    # Terminations.
    self.terminations.illegal_contact = None

    self.viewer.body_name = "Trunk"
    self.commands.twist.viz.z_offset = 1.15


@dataclass
class BoosterK1RoughEnvCfg_PLAY(BoosterK1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
