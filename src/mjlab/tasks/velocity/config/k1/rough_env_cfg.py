from dataclasses import dataclass, replace
import math

from mjlab.asset_zoo.robots.booster_k1.k1_constants import (
  K1_ROBOT_CFG,
)
from mjlab.entity.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp.events import _DEFAULT_ASSET_CFG
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import UniformCrawlVelocityCommand
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse
import torch


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation when 'up' is body x.

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.

  In this (y, z) locomotion frame, we penalize the gravity components
  along body y and z (indices 1 and 2) so that gravity aligns with body x.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    # Penalize components in (y, z) -> indices 1 and 2.
    yz_squared = torch.sum(torch.square(projected_gravity_b[:, 1:3]), dim=1)
  else:
    # Use root link projected gravity, penalizing y/z components.
    yz_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, 1:3]), dim=1)

  return torch.exp(-yz_squared / std**2)


def track_crawl_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity in crawl mode.

  Assumes:
    - command[:, 0] = v_y_cmd (lateral in body frame)
    - command[:, 1] = v_z_cmd (forward in body frame)
    - vertical / non-planar motion is along body x (actual[:, 0]) and should be ~0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  # Body-frame linear velocity: [v_x, v_y, v_z]
  actual = asset.data.root_link_lin_vel_b

  # Planar error in (y, z) plane: (v_y, v_z).
  # command: [v_y_cmd, v_z_cmd]
  # actual:  v_y = actual[:, 1], v_z = actual[:, 2]
  yz_error = (
    torch.square(command[:, 0] - actual[:, 1])  # lateral (y)
    + torch.square(command[:, 1] - actual[:, 2])  # forward (z)
  )

  # Vertical / out-of-plane velocity along x should be ~0.
  x_error = torch.square(actual[:, 0])

  lin_vel_error = yz_error + x_error
  return torch.exp(-lin_vel_error / std**2)


@dataclass
class BoosterK1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(K1_ROBOT_CFG)}

    # Constants.
    site_names = ["left_hand", "right_hand", "left_foot", "right_foot"]
    geom_names = [
      "left_hand_collision",
      "right_hand_collision",
      "left_foot_collision",
      "right_foot_collision",
    ]
    target_foot_height = 0.1

    # Sensors.
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
    self.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    def uniform_lin_vel(min_val: float, max_val: float) -> dict:
      return {
        "lin_vel_x": (min_val, max_val),
        "lin_vel_y": (min_val, max_val),
        "ang_vel_z": (0, 0),
      }

    # Events.
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names
    self.curriculum.command_vel.params["velocity_stages"] = [
      {"step": 0, **uniform_lin_vel(-0.5, 0.5)},
      {"step": 1000 * 24, **uniform_lin_vel(-1.0, 1.0)},
      {"step": 3000 * 24, **uniform_lin_vel(-1.5, 1.5)},
      {"step": 5000 * 24, **uniform_lin_vel(-1.7, 1.7)},
      {
        "step": 80000 * 24,
        **uniform_lin_vel(-2.0, 2.0),
      },
      {
        "step": 120000 * 24,
        **uniform_lin_vel(-2.5, 2.5),
      },
    ]

    self.events.reset_base.params["pose_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "yaw": (-1.57, -1.57),
    }

    # Rewards.
    self.rewards.upright.params["asset_cfg"].body_names = ["Trunk"]
    # Tight control when stationary: maintain stable default pose.
    self.rewards.pose.params["std_standing"] = {".*": 0.05}
    # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
    self.rewards.pose.params["std_walking"] = {
      # Head
      r".*Head.*": 0.1,
      # Lower body.
      r".*Hip_Pitch.*": 0.4,
      r".*Hip_Roll.*": 0.45,
      r".*Hip_Yaw.*": 0.45,
      r".*Knee.*": 0.45,
      r".*Ankle_Pitch.*": 0.25,
      r".*Ankle_Roll.*": 0.1,
      # Arms.
      r".*Shoulder_Pitch.*": 0.45,
      r".*Shoulder_Roll.*": 0.45,
      r".*Elbow.*": 0.45,
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
      r".*Shoulder_Pitch.*": 0.6,
      r".*Shoulder_Roll.*": 0.4,
      r".*Elbow.*": 0.55,
    }
    self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
    self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["target_height"] = target_foot_height
    self.rewards.foot_clearance.params["target_height"] = target_foot_height
    self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["Trunk"]

    # Disable humanoid rewards.
    self.rewards.self_collisions.weight = 0.0
    self.rewards.body_ang_vel.weight = 0.0
    self.rewards.angular_momentum.weight = 0.0
    self.rewards.track_angular_velocity.weight = 0.0

    self.rewards.upright.func = flat_orientation
    self.rewards.track_linear_velocity.func = track_crawl_linear_velocity

    # Observations.
    self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

    # Terminations.
    self.terminations.illegal_contact = None
    self.terminations.fell_over.params["limit_angle"] = math.radians(180)

    self.viewer.body_name = "Trunk"
    self.commands.twist.class_type = UniformCrawlVelocityCommand
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
