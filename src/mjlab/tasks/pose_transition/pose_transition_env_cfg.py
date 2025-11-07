"""Base configuration for pose transition tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.managers.manager_term_config import (
  CurriculumTermCfg as CurrTerm,
  EventTermCfg as EventTerm,
  ObservationGroupCfg as ObsGroup,
  ObservationTermCfg as ObsTerm,
  RewardTermCfg as RewardTerm,
  TerminationTermCfg as DoneTerm,
  term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.tasks.pose_transition import mdp as pose_mdp
from mjlab.tasks.velocity.velocity_env_cfg import (
  EventCfg as VelocityEventCfg,
  TerminationCfg as VelocityTerminationCfg,
  SIM_CFG,
)
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def _default_scene_cfg() -> SceneCfg:
  return SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    num_envs=1,
    extent=2.0,
  )


def _default_viewer_cfg() -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )


@dataclass
class ActionCfg:
  phase: pose_mdp.PoseBlendActionCfg = term(
    pose_mdp.PoseBlendActionCfg,
    asset_name="robot",
    actuator_names=(".*",),
  )


@dataclass
class CommandsCfg:
  pose: pose_mdp.PoseTransitionCommandCfg = term(
    pose_mdp.PoseTransitionCommandCfg,
    resampling_time_range=(2.0, 4.0),
    debug_vis=False,
    initial_state=0.0,
    toggle_on_resample=True,
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_vel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    command: ObsTerm = term(
      ObsTerm,
      func=mdp.generated_commands,
      params={"command_name": "pose"},
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class CriticCfg(PolicyCfg):
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)


@dataclass
class RewardCfg:
  pose_tracking: RewardTerm = term(
    RewardTerm,
    func=pose_mdp.TrackPoseKeyframeReward,
    weight=4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "command_name": "pose",
      "start_keyframe": None,
      "end_keyframe": None,
      "std": 0.35,
    },
  )
  phase_alignment: RewardTerm = term(
    RewardTerm,
    func=pose_mdp.phase_command_alignment,
    weight=1.5,
    params={"command_name": "pose", "std": 0.2},
  )
  upright: RewardTerm = term(
    RewardTerm,
    func=mdp.flat_orientation,
    weight=0.5,
    params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", body_names=[])},
  )
  body_ang_vel: RewardTerm = term(
    RewardTerm,
    func=pose_mdp.body_angular_velocity_penalty,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=[])},
  )
  action_rate_l2: RewardTerm = term(
    RewardTerm,
    func=mdp.action_rate_l2,
    weight=-0.05,
  )
  joint_pos_limits: RewardTerm = term(
    RewardTerm,
    func=mdp.joint_pos_limits,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
  )
  self_collisions: RewardTerm = term(
    RewardTerm,
    func=pose_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision"},
  )


@dataclass
class PoseTransitionEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=_default_scene_cfg)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: VelocityEventCfg = field(default_factory=VelocityEventCfg)
  terminations: VelocityTerminationCfg = field(default_factory=VelocityTerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  curriculum: CurrTerm | None = None
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=_default_viewer_cfg)
  decimation: int = 4
  episode_length_s: float = 8.0

  def __post_init__(self):
    if (
      self.actions.phase.start_keyframe is None
      or self.actions.phase.end_keyframe is None
    ):
      raise ValueError(
        "PoseTransitionEnvCfg requires start/end keyframes to be provided before initialization."
      )
