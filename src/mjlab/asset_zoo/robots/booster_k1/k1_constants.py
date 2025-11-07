"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

K1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "k1.xml"
)
assert K1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, K1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(K1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.453),
  joint_pos={
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
    "Left_Hip_Pitch": -0.2,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.2,
    "Right_Hip_Pitch": -0.2,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.2,
  },
  joint_vel={".*": 0.0},
)

CRAWL_KEYFRAME_UPSIDE_DOWN = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.353),
  rot=(0, 0, 0.707, 0.707),
  joint_pos={
    "Left_Shoulder_Pitch": -1.6,
    "Left_Shoulder_Roll": -0.3,
    "Left_Elbow_Pitch": 1.54,
    "Left_Elbow_Yaw": -1.5,
    "Right_Shoulder_Pitch": -1.6,
    "Right_Shoulder_Roll": 0.3,
    "Right_Elbow_Pitch": 1.54,
    "Right_Elbow_Yaw": 1.5,
    "Left_Hip_Pitch": -2.2,
    "Left_Hip_Roll": 1.57,
    "Left_Hip_Yaw": 0.68,
    "Left_Knee_Pitch": 1.68,
    "Left_Ankle_Pitch": 0.03,
    "Left_Ankle_Roll": -0.3,
    "Right_Hip_Pitch": -2.2,
    "Right_Hip_Roll": -1.57,
    "Right_Hip_Yaw": -0.68,
    "Right_Knee_Pitch": 1.68,
    "Right_Ankle_Pitch": 0.03,
    "Right_Ankle_Roll": 0.3,
  },
  joint_vel={".*": 0.0},
)

CRAWL_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.353),
  rot=(0, 0, -0.707, 0.707),
  joint_pos={
    "Left_Shoulder_Pitch": -1.4,
    "Left_Shoulder_Roll": 0.6,
    "Left_Elbow_Pitch": -1.7,
    "Left_Elbow_Yaw": -1.0,
    "Right_Shoulder_Pitch": -1.4,
    "Right_Shoulder_Roll": -0.6,
    "Right_Elbow_Pitch": -1.7,
    "Right_Elbow_Yaw": 1.0,
    "Left_Hip_Pitch": 0.1,
    "Left_Hip_Roll": 0.7,
    "Left_Hip_Yaw": 0,
    "Left_Knee_Pitch": 1.5,
    "Left_Ankle_Pitch": 0.03,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": 0.1,
    "Right_Hip_Roll": -0.7,
    "Right_Hip_Yaw": 0,
    "Right_Knee_Pitch": 1.5,
    "Right_Ankle_Pitch": 0.03,
    "Right_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={".*_collision": 3},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=0,
  condim={".*_collision": 3},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[r"^(left|right)_(foot|hand)_collision$"],
  contype=0,
  conaffinity=0,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

K1_ROBOT_CFG = EntityCfg(
  init_state=CRAWL_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(K1_ROBOT_CFG)

  viewer.launch(robot.spec.compile())
