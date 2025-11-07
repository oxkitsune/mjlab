import os

import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.joint_actions import JointAction
from mjlab.third_party.isaaclab.isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_velocity_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  robot: Entity = env.scene["robot"]
  onnx_path = os.path.join(path, filename)
  joint_action: JointAction | None = None
  if "joint_pos" in env.action_manager.active_terms:
    candidate = env.action_manager.get_term("joint_pos")
    if isinstance(candidate, JointAction):
      joint_action = candidate
  if joint_action is None:
    for term_name in env.action_manager.active_terms:
      candidate = env.action_manager.get_term(term_name)
      if isinstance(candidate, JointAction):
        joint_action = candidate
        break
  if joint_action is not None:
    ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    action_scale = (
      joint_action._scale[0].cpu().tolist()
      if isinstance(joint_action._scale, torch.Tensor)
      else joint_action._scale
    )
  else:
    joint_stiffness = []
    joint_damping = []
    action_scale = []
  metadata = {
    "run_path": run_path,
    "joint_names": robot.joint_names,
    "joint_stiffness": joint_stiffness.tolist()
    if hasattr(joint_stiffness, "tolist")
    else joint_stiffness,
    "joint_damping": joint_damping.tolist()
    if hasattr(joint_damping, "tolist")
    else joint_damping,
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": env.command_manager.active_terms,
    "observation_names": env.observation_manager.active_terms["policy"],
    "action_scale": action_scale,
  }

  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
