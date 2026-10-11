"""GR00T modality config for gear tasks in end-effector pose space + gripper (wrist camera only).

Shared by ManiFeel (convert_to_lerobot_v2.py --action-format eef_pose --gripper-width 0.033) and REASSEMBLE
(convert_reassemble_to_lerobot_v2.py), so both datasets can be trained together under NEW_EMBODIMENT.

    eef_pose  xyz + rot6d; GR00T turns the 16-step chunk into poses relative to the current pose
    gripper   finger opening in m, used as-is (absolute)
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

gear_eef_gripper_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_pose", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["eef_pose", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_pose",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(gear_eef_gripper_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
