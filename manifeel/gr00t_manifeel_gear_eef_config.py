"""GR00T modality config for ManiFeel gear assembly in end-effector pose space (wrist camera only).

Dataset: convert_to_lerobot_v2.py --action-format eef_pose (state = fingertip pose, action = pose
reached at the next step, both xyz + rot6d). GR00T turns each 16-step action chunk into poses relative
to the current end-effector pose -- the relative-EEF action space N1.7 was pretrained with, and one that
other embodiments (e.g. a humanoid hand frame) can share.

Pass to GR00T fine-tuning with:
    --embodiment-tag NEW_EMBODIMENT --modality-config-path <this file>
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

manifeel_gear_eef_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_pose"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step chunk, same as the delta runs
        modality_keys=["eef_pose"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,  # relative to the current EEF pose (state_key)
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_pose",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(manifeel_gear_eef_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
