"""GR00T modality config for the ManiFeel gear assembly dataset (vision-only: wrist camera + EEF state).

Pass to GR00T fine-tuning with:
    --embodiment-tag NEW_EMBODIMENT --modality-config-path <this file>

Keys must match meta/modality.json written by convert_to_lerobot_v2.py.
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

# ManiFeel actions are already normalized end-effector deltas, so GR00T should use them as-is
# (ABSOLUTE = no conversion relative to the current state).
DELTA_AS_IS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

manifeel_gear_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_position", "eef_quaternion"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step chunk, same as ManiFeel's Diffusion Policy horizon
        modality_keys=["delta_translation", "delta_rotation"],
        action_configs=[DELTA_AS_IS, DELTA_AS_IS],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(manifeel_gear_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
