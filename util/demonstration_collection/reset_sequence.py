"""Taught joint reset paths and bounded offline trajectory generation."""
import json
from pathlib import Path
import numpy as np
from joint_teleop import JOINTS, Segment


def load_sequence(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('Reset JSON must be an object containing a poses list')
    # Pose count and joint count are independent. Omitted joint names use Panda order.
    if data.get('joints', JOINTS) != JOINTS:
        raise ValueError('The joints field names the 7 arm joints, not 7 reset poses. '
                         'Use panda_joint1 through panda_joint7 in order, or omit joints.')
    data['joints'] = JOINTS.copy()
    poses = data.get('poses')
    if not isinstance(poses, list) or not poses:
        raise ValueError('Add at least one reset pose. One or two poses are valid; the last is the start pose.')
    for index, pose in enumerate(poses, 1):
        if not isinstance(pose, dict):
            raise ValueError(f'Pose {index} must contain q: a list of 7 joint angles in radians')
        try:
            q = np.asarray(pose.get('q'), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Pose {index}: q must contain 7 finite joint angles in radians') from error
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f'Pose {index}: q needs 7 finite joint angles (one per arm joint). '
                             'The sequence itself can contain just one pose.')
    return data


def reset_segment(q, end, model, speed=.25, acceleration=.5, jerk=2.):
    q, end = np.asarray(q), np.asarray(end)
    if np.any(end < model.lower) or np.any(end > model.upper):
        raise ValueError('Reset target outside model limits')
    if np.any(q < model.lower) or np.any(q > model.upper):
        raise ValueError('Reset starting joints outside model limits')
    duration = max(2., 1.875 * float(np.max(np.abs(end-q))) / speed)
    for _ in range(30):
        segment = Segment(q, np.zeros(7), np.zeros(7), end, duration)
        if segment.bounded(model.lower, model.upper, np.minimum(model.velocity, speed), acceleration, jerk):
            return segment
        duration *= 1.2
    raise ValueError('Could not generate bounded reset segment')
