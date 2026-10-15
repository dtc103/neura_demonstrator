"""Read-only episode playback and bounded reconstruction of measured joint motion."""
import bisect
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline, PPoly
from joint_teleop import JOINTS


def rows(path):
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def list_episodes(root):
    result = []
    for path in sorted(Path(root).expanduser().glob('session_*/episode_*/episode.json')):
        try:
            metadata = json.loads(path.read_text())
            result.append((path.parent, metadata))
        except (OSError, ValueError):
            continue
    return result


class Episode:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.metadata = json.loads((self.path / 'episode.json').read_text())
        self.session = json.loads((self.path.parent / 'session.json').read_text())
        self.start_ns = int(self.metadata['start_ns'])
        self.cameras = {}
        latest = self.start_ns
        for role in ('wrist', 'side', 'opposite'):
            frames = list(rows(self.path / f'camera_{role}.jsonl'))
            times = [int(frame['received_ns']) for frame in frames]
            if any(b < a for a, b in zip(times, times[1:])):
                raise ValueError(f'{role} camera timestamps are not ordered')
            self.cameras[role] = (times, frames)
            if times:
                latest = max(latest, times[-1])
        self.joint_times, self.joints = [], []
        for row in rows(self.path / 'robot_state.jsonl'):
            q = np.asarray(row['data']['q'], dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError('Invalid recorded joint positions')
            self.joint_times.append((int(row['received_ns']) - self.start_ns) / 1e9)
            self.joints.append(q)
        self.joint_times = np.asarray(self.joint_times)
        self.joints = np.asarray(self.joints)
        self.gripper = [((int(row['received_ns'])-self.start_ns)/1e9, row['data'])
                        for row in rows(self.path / 'gripper_command.jsonl')]
        self.initial_width = None
        for row in rows(self.path / 'gripper_state.jsonl'):
            data = row['data']
            positions = dict(zip(data['name'], data['position']))
            if all(name in positions for name in ('panda_finger_joint1', 'panda_finger_joint2')):
                self.initial_width = sum(positions[name] for name in ('panda_finger_joint1', 'panda_finger_joint2'))
                break
        latest = max(latest, self.start_ns + int(self.joint_times[-1]*1e9) if len(self.joint_times) else latest)
        self.duration = (int(self.metadata.get('end_ns') or latest)-self.start_ns)/1e9
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError('Episode has no positive playback duration')

    def frame_at(self, role, seconds):
        times, frames = self.cameras[role]
        index = bisect.bisect_right(times, self.start_ns + int(seconds*1e9))-1
        return None if index < 0 else frames[index]

    def image(self, frame, depth=False):
        import cv2
        name = frame['depth' if depth else 'rgb']
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError('Camera image path escapes episode directory')
        image = np.load(path, allow_pickle=False) if path.suffix == '.npy' else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Cannot read {path}')
        if depth:
            if image.ndim != 2:
                raise ValueError('Expected a depth image')
            # Fixed 0–2 m display range; preserve the original depth files.
            role = next((r for r in self.cameras if f'/{r}/' in str(name)), '')
            scale = self.metadata.get('cameras', {}).get(role, {}).get('depth_scale_m', .001)
            view = np.clip(image.astype(float)*scale/2*255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(255-view, cv2.COLORMAP_TURBO)
            colored[image == 0] = 0
            image = colored
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a three-channel camera image')
        return cv2.resize(image, (480, 270))[:, :, ::-1].copy()

    def validate_live(self):
        if self.session.get('preview', self.session.get('settings', {}).get('execute') is not True):
            raise ValueError('Preview recordings cannot be replayed on the robot')
        if self.metadata.get('outcome') not in ('success', 'failure', 'unsuccessful') or self.metadata.get('recording_error'):
            raise ValueError('Live replay requires a finalized success/failure episode without recording errors')
        if len(self.joint_times) < 6:
            raise ValueError('At least six measured joint samples are required')
        gaps = np.diff(self.joint_times)
        if np.any(gaps <= 0) or np.max(gaps) > .25:
            raise ValueError('Joint timestamps are unordered or contain a gap larger than 250 ms')
        if self.joint_times[0] < 0 or self.joint_times[0] > .25 or self.duration-self.joint_times[-1] > .25 or self.joint_times[-1] > self.duration:
            raise ValueError('Robot state does not cover the episode boundaries')
        if self.initial_width is None or not np.isfinite(self.initial_width) or not 0 <= self.initial_width <= .081:
            raise ValueError('Missing/invalid initial gripper width')
        previous = -1.
        for seconds, goal in self.gripper:
            if not previous <= seconds <= self.duration or seconds < 0:
                raise ValueError('Invalid gripper event time')
            previous = seconds
            for key, lower, upper in [('width', 0., .08), ('speed', .001, .2)]:
                value = goal.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f'Invalid gripper {key}')
            if 'force' in goal:
                if not np.isfinite(goal['force']) or not 0 < goal['force'] <= 70:
                    raise ValueError('Invalid grasp force')
                for key in ('inner', 'outer'):
                    value = goal.get('epsilon', {}).get(key)
                    if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= .08:
                        raise ValueError('Invalid grasp epsilon')


class ReplayPlan:
    """Quintic interpolation of measured q, with whole-timeline retiming.

    All polynomial extrema are checked, not only the stored samples. ROS quintic
    interpolation reconstructs the same polynomial from knot q/dq/ddq values.
    """
    def __init__(self, episode, model, speed=1., acceleration=3., jerk=30.):
        episode.validate_live()
        if not np.isfinite(speed) or not 0 < speed <= 1:
            raise ValueError('Replay speed must be in (0, 1]')
        if acceleration <= 0 or jerk <= 0:
            raise ValueError('Acceleration/jerk limits must be positive')
        # Hold initial/final pose until the first/after the last state sample.
        t = episode.joint_times - episode.joint_times[0]
        self.offset = float(episode.joint_times[0])
        zero = np.zeros(7)
        self.spline = make_interp_spline(t, episode.joints, k=5,
                                        bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
        self.end = float(t[-1])
        self.duration = episode.duration
        self.scale = 1./speed
        peaks = np.zeros((3, 7))
        for j in range(7):
            poly = PPoly.from_spline((self.spline.t, self.spline.c[:, j], self.spline.k))
            for order in range(4):
                fn = poly.derivative(order)
                roots = fn.derivative().roots(extrapolate=False)
                points = np.r_[0., self.end, roots[(roots >= 0) & (roots <= self.end)]]
                values = fn(points)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite interpolated trajectory')
                if order == 0:
                    if values.min() < model.lower[j] or values.max() > model.upper[j]:
                        raise ValueError(f'Replayed joint {j+1} exceeds model position limits')
                else:
                    peaks[order-1, j] = np.max(np.abs(values))
        self.scale = max(self.scale, float(np.max(peaks[0]/model.velocity)),
                         float(np.sqrt(np.max(peaks[1])/acceleration)),
                         float(np.cbrt(np.max(peaks[2])/jerk))) * 1.001
        self.effective_speed = 1./self.scale
        self.times = np.unique(np.r_[0., self.offset + self.spline.t[(self.spline.t >= 0) & (self.spline.t <= self.end)], self.duration])

    def sample(self, seconds):
        t = np.clip(seconds-self.offset, 0., self.end)
        q = self.spline(t)
        if seconds <= self.offset or seconds >= self.offset+self.end:
            return q, np.zeros(7), np.zeros(7)
        return q, self.spline(t, 1)/self.scale, self.spline(t, 2)/self.scale**2
