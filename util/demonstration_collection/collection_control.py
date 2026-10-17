"""Episode lifecycle and asynchronous reset actions, separate from recording I/O."""
import copy
import hashlib
import subprocess
import json
import queue
import time
from pathlib import Path
import numpy as np
from collection_cameras import Cameras
from recording import Recorder, session_name
from reset_sequence import load_sequence, reset_segment
from panda_controller import JOINTS


class Edges:
    def __init__(self):
        self.previous = {}
        self.identity = None

    def update(self, frame):
        if frame is None:
            self.previous.clear()
            self.identity = None
            return set()
        identity = frame['source'], frame['epoch']
        if identity != self.identity:
            self.previous.clear()
            self.identity = identity
        pressed = set()
        for key, hand, field in [('A', 'right', 'select'), ('X', 'left', 'select'), ('Y', 'left', 'secondary')]:
            # Pose loss must not hide episode buttons. Older clients use poses.
            pose = frame.get('controller_buttons', frame['controllers'])[hand]
            if pose is None:
                self.previous.pop(key, None)
                continue
            value = bool(pose.get(field, False))
            if self.previous.get(key) is False and value:
                pressed.add(key)
            self.previous[key] = value
        return pressed


class Reset:
    def __init__(self, node, model, sequence, speed, disable_path_tolerance=False):
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from franka_msgs.action import Move
        self.client = ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        self.gripper = ActionClient(node, Move, '/panda_gripper/move')
        self.model, self.sequence, self.speed = model, sequence, speed
        self.disable_path_tolerance = disable_path_tolerance
        self.pending = self.handle = self.result = None
        self.index = -1
        self.deadline = 0
        self.active = False
        self.closed = False
        self.on_command = lambda *args: None

    def start(self, now):
        from franka_msgs.action import Move
        if not self.client.server_is_ready() or not self.gripper.server_is_ready():
            raise RuntimeError('Reset action servers unavailable')
        self.active = True
        self.index = -1
        goal = Move.Goal()
        goal.width, goal.speed = .08, .1
        self.on_command('gripper', goal)
        self.pending = self.gripper.send_goal_async(goal)
        self.pending.add_done_callback(self._accepted)
        self.deadline = now+15

    def _accepted(self, future):
        self.pending = None
        self.handle = future.result()
        if not self.handle.accepted:
            self.active = False
            return
        self.result = self.handle.get_result_async()
        if self.closed:
            self.handle.cancel_goal_async()

    def tick(self, q, dq, now):
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rclpy.duration import Duration
        if not self.active:
            raise RuntimeError('Reset action rejected')
        if now > self.deadline:
            self.cancel()
            raise RuntimeError('Reset timed out; no automatic retry')
        if self.result is None or not self.result.done():
            return False
        result = self.result.result()
        self.on_command("result", result)
        self.handle = self.result = None
        success = result.status == 4 and (result.result.success if self.index == -1 else result.result.error_code == 0)
        if not success:
            raise RuntimeError(f'Reset action failed: {result.result}')
        if np.max(np.abs(dq)) > .03:
            raise RuntimeError('Robot still moving after reset action; reset stopped')
        self.index += 1
        if self.index == len(self.sequence['poses']):
            self.active = False
            return True
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        segment = reset_segment(q, self.sequence['poses'][self.index]['q'], self.model, self.speed)
        for seconds in np.linspace(0., segment.duration, max(8, int(segment.duration*20))):
            point = JointTrajectoryPoint()
            pos, vel, acc = segment.sample(seconds)
            point.positions, point.velocities, point.accelerations = pos.tolist(), vel.tolist(), acc.tolist()
            point.time_from_start = Duration(seconds=float(seconds)).to_msg()
            goal.trajectory.points.append(point)
        goal.goal_tolerance = [JointTolerance(name=name, position=.03) for name in JOINTS]
        goal.path_tolerance = self.path_tolerances(JointTolerance)
        goal.goal_time_tolerance = Duration(seconds=5.).to_msg()
        self.on_command('trajectory', goal)
        self.pending = self.client.send_goal_async(goal)
        self.pending.add_done_callback(self._accepted)
        self.deadline = now+segment.duration+10
        return False

    def path_tolerances(self, tolerance_type):
        if self.disable_path_tolerance:
            # -1 erases a tolerance; 0 would inherit the controller default.
            return [tolerance_type(name=name, position=-1., velocity=-1., acceleration=-1.)
                    for name in JOINTS]
        return [tolerance_type(name=name, position=.15) for name in JOINTS]

    def cancel(self):
        self.closed = True
        if self.handle is not None:
            self.handle.cancel_goal_async()


class Collection:
    def __init__(self, args, prompt, sequence_path):
        self.args = args
        self.session_name = session_name(getattr(args, 'session_name', ''))
        self.prompts = [part.strip() for part in prompt.split(';') if part.strip()]
        if not self.prompts:
            raise ValueError('Enter at least one nonempty task description')
        self.prompt = self.prompts[0]  # Preserve the legacy single-prompt field.
        self.sequence = load_sequence(sequence_path)
        self.commands = queue.Queue()
        self.phase = 'connecting'
        self.detail = 'Connecting to robot'
        self.recorder = self.cameras = self.reset = None
        self.edges = Edges()
        self.episode = None
        self.outcome = None
        self.stop_ns = None
        self.last_health = 0
        self.frame_identity = None
        self.tracking_pause_reason = None

    def connect(self, node, model, xml, settings):
        serials = json.loads(Path(self.args.camera_config).read_text())['cameras']
        source_dir = Path(__file__).parent
        source_hashes = {str(path.relative_to(source_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in source_dir.glob('*.py')}
        shared_dir = source_dir.parents[1] / 'tests/quest_tracking_alternative'
        for relative in ('webxr_reader.py', 'static/app.js', 'static/index.html'):
            source_hashes['quest/' + relative] = hashlib.sha256((shared_dir / relative).read_bytes()).hexdigest()
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source_dir,
                                  capture_output=True, text=True, timeout=3)
        metadata = {'source_sha256': source_hashes, 'git_head': revision.stdout.strip(),
            'controller_parameters': getattr(self, 'controller_parameters', {}),
            'prompt': self.prompt, 'prompts': self.prompts, 'settings': settings, 'reset_sequence': self.sequence,
            'clock_pair': {'monotonic_ns': time.monotonic_ns(), 'unix_ns': time.time_ns()},
            'camera_serials': serials, 'preview': not self.args.execute}
        if self.args.scene:
            metadata['scene_calibration'] = json.loads(Path(self.args.scene).read_text())
        self.recorder = Recorder(self.args.output, metadata, image_format=self.args.image_format,
                                 name=self.session_name)
        (self.recorder.root / 'robot.urdf').write_text(xml)
        self.cameras = Cameras(serials, self.recorder, self.args.camera_width, self.args.camera_height, self.args.camera_fps)
        self.reset = Reset(node, model, self.sequence, self.args.reset_speed,
                           disable_path_tolerance=self.args.disable_reset_path_tolerance)
        print('Reset path tolerance: ' + ('disabled' if self.args.disable_reset_path_tolerance else '0.15 rad')
              + '; final goal tolerance: 0.03 rad; goal time allowance: 5s.', flush=True)
        from rosidl_runtime_py.convert import message_to_ordereddict
        self.reset.on_command = lambda kind, msg: self.recorder.record_session(
            'reset_' + kind, message_to_ordereddict(msg))
        # Validate all saved targets before enabling any reset motion.
        for pose in self.sequence['poses']:
            reset_segment(pose['q'], pose['q'], model, self.args.reset_speed)
        self.phase = 'needs_reset'
        self.detail = 'A: open gripper and reset; then thumbstick, then A to record'
        self.connected_at = time.monotonic()

    def record(self, stream, data, received_ns=None):
        if self.recorder is not None:
            self.recorder.record(stream, data, received_ns)

    def tick(self, frame, mapper, measured, q, dq, planner, gripper_busy, now):
        if now-self.last_health > .5:
            self.recorder.health()
            ready = self.cameras.health()
            if not ready and (self.phase == 'recording' or now-self.connected_at > 15):
                raise RuntimeError('RGB/depth camera stream unavailable')
            self.last_health = now
        edges = self.edges.update(frame)
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            if command == 'shutdown':
                raise RuntimeError('Session closed by operator')
            edges.add(command)
        if self.phase == 'resetting':
            if not self.args.execute or self.reset.tick(q, dq, now):
                planner.reset()
                mapper.suspend(measured)
                mapper.target = measured.copy()
                mapper.ready = False
                mapper.reset_armed = False
                self.phase = 'reference'
                self.detail = ('Reset complete. ' if self.args.execute else 'Preview reset: no movement. ') + 'Click right thumbstick, then A to record.'
            return None, False
        if self.phase == 'stopping':
            if now-self.stop_started > 8:
                raise RuntimeError('Arm or gripper did not finish stopping; reset inhibited')
            if now-self.stop_started >= max(.5, self.args.trajectory_horizon) and np.max(np.abs(dq)) < .03 and not gripper_busy:
                self.recorder.end(self.outcome, boundary_ns=self.stop_ns)
                self.phase = 'flushing'
                self.flush_started = now
            return None, False
        if self.phase == 'flushing':
            if now-self.flush_started > 30:
                raise RuntimeError('Recording finalization timed out; reset inhibited')
            if self.episode in self.recorder.finished:
                self.begin_reset(now, mapper, measured, dq, gripper_busy)
            return None, False
        if self.phase == 'recording':
            identity = None if frame is None else (frame['source'], frame['epoch'])
            right = None if frame is None else frame['controllers']['right']
            outcome = 'interrupted' if 'Y' in edges else 'failure' if 'X' in edges else 'success' if 'A' in edges else None
            if outcome:
                self.outcome = outcome
                self.stop_ns = time.monotonic_ns()
                self.record('events', {'event': 'stop_requested', 'outcome': outcome})
                self.stop_started = now
                self.phase = 'stopping'
                mapper.suspend(measured)
                self.detail = f'{outcome}: stopping, saving, then opening/resetting'
                return None, False
            timing = {} if frame is None else frame.get('timing', {})
            reason = None
            if identity is not None and identity != self.frame_identity:
                # Reconnection/recentering changes coordinates: keep recording,
                # but require a new deliberate thumbstick reference before motion.
                self.frame_identity = identity
                mapper.ready = False
                mapper.reset_armed = False
                self.record('events', {'event': 'tracking_reference_changed', 'identity': identity})
                reason = 'Quest reference changed; click thumbstick after tracking returns'
            elif right is None:
                reason = 'Quest frame missing/stale' if frame is None else 'Right controller not tracked'
            elif timing.get('host_offset_ns') is None or timing.get('sync_age_ms', float('inf')) > 15000:
                reason = 'Waiting for Quest clock synchronization'
            if reason:
                if reason != self.tracking_pause_reason:
                    self.record('events', {'event': 'tracking_paused', 'reason': reason})
                self.tracking_pause_reason = reason
                mapper.suspend(measured)
                self.detail = f'Recording continues · motion paused: {reason}'
                return None, False
            if self.tracking_pause_reason is not None:
                self.record('events', {'event': 'tracking_restored', 'previous_reason': self.tracking_pause_reason})
                self.tracking_pause_reason = None
            self.detail = ('Recording: release grip/trigger, then grip to move; A success, X failure, Y interrupted'
                           if mapper.ready else 'Recording continues: click right thumbstick to establish the new reference')
            return frame, True
        if self.phase == 'needs_reset' and 'A' in edges:
            self.begin_reset(now, mapper, measured, dq, gripper_busy)
            return None, False
        if self.phase in ('reference', 'ready'):
            if not mapper.ready:
                self.phase = 'reference'
            else:
                self.phase = 'ready'
                self.detail = 'A: start episode. Release grip first; B remains precision.'
            if self.phase == 'ready' and 'A' in edges:
                timing = {} if frame is None else frame.get('timing', {})
                if (not self.cameras.health() or gripper_busy or timing.get('host_offset_ns') is None
                        or timing.get('sync_age_ms', float('inf')) > 15000):
                    self.detail = 'Waiting for cameras, gripper and Quest clock sync; reload Quest page if needed'
                else:
                    self.episode = self.recorder.start(self.prompt, {'prompts': self.prompts.copy(), 'cameras': self.cameras.metadata,
                        'reference_basis': mapper.basis.tolist(), 'workspace_origin': mapper.origin.tolist()})
                    self.phase = 'recording'
                    self.tracking_pause_reason = None
                    self.frame_identity = frame['source'], frame['epoch']
                    self.record('events', {'event': 'start', 'quest': frame})
                    mapper.suspend(measured)
                    self.detail = 'Recording: A success, X failure, Y interrupted'
                    # Do not arm a held grip/trigger on the start edge.
                    return None, False
            # Allow thumbstick reference changes, but never idle arm/gripper input.
            masked = copy.deepcopy(frame)
            if masked is not None and masked['controllers']['right'] is not None:
                masked['controllers']['right']['squeeze'] = 0.
                masked['controllers']['right']['trigger'] = 1.  # cannot re-arm trigger
            return masked, False
        return None, False

    def begin_reset(self, now, mapper, measured, dq, busy):
        if not self.cameras.health():
            self.detail = 'Waiting for RGB/depth camera streams; press A again when ready'
            return
        if busy or np.max(np.abs(dq)) > .03:
            self.detail = 'Reset requires stationary arm and idle gripper; press A again'
            return
        mapper.suspend(measured)
        mapper.ready = False
        self.phase = 'resetting'
        self.detail = 'Opening gripper and executing taught reset sequence'
        if self.args.execute:
            self.reset.start(now)

    def close(self, reason):
        self.phase = 'fault' if reason else 'closed'
        self.detail = reason or 'Closed'
        if self.reset is not None:
            self.reset.cancel()
        if self.recorder is not None:
            self.recorder.end('interrupted', reason or 'Session closed')
        if self.cameras is not None:
            self.cameras.close()
        if self.recorder is not None:
            self.recorder.close()
