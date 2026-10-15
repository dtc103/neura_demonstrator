#!/usr/bin/env python3
"""Quest browser to Panda Cartesian targets. Preview by default; --execute commands hardware."""
import argparse
import math
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/quest_tracking_alternative'))
from webxr_reader import WebXRReader
from teleop_mapping import Mapper, pose_matrix
from joint_teleop import Kinematics, JointPlanner, JOINTS
from gripper_mapping import ToggleCommands


def arguments(argv=None, collection=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('joint', 'cartesian'), default='joint')
    p.add_argument('--trajectory-horizon', type=float, default=.35, help='Joint trajectory and stop horizon, seconds')
    p.add_argument('--joint-speed', type=float, default=2., help='Joint command speed cap, rad/s, also bounded by URDF')
    p.add_argument('--joint-acceleration', type=float, default=5., help='Joint command acceleration cap, rad/s^2')
    p.add_argument('--joint-jerk', type=float, default=50., help='Joint command jerk cap, rad/s^3')
    p.add_argument('--smoothing-seconds', type=float, default=.05, help='Target low-pass time constant, seconds')
    p.add_argument('--translation-only', action='store_true', help='Keep TCP orientation fixed during each clutch engagement')
    p.add_argument('--log', type=Path, help='Diagnostic JSONL path (default: logs/teleop_<UTC>.jsonl)')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--port', type=int, default=8443)
    p.add_argument('--frame-alignment', choices=('world', 'tool'), default='world',
                   help='tool: align controller axes to TCP axes on thumbstick reset; world: fixed yaw mapping')
    p.add_argument('--precision-scale', type=float, default=.5, help='Hold right B for this translation and rotation ratio; release for --scale translation and 1:1 rotation')
    p.add_argument('--scale', type=float, default=1., help='Robot/controller translation ratio')
    p.add_argument('--max-speed', type=float, default=10., help='Target translation speed, m/s')
    p.add_argument('--max-angular-speed', type=float, default=10., help='Target rotation speed, rad/s')
    p.add_argument('--workspace-radius', type=float, default=.35, help='Target radius around startup TCP, m')
    p.add_argument('--yaw-deg', type=float, default=0, help='Rotate Quest-to-base mapping about robot Z')
    p.add_argument('--stale-seconds', type=float, default=.25)
    p.add_argument('--grasp-force', type=float, default=20., help='Closing grasp force, N')
    p.add_argument('--gripper-speed', type=float, default=.1, help='Opening/closing speed, m/s')
    if collection:
        p.add_argument('--image-format', choices=('npy', 'png'), default='npy', help='Both lossless; npy avoids compression CPU load')
        p.add_argument('--output', type=Path, default=ROOT / 'datasets/demonstrations')
        p.add_argument('--camera-config', type=Path, default=ROOT / 'util/scene_calibration/scene_config_auto.json')
        p.add_argument('--scene', type=Path, help='Scene calibration JSON to embed in session metadata')
        p.add_argument('--reset-sequence', type=Path)
        p.add_argument('--camera-width', type=int, default=1280)
        p.add_argument('--camera-height', type=int, default=720)
        p.add_argument('--camera-fps', type=int, default=30)
        p.add_argument('--reset-speed', type=float, default=.25, help='Reset joint speed cap, rad/s')
    a = p.parse_args(argv)
    if collection:
        if a.backend != 'joint':
            p.error('Collection currently requires the joint backend')
        if not math.isfinite(a.reset_speed) or a.reset_speed <= 0:
            p.error('reset-speed must be positive and finite')
        if min(a.camera_width, a.camera_height, a.camera_fps) <= 0:
            p.error('Camera dimensions and FPS must be positive')
    for key in ('precision_scale', 'scale', 'max_speed', 'max_angular_speed', 'workspace_radius', 'stale_seconds',
                'grasp_force', 'gripper_speed', 'trajectory_horizon', 'joint_speed', 'joint_acceleration', 'joint_jerk'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f'{key} must be finite and positive')
    if not math.isfinite(a.smoothing_seconds) or a.smoothing_seconds < 0:
        p.error('smoothing-seconds must be finite and nonnegative')
    if a.trajectory_horizon < .1:
        p.error('trajectory-horizon must be at least 0.1 seconds')
    if not math.isfinite(a.yaw_deg) or not 1 <= a.port <= 65535:
        p.error('Invalid yaw or port')
    return a


def run(a, collection=None, shared_reader=None):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.parameter_client import AsyncParameterClient
    from controller_manager_msgs.srv import ListControllers
    from geometry_msgs.msg import PoseStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from rclpy.duration import Duration
    from sensor_msgs.msg import JointState
    from franka_msgs.msg import FrankaState
    from franka_msgs.action import Move, Grasp
    from rosidl_runtime_py.convert import message_to_ordereddict

    if collection is None:
        rclpy.init(args=[])
    else:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('quest_panda_teleoperation')
    log = None
    reader = None
    pub = None
    planner = None
    selected_controller = 'joint_trajectory_controller' if a.backend == 'joint' else 'cartesian_impedance_controller'
    command_topic = '/joint_trajectory_controller/joint_trajectory' if a.backend == 'joint' else '/target_pose'
    pending = handle = result_future = None
    gripper_commands = ToggleCommands()
    missing_secondary_warned = False
    gripper_error = None
    shutting_down = False
    measured = state = width = None
    pose_at = state_at = width_at = 0.
    manager_at = 0.
    active = set()
    manager_future = None
    # OpenXR right/up/back -> robot left/up/back: forward(-Z)->+X, right(+X)->-Y.
    basis = Rotation.from_euler('z', a.yaw_deg, degrees=True).as_matrix() @ np.array(
        [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    mapper = Mapper(basis, a.scale, a.max_speed, a.max_angular_speed, a.workspace_radius,
                    smoothing=a.smoothing_seconds, require_reset=True, translation_only=a.translation_only, frame_alignment=a.frame_alignment, precision_scale=a.precision_scale)
    move = ActionClient(node, Move, '/panda_gripper/move')
    grasp = ActionClient(node, Grasp, '/panda_gripper/grasp')
    manager = node.create_client(ListControllers, '/controller_manager/list_controllers')

    def record_message(stream, msg):
        if collection is not None:
            received_ns = time.monotonic_ns()
            collection.record(stream, message_to_ordereddict(msg), received_ns)

    def receive_pose(msg):
        nonlocal measured, pose_at
        if msg.header.frame_id != 'base':
            raise RuntimeError(f'Expected pose in base, received {msg.header.frame_id!r}')
        record_message("robot_pose", msg)
        p, q = msg.pose.position, msg.pose.orientation
        measured = pose_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        pose_at = time.monotonic()

    def receive_state(msg):
        nonlocal state, state_at
        state, state_at = msg, time.monotonic()
        record_message("robot_state", msg)

    def receive_gripper(msg):
        nonlocal width, width_at
        record_message("gripper_state", msg)
        positions = dict(zip(msg.name, msg.position))
        value = positions.get('panda_finger_joint1')
        if value is not None and math.isfinite(value):
            width, width_at = 2 * value, time.monotonic()

    node.create_subscription(PoseStamped, '/current_pose', receive_pose, qos_profile_sensor_data)
    node.create_subscription(FrankaState, '/franka_robot_state_broadcaster/robot_state', receive_state,
                             qos_profile_sensor_data)
    node.create_subscription(JointState, '/panda_gripper/joint_states', receive_gripper, qos_profile_sensor_data)

    def send_joint(segment):
        if pub is None or segment is None:
            return
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        # Timestamp zero starts on receipt. Explicit initial derivatives preserve
        # continuity across replacements; transport delay is not measured by this process.
        for seconds in np.linspace(0., segment.duration, 8):
            point = JointTrajectoryPoint()
            q, v, acc = segment.sample(seconds)
            point.positions, point.velocities, point.accelerations = q.tolist(), v.tolist(), acc.tolist()
            point.time_from_start = Duration(seconds=float(seconds)).to_msg()
            msg.points.append(point)
        pub.publish(msg)
        record_message("sent_command", msg)

    def send_target(target):
        if pub is None:
            return
        msg = PoseStamped()
        msg.header.frame_id = 'base'
        msg.header.stamp = node.get_clock().now().to_msg()
        p, q = target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
        pub.publish(msg)
        record_message("sent_command", msg)

    def wait(future, seconds=5):
        deadline = time.monotonic() + seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError('ROS service timed out')
            rclpy.spin_once(node, timeout_sec=.02)
        return future.result()

    def accepted(future):
        nonlocal handle, result_future, pending, gripper_error
        pending = None
        try:
            handle = future.result()
        except Exception as error:
            gripper_error = str(error)
            return
        if handle is None or not handle.accepted:
            gripper_error = 'Gripper command rejected'
            return
        result_future = handle.get_result_async()
        if shutting_down:
            handle.cancel_goal_async()

    try:
        # Ensure measured and commanded poses use the same TCP and reference frame.
        for name in (('cartesian_impedance_controller', 'pose_broadcaster') if a.backend == 'cartesian' else ('pose_broadcaster',)):
            client = AsyncParameterClient(node, name)
            if not client.wait_for_services(timeout_sec=5):
                raise RuntimeError(f'{name} unavailable; start the Panda launch first')
            topic_key = 'topics.target_pose' if name == 'cartesian_impedance_controller' else 'topics.current_pose'
            response = wait(client.get_parameters(['base_frame', 'end_effector_frame', topic_key]))
            if [v.string_value for v in response.values[:2]] != ['base', 'panda_hand_tcp']:
                raise RuntimeError(f'{name} must use base and panda_hand_tcp; adapt mapping for other configurations')
            expected = 'target_pose' if name == 'cartesian_impedance_controller' else 'current_pose'
            if response.values[2].string_value not in ('', expected, '/' + expected):
                raise RuntimeError(f'{name} uses a custom pose topic; adapt the script before running')
        if a.backend == 'joint':
            client = AsyncParameterClient(node, 'robot_state_publisher')
            if not client.wait_for_services(timeout_sec=5):
                raise RuntimeError('robot_state_publisher unavailable')
            xml = wait(client.get_parameters(['robot_description'])).values[0].string_value
            model = Kinematics(xml, a.joint_speed)
            planner = JointPlanner(model, a.trajectory_horizon, a.joint_speed, a.joint_acceleration, a.joint_jerk)
            client = AsyncParameterClient(node, selected_controller)
            if not client.wait_for_services(timeout_sec=5):
                raise RuntimeError('Joint controller unavailable')
            names = wait(client.get_parameters(['joints'])).values[0].string_array_value
            if list(names) != JOINTS:
                raise RuntimeError('Unexpected joint controller joint order')
        if not manager.wait_for_service(timeout_sec=5):
            raise RuntimeError('Controller manager unavailable')
        response = wait(manager.call_async(ListControllers.Request()))
        active = {c.name for c in response.controller if c.state == 'active'}
        manager_at = time.monotonic()
        if a.execute and selected_controller not in active:
            raise RuntimeError(f'{selected_controller} must be active; no controllers were switched')
        if a.execute and not move.wait_for_server(timeout_sec=3):
            raise RuntimeError('Gripper Move action server unavailable')
        if a.execute and not grasp.wait_for_server(timeout_sec=3):
            raise RuntimeError('Gripper Grasp action server unavailable')
        deadline = time.monotonic() + 5
        while measured is None or state is None or width is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Missing current_pose, robot state, or gripper joint state')
            rclpy.spin_once(node, timeout_sec=.02)
        if planner is not None:
            predicted = model.fk(state.q)
            if np.linalg.norm(predicted[:3, 3] - measured[:3, 3]) > .01 or Rotation.from_matrix(predicted[:3, :3] @ measured[:3, :3].T).magnitude() > .03:
                raise RuntimeError('URDF forward kinematics disagrees with measured TCP frame')
        if collection is not None:
            controller_client = AsyncParameterClient(node, selected_controller)
            parameter_names = wait(controller_client.list_parameters()).result.names
            parameter_values = wait(controller_client.get_parameters(parameter_names)).values
            collection.controller_parameters = {
                name: message_to_ordereddict(value) for name, value in zip(parameter_names, parameter_values)}
            collection.connect(node, model, xml, {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
        reader = shared_reader if shared_reader is not None else WebXRReader(port=a.port)
        if collection is not None:
            reader.state.on_frame = lambda frame, receipt: collection.record('quest', frame, receipt)
        reader.state.stale_seconds = a.stale_seconds
        if a.execute:
            if node.count_publishers(command_topic):
                raise RuntimeError(f'Another {command_topic} publisher exists; stop other teleoperation programs')
            pub = node.create_publisher(JointTrajectory if planner is not None else PoseStamped, command_topic, 1)
        log_path = a.log or Path(__file__).parent / 'logs' / (
            'teleop_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.jsonl')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('x', buffering=1)
        log.write(json.dumps({'event': 'start', 'settings': {k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(a).items()}}) + '\n')
        print(f'Diagnostic log: {log_path}', flush=True)
        print(f'Backend: {a.backend}; translation scale: {a.scale:g}', flush=True)
        print(f'Target caps: {a.max_speed:g} m/s, {a.max_angular_speed:g} rad/s; '
              f'joint cap: {a.joint_speed:g} rad/s (also limited by URDF).', flush=True)
        if planner is not None:
            print(f'Joint acceleration: {a.joint_acceleration:g} rad/s²; jerk: {a.joint_jerk:g} rad/s³; '
                  f'horizon: {a.trajectory_horizon:g}s. These can limit achieved speed.', flush=True)
        print('LIVE CONTROL'  if a.execute else 'PREVIEW ONLY: no arm or gripper commands are sent', flush=True)
        print('Right grip = clutch; trigger click = close/open toggle; hold B = precision translation and rotation. Click right thumbstick when ready, then release grip/trigger and grip to move. Ctrl+C exits.', flush=True)
        print('Mapping: controller axes align to TCP axes on thumbstick reset (yaw ignored).' if a.frame_alignment == 'tool'
              else f'Mapping: Quest forward -> base +X, right -> base -Y, up -> base +Z, rotated by yaw {a.yaw_deg:g} deg.', flush=True)
        last = last_print = last_manager = time.monotonic()
        gripper_started = 0.
        last_clock_sample = 0.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.005)
            now = time.monotonic()
            if now - last < .02:  # Process subscriptions continuously, command at 50 Hz.
                continue
            if manager_future is not None and manager_future.done():
                response = manager_future.result()
                active = {c.name for c in response.controller if c.state == 'active'}
                manager_at, manager_future = now, None
            if now - last_manager > .5 and manager_future is None:
                manager_future = manager.call_async(ListControllers.Request())
                last_manager = now
            if gripper_error:
                raise RuntimeError(gripper_error)
            if collection is not None and now-last_clock_sample > 1:
                collection.record('clock_samples', {'host_monotonic_ns': time.monotonic_ns(),
                    'host_unix_ns': time.time_ns(), 'ros_ns': node.get_clock().now().nanoseconds})
                last_clock_sample = now
            if now - manager_at > 2:
                raise RuntimeError('Controller manager response stale')
            if a.execute and (selected_controller not in active or node.count_publishers(command_topic) > 1):
                raise RuntimeError('Selected controller inactive or competing target publisher')
            if now - min(pose_at, state_at, width_at) > a.stale_seconds:
                raise RuntimeError('Robot or gripper state stream stale; stopping teleoperation')
            errors = [key for key in state.current_errors.get_fields_and_field_types() if getattr(state.current_errors, key)]
            if errors or state.robot_mode not in (1, 2):  # Idle or Move; no reflex/guiding/recovery.
                raise RuntimeError(f'Robot not available for teleoperation: mode={state.robot_mode}, errors={errors}')
            if now - last > .2:
                mapper.suspend(measured)
            frame = reader.snapshot()
            right = None if frame is None else frame['controllers']['right']
            missing_secondary = right is not None and 'secondary' not in right
            if missing_secondary and not missing_secondary_warned:
                print('B BUTTON DATA MISSING: stop tracking and reload the Quest browser page. '
                      'Precision mode cannot work with this older page.', flush=True)
            missing_secondary_warned = missing_secondary
            collection_frame = frame
            allow_input = True
            if collection is not None:
                previous_phase = collection.phase
                collection_frame, allow_input = collection.tick(
                    frame, mapper, measured, state.q, state.dq, planner,
                    pending is not None or result_future is not None, now)
                if previous_phase == 'resetting' and collection.phase == 'reference':
                    gripper_commands.closed = False
                if collection.phase == 'resetting':
                    last = now
                    continue
            if collection is not None and collection_frame is None:
                mapper.suspend(measured)
                if mapper.target is None:
                    mapper.target = measured.copy()
                target, gripper_click = mapper.target.copy(), None
            else:
                target, gripper_click = mapper.update(collection_frame, measured, now-last)
            if collection is not None and not allow_input:
                mapper.engaged = False
                gripper_click = None
            last = now
            # Joint planning may intentionally lag the input due to derivative
            # limits. Monitor actual execution against its current planned pose.
            reference = target if planner is None else planner.commanded_pose(now, measured)
            tracking_position = float(np.linalg.norm(reference[:3, 3] - measured[:3, 3]))
            tracking_rotation = float(Rotation.from_matrix(reference[:3, :3] @ measured[:3, :3].T).magnitude())
            requested_position = float(np.linalg.norm(target[:3, 3] - measured[:3, 3]))
            requested_rotation = float(Rotation.from_matrix(target[:3, :3] @ measured[:3, :3].T).magnitude())
            if a.execute and (tracking_position > .08 or tracking_rotation > .5):
                mapper.suspend(measured)
                target, gripper_click = mapper.target.copy(), None
                mapper.status = (f'Tracking error: {tracking_position*1000:.1f} mm, '
                                 f'{tracking_rotation:.3f} rad; release grip and re-engage')
            segment = None
            solve_started = time.monotonic()
            if planner is not None and (mapper.ready or planner.moving):
                try:
                    segment = planner.update(target, state.q, state.dq, mapper.engaged, now)
                except ValueError as error:
                    mapper.suspend(measured)
                    mapper.status = str(error) + '; release grip and re-engage'
                    gripper_click = None
                    segment = planner.update(measured, state.q, state.dq, False, now)
            solve_seconds = time.monotonic() - solve_started
            if collection is not None:
                collection.record('control', {
                    'command_time_ns': time.monotonic_ns(), 'quest_sequence': None if frame is None else frame['sequence'],
                    'clutch_pressed': bool(right and right.get('squeeze', 0) > .8),
                    'tracking_available': right is not None,
                    'input_allowed': allow_input,
                    'engaged': mapper.engaged, 'status': mapper.status, 'scale': mapper.scale,
                    'precision': mapper.precision, 'rotation_scale': mapper.rotation_scale,
                    'angular_speed_limit_rad_s': mapper.angular_speed * mapper.rotation_scale,
                    'basis': mapper.basis.tolist(), 'target': target.tolist(), 'measured': measured.tolist(),
                    'gripper_click': gripper_click, 'gripper_closed': gripper_commands.closed,
                    'tracking_position_m': tracking_position, 'tracking_rotation_rad': tracking_rotation,
                    'requested_position_gap_m': requested_position, 'requested_rotation_gap_rad': requested_rotation,
                    'solve_seconds': solve_seconds})
            if log:
                log.write(json.dumps({'t_monotonic': now, 'status': mapper.status,
                    'engaged': mapper.engaged, 'scale': mapper.scale, 'precision': mapper.precision, 'rotation_scale': mapper.rotation_scale,
                    'angular_speed_limit_rad_s': mapper.angular_speed * mapper.rotation_scale, 'basis': mapper.basis.tolist(), 'target': target.tolist(), 'measured': measured.tolist(),
                    'q': list(state.q), 'dq': list(state.dq),
                    'command_success_rate': state.control_command_success_rate,
                    'quest': frame, 'gripper_click': gripper_click, 'gripper_closed': gripper_commands.closed, 'gripper_measured_width_m': width, 'solve_seconds': solve_seconds,
                    'tracking_reference': reference.tolist(),
                    'tracking_position_m': tracking_position, 'tracking_rotation_rad': tracking_rotation,
                    'requested_position_gap_m': requested_position, 'requested_rotation_gap_rad': requested_rotation,
                    'joint_endpoint': None if segment is None else segment.sample(segment.duration)[0].tolist()}) + '\n')
            if planner is not None:
                send_joint(segment)
            elif mapper.ready:
                send_target(target)
            if result_future is not None and result_future.done():
                result = result_future.result()
                record_message("gripper_result", result)
                success = result.status == 4 and result.result.success
                gripper_commands.complete(success)
                handle = result_future = None
                if not success:
                    raise RuntimeError(f'Gripper action failed: {result.result.error}')
            was_busy = gripper_commands.active is not None
            close = gripper_commands.next(gripper_click)
            if gripper_click and was_busy:
                print('Gripper busy: click ignored; release and click again after movement finishes.', flush=True)
            if close is not None:
                if a.execute:
                    if close:
                        goal = Grasp.Goal()
                        goal.width, goal.speed, goal.force = 0., a.gripper_speed, a.grasp_force
                        goal.epsilon.inner, goal.epsilon.outer = .005, .08
                        client = grasp
                    else:
                        goal = Move.Goal()
                        goal.width, goal.speed = .08, a.gripper_speed
                        client = move
                    gripper_started = now
                    record_message("gripper_command", goal)
                    pending = client.send_goal_async(goal)
                    pending.add_done_callback(accepted)
                else:
                    gripper_commands.complete(True)
                print(f'Gripper: {"close" if close else "open"}' +
                      ('' if a.execute else ' (preview)'), flush=True)
            if (pending is not None or result_future is not None) and now - gripper_started > 15:
                raise RuntimeError('Gripper command timeout')
            if now - last_print > .5:
                print(f'{mapper.status}; B={"missing" if right is not None and "secondary" not in right else bool(right and right.get("secondary"))}; scale={mapper.scale:g} ({"precision" if mapper.precision else "normal"}); target xyz={np.round(target[:3, 3], 4).tolist()}; '
                      f'gripper={"busy" if gripper_commands.active is not None else "closed" if gripper_commands.closed else "open/ready"}; '
                      f'input gap={requested_position*1000:.1f} mm/{requested_rotation:.3f} rad; '
                      f'execution error={tracking_position*1000:.1f} mm/{tracking_rotation:.3f} rad', flush=True)
                last_print = now
    finally:
        if log:
            log.write(json.dumps({'event': 'stop', 'reason': str(sys.exc_info()[1] or 'shutdown')}) + '\n')
            log.close()
        shutting_down = True
        if collection is not None and collection.reset is not None:
            collection.reset.cancel()
        # Best effort stop/hold; leave the selected controller active.
        robot_available = (state is not None and state.robot_mode in (1, 2)
                           and not any(getattr(state.current_errors, key)
                                       for key in state.current_errors.get_fields_and_field_types()))
        if (collection is None or collection.phase != "resetting") and robot_available and measured is not None and time.monotonic() - pose_at < a.stale_seconds:
            if planner is None:
                send_target(measured)
            elif planner.segment is not None:
                try:
                    send_joint(planner.update(measured, state.q, state.dq, False, time.monotonic()))
                except ValueError:
                    pass  # Last transmitted segment already ends at rest.
        if handle is not None:
            handle.cancel_goal_async()
        reset_pending = collection is not None and collection.reset is not None and collection.reset.pending is not None
        until = time.monotonic() + (5. if pending is not None or reset_pending else .3)
        while rclpy.ok() and time.monotonic() < until:
            try:
                rclpy.spin_once(node, timeout_sec=.01)
            except (Exception, KeyboardInterrupt):
                break
        if reader is not None:
            reader.state.on_frame = None
            if shared_reader is None:
                reader.stop()
        if collection is not None:
            collection.close(str(sys.exc_info()[1] or 'Session closed'))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    try:
        run(arguments())
    except KeyboardInterrupt:
        print('Stopped; selected controller remains active when executing.')
    except Exception as error:
        print(f'Teleoperation stopped: {error}', file=sys.stderr)
        sys.exit(1)
