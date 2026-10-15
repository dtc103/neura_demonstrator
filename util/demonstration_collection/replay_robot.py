"""Opt-in live episode replay. Imported ROS dependencies only inside the worker."""
import copy
import threading
import time
import numpy as np
from joint_teleop import JOINTS, Kinematics
from replay_data import ReplayPlan
from collection_control import Reset


class LiveReplay:
    def __init__(self, episode, speed=1.):
        self.episode, self.speed = episode, speed
        self.cancelled = threading.Event()
        self.thread = None
        self.status = 'Preparing live replay'
        self.position = 0.
        self.effective_speed = 0.
        self.error = None
        self.done = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=False)
        self.thread.start()

    def stop(self):
        self.cancelled.set()
        self.status = 'Cancelling replay actions; waiting for the worker to stop'

    def _run(self):
        try:
            self._execute()
            self.status = 'Replay complete. Robot holds the final pose.'
        except Exception as error:
            self.error = str(error)
            self.status = f'Replay stopped: {error}'
        finally:
            self.done = True
            print(self.status, flush=True)

    def _execute(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from rclpy.action import ActionClient
        from rclpy.duration import Duration
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.qos import qos_profile_sensor_data
        from franka_msgs.msg import FrankaState
        from franka_msgs.action import Move, Grasp
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance
        from controller_manager_msgs.srv import ListControllers
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rosidl_runtime_py.set_message import set_message_fields

        context = Context()
        rclpy.init(args=[], context=context, signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node('demonstration_replay', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        state, state_at = None, 0.
        handles, pending = [], []
        reset = None
        checking = False
        manager_future = None
        manager_at = 0.

        def receive(msg):
            nonlocal state, state_at
            state, state_at = msg, time.monotonic()

        node.create_subscription(FrankaState, '/franka_robot_state_broadcaster/robot_state', receive, qos_profile_sensor_data)
        manager = node.create_client(ListControllers, '/controller_manager/list_controllers')

        def check_controllers(response):
            active = {c.name for c in response.controller if c.state == 'active'}
            if 'joint_trajectory_controller' not in active:
                raise RuntimeError('joint_trajectory_controller must be active; no controllers were switched')
            if active & {'cartesian_impedance_controller', 'joint_impedance_controller', 'gravity_compensation'}:
                raise RuntimeError('Another motion controller is active')

        def health():
            nonlocal manager_future, manager_at
            if self.cancelled.is_set():
                raise RuntimeError('Stop requested')
            if not checking:
                return
            if state is None or time.monotonic()-state_at > .5:
                raise RuntimeError('Robot state is stale')
            errors = [key for key in state.current_errors.get_fields_and_field_types()
                      if getattr(state.current_errors, key)]
            if state.robot_mode not in (1, 2) or errors:
                raise RuntimeError(f'Robot unavailable: mode={state.robot_mode}, errors={errors}')
            if not np.isfinite(state.q).all() or not np.isfinite(state.dq).all():
                raise RuntimeError('Nonfinite robot state')
            if node.count_publishers('/joint_trajectory_controller/joint_trajectory'):
                raise RuntimeError('Another trajectory publisher is running; close teleoperation first')
            if manager_future is not None:
                if manager_future.done():
                    check_controllers(manager_future.result())
                    manager_future = None
                elif time.monotonic()-manager_at > 3:
                    raise RuntimeError('Controller manager stopped responding')
            elif time.monotonic()-manager_at > 1:
                manager_future = manager.call_async(ListControllers.Request())
                manager_at = time.monotonic()

        def spin():
            executor.spin_once(timeout_sec=.01)
            health()

        def wait(future, timeout=5.):
            until = time.monotonic()+timeout
            while not future.done():
                if time.monotonic() > until:
                    raise RuntimeError('Replay ROS request timed out')
                spin()
            health()
            return future.result()

        def send(client, goal):
            future = client.send_goal_async(goal)
            pending.append(future)
            def accepted(f):
                handle = f.result()
                if handle and handle.accepted:
                    handles.append(handle)
                    if self.cancelled.is_set():
                        handle.cancel_goal_async()
            future.add_done_callback(accepted)
            handle = wait(future)
            if not handle or not handle.accepted:
                raise RuntimeError('Replay action rejected')
            return handle.get_result_async()

        def successful(future, arm=False):
            result = future.result()
            if result.status != 4 or (result.result.error_code != 0 if arm else not result.result.success):
                raise RuntimeError(f'Replay action failed: {result.result}')

        try:
            self.episode.validate_live()
            client = AsyncParameterClient(node, 'robot_state_publisher')
            if not client.wait_for_services(timeout_sec=5):
                raise RuntimeError('Start the Panda driver first (robot_state_publisher unavailable)')
            xml = wait(client.get_parameters(['robot_description'])).values[0].string_value
            model = Kinematics(xml, speed=1.)
            plan = ReplayPlan(self.episode, model, self.speed)
            self.effective_speed = plan.effective_speed
            sequence = copy.deepcopy(self.episode.session.get('reset_sequence'))
            if not isinstance(sequence, dict) or sequence.get('joints', JOINTS) != JOINTS or not sequence.get('poses'):
                raise ValueError('Session has no valid saved reset sequence')
            first = self.episode.joints[0]
            for pose in sequence['poses']:
                q = np.asarray(pose.get('q'), dtype=float)
                if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < model.lower) or np.any(q > model.upper):
                    raise ValueError('Invalid reset pose in session')
            if np.max(np.abs(np.asarray(sequence['poses'][-1]['q'])-first)) > .05:
                raise ValueError('Recorded start is more than 0.05 rad from the saved reset endpoint; provide a checked reset path before live replay')
            sequence['poses'].append({'name': 'recorded_start', 'q': first.tolist()})
            if not manager.wait_for_service(timeout_sec=5):
                raise RuntimeError('Controller manager unavailable')
            check_controllers(wait(manager.call_async(ListControllers.Request())))
            client = AsyncParameterClient(node, 'joint_trajectory_controller')
            if not client.wait_for_services(timeout_sec=5):
                raise RuntimeError('Joint trajectory controller unavailable')
            if list(wait(client.get_parameters(['joints'])).values[0].string_array_value) != JOINTS:
                raise RuntimeError('Unexpected controller joint order')
            until = time.monotonic()+5
            while state is None:
                if time.monotonic() > until:
                    raise RuntimeError('Robot state unavailable')
                spin()
            checking = True
            health()
            if np.max(np.abs(state.dq)) > .03:
                raise RuntimeError('Robot must be stationary before replay')
            arm = ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
            move = ActionClient(node, Move, '/panda_gripper/move')
            grasp = ActionClient(node, Grasp, '/panda_gripper/grasp')
            for action in (arm, move, grasp):
                if not action.wait_for_server(timeout_sec=3):
                    raise RuntimeError('Replay action server unavailable')
            self.status = 'Opening gripper and following the saved reset path'
            reset = Reset(node, model, sequence, .25)
            reset.start(time.monotonic())
            while True:
                spin()
                if reset.tick(state.q, state.dq, time.monotonic()):
                    break
            if np.max(np.abs(np.asarray(state.q)-first)) > .03:
                raise RuntimeError('Robot did not reach the recorded starting pose within 0.03 rad')
            initial = Move.Goal()
            initial.width, initial.speed = min(.08, self.episode.initial_width), .1
            result = send(move, initial)
            wait(result, 15.)
            successful(result)
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = JOINTS
            for seconds in plan.times:
                point = JointTrajectoryPoint()
                q, v, acc = plan.sample(float(seconds))
                point.positions, point.velocities, point.accelerations = q.tolist(), v.tolist(), acc.tolist()
                point.time_from_start = Duration(seconds=float(seconds*plan.scale)).to_msg()
                goal.trajectory.points.append(point)
            goal.path_tolerance = [JointTolerance(name=name, position=.15) for name in JOINTS]
            goal.goal_tolerance = [JointTolerance(name=name, position=.03) for name in JOINTS]
            goal.goal_time_tolerance = Duration(seconds=5.).to_msg()
            health()
            # Schedule in ROS time and drive recorded-video playback from the
            # corresponding host monotonic time; this is not measured robot sync.
            goal.trajectory.header.stamp = (node.get_clock().now()+Duration(seconds=1.)).to_msg()
            start = time.monotonic()+1.
            arm_result = send(arm, goal)
            grip_result = None
            grip_at = 0.
            index = 0
            self.status = f'Live replay · {plan.effective_speed:.3f}× recorded speed'
            while True:
                spin()
                self.position = min(self.episode.duration, max(0., (time.monotonic()-start)/plan.scale))
                if grip_result is not None:
                    if grip_result.done():
                        successful(grip_result)
                        grip_result = None
                    elif time.monotonic()-grip_at > 15:
                        raise RuntimeError('Replay gripper action timed out')
                if index < len(self.episode.gripper) and self.episode.gripper[index][0] <= self.position:
                    if grip_result is not None:
                        raise RuntimeError('Previous gripper action still running at next recorded command')
                    _, data = self.episode.gripper[index]
                    command = Grasp.Goal() if 'force' in data else Move.Goal()
                    set_message_fields(command, data)
                    grip_result = send(grasp if 'force' in data else move, command)
                    grip_at = time.monotonic()
                    index += 1
                if arm_result.done():
                    successful(arm_result, arm=True)
                    if self.position >= self.episode.duration and grip_result is None and index == len(self.episode.gripper):
                        break
                if time.monotonic() > start+self.episode.duration*plan.scale+20:
                    raise RuntimeError('Live replay exceeded its deadline')
        finally:
            self.cancelled.set()
            if reset is not None:
                reset.cancel()
            for handle in handles:
                if handle.status not in (4, 5, 6):
                    handle.cancel_goal_async()
            # Allow cancel requests and late goal acceptances to be processed.
            until = time.monotonic()+1.
            while context.ok() and time.monotonic() < until:
                try:
                    executor.spin_once(timeout_sec=.02)
                except Exception:
                    break
            executor.shutdown()
            node.destroy_node()
            context.shutdown()
