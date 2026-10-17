"""Offline kinematics and bounded, replaceable Panda joint trajectories.

Reusable control core: no ROS, Quest, hardware access, or collision checking.
The simulation PD helper does not model the complete Franka firmware.
"""
import numpy as np

JOINTS = [f'panda_joint{i}' for i in range(1, 8)]


class Kinematics:
    def __init__(self, xml, speed=.5, tip='panda_hand_tcp'):
        try:
            import pinocchio as pin
            import pink
            from pink.tasks import FrameTask, PostureTask
            import qpsolvers
        except ImportError as error:
            raise RuntimeError('Joint backend needs pin-pink, qpsolvers and quadprog; see README installation instructions') from error
        if 'quadprog' not in qpsolvers.available_solvers:
            raise RuntimeError('Pink quadprog solver is unavailable')
        self.pin, self.pink = pin, pink
        raw = pin.buildModelFromXML(xml)
        lock = [i for i in range(1, raw.njoints) if raw.names[i] not in JOINTS]
        self.model = pin.buildReducedModel(raw, lock, pin.neutral(raw))
        if self.model.nq != 7 or self.model.nv != 7 or list(self.model.names)[1:] != JOINTS:
            raise ValueError('Expected seven Panda joints in standard order')
        if not self.model.existFrame(tip):
            raise ValueError(f'Missing TCP frame {tip}')
        # Same root coordinates as the installed pose broadcaster, labeled base.
        self.tip = tip
        self.model.lowerPositionLimit += .02
        self.model.upperPositionLimit -= .02
        self.model.velocityLimit = np.minimum(self.model.velocityLimit, speed)
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.velocity = self.model.velocityLimit.copy()
        self.data = self.model.createData()
        self.task = FrameTask(tip, position_cost=1., orientation_cost=1., lm_damping=1.)
        self.posture = PostureTask(cost=1e-3)

    def fk(self, q):
        self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
        return self.data.oMf[self.model.getFrameId(self.tip)].homogeneous.copy()

    def solve(self, target, seed, dt=.35):
        """One Pink differential IK step, constrained by model joint limits."""
        q = np.asarray(seed, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError('IK seed outside joint limits')
        configuration = self.pink.Configuration(self.model, self.data, q)
        self.task.set_target(self.pin.SE3(target[:3, :3], target[:3, 3]))
        self.posture.set_target(q)
        try:
            velocity = self.pink.solve_ik(configuration, [self.task, self.posture], dt,
                                          solver='quadprog', damping=1e-6, safety_break=True)
        except Exception as error:
            raise ValueError(f'Pink IK failed: {error}') from error
        result = self.pin.integrate(self.model, q, velocity * dt)
        if not np.isfinite(result).all() or np.any(result < self.lower - 1e-8) or np.any(result > self.upper + 1e-8):
            raise ValueError('Pink returned an invalid joint target')
        return result


class Segment:
    def __init__(self, q, v, a, end, duration):
        self.duration = duration
        # Polynomial in normalized time, with zero final velocity/acceleration.
        c0, c1, c2 = np.asarray(q), np.asarray(v) * duration, np.asarray(a) * duration**2 / 2
        rhs = np.array([end - c0 - c1 - c2, -c1 - 2*c2, -2*c2])
        high = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]), rhs)
        self.c = np.vstack([c0, c1, c2, high])

    def sample(self, seconds):
        u = np.clip(seconds / self.duration, 0., 1.)
        c = self.c
        q = sum(c[i] * u**i for i in range(6))
        v = sum(i*c[i]*u**(i-1) for i in range(1, 6)) / self.duration
        a = sum(i*(i-1)*c[i]*u**(i-2) for i in range(2, 6)) / self.duration**2
        return q, v, a

    def bounded(self, lower, upper, velocity, acceleration, jerk):
        # Evaluate extrema analytically, not just at output samples.
        for joint in range(7):
            coeff = self.c[:, joint].copy()
            for derivative, low, high in ((0, lower[joint], upper[joint]), (1, -velocity[joint], velocity[joint]),
                                           (2, -acceleration, acceleration), (3, -jerk, jerk)):
                if derivative:
                    coeff = np.polynomial.polynomial.polyder(coeff) / self.duration
                roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coeff))
                times = [0., 1.] + [r.real for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < 1]
                values = np.polynomial.polynomial.polyval(times, coeff)
                if np.min(values) < low - 1e-8 or np.max(values) > high + 1e-8:
                    return False
        return True


class JointPlanner:
    def __init__(self, model, horizon=.35, speed=.5, acceleration=1., jerk=5.):
        self.model, self.horizon = model, horizon
        self.velocity = np.minimum(model.velocity, speed)
        self.acceleration, self.jerk = acceleration, jerk
        self.segment = None
        self.started = 0.
        self.moving = False

    def commanded_pose(self, now, measured_pose):
        """Pose of the trajectory currently being executed, not the input goal."""
        if self.segment is None:
            return np.asarray(measured_pose).copy()
        q, _, _ = self.segment.sample(now - self.started)
        return self.model.fk(q)

    def update(self, target, measured_q, measured_dq, engaged, now):
        measured_q, measured_dq = np.asarray(measured_q), np.asarray(measured_dq)
        if not np.isfinite([measured_q, measured_dq]).all():
            raise ValueError('Nonfinite measured joints')
        if self.segment is None:
            if np.max(np.abs(measured_dq)) > .03:
                raise ValueError('Arm must be stationary before starting joint teleoperation')
            q, v, a = measured_q.copy(), np.zeros(7), np.zeros(7)
        else:
            q, v, a = self.segment.sample(now - self.started)
            if np.max(np.abs(q - measured_q)) > .15:
                raise ValueError('Joint tracking error exceeded 0.15 rad')
        if not engaged and not self.moving:
            return None  # Do not repeatedly restart the stop trajectory.
        duration = self.horizon
        # A minimum-jerk stop near the current commanded state.
        stop = q + .5 * v * duration + a * duration**2 / 12
        desired = self.model.solve(target, q, duration) if engaged else stop
        # Progress only as far as joint derivative limits allow this horizon.
        # Every candidate includes a finite stop tail if input disappears.
        for factor in [1., .5, .25, .125, .0625, .03125, .015625, 0.]:
            end = stop + factor * (desired - stop)
            segment = Segment(q, v, a, end, duration)
            if segment.bounded(self.model.lower, self.model.upper, self.velocity, self.acceleration, self.jerk):
                self.segment, self.started, self.moving = segment, now, engaged
                return segment
        raise ValueError('Cannot generate bounded trajectory from current commanded state')


class PandaCartesianController(JointPlanner):
    """Persistent Cartesian target -> IK -> bounded joint reference pipeline.

    Call set_delta_action once per policy action, step at the planner frequency
    (normally 50 Hz), and sample/compute_torques at the actuator frequency.
    All times use a single monotonic clock (simulation time is suitable).
    ROS hardware adapters publish trajectory_points instead of compute_torques.
    """
    def __init__(self, xml=None, *, model=None, horizon=.35, speed=2.,
                 acceleration=5., jerk=50., tip='panda_hand_tcp',
                 kp=(600., 600., 600., 600., 250., 150., 50.),
                 kd=(30., 30., 30., 30., 10., 10., 5.)):
        limits = np.array([horizon, speed, acceleration, jerk], dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError('Controller limits must be finite and positive')
        if (xml is None) == (model is None):
            raise ValueError('Provide exactly one of URDF xml or a kinematics model')
        super().__init__(model if model is not None else Kinematics(xml, speed, tip),
                         horizon, speed, acceleration, jerk)
        self.kp = self._vector(kp, 'kp')
        self.kd = self._vector(kd, 'kd')
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError('PD gains must be nonnegative')
        self.target = None
        self.gripper_closed = None
        self._last_time = None

    @staticmethod
    def _vector(value, name):
        value = np.asarray(value, dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f'{name} must contain seven finite values')
        return value.copy()

    def reset(self):
        """Clear references after an external reset; next step requires rest."""
        self.segment = None
        self.moving = False
        self.started = 0.
        self.target = None
        self.gripper_closed = None
        self._last_time = None

    def set_target(self, target):
        """Set an absolute TCP transform in the model root/base frame."""
        target = np.asarray(target, dtype=float)
        if (target.shape != (4, 4) or not np.isfinite(target).all()
                or not np.allclose(target[3], [0, 0, 0, 1])
                or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(target[:3, :3]), 1., atol=1e-6)):
            raise ValueError('Target must be a finite rigid transform')
        self.target = target.copy()

    def set_delta_action(self, action, measured_q):
        """Latch [base dx,dy,dz (m), base rotvec (rad), closed (0/1)].

        Reference is FK of measured_q at this call, never the previous target.
        The delta is applied once; subsequent step calls retain the target.
        """
        from scipy.spatial.transform import Rotation
        action = self._vector(action, 'action')
        if action[6] not in (0., 1.):
            raise ValueError('Gripper state must be 0 (open) or 1 (closed)')
        pose = self.model.fk(self._vector(measured_q, 'measured_q')).copy()
        pose[:3, 3] += action[:3]
        pose[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix() @ pose[:3, :3]
        self.set_target(pose)
        self.gripper_closed = bool(action[6])

    def step(self, joint_positions, joint_velocities, time, *, enabled=True):
        """Replan once; return a new Segment, or None if already stopped."""
        if enabled and self.target is None:
            raise ValueError('Set a target before stepping the controller')
        return self.update(self.target, joint_positions, joint_velocities, enabled, time)

    def update(self, target, measured_q, measured_dq, engaged, now):
        # Compatibility entry point for absolute-target teleoperation adapters.
        q = self._vector(measured_q, 'measured_q')
        dq = self._vector(measured_dq, 'measured_dq')
        if not np.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError('Controller time must be finite and nondecreasing; reset after clock restart')
        if target is not None:
            self.set_target(target)
        if engaged and self.target is None:
            raise ValueError('Set a target before stepping the controller')
        result = super().update(self.target, q, dq, engaged, now)
        self._last_time = now
        return result

    @staticmethod
    def trajectory_points(segment):
        """Eight (time, q, dq, ddq) points, matching the ROS adapter."""
        return [(float(t), *segment.sample(t)) for t in np.linspace(0., segment.duration, 8)]

    def sample(self, time):
        """Sample the active reference without replanning or reapplying an action."""
        if self.segment is None:
            raise ValueError('No active trajectory; call step first')
        if not np.isfinite(time) or time < self.started:
            raise ValueError('Sample time must be at or after trajectory start')
        return self.segment.sample(time - self.started)

    def compute_torques(self, joint_positions, joint_velocities, time, *,
                        effort_limits, feedforward=None):
        """Explicit simulation PD + supplied compensation, clipped per joint.

        Caller supplies positive actuator limits and, if needed, gravity/payload
        compensation. No implicit drive should add a second PD controller.
        This is not a hardware torque interface or a full firmware model.
        """
        q = self._vector(joint_positions, 'joint_positions')
        dq = self._vector(joint_velocities, 'joint_velocities')
        limits = self._vector(effort_limits, 'effort_limits')
        if np.any(limits <= 0):
            raise ValueError('Effort limits must be positive')
        ff = np.zeros(7) if feedforward is None else self._vector(feedforward, 'feedforward')
        desired_q, desired_dq, _ = self.sample(time)
        return np.clip(self.kp * (desired_q-q) + self.kd * (desired_dq-dq) + ff, -limits, limits)
