"""Offline kinematics and bounded, replaceable Panda joint trajectories.

No inertia model, ROS imports, hardware access, or collision checking.
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
