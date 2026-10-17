"""Offline controller interface and legacy trajectory equivalence tests."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from panda_controller import PandaCartesianController, JointPlanner
from test_joint_teleop import Model


class ControllerTests(unittest.TestCase):
    def make(self, **kwargs):
        return PandaCartesianController(model=Model(), **kwargs)

    def test_delta_latched_once_and_referenced_to_measurement(self):
        c = self.make()
        q = np.full(7, .1)
        c.set_delta_action([.01, .02, .03, .1, .2, -.1, 1], q)
        np.testing.assert_allclose(c.target[:3, 3], [.11, .02, .03])
        np.testing.assert_allclose(c.target[:3, :3], Rotation.from_rotvec([.1, .2, -.1]).as_matrix())
        target = c.target.copy()
        c.step(q, np.zeros(7), 0.)
        c.step(q, np.zeros(7), .02)
        np.testing.assert_array_equal(c.target, target)
        c.set_delta_action([.01, 0, 0, 0, 0, 0, 0], q)
        self.assertAlmostEqual(c.target[0, 3], .11)
        self.assertFalse(c.gripper_closed)

    def test_base_rotation_order(self):
        model = Model()
        original = np.eye(4)
        original[:3, :3] = Rotation.from_euler('x', .7).as_matrix()
        model.fk = lambda q: original.copy()
        c = PandaCartesianController(model=model)
        c.set_delta_action([0, 0, 0, 0, 0, .4, 0], np.zeros(7))
        np.testing.assert_allclose(c.target[:3, :3], Rotation.from_euler('z', .4).as_matrix() @ original[:3, :3])

    def test_matches_previous_planner_through_motion_and_stop(self):
        c = self.make()
        legacy = JointPlanner(Model(), .35, 2., 5., 50.)
        q, dq = np.zeros(7), np.zeros(7)
        for i in range(30):
            now = i*.02
            if c.segment is not None:
                q, dq, _ = c.sample(now)
            target = np.eye(4); target[0, 3] = .06 if i < 10 else -.03
            a = c.update(target, q, dq, i < 25, now)
            b = legacy.update(target, q, dq, i < 25, now)
            self.assertEqual(a is None, b is None)
            if a is not None:
                np.testing.assert_array_equal(a.c, b.c)
                points = c.trajectory_points(a)
                self.assertEqual(len(points), 8)
                for t, pos, vel, acc in points:
                    for actual, expected in zip((pos, vel, acc), b.sample(t)):
                        np.testing.assert_array_equal(actual, expected)

    def test_pd_sampling_limits_and_feedforward(self):
        c = self.make(kp=np.ones(7)*10, kd=np.ones(7)*2)
        c.set_delta_action([.02, 0, 0, 0, 0, 0, 1], np.zeros(7))
        c.step(np.zeros(7), np.zeros(7), 10.)
        q, dq, _ = c.sample(10.1)
        torque = c.compute_torques(q-.01, dq-.02, 10.1,
                                  effort_limits=np.ones(7), feedforward=np.ones(7)*.2)
        np.testing.assert_allclose(torque, .34)
        clipped = c.compute_torques(q-1, dq, 10.1, effort_limits=np.ones(7)*.5)
        np.testing.assert_allclose(clipped, .5)
        before = c.segment.c.copy()
        c.sample(10.11)
        np.testing.assert_array_equal(c.segment.c, before)
        c.reset()
        self.assertIsNone(c.target)
        with self.assertRaises(ValueError): c.sample(0.)
        c.set_delta_action([0]*7, np.zeros(7))
        c.step(np.zeros(7), np.zeros(7), 0.)

    def test_invalid_inputs(self):
        c = self.make()
        with self.assertRaises(ValueError): c.step(np.zeros(7), np.zeros(7), 0.)
        with self.assertRaises(ValueError): c.set_delta_action([0]*6+[.5], np.zeros(7))
        with self.assertRaises(ValueError): c.set_delta_action([float('nan')]*7, np.zeros(7))
        with self.assertRaises(ValueError): c.set_target(np.zeros((4, 4)))
        with self.assertRaises(ValueError): self.make(speed=-1)
        c.set_target(np.eye(4)); c.step(np.zeros(7), np.zeros(7), 1.)
        with self.assertRaises(ValueError): c.step(np.zeros(7), np.zeros(7), .5)
        with self.assertRaises(ValueError): c.compute_torques(np.zeros(7), np.zeros(7), 1., effort_limits=np.zeros(7))


if __name__ == '__main__':
    unittest.main()
