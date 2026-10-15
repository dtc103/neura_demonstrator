import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.interpolate import BPoly
from replay_data import Episode, ReplayPlan, list_episodes
from replay_robot import LiveReplay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'session_test'/'episode_000001'
        self.path.mkdir(parents=True)
        (self.path.parent/'session.json').write_text(json.dumps({'preview': False}))
        self.manifest = {'start_ns': 1_000_000_000, 'end_ns': 3_000_000_000, 'outcome': 'success', 'prompt': 'test'}
        self.write_manifest()
        # Smooth rest-to-rest movement, represented exactly by the quintic spline.
        t = np.linspace(0, 2, 41)
        q = (.1*(10*(t/2)**3-15*(t/2)**4+6*(t/2)**5))[:, None]*np.ones((1, 7))
        self.states = [{'received_ns': int(1e9+x*1e9), 'data': {'q': y.tolist()}} for x, y in zip(t, q)]
        self.write_rows('robot_state', self.states)
        self.write_rows('gripper_state', [{'received_ns': int(1e9), 'data': {
            'name': ['panda_finger_joint1', 'panda_finger_joint2'], 'position': [.04, .04]}}])
        self.model = SimpleNamespace(lower=np.full(7, -3.), upper=np.full(7, 3.), velocity=np.ones(7))

    def write_manifest(self):
        (self.path/'episode.json').write_text(json.dumps(self.manifest))

    def write_rows(self, stream, values):
        (self.path/(stream+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))

    def test_episode_discovery_and_duration(self):
        self.assertEqual(len(list_episodes(self.root)), 1)
        self.assertEqual(Episode(self.path).duration, 2.)

    def test_camera_uses_timestamps_not_frame_indices(self):
        self.write_rows('camera_wrist', [{'received_ns': 1_100_000_000, 'rgb': 'first.npy'},
                                         {'received_ns': 1_400_000_000, 'rgb': 'second.npy'}])
        episode = Episode(self.path)
        self.assertIsNone(episode.frame_at('wrist', .05))
        self.assertEqual(episode.frame_at('wrist', .35)['rgb'], 'first.npy')
        self.assertEqual(episode.frame_at('wrist', .4)['rgb'], 'second.npy')

    def test_bgr_lossless_image_display(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = 123
        np.save(self.path/'image.npy', image)
        shown = Episode(self.path).image({'rgb': 'image.npy'})
        self.assertTrue(np.all(shown[:, :, 2] == 123))
        self.assertTrue(np.all(shown[:, :, 0] == 0))

    def test_image_path_cannot_escape_episode(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            Episode(self.path).image({'rgb': '../outside.npy'})

    def test_invalid_or_incomplete_recordings_cannot_move_robot(self):
        for outcome in ('incomplete', 'interrupted'):
            self.manifest['outcome'] = outcome
            self.write_manifest()
            with self.assertRaises(ValueError):
                Episode(self.path).validate_live()
        self.manifest['outcome'] = 'success'
        self.write_manifest()
        (self.path.parent/'session.json').write_text('{"preview": true}')
        with self.assertRaisesRegex(ValueError, 'Preview'):
            Episode(self.path).validate_live()

    def test_duplicate_and_missing_robot_samples_rejected(self):
        for samples in (self.states[:4], self.states[:5]+self.states[4:], self.states[:5]+self.states[20:]):
            self.write_rows('robot_state', samples)
            with self.assertRaises(ValueError):
                Episode(self.path).validate_live()

    def test_gripper_validation(self):
        self.write_rows('gripper_command', [{'received_ns': int(2e9), 'data': {'width': .5, 'speed': .1}}])
        with self.assertRaisesRegex(ValueError, 'width'):
            Episode(self.path).validate_live()

    def test_live_plan_rest_endpoints_and_bounds(self):
        plan = ReplayPlan(Episode(self.path), self.model)
        for seconds in (0., 2.):
            _, velocity, acceleration = plan.sample(seconds)
            np.testing.assert_allclose(velocity, 0., atol=1e-10)
            np.testing.assert_allclose(acceleration, 0., atol=1e-10)
        self.assertLessEqual(plan.effective_speed, 1.)
        self.model.upper[:] = .05
        with self.assertRaisesRegex(ValueError, 'position limits'):
            ReplayPlan(Episode(self.path), self.model)

    def test_retime_respects_velocity_acceleration_and_jerk(self):
        self.model.velocity[:] = .01
        plan = ReplayPlan(Episode(self.path), self.model, speed=.5, acceleration=.02, jerk=.03)
        t = np.linspace(0, 2, 2001)
        self.assertLessEqual(np.max(np.abs(plan.spline(t, 1)))/plan.scale, .010001)
        self.assertLessEqual(np.max(np.abs(plan.spline(t, 2)))/plan.scale**2, .020001)
        self.assertLessEqual(np.max(np.abs(plan.spline(t, 3)))/plan.scale**3, .030001)
        self.assertLessEqual(plan.effective_speed, .5)

    def test_ros_quintic_knots_reproduce_planned_curve(self):
        plan = ReplayPlan(Episode(self.path), self.model, speed=.5)
        derivatives = [plan.sample(float(t)) for t in plan.times]
        # JointTrajectoryController uses quintic interpolation when q/dq/ddq are supplied.
        reconstruction = BPoly.from_derivatives(plan.times*plan.scale, derivatives)
        for t in np.linspace(0, 2, 250):
            np.testing.assert_allclose(reconstruction(t*plan.scale), plan.sample(t)[0], atol=1e-9)

    def test_live_worker_reports_failure_and_stop_without_retry(self):
        from unittest.mock import Mock
        replay = LiveReplay(Episode(self.path))
        self.assertIsNone(replay.thread)
        replay._execute = Mock(side_effect=RuntimeError('test controller fault'))
        replay.start()
        replay.thread.join(2)
        self.assertTrue(replay.done)
        self.assertEqual(replay.error, 'test controller fault')
        replay._execute.assert_called_once()
        replay.stop()
        self.assertTrue(replay.cancelled.is_set())


if __name__ == '__main__':
    unittest.main()
