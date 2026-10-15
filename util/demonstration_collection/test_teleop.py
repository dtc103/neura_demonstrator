import copy
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from teleop_mapping import Mapper
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests/quest_tracking_alternative'))
from webxr_reader import validate_frame


def frame(grip=0., trigger=0., xyz=(0., 0., 0.), quaternion=(0., 0., 0., 1.), epoch=0):
    return dict(source='test', sequence=1, epoch=epoch, reference_space='local-floor', controllers={
        'left': None, 'right': dict(input_type='controller', position=list(xyz), orientation=list(quaternion),
                                   squeeze=grip, trigger=trigger, select=False)})


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.m = Mapper(np.eye(3), scale=1., speed=1., angular_speed=2., radius=1.)
        self.robot = np.eye(4)

    def update(self, f):
        return self.m.update(f, self.robot, .02)

    def test_startup_held_grip_requires_release(self):
        self.update(frame(grip=1))
        self.assertFalse(self.m.engaged)
        self.update(frame())
        self.update(frame(grip=1))
        self.assertTrue(self.m.engaged)

    def test_release_cancels_catchup_and_reengagement_rebases(self):
        self.update(frame()); self.update(frame(grip=1))
        moved, _ = self.update(frame(grip=1, xyz=(.5, 0, 0)))
        self.assertAlmostEqual(moved[0, 3], .02)
        self.robot[0, 3] = .01
        held, _ = self.update(frame(xyz=(.5, 0, 0)))
        self.assertAlmostEqual(held[0, 3], .01)
        held, _ = self.update(frame(xyz=(2., 0, 0)))
        self.assertAlmostEqual(held[0, 3], .01)
        resumed, _ = self.update(frame(grip=1, xyz=(2., 0, 0)))
        np.testing.assert_allclose(resumed, self.robot)
        moved, _ = self.update(frame(grip=1, xyz=(2.01, 0, 0)))
        self.assertAlmostEqual(moved[0, 3], .02)

    def test_trigger_click_edges_independent_of_clutch(self):
        self.assertIsNone(self.update(frame(trigger=1))[1])
        self.assertIsNone(self.update(frame(trigger=0))[1])
        self.assertIsNone(self.update(frame(trigger=.5))[1])
        self.assertTrue(self.update(frame(trigger=1))[1])
        self.assertIsNone(self.update(frame(trigger=1))[1])
        self.assertIsNone(self.update(frame(trigger=.5))[1])
        self.assertIsNone(self.update(frame(trigger=0))[1])
        self.assertTrue(self.update(frame(trigger=1, grip=1))[1])
        self.update(None)
        self.assertIsNone(self.update(frame(trigger=1))[1])
        self.assertIsNone(self.update(frame(trigger=0))[1])
        self.assertTrue(self.update(frame(trigger=1))[1])

    def test_loss_and_epoch_reset_require_release(self):
        for reset in (None, frame(grip=1, epoch=1)):
            self.update(frame()); self.update(frame(grip=1))
            self.update(reset)
            self.update(frame(grip=1, epoch=1))
            self.assertFalse(self.m.engaged)
            self.update(frame(epoch=1)); self.update(frame(grip=1, epoch=1))
            self.assertTrue(self.m.engaged)

    def test_rotation_maps_axes_and_is_rate_limited(self):
        self.m.basis = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        self.update(frame()); self.update(frame(grip=1))
        q = Rotation.from_euler('x', .5).as_quat()
        target, _ = self.update(frame(grip=1, quaternion=q))
        np.testing.assert_allclose(Rotation.from_matrix(target[:3, :3]).as_rotvec(), [0, .04, 0], atol=1e-10)

    def test_workspace_violation_holds_and_disarms(self):
        self.update(frame()); self.update(frame(grip=1))
        target, toggle = self.update(frame(grip=1, xyz=(2, 0, 0), trigger=1))
        np.testing.assert_allclose(target, self.robot)
        self.assertFalse(self.m.engaged)
        self.assertFalse(toggle)

    def test_manual_reset_waits_and_rebases_without_motion(self):
        self.m = Mapper(np.eye(3), require_reset=True)
        self.update(frame()); self.update(frame(grip=1))
        self.assertFalse(self.m.engaged)
        reset = frame(grip=1); reset['controllers']['right']['thumbstick'] = True
        self.robot[0, 3] = .4
        target, toggle = self.update(reset)
        np.testing.assert_allclose(target, self.robot)
        np.testing.assert_allclose(self.m.origin, [.4, 0, 0])
        self.assertFalse(toggle)
        self.assertFalse(self.m.engaged)
        self.update(reset)
        self.assertFalse(self.m.engaged)
        self.update(frame()); self.update(frame(grip=1))
        self.assertTrue(self.m.engaged)
        self.update(frame(grip=1, epoch=2))
        self.assertFalse(self.m.ready)

    def test_smoothing_reduces_small_target_jitter_and_hold_is_constant(self):
        self.m = Mapper(np.eye(3), smoothing=.15)
        self.update(frame()); self.update(frame(grip=1))
        target, _ = self.update(frame(grip=1, xyz=(.001, 0, 0)))
        self.assertGreater(target[0, 3], 0)
        self.assertLess(target[0, 3], .0005)
        held, _ = self.update(frame())
        self.robot[0, 3] = .01
        held_again, _ = self.update(frame())
        np.testing.assert_allclose(held, held_again)

    def test_translation_only_ignores_controller_rotation(self):
        self.m = Mapper(np.eye(3), translation_only=True)
        self.update(frame()); self.update(frame(grip=1))
        q = Rotation.from_euler('y', 1.).as_quat()
        target, _ = self.update(frame(grip=1, quaternion=q))
        np.testing.assert_allclose(target[:3, :3], np.eye(3))

    def test_tool_alignment_maps_local_axes_and_stays_fixed_across_clutch(self):
        self.m = Mapper(np.eye(3), scale=1., speed=100., angular_speed=100.,
                        require_reset=True, frame_alignment='tool')
        rc = Rotation.from_euler('xyz', [.3, -.5, .7]).as_matrix()
        rr = Rotation.from_euler('xyz', [-.2, .4, -1.]).as_matrix()
        self.robot[:3, :3] = rr
        q = Rotation.from_matrix(rc).as_quat()
        self.update(frame(quaternion=q))
        reset = frame(quaternion=q); reset['controllers']['right']['thumbstick'] = True
        self.update(reset)
        basis = self.m.basis.copy()
        self.update(frame(quaternion=q)); self.update(frame(grip=1, quaternion=q))
        delta_r = Rotation.from_euler('x', .1).as_matrix()
        target, _ = self.update(frame(grip=1, xyz=rc @ [.01, 0, 0],
                                     quaternion=Rotation.from_matrix(rc @ delta_r).as_quat()))
        np.testing.assert_allclose(target[:3, 3], rr @ [.01, 0, 0], atol=1e-10)
        np.testing.assert_allclose(target[:3, :3], rr @ delta_r, atol=1e-10)
        self.update(frame()); self.update(frame(grip=1))
        np.testing.assert_allclose(self.m.basis, basis)

    def test_precision_is_held_and_both_edges_have_no_target_jump(self):
        self.m = Mapper(np.eye(3), scale=1., speed=100., angular_speed=100., radius=2.)
        self.update(frame()); self.update(frame(grip=1))
        before, _ = self.update(frame(grip=1, xyz=(.2, 0, 0)))
        button = frame(grip=1, xyz=(.2, 0, 0))
        button['controllers']['right']['secondary'] = True
        after, _ = self.update(button)
        np.testing.assert_allclose(after, before)
        self.assertEqual(self.m.scale, .5)
        button = frame(grip=1, xyz=(.3, 0, 0)); button['controllers']['right']['secondary'] = True
        moved, _ = self.update(button)
        self.assertAlmostEqual(moved[0, 3], .25)
        after, _ = self.update(frame(grip=1, xyz=(.3, 0, 0)))
        np.testing.assert_allclose(after, moved)
        self.assertEqual(self.m.scale, 1.)
        moved, _ = self.update(frame(grip=1, xyz=(.4, 0, 0)))
        self.assertAlmostEqual(moved[0, 3], .35)

    def test_secondary_validation_and_reconnection_uses_current_button(self):
        data = frame(); data['controllers']['right']['secondary'] = True
        self.assertTrue(validate_frame(data)['controllers']['right']['secondary'])
        data['controllers']['right']['secondary'] = 1
        with self.assertRaises(ValueError): validate_frame(data)
        self.update(frame()); self.update(None)
        button = frame(); button['controllers']['right']['secondary'] = True
        self.update(button)
        self.assertTrue(self.m.precision)
        self.update(frame())
        self.assertFalse(self.m.precision)

    def test_precision_scales_rotation_and_rebases_both_edges(self):
        self.m = Mapper(np.eye(3), speed=100., angular_speed=100.)
        def rotated(angle, precision=False):
            f = frame(grip=1, quaternion=Rotation.from_euler('y', angle).as_quat())
            f['controllers']['right']['secondary'] = precision
            return f
        self.update(frame()); self.update(frame(grip=1))
        before, _ = self.update(rotated(.2))
        after, _ = self.update(rotated(.2, True))
        np.testing.assert_allclose(after, before, atol=1e-12)
        moved, _ = self.update(rotated(.4, True))
        np.testing.assert_allclose(moved[:3, :3], Rotation.from_euler('y', .3).as_matrix(), atol=1e-12)
        self.assertEqual(self.m.rotation_scale, .5)
        after, _ = self.update(rotated(.4))
        np.testing.assert_allclose(after, moved, atol=1e-12)
        moved, _ = self.update(rotated(.6))
        np.testing.assert_allclose(moved[:3, :3], Rotation.from_euler('y', .5).as_matrix(), atol=1e-12)
        self.assertEqual(self.m.rotation_scale, 1.)

    def test_precision_reduces_angular_speed_cap(self):
        self.update(frame()); self.update(frame(grip=1))
        f = frame(grip=1); f['controllers']['right']['secondary'] = True
        self.update(f)
        f['controllers']['right']['orientation'] = Rotation.from_euler('x', 1.).as_quat().tolist()
        target, _ = self.update(f)
        self.assertAlmostEqual(Rotation.from_matrix(target[:3, :3]).magnitude(), 2.*.5*.02)

    def test_precision_rotation_crosses_180_without_jump(self):
        self.m = Mapper(np.eye(3), angular_speed=100., precision_scale=.25)
        self.update(frame()); self.update(frame(grip=1))
        for degrees in range(0, 271, 10):
            f = frame(grip=1, quaternion=Rotation.from_euler('z', degrees, degrees=True).as_quat())
            f['controllers']['right']['secondary'] = True
            target, _ = self.update(f)
            expected = Rotation.from_euler('z', degrees*.25, degrees=True).as_matrix()
            np.testing.assert_allclose(target[:3, :3], expected, atol=1e-12)

    def test_thumbstick_validation(self):
        data = frame(); data['controllers']['right']['thumbstick'] = True
        self.assertTrue(validate_frame(data)['controllers']['right']['thumbstick'])
        data['controllers']['right']['thumbstick'] = 1
        with self.assertRaises(ValueError): validate_frame(data)

    def test_relay_grip_validation_and_legacy_compatibility(self):
        self.assertEqual(validate_frame(frame(grip=.9))['controllers']['right']['squeeze'], .9)
        for value in (float('nan'), 1.2, True, 'pressed'):
            with self.assertRaises(ValueError): validate_frame(frame(grip=value))
        old = frame(); del old['controllers']['right']['squeeze']
        old = validate_frame(old)
        self.update(old)
        self.assertFalse(self.m.armed)


if __name__ == '__main__':
    unittest.main()
