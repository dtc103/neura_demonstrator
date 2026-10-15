import unittest

import numpy as np

import visualize_controller as viewer


class ViewerTests(unittest.TestCase):
    def test_controller_selection_uses_button_press_edges(self):
        self.assertEqual(viewer.selected_hand({"X": True}, {}, "right"), "left")
        self.assertEqual(viewer.selected_hand({"A": True}, {}, "left"), "right")
        self.assertEqual(viewer.selected_hand({"A": True}, {"A": True}, "left"), "left")
        self.assertEqual(viewer.selected_hand({}, {"X": True}, "right"), "right")

    def test_rotation_and_reference_preserve_cuboid_geometry(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = viewer.VERTICES @ rotation.T
        np.testing.assert_allclose(moved[:, 0], -viewer.VERTICES[:, 1])
        np.testing.assert_allclose(moved[:, 1], viewer.VERTICES[:, 0])
        np.testing.assert_allclose(moved @ rotation, viewer.VERTICES)

    def test_projection_has_positive_depth_and_centred_origin(self):
        for frame in ("ros", "openxr"):
            points, depth = viewer.project(np.vstack((np.zeros(3), viewer.VERTICES)), (1100, 760), frame)
            self.assertTrue(np.all(depth > 0))
            np.testing.assert_allclose(points[0], [550, 760 * .48])
            eye, basis = viewer.camera_basis(frame)
            np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
            np.testing.assert_allclose((-eye) @ basis, [0, 0, np.linalg.norm(eye)], atol=1e-12)

    def test_missing_or_invalid_rotations_are_not_rendered_as_live_data(self):
        self.assertIsNone(viewer.valid_rotation(None))
        self.assertIsNone(viewer.valid_rotation(np.zeros((4, 4))))
        invalid = np.eye(4)
        invalid[0, 0] = float("nan")
        self.assertIsNone(viewer.valid_rotation(invalid))
        np.testing.assert_allclose(viewer.valid_rotation(np.eye(4)), np.eye(3))


if __name__ == "__main__":
    unittest.main()
