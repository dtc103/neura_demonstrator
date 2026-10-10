import unittest

import cv2
import numpy as np
import pyrealsense2 as rs

from calibration_core import reprojection_error, solve_dataset, transform
from camera_geometry import draw_axes, estimate_pose, project_points, sdk_intrinsics, supported_model
from generate_board import definition, make_board
from test_calibration import synthetic_dataset


class CameraGeometryTests(unittest.TestCase):
    def inverse_intrinsics(self):
        return {"width": 1280, "height": 720, "fx": 800., "fy": 810., "cx": 640., "cy": 360.,
                "distortion_model": "distortion.inverse_brown_conrady",
                "distortion_coefficients": [0.15, -0.03, 0.012, -0.008, 0.005]}

    def test_nonzero_inverse_distortion_recovers_known_pose_from_sdk_pixels(self):
        intrinsics = self.inverse_intrinsics()
        self.assertTrue(supported_model(intrinsics))
        objects = make_board(definition()).getChessboardCorners().astype(float)
        expected_rotation = np.array([0.2, -0.25, 0.08])
        expected_translation = np.array([0.03, -0.02, 0.45])
        rotation_matrix = cv2.Rodrigues(expected_rotation)[0]
        points_in_camera = objects @ rotation_matrix.T + expected_translation
        native = sdk_intrinsics(intrinsics)
        pixels = np.array([rs.rs2_project_point_to_pixel(native, p.tolist()) for p in points_in_camera])
        valid, rotation, translation = estimate_pose(objects, pixels, intrinsics)
        self.assertTrue(valid)
        np.testing.assert_allclose(translation.ravel(), expected_translation, atol=2e-6)
        np.testing.assert_allclose(rotation.ravel(), expected_rotation, atol=2e-6)
        observation = {"object_points_m": objects.tolist(), "image_points_px": pixels.tolist()}
        rms = reprojection_error(observation, transform(cv2.Rodrigues(rotation)[0], translation), intrinsics)
        self.assertLess(rms, 0.001)
        # An ordinary OpenCV Brown projection is not interchangeable with the
        # SDK's nonzero inverse model; this regression catches accidental reuse.
        brown = {**intrinsics, "distortion_model": "distortion.brown_conrady"}
        wrong = project_points(objects, expected_rotation, expected_translation, brown)
        self.assertGreater(float(np.max(np.abs(wrong - pixels))), 0.01)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_axes(image, rotation, translation, intrinsics)
        self.assertTrue(image.any())

    def test_inverse_model_with_zero_coefficients_is_pinhole(self):
        inverse = self.inverse_intrinsics()
        inverse["distortion_coefficients"] = [0.] * 5
        pinhole = {**inverse, "distortion_model": "distortion.none"}
        objects = np.array([[0, 0, 0], [0.1, 0.1, 0], [-0.1, 0.03, 0]])
        np.testing.assert_allclose(project_points(objects, [0., 0., 0.], [0., 0., 0.5], inverse),
                                   project_points(objects, [0., 0., 0.], [0., 0., 0.5], pinhole), atol=1e-4)

    def test_full_scene_validation_uses_native_inverse_projection(self):
        metadata, samples, *_ = synthetic_dataset()
        for role in metadata["cameras"]:
            metadata["cameras"][role]["intrinsics"].update(
                distortion_model="distortion.inverse_brown_conrady",
                distortion_coefficients=[0.03, -0.01, 0.001, -0.002, 0.002])
        for sample in samples:
            for role, observation in sample["observations"].items():
                pose = np.array(observation["T_camera_from_board"])
                observation["image_points_px"] = project_points(observation["object_points_m"],
                    cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], metadata["cameras"][role]["intrinsics"]).tolist()
        *_, report = solve_dataset(metadata, samples)
        self.assertTrue(report["passed"])
        for result in report["summary"]["validation"].values():
            self.assertLess(result["reprojection_rms_px"]["max"], 0.001)

    def test_unknown_models_fail_instead_of_silently_ignoring_distortion(self):
        unsupported = {**self.inverse_intrinsics(), "distortion_model": "distortion.ftheta"}
        self.assertFalse(supported_model(unsupported))
        with self.assertRaises(ValueError):
            sdk_intrinsics(unsupported)


if __name__ == "__main__":
    unittest.main()
