import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from generate_board import definition, make_board, svg_board


class BoardTests(unittest.TestCase):
    def test_larger_a4_board_has_matching_physical_geometry(self):
        config = definition(35)
        svg = ET.fromstring(svg_board(config))
        namespace = {"s": "http://www.w3.org/2000/svg"}
        pattern = svg.find("s:svg", namespace)
        self.assertEqual(float(pattern.attrib["width"]), 175)
        self.assertEqual(float(pattern.attrib["height"]), 245)
        self.assertGreaterEqual(float(pattern.attrib["x"]), 10)
        self.assertGreaterEqual(float(pattern.attrib["y"]), 10)
        np.testing.assert_allclose(make_board(config).getChessboardCorners(),
                                   make_board(definition(25)).getChessboardCorners() * 1.4, atol=2e-8)
        self.assertAlmostEqual(config["marker_length_m"], 0.0245)

    def test_printable_vector_pattern_matches_detectable_board(self):
        config = definition()
        board = make_board(config)
        reference = board.generateImage((1600, 2240), marginSize=0)
        svg = ET.fromstring(svg_board(config))
        namespace = {"s": "http://www.w3.org/2000/svg"}
        pattern = svg.find("s:svg", namespace)
        self.assertEqual(float(pattern.attrib["width"]), 125)
        self.assertEqual(float(pattern.attrib["height"]), 175)
        reconstructed = np.full(reference.shape, 255, dtype=np.uint8)
        for rect in pattern.findall("s:g/s:rect", namespace):
            x, y, width, height = (int(rect.attrib[name]) for name in ("x", "y", "width", "height"))
            reconstructed[y:y + height, x:x + width] = 0
        np.testing.assert_array_equal(reconstructed, reference)
        # Detection runs on the actual printable pattern, including a white margin.
        image = np.pad(reconstructed, 80, constant_values=255)
        corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
        self.assertEqual(len(corners), 24)
        self.assertEqual(set(ids.ravel()), set(range(24)))

    def test_known_metric_board_geometry_recovers_synthetic_pose(self):
        objects = make_board(definition()).getChessboardCorners()
        matrix = np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]])
        expected_rotation = np.array([0.15, -0.2, 0.1])
        expected_translation = np.array([-0.06, -0.08, 0.35])
        pixels, _ = cv2.projectPoints(objects, expected_rotation, expected_translation, matrix, None)
        valid, rotation, translation = cv2.solvePnP(objects, pixels, matrix, None)
        self.assertTrue(valid)
        np.testing.assert_allclose(translation.ravel(), expected_translation, atol=1e-6)
        np.testing.assert_allclose(rotation.ravel(), expected_rotation, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
