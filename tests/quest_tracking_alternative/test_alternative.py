import copy
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from webxr_reader import TrackingState, WebXRReader, validate_frame
import visualize_controller as viewer


def frame():
    return {"source": "quest", "sequence": 1, "epoch": 0, "reference_space": "local-floor",
            "controllers": {"left": None, "right": {"input_type": "controller", "position": [1., 2., 3.],
                "orientation": [0., 0., 0., 1.], "trigger": .75, "select": True}}}


class AlternativeTests(unittest.TestCase):
    def test_world_pose_and_buttons_match_frame(self):
        reader = WebXRReader.__new__(WebXRReader)
        reader.state = TrackingState()
        reader.state.publish(frame())
        pose = reader.get_hand_controller_transform_openxr("right")
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))
        np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
        self.assertEqual(reader.get_trigger_value("right"), .75)
        self.assertTrue(reader.get_button_state("A"))
        self.assertFalse(reader.get_button_state("X"))
        self.assertIsNone(reader.get_hand_controller_transform_openxr("left"))
        self.assertEqual(reader.get_frame_epoch(), ("quest", 0))

    def test_hand_and_viewer_spaces_are_rejected(self):
        bad = frame(); bad["reference_space"] = "viewer"
        with self.assertRaises(ValueError): validate_frame(bad)
        bad = frame(); bad["controllers"]["right"]["input_type"] = "hand"
        with self.assertRaises(ValueError): validate_frame(bad)
        bad = frame(); bad["controllers"]["right"]["emulatedPosition"] = True
        self.assertIsNone(validate_frame(bad)["controllers"]["right"])

    def test_stale_samples_disappear_and_reordering_is_rejected(self):
        state = TrackingState()
        with patch("webxr_reader.time.monotonic", return_value=10):
            state.publish(frame())
            with self.assertRaises(ValueError): state.publish(frame())
            self.assertIsNotNone(state.snapshot())
        with patch("webxr_reader.time.monotonic", return_value=11):
            self.assertIsNone(state.snapshot())

    def test_reference_reset_changes_epoch(self):
        state = TrackingState(); state.publish(frame())
        changed = frame(); changed.update(sequence=2, epoch=1)
        state.publish(changed)
        self.assertEqual(state.snapshot()["epoch"], 1)

    def test_cube_renders_and_changes_orientation_without_headset(self):
        import pygame
        pygame.display.init(); pygame.font.init()
        try:
            screen = pygame.Surface((1100, 760))
            font = pygame.font.Font(None, 23)
            viewer.render_scene(pygame, screen, np.eye(3), "ros", font)
            before = pygame.image.tostring(screen, "RGB")
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            viewer.render_scene(pygame, screen, rotation, "ros", font)
            self.assertNotEqual(before, pygame.image.tostring(screen, "RGB"))
            viewer.render_scene(pygame, screen, np.eye(3), "openxr", font)
        finally: pygame.quit()
        self.assertEqual(viewer.selected_hand({"X": True}, {}, "right"), "left")
        self.assertEqual(viewer.selected_hand({"A": True}, {}, "left"), "right")


if __name__ == "__main__":
    unittest.main()
