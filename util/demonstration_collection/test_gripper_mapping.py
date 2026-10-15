import unittest
from gripper_mapping import ToggleCommands


class ToggleTests(unittest.TestCase):
    def test_first_click_closes_second_opens(self):
        c = ToggleCommands()
        self.assertIsNone(c.next(None))
        self.assertIs(c.next(True), True)
        c.complete(True)
        self.assertTrue(c.closed)
        self.assertIs(c.next(True), False)
        c.complete(True)
        self.assertFalse(c.closed)

    def test_busy_clicks_are_not_queued(self):
        c = ToggleCommands()
        self.assertIs(c.next(True), True)
        self.assertIsNone(c.next(True))
        c.complete(True)
        self.assertIsNone(c.next(None))
        self.assertIs(c.next(True), False)

    def test_failed_action_does_not_change_state(self):
        c = ToggleCommands()
        c.next(True)
        c.complete(False)
        self.assertFalse(c.closed)
        self.assertIs(c.next(True), True)
