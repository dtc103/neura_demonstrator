from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import track_controllers as tracking


class TrackingTests(unittest.TestCase):
    def test_position_format_and_missing_tracking(self):
        transform = [[1, 0, 0, .1], [0, 1, 0, -.2], [0, 0, 1, .3], [0, 0, 0, 1]]
        line = tracking.format_pose("right", transform, True)
        self.assertIn("x=+0.1000 m", line)
        self.assertIn("y=-0.2000 m", line)
        self.assertIn("z=+0.3000 m", line)
        self.assertIn("roll=+0.00 deg", line)
        self.assertIn("pitch=+0.00 deg", line)
        self.assertIn("yaw=+0.00 deg", line)
        self.assertIn("no tracking", tracking.format_pose("left", None))

    def test_quarter_turn_about_z_is_yaw(self):
        transform = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        line = tracking.format_pose("right", transform, True)
        self.assertIn("roll=+0.00 deg", line)
        self.assertIn("pitch=+0.00 deg", line)
        self.assertIn("yaw=+90.00 deg", line)

    def test_rejects_git_lfs_pointer_before_headset_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "APK").mkdir()
            (root / "APK/teleop-pointer-frame-relative.apk").write_text("version https://git-lfs.github.com/spec/v1\n")
            with patch.object(tracking.importlib.util, "find_spec", return_value=SimpleNamespace(submodule_search_locations=[str(root)])):
                with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
                    tracking.check_apk()

    def test_listing_only_runs_adb_and_does_not_load_reader(self):
        with patch.object(tracking.shutil, "which", return_value="/usr/bin/adb"), \
             patch.object(tracking.subprocess, "run") as run, patch.object(tracking, "check_apk") as apk:
            tracking.run(SimpleNamespace(list=True))
        apk.assert_not_called()
        self.assertEqual(run.call_args.args[0], ["/usr/bin/adb", "devices", "-l"])

    def test_bad_rate_never_starts_tracking(self):
        with patch.object(sys, "argv", ["track_controllers.py", "--rate", "nan"]), \
             patch.object(tracking, "run") as run:
            with self.assertRaises(SystemExit):
                tracking.main()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
