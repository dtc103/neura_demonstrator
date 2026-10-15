import copy
import subprocess
import unittest
from unittest.mock import patch

from native_reader import NativeReader, PACKAGE
from webxr_reader import TrackingState
from test_alternative import frame


def native_frame():
    data = frame()
    data.update(schema="neura.quest.controllers.v1", reference_space="stage", sample_time_ns=1000)
    return data


class NativeTests(unittest.TestCase):
    def test_native_controller_validation_rejects_head_space_and_repeated_samples(self):
        reader = NativeReader.__new__(NativeReader)
        reader.state = TrackingState(allowed_spaces=("stage", "local"))
        reader.accept_frame(native_frame())
        self.assertTrue(reader.get_button_state("A"))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            reader.accept_frame(native_frame())
        bad = native_frame(); bad.update(sequence=2, sample_time_ns=2000, reference_space="viewer")
        with self.assertRaisesRegex(ValueError, "reference"):
            reader.accept_frame(bad)

    def test_startup_applies_keep_awake_before_launch_and_restores_on_exit(self):
        for keep_awake in (True, False):
            with self.subTest(keep_awake=keep_awake):
                commands = []
                def command(instance, *args, **kwargs):
                    commands.append(args)
                    return "package:/data/app/base.apk" if args[:3] == ("shell", "pm", "path") else ""
                def receive(instance):
                    instance.accept_frame(native_frame())
                with patch("native_reader.shutil.which", return_value="/usr/bin/adb"), \
                     patch.object(NativeReader, "command", command), patch.object(NativeReader, "receive", receive):
                    reader = NativeReader(keep_awake=keep_awake)
                    reader.stop()
                prox = ("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close")
                restore = ("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable")
                if keep_awake:
                    self.assertIn(prox, commands); self.assertIn(restore, commands)
                    launch = next(command for command in commands if command[:3] == ("shell", "am", "start"))
                    self.assertLess(commands.index(prox), commands.index(launch))
                else:
                    self.assertNotIn(prox, commands); self.assertNotIn(restore, commands)
                self.assertIn(("shell", "am", "force-stop", PACKAGE), commands)
                self.assertIn(("forward", "--remove", "tcp:9201"), commands)

    def test_startup_failure_restores_proximity_and_forwarding(self):
        commands = []
        def command(instance, *args, **kwargs):
            commands.append(args)
            if args[:3] == ("shell", "pm", "path"): return "package:/base.apk"
            if args[:3] == ("shell", "am", "start"): raise RuntimeError("launch failed")
            return ""
        with patch("native_reader.shutil.which", return_value="/usr/bin/adb"), patch.object(NativeReader, "command", command):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                NativeReader()
        self.assertIn(("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable"), commands)
        self.assertIn(("forward", "--remove", "tcp:9201"), commands)


if __name__ == "__main__":
    unittest.main()
