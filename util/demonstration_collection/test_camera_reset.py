import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from camera_reset import reset_cameras


class ResetTests(unittest.TestCase):
    def run_reset(self, batches, cancel=None, fail=False):
        devices = {}
        for serial in ('1', '2', '3', 'unrelated'):
            device = Mock()
            device.get_info.return_value = serial
            devices[serial] = device
        if fail:
            devices['1'].hardware_reset.side_effect = RuntimeError('busy')
        context = Mock()
        context.query_devices.side_effect = [[devices[s] for s in batch] for batch in batches]
        sdk = SimpleNamespace(context=lambda: context,
                              camera_info=SimpleNamespace(serial_number=0))
        events = []
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({'cameras': dict(wrist='1', side='2', opposite='3')}))
            reset_cameras(config, cancel or threading.Event(), lambda *args: events.append(args),
                          sdk=sdk, reboot_seconds=0, timeout=0)
        return devices, events

    def test_only_configured_cameras_reset_and_reconnect(self):
        devices, events = self.run_reset([['1', '2', '3', 'unrelated'], ['1', '2', '3']])
        for serial in ('1', '2', '3'):
            devices[serial].hardware_reset.assert_called_once()
        devices['unrelated'].hardware_reset.assert_not_called()
        self.assertEqual(events[-1][0], 'ready')

    def test_missing_camera_prevents_partial_reset(self):
        devices, events = self.run_reset([['1', '2']])
        self.assertEqual(events[-1][0], 'failed')
        for device in devices.values():
            device.hardware_reset.assert_not_called()

    def test_timeout_and_sdk_failure_report_failure(self):
        for batches, fail in (([['1', '2', '3'], ['1']], False), ([['1', '2', '3']], True)):
            _, events = self.run_reset(batches, fail=fail)
            self.assertEqual(events[-1][0], 'failed')

    def test_cancel_prevents_reset(self):
        cancel = threading.Event()
        cancel.set()
        devices, events = self.run_reset([['1', '2', '3']], cancel=cancel)
        self.assertEqual(events, [])
        for device in devices.values():
            device.hardware_reset.assert_not_called()
