import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import configure_collision as cli


class CollisionTests(unittest.TestCase):
    def test_array_validation(self):
        self.assertEqual(cli.parse_values('[1, 2, 3]', 3), [1, 2, 3])
        for value in ('1 2', '1 nan 3', '1 inf 3', '1 -2 3', '1 0 3'):
            with self.assertRaises(ValueError):
                cli.parse_values(value, 3)
        values = cli.defaults()
        cli.validate(values)
        values['lower_force_thresholds_nominal'][0] = 21
        with self.assertRaises(ValueError):
            cli.validate(values)

    def test_edit_and_cancel(self):
        answers = iter(['', '21 21 19 19 17 15 13'] + [''] * 6)
        result = cli.edit(cli.defaults(), 'default', read=lambda _: next(answers), write=lambda _: None)
        self.assertEqual(result['upper_torque_thresholds_nominal'][0], 21)
        self.assertIsNone(cli.edit(cli.defaults(), 'default', read=lambda _: 'q', write=lambda _: None))

    def test_cache_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            self.assertIn('active value unknown', cli.load_baseline(path, cli.SERVICE)[1])
            cli.save_applied(path, cli.defaults(), cli.SERVICE)
            values, label = cli.load_baseline(path, cli.SERVICE)
            self.assertEqual(values, cli.defaults())
            self.assertIn('not live readback', label)
            with self.assertRaises(ValueError):
                cli.load_baseline(path, '/other')

    def test_main_never_caches_failed_application(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            with patch.object(cli, 'edit', return_value=cli.defaults()), \
                    patch('builtins.input', return_value='APPLY'), \
                    patch.object(cli, 'apply_ros', side_effect=RuntimeError('failed')) as apply, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--state-file', str(path)]), 1)
                apply.assert_called_once()
                self.assertFalse(path.exists())

    def test_dry_run_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            with patch.object(cli, 'edit', return_value=cli.defaults()), \
                    patch('builtins.input', return_value='APPLY'), \
                    patch.object(cli, 'apply_ros') as apply, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--state-file', str(path), '--dry-run']), 0)
                apply.assert_not_called()
                self.assertFalse(path.exists())
                self.assertEqual(cli.main(['--state-file', str(path)]), 0)
                apply.assert_called_once()
                self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
