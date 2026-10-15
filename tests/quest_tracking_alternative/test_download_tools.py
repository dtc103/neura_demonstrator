import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download_tools import download


class DownloadTests(unittest.TestCase):
    def test_timeout_retries_and_successful_payload(self):
        with patch("download_tools.urllib.request.urlopen", side_effect=[TimeoutError("read timed out"), io.BytesIO(b"success")]) as request, \
             patch("download_tools.time.sleep") as sleep:
            self.assertEqual(download("https://example.com/file", timeout=120, retries=1), b"success")
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args.kwargs["timeout"], 120)
            sleep.assert_called_once_with(1)

    def test_failed_download_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "archive.zip"
            target.write_bytes(b"original")
            with patch("download_tools.urllib.request.urlopen", side_effect=TimeoutError("timeout")):
                with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                    download("https://example.com/file", target, retries=0)
            self.assertEqual(target.read_bytes(), b"original")
            self.assertFalse(target.with_name("archive.zip.part").exists())

    def test_successful_file_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "file.zip"
            with patch("download_tools.urllib.request.urlopen", return_value=io.BytesIO(b"archive")):
                download("https://example.com/file", target)
            self.assertEqual(target.read_bytes(), b"archive")


if __name__ == "__main__":
    unittest.main()
