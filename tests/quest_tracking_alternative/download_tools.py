"""Bounded retries and atomic files for Android/Gradle tooling downloads."""

import io
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def byte_size(size):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024


class Progress:
    """Terminal progress without requiring another package download."""
    def __init__(self, label, total):
        self.label, self.total = label, total
        self.size = 0
        self.started = time.monotonic()
        self.last_update = self.started
        self.terminal = sys.stdout.isatty()

    def update(self, size, final=False, failed=False):
        self.size = size
        now = time.monotonic()
        if not final and now - self.last_update < (.2 if self.terminal else 2):
            return
        self.last_update = now
        elapsed = max(now - self.started, .001)
        speed = size / elapsed
        if self.total:
            fraction = min(size / self.total, 1)
            filled = int(fraction * 28)
            bar = "#" * filled + "-" * (28 - filled)
            eta = max(self.total - size, 0) / speed if speed else None
            text = f"{self.label} [{bar}] {fraction * 100:5.1f}%  {byte_size(size)} / {byte_size(self.total)}"
            if eta is not None:
                text += f"  ETA {eta:.0f}s"
        else:
            text = f"{self.label} {byte_size(size)} (total size unknown)"
        text += f"  {byte_size(speed)}/s"
        if failed:
            text += "  interrupted"
        if self.terminal:
            print("\r" + text + "\033[K", end="\n" if final else "", flush=True)
        else:
            print(text, flush=True)


def download(url, destination=None, timeout=120, retries=2):
    target = None if destination is None else Path(destination)
    partial = None if target is None else target.with_name(target.name + ".part")
    label = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1] or "Download"
    for attempt in range(retries + 1):
        progress = None
        size = 0
        try:
            print(f"Connecting: {label} (attempt {attempt + 1}/{retries + 1}, timeout {timeout:g}s)…", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "NeuraQuestBuild/1.0"})
            output = io.BytesIO() if target is None else partial.open("wb")
            with output:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    headers = getattr(response, "headers", {})
                    try:
                        total = int(headers.get("Content-Length", "0")) or None
                    except (ValueError, TypeError):
                        total = None
                    if total is not None and total < 0:
                        total = None
                    progress = Progress(label, total)
                    read_chunk = getattr(response, "read1", response.read)
                    while chunk := read_chunk(64 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                        progress.update(size)
                    if total is not None and size != total:
                        raise OSError(f"Incomplete download: received {size} of {total} bytes")
                data = output.getvalue() if target is None else None
            if size == 0:
                raise OSError("Server returned an empty download")
            if target is not None:
                partial.replace(target)
            progress.update(size, final=True)
            return data
        except (OSError, urllib.error.URLError) as error:
            if progress is not None:
                progress.update(size, final=True, failed=True)
            if partial is not None:
                partial.unlink(missing_ok=True)
            # A bad URL/permission response is not fixed by waiting longer.
            if isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500 and error.code != 429:
                raise
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries + 1} attempts: {url}: {error}. "
                                   "Check access to this address from your PC or the proxy/VPN configuration.") from error
            pause = min(attempt + 1, 5)
            print(f"Download attempt {attempt + 1} failed: {error}. Retrying after {pause}s "
                  f"(timeout {timeout:g}s).", flush=True)
            time.sleep(pause)
