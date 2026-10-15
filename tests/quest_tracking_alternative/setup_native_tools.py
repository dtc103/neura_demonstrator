"""Download Android command-line tools and install the pinned APK toolchain."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from download_tools import download

REPOSITORY = "https://dl.google.com/android/repository/"


def child(node, name):
    return next((item for item in node if item.tag.rsplit("}", 1)[-1] == name), None)


def text(node, name, default=""):
    item = child(node, name)
    return default if item is None else item.text or default


def setup(sdk, timeout=120, retries=2):
    sdk = Path(sdk).resolve()
    manager = sdk / "cmdline-tools/latest/bin/sdkmanager"
    if not manager.exists():
        print("Reading Google's Android SDK repository…", flush=True)
        root = ET.fromstring(download(REPOSITORY + "repository2-1.xml", timeout=timeout, retries=retries))
        candidates = []
        for package in root:
            if not package.attrib.get("path", "").startswith("cmdline-tools;"):
                continue
            channel = child(package, "channelRef")
            if channel is not None and channel.attrib.get("ref") != "channel-0":
                continue
            revision = child(package, "revision")
            if revision is None:
                continue
            version = tuple(int(text(revision, name, "0")) for name in ("major", "minor", "micro"))
            archives = child(package, "archives")
            if archives is None:
                continue
            for archive in archives:
                if text(archive, "host-os") == "linux":
                    complete = child(archive, "complete")
                    if complete is not None:
                        candidates.append((version, complete))
        if not candidates:
            raise RuntimeError("No stable Linux Android command-line tools found in Google's repository")
        complete = max(candidates, key=lambda entry: entry[0])[1]
        url = text(complete, "url")
        if not url or "/" in url or ".." in url:
            raise RuntimeError("Unexpected Android tools archive URL")
        checksum = child(complete, "checksum")
        algorithm = checksum.attrib.get("type", "sha1") if checksum is not None else ""
        if algorithm not in ("sha1", "sha256") or checksum is None:
            raise RuntimeError("Android tools checksum is unavailable")
        sdk.mkdir(parents=True, exist_ok=True)
        archive = sdk / "command-tools.zip"
        print(f"Downloading {url}…", flush=True)
        download(REPOSITORY + url, archive, timeout=timeout, retries=retries)
        digest = hashlib.new(algorithm, archive.read_bytes()).hexdigest()
        if digest != checksum.text.strip():
            raise RuntimeError("Android command-line tools checksum mismatch")
        staging = sdk / ".command-tools-unpack"
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if not (staging / entry.filename).resolve().is_relative_to(staging):
                    raise RuntimeError("Invalid path in Android tools archive")
            z.extractall(staging)
        destination = sdk / "cmdline-tools/latest"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Incomplete command-line tools already exist at {destination}; inspect before replacing")
        (staging / "cmdline-tools").rename(destination)
        for executable in (destination / "bin").iterdir():
            if executable.is_file(): executable.chmod(0o755)
        archive.unlink()
    print("Android SDK licenses require your review and acceptance below.", flush=True)
    subprocess.run([str(manager), f"--sdk_root={sdk}", "--licenses"], check=True)
    subprocess.run([str(manager), f"--sdk_root={sdk}", "platforms;android-34", "build-tools;34.0.0",
                    "ndk;27.2.12479018", "cmake;3.22.1"], check=True)
