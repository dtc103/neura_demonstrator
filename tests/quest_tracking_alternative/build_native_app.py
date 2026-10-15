"""Check/build the native APK. Android SDK/NDK must be installed separately."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from setup_native_tools import setup
from download_tools import download

ROOT = Path(__file__).parent
GRADLE_VERSION = "8.9"


def requirements(sdk):
    paths = (sdk / "platforms/android-34/android.jar", sdk / "build-tools/34.0.0",
             sdk / "ndk/27.2.12479018", sdk / "cmake/3.22.1")
    return [str(path) for path in paths if not path.exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=Path(os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or str(ROOT / ".android-sdk")))
    parser.add_argument("--setup", action="store_true", help="Download SDK tools locally and install prerequisites; prompts for SDK license acceptance")
    parser.add_argument("--check", action="store_true", help="Report prerequisites without downloading/building")
    parser.add_argument("--gradle", type=Path, help="Existing Gradle 8.9 executable; otherwise downloads a verified distribution locally")
    parser.add_argument("--download-timeout", type=float, default=120, help="Network read/connect timeout in seconds for bootstrap downloads (default: 120)")
    parser.add_argument("--download-retries", type=int, default=2, help="Retries after a failed bootstrap download (default: 2)")
    args = parser.parse_args()
    if not math.isfinite(args.download_timeout) or args.download_timeout <= 0:
        parser.error("--download-timeout must be finite and positive")
    if args.download_retries < 0:
        parser.error("--download-retries must be nonnegative")
    sdk = args.sdk.expanduser().resolve()
    if args.setup:
        setup(sdk, args.download_timeout, args.download_retries)
    missing = requirements(sdk)
    if shutil.which("java") is None:
        missing.append("Java 17 or newer")
    if missing:
        print("Missing Android build prerequisites:\n  " + "\n  ".join(missing))
        print("Install Android SDK command-line tools, then run:\n"
              '  sdkmanager --licenses\n'
              '  sdkmanager "platforms;android-34" "build-tools;34.0.0" "ndk;27.2.12479018" "cmake;3.22.1"\n'
              "SDK command-line tools: https://developer.android.com/studio#command-tools")
        return 1
    print(f"SDK prerequisites found: {sdk}")
    if args.check:
        return 0
    gradle = args.gradle
    if gradle is None:
        tools = ROOT / ".build-tools"
        tools.mkdir(exist_ok=True)
        gradle = tools / f"gradle-{GRADLE_VERSION}/bin/gradle"
        if not gradle.exists():
            url = f"https://services.gradle.org/distributions/gradle-{GRADLE_VERSION}-bin.zip"
            expected = download(url + ".sha256", timeout=args.download_timeout, retries=args.download_retries).decode().strip()
            archive = tools / "gradle.zip"
            print(f"Downloading Gradle {GRADLE_VERSION}…", flush=True)
            download(url, archive, timeout=args.download_timeout, retries=args.download_retries)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Gradle checksum mismatch")
            with zipfile.ZipFile(archive) as z:
                for entry in z.infolist():
                    target = (tools / entry.filename).resolve()
                    if not target.is_relative_to(tools.resolve()):
                        raise RuntimeError("Invalid path in Gradle archive")
                z.extractall(tools)
            gradle.chmod(0o755)
            archive.unlink()
    (ROOT / "android/local.properties").write_text(f"sdk.dir={sdk}\n")
    code = subprocess.run([str(gradle.resolve()), "--no-daemon", ":app:assembleDebug"],
                          cwd=ROOT / "android", env={**os.environ, "GRADLE_USER_HOME": str(ROOT / ".gradle-cache")}).returncode
    if code == 0:
        print(f"APK: {ROOT / 'android/app/build/outputs/apk/debug/app-debug.apk'}")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
        raise SystemExit(f"APK build could not complete: {error}") from error
