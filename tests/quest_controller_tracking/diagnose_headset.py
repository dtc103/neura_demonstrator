"""Read installed Quest APK identity and compare it with the local reader APK."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Optional ADB device serial")
    parser.add_argument("--package", default="com.rail.oculus.teleop")
    parser.add_argument("--output", type=Path, default=ROOT / "headset_diagnostics.json")
    args = parser.parse_args()
    adb = shutil.which("adb")
    if adb is None:
        parser.error("ADB is not installed")
    prefix = [adb] + (["-s", args.serial] if args.serial else [])

    def query(*command):
        result = subprocess.run([*prefix, *command], check=True, text=True, capture_output=True, timeout=15)
        return result.stdout.strip()

    try:
        report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "package": args.package,
                  "devices": query("devices", "-l"), "installed_apks": [], "local_apks": []}
        paths = query("shell", "pm", "path", args.package)
        package = query("shell", "dumpsys", "package", args.package)
        report["version_details"] = [line.strip() for line in package.splitlines()
                                     if any(key in line for key in ("versionCode=", "versionName=", "firstInstallTime=", "lastUpdateTime="))]
        for line in paths.splitlines():
            if line.startswith("package:"):
                path = line[len("package:"):].strip()
                # Restrict the shell argument to normal Android package paths.
                if not re.fullmatch(r"/[A-Za-z0-9_./+=~-]+", path):
                    raise ValueError(f"Unexpected APK path: {path}")
                digest = query("shell", "sha256sum", path).split()[0]
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Could not read the installed APK hash")
                report["installed_apks"].append({"path": path, "sha256": digest})
        if not report["installed_apks"]:
            raise ValueError("App is not installed under that package name")
        installed_hashes = {entry["sha256"] for entry in report["installed_apks"]}
        folder = ROOT / "vendor/meta_quest_teleop/meta_quest_teleop/APK"
        for apk in folder.glob("*.apk"):
            digest = hashlib.sha256(apk.read_bytes()).hexdigest()
            report["local_apks"].append({"name": apk.name, "sha256": digest, "matches_installed": digest in installed_hashes})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report["devices"])
        print("\n".join(report["version_details"]))
        for entry in report["installed_apks"]:
            print(f"Installed APK SHA256: {entry['sha256']}")
        for entry in report["local_apks"]:
            print(f"{entry['name']}: {'MATCHES installed APK' if entry['matches_installed'] else 'different from installed APK'}")
        print(f"Report: {args.output}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Headset diagnosis failed: {error}") from error


if __name__ == "__main__":
    main()
