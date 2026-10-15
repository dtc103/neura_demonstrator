#!/usr/bin/env bash
# Configure the USB-connected Quest for the collection GUI.
set -euo pipefail

port="${1:-8443}"
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Expected a TCP port between 1 and 65535" >&2
    exit 2
fi
command -v adb >/dev/null || { echo "adb is not installed or is not on PATH" >&2; exit 1; }

# Bound each command so an unavailable headset cannot hang startup.
# ANDROID_SERIAL can select a device when several are attached.
timeout 10s adb reverse "tcp:$port" "tcp:$port"
timeout 10s adb shell am broadcast -a com.oculus.vrpowermanager.prox_close
echo "Quest ready: USB port $port forwarded; proximity override requested."
