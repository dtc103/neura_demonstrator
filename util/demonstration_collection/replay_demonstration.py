#!/usr/bin/env python3
"""Browse saved demonstrations. --execute unlocks the explicit live-replay checkbox."""
import argparse
import signal
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2]/'datasets/demonstrations')
    parser.add_argument('--episode', type=Path, help='Episode folder to select initially')
    parser.add_argument('--execute', action='store_true', help='Enable opt-in live robot replay controls')
    args = parser.parse_args()
    from run_collection import ensure_qt_runtime, configure_qt_plugins
    ensure_qt_runtime()
    from PySide6 import QtWidgets
    configure_qt_plugins()
    from replay_window import ReplayWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    root = args.episode.resolve().parent.parent if args.episode else args.root
    window = ReplayWindow(root, args.execute, episode=args.episode)
    window.show()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
