#!/usr/bin/env python3
"""Desktop demonstration recorder. Same teleoperation options; preview by default."""
import sys
import os
import signal
import subprocess
import threading
import queue
import traceback
from pathlib import Path
from teleoperate import arguments, run, WebXRReader
from collection_control import Collection
from camera_reset import reset_cameras


def setup_quest(port):
    """Best-effort USB setup; missing/disconnected headsets must not block the GUI."""
    script = Path(__file__).with_name('setup_quest.sh')
    try:
        result = subprocess.run(['bash', str(script), str(port)],
                                capture_output=True, text=True, timeout=25)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
        if result.returncode:
            print(f'Quest USB setup failed: {result.stderr.strip()}\n'
                  f'Connect/authorize the headset, then retry: bash "{script}" {port}',
                  file=sys.stderr, flush=True)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f'Quest USB setup could not complete: {error}', file=sys.stderr, flush=True)


def ensure_qt_runtime():
    """Restart before Qt import so pip PySide and Pixi Qt never mix libraries."""
    import importlib.util
    spec = importlib.util.find_spec('PySide6')
    if sys.platform != 'linux' or spec is None or not spec.submodule_search_locations:
        return
    qt_lib = Path(next(iter(spec.submodule_search_locations))) / 'Qt/lib'
    if not (qt_lib / 'libQt6Core.so.6').is_file():
        return  # Non-wheel installations use their environment's own Qt.
    current = os.environ.get('LD_LIBRARY_PATH', '')
    if current.split(':')[0] == str(qt_lib):
        return
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(qt_lib) + (':' + current if current else '')
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def configure_qt_plugins():
    """Use the whole PySide plugin tree, including Wayland shell integrations."""
    import PySide6
    from PySide6.QtCore import QCoreApplication
    plugins = Path(PySide6.__file__).parent / 'Qt/plugins'
    if plugins.is_dir():
        os.environ['QT_PLUGIN_PATH'] = str(plugins)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins / 'platforms')
        QCoreApplication.setLibraryPaths([str(plugins)])


def main():
    args = arguments(collection=True)
    import importlib.util
    missing = [name for name in ('PySide6', 'pyrealsense2', 'cv2')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Missing collection dependencies: ' + ', '.join(missing) + '; see COLLECTION.md')
    ensure_qt_runtime()
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise SystemExit(f'Cannot load PySide6 with {sys.executable}: {error}') from error
    # Pixi's qt.conf can select Qt 5/older Qt 6 platform plugins. Use the
    # plugins belonging to the PySide6 runtime in this process only.
    configure_qt_plugins()
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    try:
        reader = WebXRReader(port=args.port)
    except OSError as error:
        raise SystemExit(f'Cannot start Quest server on port {args.port}: {error}') from error
    reader.state.stale_seconds = args.stale_seconds

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.collection = None
            self.worker = None
            self.replay_dialog = None
            self.reset_events = queue.SimpleQueue()
            self.reset_cancel = threading.Event()
            self.reset_phase = 'resetting'
            self.setWindowTitle('Panda demonstration collection')
            self.resize(1240, 900)
            self.setMinimumSize(960, 850)
            self.setStyleSheet("""
                QWidget { background: #101720; color: #e7edf5; font-family: 'DejaVu Sans'; font-size: 13px; }
                QFrame#card { background: #192330; border: 1px solid #2b394a; border-radius: 14px; }
                QFrame#card QLabel { background: transparent; border: none; }
                QLabel#title { font-size: 27px; font-weight: 700; }
                QLabel#muted { color: #9aadc2; }
                QLabel#section { font-size: 15px; font-weight: 700; }
                QLabel#badge { background: #24394b; color: #8bd9f3; border-radius: 9px; padding: 7px 13px; font-weight: 700; }
                QLabel#metric { font-size: 26px; font-weight: 700; color: #79dfc1; }
                QLineEdit { min-height: 20px; background: #101923; border: 1px solid #35465b; border-radius: 8px; padding: 8px; selection-background-color: #347e94; }
                QLineEdit:focus { border: 1px solid #65cbb9; }
                QLineEdit:disabled { color: #8392a6; background: #18212d; }
                QPushButton { min-height: 20px; background: #26364a; border: 1px solid #3b4d64; border-radius: 8px; padding: 8px 16px; font-weight: 600; }
                QPushButton:hover { background: #344963; border-color: #6784a4; }
                QPushButton:pressed { background: #1b293b; }
                QPushButton:disabled { background: #202a38; color: #708096; border-color: #293648; }
                QPushButton#primary { background: #6ad6bd; color: #10251f; border-color: #6ad6bd; }
                QPushButton#primary:hover { background: #8ae8d2; }
                QPushButton#primary:disabled { background: #294940; color: #8bafa5; border-color: #34584e; }
                QPushButton#danger { color: #ffb6b4; border-color: #73484d; background: #35272f; }
                QPushButton#danger:hover { background: #51313a; }
                QFrame#card QLabel#preview { background: #0c121b; border: 1px solid #29394b; border-radius: 8px; color: #71859d; }
            """)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(26, 22, 26, 22)
            layout.setSpacing(18)
            menu = QtWidgets.QMenuBar(self)
            replay_action = menu.addMenu('Demonstrations').addAction('Replay recordings…')
            replay_action.triggered.connect(self.open_replay)
            layout.setMenuBar(menu)
            header = QtWidgets.QHBoxLayout()
            titles = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel('Demonstration studio')
            title.setObjectName('title')
            subtitle = QtWidgets.QLabel('Panda · Quest teleoperation · RGB + depth')
            subtitle.setObjectName('muted')
            titles.addWidget(title)
            titles.addWidget(subtitle)
            header.addLayout(titles)
            header.addStretch()
            mode = QtWidgets.QLabel('LIVE CONTROL' if args.execute else 'PREVIEW · NO MOTION')
            mode.setObjectName('badge')
            header.addWidget(mode, 0, QtCore.Qt.AlignVCenter)
            layout.addLayout(header)

            def card():
                widget = QtWidgets.QFrame()
                widget.setObjectName('card')
                box = QtWidgets.QVBoxLayout(widget)
                box.setContentsMargins(18, 16, 18, 16)
                box.setSpacing(10)
                return widget, box

            setup, box = card()
            setup.setMinimumHeight(265)
            label = QtWidgets.QLabel('01  Set up your task')
            label.setObjectName('section')
            box.addWidget(label)
            self.session_name = QtWidgets.QLineEdit(args.session_name)
            self.session_name.setAccessibleName('Session name')
            self.session_name.setPlaceholderText('Session name (optional), e.g. pbluegyellow')
            self.session_name.setToolTip('Used as the session folder name. Repeated names get _002, _003, etc. Leave blank for a timestamp.')
            box.addWidget(self.session_name)
            self.prompt = QtWidgets.QLineEdit()
            self.prompt.setAccessibleName('Task prompt')
            self.prompt.setPlaceholderText('Task descriptions separated by ;  e.g. Place the gear on the plate; Move the gear onto the plate')
            self.prompt.setToolTip('Semicolons separate alternative descriptions for every episode in this session. Whitespace is trimmed; empty entries are ignored.')
            box.addWidget(self.prompt)
            self.sequence = QtWidgets.QLineEdit(str(args.reset_sequence or ''))
            self.sequence.setPlaceholderText('Choose a taught reset sequence')
            self.output = QtWidgets.QLineEdit(str(args.output))
            self.browse_buttons = []
            for title, field, choose in [('Reset sequence', self.sequence, self.choose_sequence),
                                         ('Save recordings to', self.output, self.choose_output)]:
                row = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel(title)
                label.setMinimumWidth(135)
                label.setObjectName('muted')
                row.addWidget(label)
                field.setAccessibleName(title)
                row.addWidget(field, 1)
                button = QtWidgets.QPushButton('Browse…')
                button.clicked.connect(choose)
                self.browse_buttons.append(button)
                row.addWidget(button)
                box.addLayout(row)
            footer = QtWidgets.QHBoxLayout()
            footer.addStretch()
            self.prepare = QtWidgets.QPushButton('Prepare session  →')
            self.prepare.setObjectName('primary')
            self.prepare.clicked.connect(self.start)
            self.prepare.setEnabled(False)
            footer.addWidget(self.prepare)
            box.addLayout(footer)
            layout.addWidget(setup)

            state_card, box = card()
            row = QtWidgets.QHBoxLayout()
            self.phase_badge = QtWidgets.QLabel('NOT STARTED')
            self.phase_badge.setObjectName('badge')
            row.addWidget(self.phase_badge)
            row.addStretch()
            caption = QtWidgets.QLabel('Saved episodes')
            caption.setObjectName('muted')
            row.addWidget(caption)
            self.episode_count = QtWidgets.QLabel('0')
            self.episode_count.setObjectName('metric')
            row.addWidget(self.episode_count)
            box.addLayout(row)
            self.status = QtWidgets.QLabel('Enter a task prompt and select your reset sequence to begin.')
            self.status.setWordWrap(True)
            box.addWidget(self.status)
            self.location = QtWidgets.QLabel('Each run is saved as a separate episode.')
            self.location.setObjectName('muted')
            self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.location.setWordWrap(True)
            box.addWidget(self.location)
            row = QtWidgets.QHBoxLayout()
            self.action_buttons = {}
            for label, command in [('A  ·  Reset to start', 'A'), ('X  ·  Unsuccessful', 'X'),
                                   ('Y  ·  Interrupted', 'Y'), ('Close session', 'shutdown')]:
                button = QtWidgets.QPushButton(label)
                button.setObjectName('primary' if command == 'A' else 'danger' if command == 'shutdown' else '')
                button.clicked.connect(lambda checked=False, c=command: self.command(c))
                button.setEnabled(False)
                self.action_buttons[command] = button
                row.addWidget(button)
            box.addLayout(row)
            layout.addWidget(state_card)

            label = QtWidgets.QLabel('02  Camera views')
            label.setObjectName('section')
            layout.addWidget(label)
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(14)
            self.images = {}
            self.camera_status = {}
            for role, name in [('wrist', 'Wrist'), ('side', 'Side'), ('opposite', 'Opposite')]:
                panel, box = card()
                heading = QtWidgets.QHBoxLayout()
                title = QtWidgets.QLabel(name)
                title.setObjectName('section')
                health = QtWidgets.QLabel('Waiting')
                health.setObjectName('muted')
                self.camera_status[role] = health
                heading.addWidget(title)
                heading.addStretch()
                heading.addWidget(health)
                box.addLayout(heading)
                label = QtWidgets.QLabel('Camera preview\nAvailable after preparation')
                label.setObjectName('preview')
                label.setMinimumSize(220, 150)
                label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
                label.setAlignment(QtCore.Qt.AlignCenter)
                self.images[role] = label
                box.addWidget(label, 1)
                row.addWidget(panel, 1)
            layout.addLayout(row, 1)
            guide = QtWidgets.QLabel('A reset  →  Thumbstick reference  →  A record  →  A finish\n'
                'Grip: move   ·   Trigger: open / close   ·   Hold B: precision   ·   Clutch pauses stay recorded')
            guide.setObjectName('muted')
            guide.setAlignment(QtCore.Qt.AlignCenter)
            guide.setWordWrap(True)
            layout.addWidget(guide)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(150)

        def start_camera_reset(self):
            self.reset_phase = 'resetting'
            self.prepare.setEnabled(False)
            self.prepare.setText('Restarting cameras…')
            self.status.setText('Restarting cameras in the background. You can prepare the Quest and task fields.')
            def report(phase, message):
                print(message, flush=True)
                self.reset_events.put((phase, message))
            threading.Thread(target=reset_cameras,
                             args=(args.camera_config, self.reset_cancel, report),
                             name='camera-startup-reset', daemon=True).start()

        def open_replay(self):
            if self.worker and self.worker.is_alive():
                QtWidgets.QMessageBox.information(self, 'Close collection first',
                    'Close the collection session before opening replay. The Quest server will stay running.')
                return
            from replay_window import ReplayWindow
            dialog = ReplayWindow(Path(self.output.text()).expanduser(), args.execute, self)
            self.replay_dialog = dialog
            try:
                dialog.exec()
            finally:
                self.replay_dialog = None

        def choose_sequence(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Reset sequence', '', 'JSON (*.json)')
            if path:
                self.sequence.setText(path)

        def choose_output(self):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Dataset directory', self.output.text())
            if path:
                self.output.setText(path)

        def command(self, value):
            if self.collection and self.worker and self.worker.is_alive():
                self.collection.commands.put(value)

        def start(self):
            if self.reset_phase == 'failed':
                self.start_camera_reset()
                return
            if self.reset_phase != 'ready':
                return
            if self.worker and self.worker.is_alive():
                return
            args.output = Path(self.output.text()).expanduser()
            args.session_name = self.session_name.text()
            try:
                self.collection = Collection(args, self.prompt.text(), Path(self.sequence.text()).expanduser())
            except Exception as error:
                self.status.setText(str(error))
                return
            self.prepare.setEnabled(False)
            for widget in (self.session_name, self.prompt, self.sequence, self.output, *self.browse_buttons):
                widget.setEnabled(False)
            def work():
                try:
                    run(args, collection=self.collection, shared_reader=reader)
                except BaseException as error:
                    self.collection.detail = str(error)
                    self.collection.phase = 'fault'
                    traceback.print_exc()
            self.worker = threading.Thread(target=work, daemon=False)
            self.worker.start()

        def refresh(self):
            while not self.reset_events.empty():
                self.reset_phase, message = self.reset_events.get()
                self.status.setText(message)
                self.phase_badge.setText({'resetting': 'CAMERAS RESTARTING',
                                         'ready': 'NOT STARTED', 'failed': 'CAMERA RESET FAILED'}[self.reset_phase])
                self.prepare.setText({'resetting': 'Restarting cameras…',
                                      'ready': 'Prepare session  →', 'failed': 'Retry camera reset'}[self.reset_phase])
                self.prepare.setEnabled(self.reset_phase != 'resetting')
            c = self.collection
            if c is None:
                return
            count = 0 if c.recorder is None else len(c.recorder.finished)
            location = '' if c.recorder is None else str(c.recorder.root)
            self.phase_badge.setText(c.phase.replace('_', ' ').upper())
            color = '#6ad6bd' if c.phase == 'recording' else '#ffb6b4' if c.phase == 'fault' else '#8bd9f3'
            self.phase_badge.setStyleSheet(f'color: {color};')
            self.episode_count.setText(str(count))
            self.status.setText(c.detail)
            self.location.setText(location or 'Preparing session…')
            alive = self.worker is not None and self.worker.is_alive()
            self.action_buttons['A'].setText({
                'needs_reset': 'A  ·  Reset to start', 'ready': 'A  ·  Start recording',
                'recording': 'A  ·  Finish successfully', 'reference': 'Set thumbstick reference',
                'resetting': 'Returning to start…', 'stopping': 'Stopping…',
                'flushing': 'Saving episode…'}.get(c.phase, 'A  ·  Waiting'))
            self.action_buttons['A'].setEnabled(alive and c.phase in ('needs_reset', 'ready', 'recording'))
            for command in ('X', 'Y'):
                self.action_buttons[command].setEnabled(alive and c.phase == 'recording')
            self.action_buttons['shutdown'].setEnabled(alive)
            import time
            for role, label in self.camera_status.items():
                received = None if c.cameras is None else c.cameras.received.get(role)
                age = None if received is None else time.monotonic()-received
                label.setText('Waiting' if age is None else f'{age:.1f}s old' if age > 1.5 else '● Live')
            if self.worker and not self.worker.is_alive():
                self.prepare.setEnabled(True)
                for widget in (self.session_name, self.prompt, self.sequence, self.output, *self.browse_buttons):
                    widget.setEnabled(True)
            if c.cameras:
                for role, bgr in list(c.cameras.latest.items()):
                    rgb = bgr[:, :, ::-1].copy()
                    h, w, _ = rgb.shape
                    image = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                    self.images[role].setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                        self.images[role].size(), QtCore.Qt.KeepAspectRatio))

        def closeEvent(self, event):
            if self.replay_dialog is not None and self.replay_dialog.busy():
                self.replay_dialog.stop_playback()
                event.ignore()
            elif self.worker and self.worker.is_alive():
                self.command('shutdown')
                self.status.setText('Stopping control and flushing recordings. Close again once finished.')
                event.ignore()
            else:
                self.reset_cancel.set()
                event.accept()

    try:
        window = Window()
        window.show()
        window.start_camera_reset()
        threading.Thread(target=setup_quest, args=(args.port,), daemon=True).start()
        signal.signal(signal.SIGINT, lambda *_: window.close())
        signal.signal(signal.SIGTERM, lambda *_: window.close())
        return app.exec()
    finally:
        if 'window' in locals():
            window.reset_cancel.set()
        reader.stop()


if __name__ == '__main__':
    sys.exit(main())
