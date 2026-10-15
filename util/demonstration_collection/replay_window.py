"""Episode browser and camera playback. Live ROS control is explicitly opt-in."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from PySide6 import QtCore, QtGui, QtWidgets
from replay_data import Episode, list_episodes
from replay_robot import LiveReplay


class SpeedBox(QtWidgets.QDoubleSpinBox):
    """Keep the editor visible with Pixi/PySide styles and external step buttons."""
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.lineEdit().setGeometry(self.rect().adjusted(10, 3, -10, -3))

    def showEvent(self, event):
        super().showEvent(event)
        self.lineEdit().setGeometry(self.rect().adjusted(10, 3, -10, -3))


class ReplayWindow(QtWidgets.QDialog):
    def __init__(self, root, allow_live=False, parent=None, episode=None):
        super().__init__(parent)
        self.setWindowTitle('Demonstration replay')
        self.resize(1240, 800)
        self.setMinimumSize(900, 650)
        self.setModal(True)
        self.episode = None
        self.live = None
        self.playing = False
        self.position = 0.
        self.started = 0.
        self.preview_start = 0.
        self.preview_speed = 1.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='replay_images')
        self.future = None
        self.generation = 0
        self.last_frames = {}
        self.setStyleSheet('''
            QWidget { background: #101720; color: #e7edf5; font-size: 13px; }
            QLabel#title { font-size: 25px; font-weight: 700; }
            QLabel#muted { color: #9aadc2; }
            QLineEdit, QTreeWidget, QDoubleSpinBox { background: #192330; border: 1px solid #35465b; border-radius: 7px; padding: 7px; }
            QPushButton { background: #26364a; border: 1px solid #3b4d64; border-radius: 8px; padding: 10px 16px; }
            QPushButton:hover { background: #344963; }
            QPushButton:disabled { color: #708096; }
            QPushButton#start { background: #6ad6bd; color: #10251f; font-weight: 700; }
            QPushButton#stop { background: #51313a; color: #ffb6b4; }
            QLabel#camera { background: #0c121b; border: 1px solid #29394b; border-radius: 10px; }
            QHeaderView::section { background: #26364a; padding: 7px; }
        ''')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        title = QtWidgets.QLabel('Replay your demonstrations')
        title.setObjectName('title')
        layout.addWidget(title)
        row = QtWidgets.QHBoxLayout()
        self.root = QtWidgets.QLineEdit(str(root))
        row.addWidget(self.root, 1)
        self.browse = QtWidgets.QPushButton('Dataset folder…')
        self.browse.clicked.connect(self.choose_root)
        row.addWidget(self.browse)
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh_list)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['Task prompt', 'Session / episode', 'Outcome', 'Duration'])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(130)
        self.tree.setMaximumHeight(230)
        self.tree.itemSelectionChanged.connect(self.select_episode)
        layout.addWidget(self.tree)
        self.details = QtWidgets.QLabel('Select an episode. Camera playback does not require the robot or Quest.')
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        cameras = QtWidgets.QHBoxLayout()
        self.images = {}
        for role in ('wrist', 'side', 'opposite'):
            box = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(role.title() + ' · recorded')
            box.addWidget(label)
            image = QtWidgets.QLabel('No episode selected')
            image.setObjectName('camera')
            image.setAlignment(QtCore.Qt.AlignCenter)
            image.setMinimumSize(200, 160)
            image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
            box.addWidget(image, 1)
            self.images[role] = image
            cameras.addLayout(box, 1)
        layout.addLayout(cameras, 1)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 10000)
        self.slider.sliderMoved.connect(self.seek)
        layout.addWidget(self.slider)
        self.clock = QtWidgets.QLabel('00:00 / 00:00')
        layout.addWidget(self.clock)
        controls = QtWidgets.QHBoxLayout()
        self.speed = SpeedBox()
        self.speed.setRange(.1, 1.)
        self.speed.setSingleStep(.1)
        self.speed.setValue(1.)
        self.speed.setSuffix('×')
        self.speed.setLocale(QtCore.QLocale.c())
        self.speed.setKeyboardTracking(False)
        self.speed.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.speed.setMinimumWidth(95)
        self.speed.setFixedHeight(40)
        # The generic QLineEdit padding otherwise clips the spinbox's inner editor.
        self.speed.setStyleSheet('QDoubleSpinBox { padding: 0px 8px; color: #e7edf5; } '
                                'QLineEdit { padding: 0px; border: none; background: transparent; color: #e7edf5; }')
        self.speed.setToolTip('0.10–1.00×. Editable during camera playback; locked during live replay.')
        self.speed.valueChanged.connect(self.change_speed)
        controls.addWidget(QtWidgets.QLabel('Playback speed'))
        self.slower = QtWidgets.QPushButton('−')
        self.slower.clicked.connect(lambda: self.speed.stepDown())
        controls.addWidget(self.slower)
        controls.addWidget(self.speed)
        self.faster = QtWidgets.QPushButton('+')
        self.faster.clicked.connect(lambda: self.speed.stepUp())
        controls.addWidget(self.faster)
        self.slower.setFixedWidth(44)
        self.faster.setFixedWidth(44)
        self.speed.setFixedWidth(110)
        controls.addStretch()
        self.depth = QtWidgets.QCheckBox('Show depth')
        self.depth.toggled.connect(self.clear_images)
        controls.addWidget(self.depth)
        self.live_check = QtWidgets.QCheckBox('Replay on the live robot')
        self.live_check.setEnabled(allow_live)
        if not allow_live:
            self.live_check.setToolTip('Launch with --execute to enable live replay')
        self.live_check.toggled.connect(lambda _: self.update_buttons())
        controls.addWidget(self.live_check)
        layout.addLayout(controls)
        controls = QtWidgets.QHBoxLayout()
        self.speed_display = QtWidgets.QLabel('Playback speed: 1.00×')
        controls.addWidget(self.speed_display)
        controls.addStretch()
        self.rewind_button = QtWidgets.QPushButton('Reset playback to 0')
        self.rewind_button.clicked.connect(self.rewind)
        controls.addWidget(self.rewind_button)
        self.start_button = QtWidgets.QPushButton('Start playback')
        self.start_button.setObjectName('start')
        self.start_button.clicked.connect(self.start_playback)
        controls.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.setObjectName('stop')
        self.stop_button.clicked.connect(self.stop_playback)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)
        # Enter commits a speed edit; it must never trigger live motion implicitly.
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.status = QtWidgets.QLabel('Ready')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        note = QtWidgets.QLabel('Recorded images follow the replay timeline; they are not a live view of the workspace.\n'
                               'Live replay opens the gripper, follows the saved reset path, then replays measured joints and gripper commands. '
                               'Paths are not collision checked; arrange objects and clearance before starting.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.allow_live = allow_live
        self.refresh_list()
        if episode:
            wanted = Path(episode).resolve()
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                if Path(item.data(0, QtCore.Qt.UserRole)).resolve() == wanted:
                    self.tree.setCurrentItem(item)
                    break
        self.update_buttons()

    def busy(self):
        return self.live is not None and not self.live.done

    def choose_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Demonstration dataset', self.root.text())
        if path:
            self.root.setText(path)
            self.refresh_list()

    def refresh_list(self):
        if self.busy():
            return
        self.stop_playback()
        self.tree.clear()
        for path, metadata in list_episodes(self.root.text()):
            duration = max(0, ((metadata.get('end_ns') or metadata['start_ns'])-metadata['start_ns'])/1e9)
            descriptions = metadata.get('prompts') or [metadata.get('prompt', '')]
            item = QtWidgets.QTreeWidgetItem(['; '.join(descriptions), f'{path.parent.name} / {path.name}',
                                             metadata.get('outcome', 'unknown'), f'{duration:.1f}s'])
            item.setData(0, QtCore.Qt.UserRole, str(path))
            self.tree.addTopLevelItem(item)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tree.resizeColumnToContents(1)
        self.status.setText(f'{self.tree.topLevelItemCount()} episode(s) found')

    def select_episode(self):
        if self.busy():
            return
        self.playing = False
        self.live = None
        self.episode = None
        self.position = 0.
        self.clear_images()
        items = self.tree.selectedItems()
        if items:
            try:
                self.episode = Episode(items[0].data(0, QtCore.Qt.UserRole))
                descriptions = self.episode.metadata.get('prompts') or [self.episode.metadata.get('prompt', '')]
                self.details.setText(f'{"; ".join(descriptions)} · {self.episode.duration:.1f}s · '
                                     f'{len(self.episode.joints)} measured robot samples')
                self.status.setText('Loaded. Playback uses original host receipt timestamps; device-clock uncertainty is retained in the dataset.')
            except Exception as error:
                self.status.setText(f'Cannot load episode: {error}')
        self.update_buttons()

    def clear_images(self, *_):
        self.generation += 1
        self.last_frames.clear()
        for image in self.images.values():
            image.clear()
            image.setText('Waiting for recorded frame')

    def update_buttons(self):
        active = self.playing or self.busy()
        self.start_button.setEnabled(self.episode is not None and not active)
        self.start_button.setText('Start live replay' if self.live_check.isChecked() else 'Start playback')
        self.stop_button.setEnabled(active)
        for widget in (self.root, self.browse, self.refresh_button, self.tree):
            widget.setEnabled(not active)
        self.speed.setEnabled(not self.busy())
        self.slower.setEnabled(not self.busy() and self.speed.value() > self.speed.minimum())
        self.faster.setEnabled(not self.busy() and self.speed.value() < self.speed.maximum())
        self.slider.setEnabled(self.episode is not None and not self.busy())
        self.rewind_button.setEnabled(self.episode is not None and not self.busy())
        self.live_check.setEnabled(self.allow_live and not active)
        if self.busy() and self.live.effective_speed:
            self.speed_display.setText(f'Live speed: {self.live.effective_speed:.3f}× (requested {self.speed.value():.2f}×)')
        else:
            self.speed_display.setText(f'Playback speed: {self.speed.value():.2f}×')

    def start_playback(self):
        if self.episode is None or self.busy():
            return
        if self.live_check.isChecked():
            try:
                self.episode.validate_live()
            except ValueError as error:
                self.status.setText(str(error))
                return
            self.position = 0.
            self.clear_images()
            self.live = LiveReplay(self.episode, self.speed.value())
            self.live.start()
        else:
            self.live = None
            if self.position >= self.episode.duration:
                self.position = 0.
                self.clear_images()
            self.preview_start = self.position
            self.preview_speed = self.speed.value()
            self.started = time.monotonic()
            self.playing = True
            self.status.setText('Playing recorded cameras · no robot commands')
        self.update_buttons()

    def change_speed(self, value):
        now = time.monotonic()
        if self.playing and self.episode is not None:
            self.position = min(self.episode.duration,
                                self.preview_start + (now-self.started)*self.preview_speed)
            self.preview_start = self.position
            self.started = now
        self.preview_speed = value
        self.update_buttons()

    def stop_playback(self):
        was_playing = self.playing
        self.playing = False
        if self.busy():
            self.live.stop()
            self.status.setText('Requesting action cancellation. Use the physical stop if motion does not stop.')
        elif was_playing:
            self.status.setText('Playback paused')
        self.update_buttons()

    def seek(self, value):
        if self.episode is not None and not self.busy():
            if self.live is not None and self.live.done:
                self.live = None
            self.position = self.episode.duration*value/10000
            self.preview_start = self.position
            self.started = time.monotonic()
            self.clear_images()

    def rewind(self):
        if self.episode is None or self.busy():
            return
        self.playing = False
        self.live = None
        self.position = self.preview_start = 0.
        self.slider.setValue(0)
        self.clear_images()
        self.status.setText('Playback reset to the beginning. No robot motion.')
        self.update_buttons()
        self.tick()

    def tick(self):
        if self.live is not None:
            self.position = self.live.position
            self.status.setText(self.live.status)
            self.update_buttons()
        elif self.playing and self.episode is not None:
            self.position = min(self.episode.duration, self.preview_start+(time.monotonic()-self.started)*self.preview_speed)
            if self.position >= self.episode.duration:
                self.playing = False
                self.status.setText('Playback complete')
                self.update_buttons()
        if self.episode is None:
            return
        if not self.slider.isSliderDown():
            self.slider.setValue(round(self.position/self.episode.duration*10000))
        self.clock.setText(f'{self.position:.2f}s / {self.episode.duration:.2f}s')
        if self.future is not None:
            if not self.future.done():
                return
            try:
                generation, frames = self.future.result()
                if generation == self.generation:
                    for role, (image, error) in frames.items():
                        label = self.images[role]
                        if error:
                            label.clear()
                            label.setText(error)
                        else:
                            h, w, _ = image.shape
                            qimage = QtGui.QImage(image.data, w, h, image.strides[0], QtGui.QImage.Format_RGB888).copy()
                            label.setPixmap(QtGui.QPixmap.fromImage(qimage).scaled(label.size(), QtCore.Qt.KeepAspectRatio))
            finally:
                self.future = None
        requested = {}
        for role in self.images:
            frame = self.episode.frame_at(role, self.position)
            if frame is None and self.position == 0.:
                # Show the first recorded image on rewind, even if acquisition
                # began a few milliseconds after the episode start marker.
                frames = self.episode.cameras[role][1]
                frame = frames[0] if frames else None
            if frame is not None:
                key = frame.get('depth' if self.depth.isChecked() else 'rgb')
                if self.last_frames.get(role) != key:
                    self.last_frames[role] = key
                    requested[role] = frame
        if requested:
            episode, generation, depth = self.episode, self.generation, self.depth.isChecked()
            def load():
                result = {}
                for role, frame in requested.items():
                    try:
                        result[role] = (episode.image(frame, depth), None)
                    except Exception as error:
                        result[role] = (None, f'Frame unavailable: {error}')
                return generation, result
            self.future = self.executor.submit(load)

    def closeEvent(self, event):
        if self.busy():
            self.stop_playback()
            event.ignore()
        else:
            self.timer.stop()
            self.executor.shutdown(wait=False, cancel_futures=True)
            event.accept()

    def reject(self):
        # Escape must follow the same cancellation path as the window close button.
        self.close()
