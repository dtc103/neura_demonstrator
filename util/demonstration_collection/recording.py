"""Asynchronous, lossless episode storage. No ROS or hardware dependencies."""
import json
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def session_name(value):
    name = (value or '').strip()
    if name and (name in ('.', '..') or name.startswith('.') or name.endswith('.')
                 or any(c in '/\\<>:"|?*' or ord(c) < 32 or ord(c) == 127 for c in name)
                 or len(name.encode('utf-8')) > 180):
        raise ValueError('Session name must be a folder name (max 180 UTF-8 bytes), without path separators or special filename characters.')
    return name


class Recorder:
    def __init__(self, root, metadata, capacity=256, image_format="npy", name=''):
        if image_format not in ('npy', 'png'):
            raise ValueError('Unknown lossless image format')
        self.image_format = image_format
        name = session_name(name)
        base = name or ('session_' + stamp())
        suffix = 1
        while True:
            self.root = Path(root) / (base if suffix == 1 else f'{base}_{suffix:03d}')
            try:
                self.root.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
        atomic_json(self.root / 'session.json', {
            'schema_version': 1, 'created_utc': stamp(),
            'clock': 'host monotonic_ns; original clocks retained separately',
            'image_encoding': image_format, 'color_order_on_disk': 'BGR uint8',
            'depth_encoding': 'native unaligned uint16; multiply by depth_scale_m',
            **metadata, 'session_name': name, 'session_directory': self.root.name})
        self.queue = queue.Queue(capacity)
        self.lock = threading.Lock()
        self.active = None
        self.error = None
        self.finished = {}
        self.count = 0
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def start(self, prompt, metadata=None):
        with self.lock:
            if self.active is not None or self.error:
                raise RuntimeError(self.error or 'Episode already active')
            self.count += 1
            episode = f'episode_{self.count:06d}'
            path = self.root / episode
            path.mkdir()
            atomic_json(path / 'episode.json', {
                'episode': episode, 'prompt': prompt, 'outcome': 'incomplete',
                'start_ns': time.monotonic_ns(), **(metadata or {})})
            self.active = episode
            return episode

    def _put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.error = 'Recording queue overflow; data incomplete'

    def record(self, stream, data, received_ns=None):
        with self.lock:
            if self.active is not None and not self.error:
                self._put(('sample', self.active, stream,
                           {'received_ns': received_ns or time.monotonic_ns(), 'data': data}))

    def record_session(self, stream, data):
        with self.lock:
            if not self.error:
                self._put(('sample', '_session', stream,
                           {'received_ns': time.monotonic_ns(), 'data': data}))

    def image(self, role, rgb, depth, metadata):
        with self.lock:
            if self.active is not None and not self.error:
                self._put(('image', self.active, role, (rgb, depth, metadata)))

    def end(self, outcome, reason='', boundary_ns=None):
        with self.lock:
            episode, self.active = self.active, None
        if episode is not None:
            # Control loop never blocks on disk or a full writer queue.
            self._put(('end', episode, outcome, {
                'reason': reason, 'end_ns': time.monotonic_ns(),
                'stop_requested_ns': boundary_ns, 'recording_error': self.error}))
        return episode

    def _write(self):
        files = {}
        counts = {}
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                kind, episode, name, value = item
                path = self.root if episode == "_session" else self.root / episode
                if kind == 'end':
                    for key in list(files):
                        if key[0] == episode:
                            files.pop(key).close()
                    manifest = json.loads((path / 'episode.json').read_text())
                    manifest.update(value, outcome='interrupted' if self.error else name,
                                    counts=counts.get(episode, {}), recording_error=self.error)
                    atomic_json(path / 'episode.json', manifest)
                    self.finished[episode] = manifest
                    continue
                if kind == 'image':
                    import cv2
                    rgb, depth, metadata = value
                    directory = path / 'cameras' / name
                    directory.mkdir(parents=True, exist_ok=True)
                    index = counts.setdefault(episode, {}).get(name + '_frames', 0)
                    basename = f'{index:08d}'
                    for suffix, image in (('rgb', rgb), ('depth', depth)):
                        target = directory / (basename + '_' + suffix + '.' + self.image_format)
                        if self.image_format == 'npy':
                            import numpy as np
                            np.save(target, image, allow_pickle=False)
                        elif not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                            raise OSError(f'Cannot write {target}')
                    counts[episode][name + '_frames'] = index + 1
                    value = dict(metadata, rgb=f'cameras/{name}/{basename}_rgb.{self.image_format}',
                                 depth=f'cameras/{name}/{basename}_depth.{self.image_format}')
                    name = 'camera_' + name
                key = episode, name
                if key not in files:
                    files[key] = (path / (name + '.jsonl')).open('x')
                files[key].write(json.dumps(value, allow_nan=False) + '\n')
                counter = counts.setdefault(episode, {})
                counter[name] = counter.get(name, 0) + 1
        except Exception as error:
            self.error = f'Recording failed: {error}'
        finally:
            for file in files.values():
                file.close()

    def health(self):
        if self.error:
            raise RuntimeError(self.error)
        if shutil.disk_usage(self.root).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB free: recording stopped')

    def close(self):
        self.end('interrupted', 'Recorder closed while episode active')
        # Called after control has stopped, never in the control loop.
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.2)
                break
            except queue.Full:
                pass
        self.thread.join()
        # A crash/overflow must never leave an apparently successful episode.
        for path in self.root.glob('episode_*/episode.json'):
            data = json.loads(path.read_text())
            if data['outcome'] == 'incomplete':
                data.update(outcome='interrupted', reason=self.error or 'Unfinished recording',
                            recording_error=self.error)
                atomic_json(path, data)
