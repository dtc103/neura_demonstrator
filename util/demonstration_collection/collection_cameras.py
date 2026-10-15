"""Independent RealSense RGB/depth acquisition; PNG encoding lives in the writer."""
import threading
import time
import numpy as np


class Cameras:
    def __init__(self, serials, recorder, width=1280, height=720, fps=30):
        if set(serials) != {'wrist', 'side', 'opposite'} or len(set(serials.values())) != 3:
            raise ValueError('Expected three distinct serials: wrist, side, opposite')
        self.recorder = recorder
        self.stop_event = threading.Event()
        self.error = None
        self.latest = {}
        self.metadata = {}
        self.received = {}
        self.threads = []
        self.serials = serials
        for role, serial in serials.items():
            thread = threading.Thread(target=self._capture, args=(role, serial, width, height, fps), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _capture(self, role, serial, width, height, fps):
        pipeline = None
        try:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            profile = pipeline.start(config)
            def intrinsics(stream):
                intr = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
                return {k: getattr(intr, k) for k in ('width', 'height', 'ppx', 'ppy', 'fx', 'fy', 'coeffs')} | {'model': str(intr.model)}
            extr = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
            self.metadata[role] = {'serial': serial, 'color': intrinsics(rs.stream.color),
                'depth': intrinsics(rs.stream.depth), 'depth_scale_m': profile.get_device().first_depth_sensor().get_depth_scale(),
                'depth_to_color': {'rotation_column_major': list(extr.rotation), 'translation_m': list(extr.translation)},
                'fps': fps, 'depth_aligned': False}
            previous = {}
            offsets = {}
            def timing(frame, label, receipt):
                source = frame.get_timestamp() * 1e6
                domain = str(frame.get_frame_timestamp_domain())
                # Lower envelope of arrival offsets. Includes unknown minimum USB
                # delay: an estimate, not a hardware-synchronization guarantee.
                key = label, domain
                if label in previous and frame.get_frame_number() <= previous[label][0]:
                    raise RuntimeError(f'{role} {label} frame counter reset/stalled')
                if label in previous and source < previous[label][1]:
                    raise RuntimeError(f'{role} {label} timestamp reset')
                dropped = max(0, frame.get_frame_number() - previous.get(label, (frame.get_frame_number()-1, 0))[0] - 1)
                previous[label] = (frame.get_frame_number(), source)
                offsets[key] = min(offsets.get(key, float('inf')), receipt - source)
                result = {'source_timestamp_ms': frame.get_timestamp(), 'timestamp_domain': domain,
                    'frame_number': frame.get_frame_number(), 'gaps_before': dropped,
                    'estimated_host_ns': int(source + offsets[key]), 'offset_estimate_ns': offsets[key],
                    'mapping_method': 'minimum_observed_arrival_offset', 'uncertainty_ns': None}
                for attr in ('actual_exposure', 'sensor_timestamp', 'frame_timestamp'):
                    enum = getattr(rs.frame_metadata_value, attr)
                    if frame.supports_frame_metadata(enum):
                        result[attr] = frame.get_frame_metadata(enum)
                return result
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1500)
                receipt = time.monotonic_ns()
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    raise RuntimeError(f'{role}: missing RGB or depth')
                meta = {'received_ns': receipt, 'color': timing(color, 'color', receipt),
                        'depth': timing(depth, 'depth', receipt)}
                rgb = np.asanyarray(color.get_data()).copy()
                dep = np.asanyarray(depth.get_data()).copy()
                self.received[role] = time.monotonic()
                self.latest[role] = rgb[::4, ::4].copy()
                self.recorder.image(role, rgb, dep, meta)
        except Exception as error:
            self.error = f'Camera {role}: {error}'
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def health(self):
        if self.error:
            raise RuntimeError(self.error)
        return len(self.received) == 3 and all(time.monotonic()-t < 1.5 for t in tuple(self.received.values()))

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(3)
