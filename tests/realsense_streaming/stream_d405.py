"""Run with: uv run stream_d405.py. Press Q or Escape to quit."""

import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs

context = rs.context()
devices = list(context.query_devices())
if not devices:
    raise SystemExit("No RealSense devices found. Check the USB connections.")

# These are preview tile dimensions, independent of the capture resolution.
width, height = 640, 360
cameras = []
try:
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Detected: {name} ({serial})")
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, rs.format.bgr8, 0)
        pipeline = rs.pipeline(context)
        try:
            profile = pipeline.start(config)
        except RuntimeError as error:
            print(f"Cannot stream {name} ({serial}): {error}")
            continue
        cameras.append((pipeline, f"{name} ({serial})"))
        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"Started {serial}: {video.width()}x{video.height()} @ {video.fps()} FPS", flush=True)

    if not cameras:
        raise SystemExit("No cameras could start a color stream.")
    columns = math.ceil(math.sqrt(len(cameras)))
    rows = math.ceil(len(cameras) / columns)
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    cv2.namedWindow("RealSense cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RealSense cameras", columns * width, rows * height)
    started = time.monotonic()
    last_frame = [None] * len(cameras)
    stalled = [False] * len(cameras)
    for index, (_, label) in enumerate(cameras):
        row, column = divmod(index, columns)
        tile = grid[row * height:(row + 1) * height,
                    column * width:(column + 1) * width]
        cv2.putText(tile, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, "Waiting for first color frame...", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
    while True:
        for index, (pipeline, label) in enumerate(cameras):
            frames = pipeline.poll_for_frames()
            frame = frames.get_color_frame() if frames else None
            if not frame:
                if time.monotonic() - (last_frame[index] or started) > 5 and not stalled[index]:
                    print(f"No color frames for 5 seconds: {label}", flush=True)
                    row, column = divmod(index, columns)
                    tile = grid[row * height:(row + 1) * height,
                                column * width:(column + 1) * width]
                    tile[35:80] = 0
                    cv2.putText(tile, "NO RECENT FRAMES", (8, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    stalled[index] = True
                continue
            if last_frame[index] is None or stalled[index]:
                print(f"Receiving color frames: {label}", flush=True)
            last_frame[index] = time.monotonic()
            stalled[index] = False
            image = cv2.resize(np.asanyarray(frame.get_data()), (width, height))
            cv2.drawMarker(image, (width // 2, height // 2), (0, 255, 0))
            cv2.rectangle(image, (0, 0), (width, 30), (0, 0, 0), -1)
            cv2.putText(image, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row, column = divmod(index, columns)
            grid[row * height:(row + 1) * height,
                 column * width:(column + 1) * width] = image
        cv2.imshow("RealSense cameras", grid)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        try:
            visible = cv2.getWindowProperty("RealSense cameras", cv2.WND_PROP_VISIBLE)
        except cv2.error as error:
            # Qt can destroy its receiver when the last window is closed,
            # before this property query runs after waitKey().
            if "NULL guiReceiver" not in str(error):
                raise
            break
        if visible < 1:
            break
except KeyboardInterrupt:
    pass
finally:
    for pipeline, _ in cameras:
        pipeline.stop()
    cv2.destroyAllWindows()
