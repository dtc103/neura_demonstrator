"""Run with: uv run stream_d405.py. Press Q or Escape to quit."""

import math

import cv2
import numpy as np
import pyrealsense2 as rs

context = rs.context()
devices = list(context.query_devices())
if not devices:
    raise SystemExit("No RealSense devices found. Check the USB connections.")

width, height = 1280, 720
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
            pipeline.start(config)
        except RuntimeError as error:
            print(f"Cannot stream {name} ({serial}): {error}")
            continue
        cameras.append((pipeline, f"{name} ({serial})"))

    if not cameras:
        raise SystemExit("No cameras could start a color stream.")
    columns = math.ceil(math.sqrt(len(cameras)))
    rows = math.ceil(len(cameras) / columns)
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    cv2.namedWindow("RealSense cameras", cv2.WINDOW_NORMAL)
    while True:
        for index, (pipeline, label) in enumerate(cameras):
            frames = pipeline.poll_for_frames()
            frame = frames.get_color_frame() if frames else None
            if not frame:
                continue
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
        if cv2.getWindowProperty("RealSense cameras", cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    pass
finally:
    for pipeline, _ in cameras:
        pipeline.stop()
    cv2.destroyAllWindows()
