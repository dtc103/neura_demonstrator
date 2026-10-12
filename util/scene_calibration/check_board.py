"""Preview ChArUco detections in one D405 color stream; does not move the robot."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from generate_board import make_board
from camera_geometry import draw_axes, estimate_pose, supported_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List camera serial numbers and exit")
    parser.add_argument("--serial", help="Serial of the camera to preview")
    parser.add_argument("--board", type=Path, default=Path(__file__).parent / "board/board.json")
    args = parser.parse_args()
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except RuntimeError as error:
        raise SystemExit(f"Cannot access RealSense devices: {error}") from error
    for device in devices:
        print(f"{device.get_info(rs.camera_info.serial_number)}: {device.get_info(rs.camera_info.name)}")
    if not devices:
        raise SystemExit("No RealSense cameras detected. Check USB connections and device permissions.")
    if args.list:
        return
    if not args.serial:
        parser.error("Choose a camera with --serial SERIAL; use --list to identify it")
    if args.serial not in [d.get_info(rs.camera_info.serial_number) for d in devices]:
        parser.error("The selected serial number is not connected")
    if not args.board.exists():
        parser.error("Board definition missing. Run uv run generate_board.py first")
    board = make_board(json.loads(args.board.read_text()))
    detector = cv2.aruco.CharucoDetector(board)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        geometry = {"width": intrinsics.width, "height": intrinsics.height,
                    "fx": intrinsics.fx, "fy": intrinsics.fy, "cx": intrinsics.ppx, "cy": intrinsics.ppy,
                    "distortion_model": str(intrinsics.model), "distortion_coefficients": list(intrinsics.coeffs)}
        supported = supported_model(geometry)
        print(f"Color stream: {intrinsics.width} x {intrinsics.height}; distortion: {intrinsics.model}")
        print(f"fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.ppx}, cy={intrinsics.ppy}")
        if not supported:
            print("Corner detection enabled; pose disabled for this distortion model.")
        print("Check that many corners are detected reliably. Q / Escape to quit.")
        window = f"Board check — {args.serial}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue
            image = np.asanyarray(frame.get_data()).copy()
            corners, ids, markers, marker_ids = detector.detectBoard(image)
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(image, markers, marker_ids)
            count = 0 if ids is None else len(ids)
            text = f"ChArUco corners: {count}/{len(board.getChessboardCorners())}"
            if count:
                corners = np.asarray(corners).reshape(-1, 1, 2)
                ids = np.asarray(ids).reshape(-1, 1)
                cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)
            if supported and count >= 4 and not board.checkCharucoCornersCollinear(ids):
                objects, pixels = board.matchImagePoints(corners, ids)
                valid, rotation, translation = estimate_pose(objects, pixels, geometry)
                if valid and translation[2, 0] > 0:
                    draw_axes(image, rotation, translation, geometry)
                    text += f" | board distance: {np.linalg.norm(translation):.3f} m"
            cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
            cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(window, image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
