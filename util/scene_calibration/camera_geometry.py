"""PnP and overlays that respect the RealSense stream's distortion model."""

import cv2
import numpy as np
import pyrealsense2 as rs


def model_name(intrinsics):
    # Older saved/test pinhole metadata omitted the model and used OpenCV's Brown model.
    return intrinsics.get("distortion_model", "brown_conrady").rsplit(".", 1)[-1]


def supported_model(intrinsics):
    return model_name(intrinsics) in ("none", "brown_conrady", "inverse_brown_conrady")


def camera_matrix(intrinsics):
    return np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                     [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]], dtype=float)


def sdk_intrinsics(intrinsics):
    if not supported_model(intrinsics):
        raise ValueError(f"Unsupported camera distortion model: {model_name(intrinsics)}")
    result = rs.intrinsics()
    result.width = intrinsics.get("width", 0)
    result.height = intrinsics.get("height", 0)
    result.fx, result.fy = intrinsics["fx"], intrinsics["fy"]
    result.ppx, result.ppy = intrinsics["cx"], intrinsics["cy"]
    result.model = getattr(rs.distortion, model_name(intrinsics))
    result.coeffs = intrinsics["distortion_coefficients"]
    return result


def estimate_pose(objects, pixels, intrinsics):
    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if not supported_model(intrinsics):
        raise ValueError(f"Unsupported camera distortion model: {model_name(intrinsics)}")
    if model_name(intrinsics) == "inverse_brown_conrady":
        native = sdk_intrinsics(intrinsics)
        # Depth=1 obtains a bearing ray; it does not read or require a depth frame.
        rays = np.array([rs.rs2_deproject_pixel_to_point(native, p.tolist(), 1.0) for p in pixels])
        normalized = rays[:, :2] / rays[:, 2:3]
        return cv2.solvePnP(objects, normalized, np.eye(3), None)
    distortion = None if model_name(intrinsics) == "none" else np.array(intrinsics["distortion_coefficients"])
    return cv2.solvePnP(objects, pixels, camera_matrix(intrinsics), distortion)


def project_points(objects, rotation, translation, intrinsics):
    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 3)
    if not supported_model(intrinsics):
        raise ValueError(f"Unsupported camera distortion model: {model_name(intrinsics)}")
    if model_name(intrinsics) == "inverse_brown_conrady":
        native = sdk_intrinsics(intrinsics)
        camera_points = objects @ cv2.Rodrigues(np.asarray(rotation, dtype=float).reshape(3))[0].T + np.asarray(translation).reshape(3)
        return np.array([rs.rs2_project_point_to_pixel(native, p.tolist()) for p in camera_points])
    distortion = None if model_name(intrinsics) == "none" else np.array(intrinsics["distortion_coefficients"])
    pixels, _ = cv2.projectPoints(objects, np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float),
                                camera_matrix(intrinsics), distortion)
    return pixels.reshape(-1, 2)


def draw_axes(image, rotation, translation, intrinsics, length=0.05):
    axes = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    camera_points = axes @ cv2.Rodrigues(np.asarray(rotation, dtype=float).reshape(3))[0].T + np.asarray(translation).reshape(3)
    if np.any(camera_points[:, 2] <= 0):
        return
    pixels = project_points(axes, rotation, translation, intrinsics)
    if not np.isfinite(pixels).all():
        return
    pixels = np.clip(np.rint(pixels), -100000, 100000).astype(int)
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(image, tuple(pixels[0]), tuple(endpoint), color, 2, cv2.LINE_AA)
