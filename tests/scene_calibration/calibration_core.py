"""Rigid-transform math and offline scene fitting; translations are in meters."""

import cv2
import numpy as np

from camera_geometry import camera_matrix, project_points

ROLES = ("wrist", "side", "opposite")


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation).reshape(3)
    return check_transform(matrix)


def check_transform(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid transform bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Invalid rigid rotation")
    return matrix


def inverse(value):
    matrix = check_transform(value)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def franka_flange_pose(sample):
    # libfranka serializes these arrays in COLUMN-major order.
    base_ee = check_transform(np.array(sample["o_t_ee"]).reshape(4, 4, order="F"))
    flange_ee = check_transform(np.array(sample["f_t_ee"]).reshape(4, 4, order="F"))
    return base_ee @ inverse(flange_ee)


def average_transform(values):
    matrices = np.array([check_transform(v) for v in values])
    if not len(matrices):
        raise ValueError("No transforms to average")
    u, _, vt = np.linalg.svd(matrices[:, :3, :3].sum(axis=0))
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    return transform(u @ correction @ vt, matrices[:, :3, 3].mean(axis=0))


def difference(a, b):
    a, b = check_transform(a), check_transform(b)
    translation_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return {"translation_mm": translation_mm, "rotation_deg": float(np.degrees(np.arccos(cosine)))}


def stable_robot_sample(history, start_ns, end_ns, translation_mm=1.0, rotation_deg=0.5):
    """Require recent robot observations spanning a stationary capture interval."""
    if end_ns <= start_ns:
        raise ValueError("Invalid stability interval")
    samples = [s for s in history if start_ns - 150_000_000 <= s["received_monotonic_ns"] <= end_ns + 150_000_000]
    if len(samples) < 5:
        raise ValueError("Waiting for at least five robot state observations")
    times = [s["received_monotonic_ns"] for s in samples]
    if min(times) > start_ns + 100_000_000 or max(times) < end_ns - 100_000_000:
        raise ValueError("Robot state history does not span the capture interval")
    if max(np.diff(sorted(times)), default=0) > 200_000_000:
        raise ValueError("Robot state stream has a gap larger than 200 ms")
    poses = [franka_flange_pose(s) for s in samples]
    anchor = average_transform(poses)
    for sample, pose in zip(samples, poses):
        error = difference(pose, anchor)
        if error["translation_mm"] > translation_mm or error["rotation_deg"] > rotation_deg:
            raise ValueError("Arm moved during capture; let it settle before saving")
        velocities = np.asarray(sample["dq"], dtype=float)
        if velocities.shape != (7,) or not np.isfinite(velocities).all() or np.max(np.abs(velocities)) > 0.01:
            raise ValueError("Measured joint velocities are not stationary")
        if sample.get("current_error_flags"):
            raise ValueError("Robot reports an active error")
        if sample.get("robot_mode") in (4, 5, 6):
            raise ValueError("Robot is stopped, in reflex, or recovering")
    middle = (start_ns + end_ns) // 2
    return min(samples, key=lambda s: abs(s["received_monotonic_ns"] - middle)), anchor


def motion_diversity(poses):
    first = check_transform(poses[0])
    vectors = [cv2.Rodrigues(first[:3, :3].T @ check_transform(p)[:3, :3])[0].ravel() for p in poses[1:]]
    singular = np.linalg.svd(np.array(vectors), compute_uv=False)
    if len(singular) < 2 or singular[1] < 0.15:
        raise ValueError("Insufficient rotation diversity: tilt the wrist about at least two different axes")
    return singular.tolist()


def reprojection_error(observation, pose, intrinsics):
    pose = check_transform(pose)
    objects = np.array(observation["object_points_m"], dtype=float).reshape(-1, 3)
    pixels = np.array(observation["image_points_px"], dtype=float).reshape(-1, 2)
    predicted = project_points(objects, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], intrinsics)
    return float(np.sqrt(np.mean(np.sum((predicted - pixels) ** 2, axis=1))))


def summarize(errors):
    return {field: {"mean": float(np.mean([e[field] for e in errors])),
                    "max": float(max(e[field] for e in errors))}
            for field in ("translation_mm", "rotation_deg", "reprojection_rms_px")}


def hand_eye(grippers, cameras):
    if hasattr(cv2, "calibrateHandEye"):
        rotation, translation = cv2.calibrateHandEye(
            [g[:3, :3] for g in grippers], [g[:3, 3] for g in grippers],
            [c[:3, :3] for c in cameras], [c[:3, 3] for c in cameras], method=cv2.CALIB_HAND_EYE_PARK)
        return transform(rotation, translation), "OpenCV CALIB_HAND_EYE_PARK"
    # OpenCV 5.0.0.93 omits the Python binding (upstream issue #29565).
    # Equivalent Park formulation: log(R_A) = R_X log(R_B), fit R_X via
    # orthogonal Procrustes; then solve (R_A-I)t_X = R_X t_B - t_A.
    pairs = [(inverse(grippers[j]) @ grippers[i], cameras[j] @ inverse(cameras[i]))
             for i in range(len(grippers)) for j in range(i + 1, len(grippers))]
    covariance = np.zeros((3, 3))
    for a, b in pairs:
        alpha = cv2.Rodrigues(a[:3, :3])[0].ravel()
        beta = cv2.Rodrigues(b[:3, :3])[0].ravel()
        covariance += np.outer(alpha, beta)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a, _ in pairs])
    rhs = np.concatenate([rotation @ b[:3, 3] - a[:3, 3] for a, b in pairs])
    translation, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank < 3:
        raise ValueError("Hand-eye translation is underconstrained; collect more varied rotations")
    return transform(rotation, translation), "Park log-rotation SVD / least-squares (NumPy fallback for missing OpenCV binding)"


def solve_dataset(metadata, samples, max_translation_mm=5.0, max_rotation_deg=2.0, max_reprojection_px=3.0):
    """Fit on training captures and validate against explicit held-out captures."""
    training = [s for s in samples if s["split"] == "train"]
    validation = [s for s in samples if s["split"] == "validation"]
    if len(training) < 10 or len(validation) < 3:
        raise ValueError("Need at least 10 training captures and 3 separate validation captures; aim for 20–30 training poses")
    grippers = [check_transform(s["T_base_from_flange"]) for s in training]
    cameras = [check_transform(s["observations"]["wrist"]["T_camera_from_board"]) for s in training]
    diversity = motion_diversity(grippers)
    flange_wrist, method = hand_eye(grippers, cameras)
    base_board = average_transform([g @ flange_wrist @ c for g, c in zip(grippers, cameras)])
    base_cameras = {}
    for role in ("side", "opposite"):
        observations = [s["observations"][role] for s in training if role in s["observations"]]
        validation_count = sum(role in s["observations"] for s in validation)
        if len(observations) < 3 or validation_count < 1:
            raise ValueError(f"Need at least 3 training observations and 1 held-out observation for {role}; "
                             f"found {len(observations)} training and {validation_count} held-out observations. "
                             "Saved poses only count for a camera when its board detection was accepted.")
        base_cameras[role] = average_transform([base_board @ inverse(o["T_camera_from_board"]) for o in observations])
    errors = {"train": {r: [] for r in ROLES}, "validation": {r: [] for r in ROLES}}
    for sample in samples:
        base_wrist = check_transform(sample["T_base_from_flange"]) @ flange_wrist
        for role, observation in sample["observations"].items():
            base_camera = base_wrist if role == "wrist" else base_cameras[role]
            error = difference(base_camera @ check_transform(observation["T_camera_from_board"]), base_board)
            error["reprojection_rms_px"] = reprojection_error(observation, inverse(base_camera) @ base_board,
                                                            metadata["cameras"][role]["intrinsics"])
            error["sample_id"] = sample["id"]
            errors[sample["split"]][role].append(error)
    summary = {split: {role: summarize(values) for role, values in roles.items()} for split, roles in errors.items()}
    # Gate both fit consistency and held-out measurements; never mark an unchecked
    # result as validated simply because OpenCV returned a matrix.
    passed = all(e["translation_mm"] <= max_translation_mm and e["rotation_deg"] <= max_rotation_deg
                 and e["reprojection_rms_px"] <= max_reprojection_px
                 for split in errors.values() for role in split.values() for e in role)
    report = {"passed": passed, "method": method, "rotation_diversity_singular_values_rad": diversity,
              "limits": {"translation_mm": max_translation_mm, "rotation_deg": max_rotation_deg,
                         "reprojection_rms_px": max_reprojection_px}, "summary": summary, "observations": errors,
              "note": "Consistency checks do not independently certify absolute metric accuracy or camera intrinsics."}
    return flange_wrist, base_board, base_cameras, report
