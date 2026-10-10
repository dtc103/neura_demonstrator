"""Fit hand-eye calibration and export a validated robot/camera scene JSON."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from calibration_core import ROLES, check_transform, inverse, solve_dataset


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def export_scene(metadata, samples, flange_wrist, base_board, base_cameras, report):
    reference = samples[0]
    base_flange = check_transform(reference["T_base_from_flange"])
    initial = {"wrist": base_flange @ flange_wrist, **base_cameras}
    cameras = {}
    for role in ROLES:
        cameras[role] = {**metadata["cameras"][role], "frame": f"{role}_color_optical",
                         "parent": metadata["robot"]["flange_frame"] if role == "wrist" else metadata["robot"]["base_frame"],
                         "T_parent_from_camera": (flange_wrist if role == "wrist" else base_cameras[role]).tolist(),
                         "T_base_from_camera_at_reference": initial[role].tolist()}
    distances = {}
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            distances[f"{a}_to_{b}"] = float(np.linalg.norm(initial[a][:3, 3] - initial[b][:3, 3]))
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "units": {"translation": "meters", "joint_angles": "radians"},
            "matrix_convention": "Row-major nested arrays; column vectors; T_A_from_B maps B coordinates into A.",
            "camera_axes": "RealSense color optical: +X right, +Y down, +Z forward. Convert explicitly for your simulator.",
            "world_frame": metadata["robot"]["base_frame"],
            "robot": {**metadata["robot"], "T_world_from_base": np.eye(4).tolist(), "urdf": "panda.urdf",
                      "mesh_assets": "URDF package:// paths require the matching franka_description assets; meshes are not embedded.",
                      "reference_sample_id": reference["id"], "reference_joint_positions_rad": reference["robot"]["q"],
                      "reference_joint_names": [f"panda_joint{i}" for i in range(1, 8)],
                      "T_base_from_flange_at_reference": base_flange.tolist()},
            "board": {**metadata["board"], "T_base_from_board": base_board.tolist(),
                      "origin": "OpenCV board origin at the patterned rectangle corner; axes follow board object points.",
                      "removable_after_calibration": True},
            "cameras": cameras, "camera_origin_distances_m_at_reference": distances,
            "wrist_motion_rule": "T_base_from_wrist(t) = T_base_from_flange(t) @ cameras.wrist.T_parent_from_camera",
            "side_from_opposite": (inverse(base_cameras["side"]) @ base_cameras["opposite"]).tolist(),
            "calibration": {"passed": report["passed"], "report": "calibration_report.json", "sample_count": len(samples),
                            "intrinsics_source": "RealSense SDK factory stream intrinsics; not independently recalibrated here."},
            "unmeasured_geometry": ["table boundaries and thickness", "fixtures", "objects", "camera housing/mount offsets"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path(__file__).parent / "captures/current")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--max-translation-mm", type=float, default=5)
    parser.add_argument("--max-rotation-deg", type=float, default=2)
    parser.add_argument("--max-reprojection-px", type=float, default=3)
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.max_translation_mm, args.max_rotation_deg, args.max_reprojection_px)):
        parser.error("Validation limits must be positive finite numbers")
    args.output.mkdir(parents=True, exist_ok=True)
    # Avoid leaving a previous accepted scene next to a report for a failed run.
    if (args.output / "scene.json").exists():
        parser.error("Output already contains scene.json. Use a new --output directory to preserve that calibration.")
    try:
        metadata = json.loads((args.session / "session.json").read_text())
        if not metadata.get("printed_square_size_verified"):
            raise ValueError("Printed board scale has not been confirmed")
        paths = sorted(args.session.glob("sample_*/sample.json"))
        samples = [json.loads(path.read_text()) for path in paths]
        urdf_path = args.session / "panda.urdf"
        urdf = ET.fromstring(urdf_path.read_text())
        links = {link.attrib["name"] for link in urdf.findall("link")}
        if not {metadata["robot"]["base_frame"], metadata["robot"]["flange_frame"]} <= links:
            raise ValueError("Configured robot base/flange frames do not exist in the captured URDF")
        x, board, cameras, report = solve_dataset(metadata, samples, args.max_translation_mm,
                                                  args.max_rotation_deg, args.max_reprojection_px)
        report["source_session"] = str(args.session.resolve())
        report["input_sha256"] = {str(p.relative_to(args.session)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [args.session / "session.json", urdf_path, *paths]}
        write_json(args.output / "calibration_report.json", report)
        if not report["passed"]:
            raise ValueError("Validation failed. Inspect calibration_report.json; no scene.json was exported.")
        shutil.copy2(urdf_path, args.output / "panda.urdf")
        scene = export_scene(metadata, samples, x, board, cameras, report)
        write_json(args.output / "scene.json", scene)
        print(f"Scene: {args.output / 'scene.json'}")
        print(f"Robot model: {args.output / 'panda.urdf'}")
        print(f"Validation: {args.output / 'calibration_report.json'}")
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
