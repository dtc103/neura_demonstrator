"""Convert a ManiFeel zarr dataset into a GR00T-flavored LeRobot v2.1 dataset (vision + state + action).

Tactile arrays are never read. Output layout matches the IPEC LIBERO datasets GR00T was trained on:

    <out>/meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, modality.json}
    <out>/data/chunk-000/episode_000000.parquet
    <out>/videos/chunk-000/observation.images.<camera>/episode_000000.mp4   (H.264)

GR00T computes meta/stats.json itself on first use.

Two action formats:
  delta     ManiFeel's own actions: normalized EEF delta translation/rotation per step (6-D).
  eef_pose  Embodiment-agnostic EEF poses for GR00T's relative-EEF action space: state = current
            fingertip pose, action = pose actually reached at the next step, both xyz + rot6d (9-D,
            rot6d = first two rows of the rotation matrix, GR00T's convention). The reached pose is used
            rather than ManiFeel's commanded target, which its impedance controller only tracks ~30%
            per step and is therefore controller-specific.

    uv run python convert_to_lerobot_v2.py data/gear_quan_Sep15
    uv run python convert_to_lerobot_v2.py data/gear_quan_Sep15 --cameras wrist front
    uv run python convert_to_lerobot_v2.py data/gear_quan_Sep15 --action-format eef_pose
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import zarr

FPS = 10  # ManiFeel env runner / recording rate
CHUNKS_SIZE = 1000
TASKS = {"gear_quan_Sep15": "assemble the gear onto the shaft"}

# ManiFeel state: end-effector position + orientation quaternion (Isaac Gym order x, y, z, w)
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
# ManiFeel action: normalized end-effector delta translation + delta rotation, no gripper
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
# eef_pose format: position + first two rows of the rotation matrix (GR00T's xyz+rot6d)
POSE_NAMES = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"]


def to_xyz_rot6d(state: np.ndarray) -> np.ndarray:
    """(T, 7) position + quaternion (x, y, z, w) -> (T, 9) position + rot6d."""
    rot6d = Rotation.from_quat(state[:, 3:7]).as_matrix()[:, :2, :].reshape(len(state), 6)
    return np.concatenate([state[:, :3], rot6d], axis=1).astype(np.float32)


def canonicalize_quaternions(quat: np.ndarray) -> np.ndarray:
    """q and -q are the same rotation; flip signs so consecutive quaternions stay continuous."""
    quat = quat.copy()
    if quat[0, np.argmax(np.abs(quat[0]))] < 0:
        quat[0] *= -1
    for t in range(1, len(quat)):
        if np.dot(quat[t], quat[t - 1]) < 0:
            quat[t] *= -1
    return quat


def write_video(frames: np.ndarray, path: Path) -> None:
    """frames: (T, H, W, 3) uint8 RGB -> H.264 mp4 (yuv420p), small GOP for fast random access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "10",
        str(path),
    ]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def feature_stats(x: np.ndarray, image: bool = False) -> dict:
    if image:  # per-channel over all pixels, values in [0, 1], shaped like LeRobot's (3, 1, 1)
        x = x.reshape(-1, 3).astype(np.float64) / 255.0
        stats = {k: getattr(np, k)(x, axis=0) for k in ["min", "max", "mean", "std"]}
        return {k: v.reshape(3, 1, 1).tolist() for k, v in stats.items()} | {"count": [len(x)]}
    x = x.reshape(len(x), -1).astype(np.float64)
    stats = {k: getattr(np, k)(x, axis=0).tolist() for k in ["min", "max", "mean", "std"]}
    return stats | {"count": [len(x)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--cameras", nargs="+", default=["wrist"], help="wrist | front | side | wrist_2")
    parser.add_argument("--out", type=Path, help="default: <zarr_path>_lerobot_v2.1[_<cameras>]")
    parser.add_argument("--task", help="language instruction (default: per-dataset entry in TASKS)")
    parser.add_argument("--action-format", choices=["delta", "eef_pose"], default="delta")
    parser.add_argument("--gripper-width", type=float, default=None,
                        help="eef_pose: add a constant gripper-opening column (m) to state and action, e.g. 0.033 "
                             "(gear held; matches REASSEMBLE for co-training)")
    args = parser.parse_args()
    eef = args.action_format == "eef_pose"
    gripper = args.gripper_width is not None
    if gripper and not eef:
        raise SystemExit("--gripper-width requires --action-format eef_pose")

    name = args.zarr_path.name
    suffix = "" if args.cameras == ["wrist"] else "_" + "_".join(args.cameras)
    suffix += "_eef" if eef else ""
    suffix += "_gripper" if gripper else ""
    out = args.out or args.zarr_path.with_name(f"{name}_lerobot_v2.1{suffix}")
    task = args.task or TASKS[name]
    if out.exists():
        raise SystemExit(f"{out} already exists, delete it first")

    store = zarr.open(str(args.zarr_path), mode="r")
    ends = store["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    image_shape = store["data"][args.cameras[0]].shape[1:]

    (out / "meta").mkdir(parents=True)
    episodes, episodes_stats = [], []
    for ep, (start, end) in enumerate(zip(starts, ends)):
        length = int(end - start)
        state = store["data/state"][start:end].astype(np.float32)
        state[:, 3:7] = canonicalize_quaternions(state[:, 3:7])
        action = store["data/action"][start:end].astype(np.float32)
        if eef:
            state = to_xyz_rot6d(state)
            # action_t = pose reached after step t; the final step holds its pose
            action = np.concatenate([state[1:], state[-1:]], axis=0)
            if gripper:  # ManiFeel holds the gear throughout: constant opening
                width = np.full((length, 1), args.gripper_width, dtype=np.float32)
                state, action = np.concatenate([state, width], 1), np.concatenate([action, width], 1)
        frame_index = np.arange(length, dtype=np.int64)
        columns = {
            "observation.state": state,
            "action": action,
            "timestamp": (frame_index / FPS).astype(np.float32),
            "frame_index": frame_index,
            "episode_index": np.full(length, ep, dtype=np.int64),
            "index": np.arange(start, end, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }

        table = pa.table({
            k: pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), v.shape[1]) if v.ndim == 2 else pa.array(v)
            for k, v in columns.items()
        })
        parquet_path = out / f"data/chunk-{ep // CHUNKS_SIZE:03d}/episode_{ep:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)

        stats = {k: feature_stats(v) for k, v in columns.items()}
        for cam in args.cameras:
            frames = (np.clip(store["data"][cam][start:end], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            write_video(frames, out / f"videos/chunk-{ep // CHUNKS_SIZE:03d}/observation.images.{cam}/episode_{ep:06d}.mp4")
            stats[f"observation.images.{cam}"] = feature_stats(frames, image=True)

        episodes.append({"episode_index": ep, "tasks": [task], "length": length})
        episodes_stats.append({"episode_index": ep, "stats": stats})
        print(f"episode {ep + 1}/{len(ends)}: {length} steps", flush=True)

    video_info = {
        "video.height": image_shape[0], "video.width": image_shape[1], "video.codec": "h264",
        "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": FPS,
        "video.channels": 3, "has_audio": False,
    }
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}  # noqa: E731
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(ends),
        "total_frames": int(ends[-1]),
        "total_tasks": 1,
        "total_videos": len(ends) * len(args.cameras),
        "total_chunks": (len(ends) - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(ends)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{
                f"observation.images.{cam}": {
                    "dtype": "video", "shape": list(image_shape),
                    "names": ["height", "width", "rgb"], "info": video_info,
                }
                for cam in args.cameras
            },
            "observation.state": {
                "dtype": "float32", "shape": [state.shape[1]],
                "names": {"motors": POSE_NAMES + ["gripper_width"] * gripper if eef else STATE_NAMES},
            },
            "action": {
                "dtype": "float32", "shape": [action.shape[1]],
                "names": {"motors": POSE_NAMES + ["gripper_width"] * gripper if eef else ACTION_NAMES},
            },
            "timestamp": scalar("float32"),
            "frame_index": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
            "task_index": scalar("int64"),
        },
    }
    if eef:
        state_modality = {"eef_pose": {"start": 0, "end": 9}}
        action_modality = {"eef_pose": {"start": 0, "end": 9}}
        if gripper:
            state_modality["gripper"] = {"start": 9, "end": 10}
            action_modality["gripper"] = {"start": 9, "end": 10}
    else:
        state_modality = {"eef_position": {"start": 0, "end": 3}, "eef_quaternion": {"start": 3, "end": 7}}
        action_modality = {"delta_translation": {"start": 0, "end": 3}, "delta_rotation": {"start": 3, "end": 6}}
    modality = {
        "state": state_modality,
        "action": action_modality,
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in args.cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }

    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    (out / "meta/modality.json").write_text(json.dumps(modality, indent=4))
    (out / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes_stats))
    print(f"\nWrote {len(ends)} episodes / {int(ends[-1])} frames to {out}")


if __name__ == "__main__":
    main()
