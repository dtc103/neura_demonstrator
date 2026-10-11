"""Shared helpers for benchmark_latency_gr00t.py / benchmark_latency_lerobot.py (two different Python envs).

Both GR00T N1.7 (Isaac-GR00T) and the drifting policy (LeRobot fork) compute an action chunk as
    model.backbone(...)                 Cosmos-Reason2-2B vision-language backbone
    model.action_head.get_action(...)   flow-matching DiT (4 steps)  /  one-step drifting transformer
StageTimer wraps these two calls on the model instance (CUDA-synchronized), so the end-to-end time of a policy
call can be split into backbone, action head and the rest (pre/post-processing, normalization, host<->GPU copies).
"""

import json
import os
import time

import numpy as np
import torch


class StageTimer:
    def __init__(self, model):
        self.enabled = False
        self.times = {"backbone_ms": [], "action_head_ms": []}
        backbone_forward = model.backbone.forward
        head_get_action = model.action_head.get_action

        def timed(name, fn):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return fn(*args, **kwargs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                torch.cuda.synchronize()
                self.times[name].append((time.perf_counter() - t0) * 1000)
                return out
            return wrapper

        model.backbone.forward = timed("backbone_ms", backbone_forward)
        model.action_head.get_action = timed("action_head_ms", head_get_action)

    def reset(self):
        for v in self.times.values():
            v.clear()


def load_observations(npz_path, n, seed=0):
    """n (wrist (H, W, 3) uint8, state (7,) float32) pairs from the ManiFeel demos, spread over all episodes."""
    data = np.load(npz_path)
    idx = np.random.default_rng(seed).choice(len(data["state"]), n, replace=False)
    state = data["state"][idx].astype(np.float32)
    quat = state[:, 3:7]  # sign-canonicalized as in training: largest component positive
    quat[quat[np.arange(n), np.abs(quat).argmax(1)] < 0] *= -1
    return data["wrist"][idx], state


def stats(ms):
    ms = np.asarray(ms)
    return {"median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()), "p90_ms": float(np.percentile(ms, 90)),
            "min_ms": float(ms.min()), "max_ms": float(ms.max()), "n": int(len(ms))}


def benchmark(call, timer, wrist, state, batch_sizes, repeats, warmup):
    """call(wrist (B, H, W, 3) uint8, state (B, 7)) -> action chunk; returns timing stats per batch size."""
    results = []
    for b in batch_sizes:
        idx = [(np.arange(b) + i * b) % len(state) for i in range(repeats)]  # different frames every call
        batches = [(wrist[i], state[i]) for i in idx]
        for i in range(warmup):
            call(*batches[i % len(batches)])
        # 1. end to end, no stage synchronization
        timer.enabled = False
        total = []
        for w, s in batches:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = call(w, s)
            torch.cuda.synchronize()
            total.append((time.perf_counter() - t0) * 1000)
        # 2. again with backbone / action head timed separately
        timer.reset()
        timer.enabled = True
        for w, s in batches:
            call(w, s)
        timer.enabled = False
        row = {"batch_size": b, "end_to_end": stats(total),
               **{k: stats(v) for k, v in timer.times.items()}}
        row["other_median_ms"] = (row["end_to_end"]["median_ms"] - row["backbone_ms"]["median_ms"]
                                  - row["action_head_ms"]["median_ms"])
        row["chunk_shape"] = list(np.asarray(out).shape)
        results.append(row)
        print(f"[{b:3d}] end-to-end {row['end_to_end']['median_ms']:8.2f} ms | backbone "
              f"{row['backbone_ms']['median_ms']:8.2f} ms | action head {row['action_head_ms']['median_ms']:7.2f} ms"
              f" | other {row['other_median_ms']:7.2f} ms", flush=True)
    return results


def write(out, model_name, checkpoint, extra, results):
    result = {"model": model_name, "checkpoint": checkpoint, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, **extra, "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
