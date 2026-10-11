"""Print the structure of a ManiFeel zarr dataset: arrays, shapes, sizes, episodes, value ranges.

    uv run --with "zarr<3" python inspect_dataset.py data/gear_quan_Sep15
"""

from pathlib import Path
import sys

import numpy as np
import zarr

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/gear_quan_Sep15")
store = zarr.open(str(root), mode="r")

ends = store["meta/episode_ends"][:]
lengths = np.diff(np.concatenate([[0], ends]))
print(f"{root.name}: {len(ends)} episodes, {ends[-1]} steps, "
      f"length min/mean/max {lengths.min()}/{lengths.mean():.0f}/{lengths.max()}\n")

print(f"{'array':32s} {'shape':22s} {'dtype':8s} {'chunks':22s} {'on disk':>9s}")
for name, arr in store["data"].arrays():
    disk = sum(f.stat().st_size for f in (root / "data" / name).iterdir())
    print(f"{name:32s} {str(arr.shape):22s} {str(arr.dtype):8s} {str(arr.chunks):22s} {disk / 1e6:7.0f} MB")

for key in ["state", "action"]:
    x = store["data"][key][:]
    print(f"\n{key}: min  {np.round(x.min(0), 4)}")
    print(f"{' ' * len(key)}  max  {np.round(x.max(0), 4)}")
    print(f"{' ' * len(key)}  ep0[:3]\n{np.round(x[:3], 4)}")
