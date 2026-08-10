"""Quick desktop preview of a fused point cloud.

DEBUG TOOL ONLY, and the only one left (MPO-234). It opens a blocking window
and needs a display, so it cannot run in the worker container, in a browser or
on a phone. The product viewer is the web viewer (MPO-246), reused by the
desktop shell (MPO-249) and mobile (MPO-248).

Nothing in the pipeline imports this module, and its dependencies are not in
the base install:

    uv sync --extra viewer
    python fusion/view.py runs/my-clip/cloud/fused.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pointcloud_io import read_ply  # noqa: E402

_MISSING_VIEWER_EXTRA = (
    "The debug preview needs the viewer extra, which is not part of the base "
    "install because it requires a desktop GL context: `uv sync --extra viewer`."
)


def preview(ply_path: Path, max_points: int = 50_000, backend: str = "matplotlib") -> None:
    pts, cols = read_ply(ply_path)
    n = len(pts)
    if n == 0:
        print(f"No points in {ply_path}")
        return
    print(f"{ply_path}: {n:,} points")

    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None
        print(f"Subsampled to {max_points:,} for display")

    if backend == "trimesh":
        try:
            import trimesh
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_MISSING_VIEWER_EXTRA) from exc

        cloud = trimesh.PointCloud(pts, colors=cols)
        cloud.show()
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_VIEWER_EXTRA) from exc

    fig = plt.figure(figsize=(9, 7))
    # Axes3D-only methods (set_zlabel/set_zlim) are not on the Axes stub.
    ax: Any = fig.add_subplot(projection="3d")
    ax.scatter(
        pts[:, 0], pts[:, 1], zs=pts[:, 2],
        c=(cols / 255.0) if cols is not None else "k", s=1, linewidths=0,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Equal aspect, so the scene is not visually distorted.
    span = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0 + 1e-6
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    ax.set_title(ply_path.name)
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preview a point cloud PLY (debug tool).")
    p.add_argument("ply", type=Path, help="Path to a .ply file.")
    p.add_argument("--max-points", type=int, default=50_000)
    p.add_argument("--backend", choices=["matplotlib", "trimesh"], default="matplotlib")
    args = p.parse_args()

    if not args.ply.is_file():
        raise SystemExit(f"No such file: {args.ply}")
    preview(args.ply, max_points=args.max_points, backend=args.backend)
