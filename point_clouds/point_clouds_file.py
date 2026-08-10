"""Per-frame depth -> point cloud conversion.

The matplotlib preview that used to live here was removed (MPO-234): it opened
a blocking window, needed a display, and duplicated `fusion/view.py`, which is
now the single dev-only preview and sits behind the `viewer` extra.

NOTE ON SCALE: this module unprojects depth values as if they were metric Z.
MiDaS and Depth-Anything-v2 emit relative *inverse* depth, so the clouds it
produces are geometrically distorted and are not comparable between frames.
It is retained as a single-frame debugging aid.

For real reconstruction use `fusion.fuse`, which fits per-frame scale against
the COLMAP sparse points first (see `depth_scale`, MPO-225).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pointcloud_io import write_ply  # noqa: E402

def depth_to_point_cloud(depth: np.ndarray,
                         fx: float,
                         fy: float,
                         cx: float,
                         cy: float,
                         depth_scale: float = 1.0,
                         color: Optional[np.ndarray] = None,
                         mask_zero: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert a depth map (H x W) to an (N x 3) point cloud and optional (N x 3) colors.
    depth_scale: multiply depth values by this to convert to meters (or desired units).
    If mask_zero is True, pixels with depth == 0 are discarded.
    """
    h, w = depth.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float32) * depth_scale
    if mask_zero:
        valid = z > 0
    else:
        valid = np.ones_like(z, dtype=bool)

    uu = uu[valid]
    vv = vv[valid]
    zz = z[valid]

    x = (uu.astype(np.float32) - cx) * zz / fx
    y = (vv.astype(np.float32) - cy) * zz / fy

    points = np.stack([x, y, zz], axis=-1)  # (N,3)

    colors = None
    if color is not None:
        if color.shape[:2] != depth.shape:
            # try resizing color to depth shape if needed
            color_img = Image.fromarray(color)
            color = np.asarray(color_img.resize((w, h), resample=Image.BILINEAR))
        colors = color[valid]  # (N,3) RGB

    return points, colors

def save_ply(filename: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Save a point cloud to PLY.

    Thin wrapper over `pointcloud_io.write_ply`. The previous implementation here
    wrote ASCII with one formatted f-string per point — the third duplicate PLY
    writer in the repo, and the slowest.
    """
    write_ply(filename, points, colors)

def find_corresponding_color(depth_file: Path, frames_dir: Path) -> Optional[np.ndarray]:
    """
    Try to find a source color image in frames_dir that matches the depth file stem.
    depth files are expected like: frame_000001_depth.npy -> try frame_000001.* in frames_dir
    """
    stem = depth_file.stem
    if stem.endswith("_depth"):
        base = stem[:-6]
    else:
        base = stem

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    for ext in exts:
        candidate = frames_dir / f"{base}{ext}"
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGB")
                return np.asarray(img)
            except Exception:
                continue
    return None

def convert_all_depths(depth_dir: str = "./depth_maps",
                       out_dir: str = "./point_clouds",
                       frames_dir: str = "./frames",
                       focal_factor: float = 0.5,
                       depth_scale: float = 1.0,
                       include_color: bool = True) -> None:
    """
    Convert all *_depth.npy files in depth_dir to PLY point clouds in out_dir.

    - focal_factor: fx = fy = focal_factor * width (default 0.5 * width).
    - depth_scale: multiply depth values by this (useful if depths are normalized).
    """
    depth_path = Path(depth_dir)
    out_path = Path(out_dir)
    frames_path = Path(frames_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for p in sorted(depth_path.iterdir()):
        if not p.is_file() or not p.suffix.lower() == ".npy":
            continue
        try:
            depth = np.load(str(p))
        except Exception:
            print(f"Failed to load {p}, skipping.")
            continue

        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            print(f"Depth file {p} has unexpected shape {depth.shape}, skipping.")
            continue

        h, w = depth.shape
        fx = fy = float(w) * float(focal_factor)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        color = None
        if include_color:
            color = find_corresponding_color(p, frames_path)

        pts, cols = depth_to_point_cloud(depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=depth_scale, color=color)
        out_file = out_path / f"{p.stem}.ply"
        save_ply(str(out_file), pts, cols)
        print(f"Wrote {out_file} ({pts.shape[0]} points)")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Single-frame depth -> point cloud debug tool. See the module "
                    "docstring: output is NOT metrically correct. Use fusion.fuse "
                    "for real reconstruction. To look at a cloud, write it out "
                    "here and open it with `python fusion/view.py` (needs the "
                    "viewer extra)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Convert every depth map in a directory to PLY.")
    c.add_argument("--depth-dir", type=Path, default=Path("depth"))
    c.add_argument("--out-dir", type=Path, default=Path("cloud/per_frame"))
    c.add_argument("--frames-dir", type=Path, default=Path("frames"))
    c.add_argument("--focal-factor", type=float, default=0.5)
    c.add_argument("--depth-scale", type=float, default=1.0)
    c.add_argument("--no-color", dest="include_color", action="store_false")

    args = p.parse_args()

    convert_all_depths(
        depth_dir=str(args.depth_dir), out_dir=str(args.out_dir),
        frames_dir=str(args.frames_dir), focal_factor=args.focal_factor,
        depth_scale=args.depth_scale, include_color=args.include_color,
    )