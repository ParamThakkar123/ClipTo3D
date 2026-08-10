"""Fuse per-frame depth maps into a single consistent point cloud.

Rewritten to fix three things the previous `3d_pc/3dpc.py` got wrong:

1. **Scale (MPO-225).** It renormalized each frame's depth to a fixed
   `GLOBAL_MAX_DEPTH = 5.0`, giving every frame its own arbitrary scale, and it
   used relative inverse depth directly as Z. Metric depth is now recovered per
   frame by fitting `d = a/z + b` against that frame's COLMAP sparse points
   (see `depth_scale`).
2. **Wiring (MPO-224).** It read a hardcoded `gsplat_output/transforms.json`
   that no stage produced. It now reads the COLMAP model directly, which is
   also what the scale fit needs (transforms.json carries no sparse points).
3. **Fusion (MPO-239).** It sampled 50k random pixels per frame and `vstack`ed
   everything, so 300 frames meant up to 15M mostly-duplicate points held in
   RAM at once. Frames now stream into a `VoxelAccumulator`: one running entry
   per occupied voxel, so both the output size *and* peak memory follow scene
   extent rather than frame count.

A useful side effect: each frame counts as a single observation per voxel, so a
voxel's occupancy is the number of *frames* that saw it, and `--min-views`
filters on cross-frame agreement — a much better speckle signal than a raw
point count.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from colmap_io import Camera, find_model_dir, read_model  # noqa: E402
from depth_scale import fit_and_convert, robust_scene_extent  # noqa: E402
from depth_io import find_depth as _find_depth_file, load_depth  # noqa: E402
from pointcloud_io import VoxelAccumulator, write_ply  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _find_depth(depth_dir: Path, stem: str) -> Optional[Path]:
    """Locate a depth map in any supported storage format (MPO-240)."""
    return _find_depth_file(depth_dir, stem)


def _find_frame(frames_dir: Path, name: str) -> Optional[Path]:
    direct = frames_dir / Path(name).name
    if direct.is_file():
        return direct
    stem = Path(name).stem
    for ext in IMAGE_EXTS:
        p = frames_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _scaled_intrinsics(
    cam: Camera, depth_shape: Tuple[int, int]
) -> Tuple[float, float, float, float, float, float]:
    """Adjust intrinsics when the depth map resolution differs from the camera's.

    Returns (fx, fy, cx, cy, sx, sy) where sx/sy also convert COLMAP pixel
    coordinates into depth-map coordinates.
    """
    dh, dw = depth_shape
    sx = dw / float(cam.width)
    sy = dh / float(cam.height)
    fx, fy, cx, cy = cam.intrinsics
    return fx * sx, fy * sy, cx * sx, cy * sy, sx, sy


def _unproject(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject a metric depth map to (N,3) camera points plus a flat valid mask."""
    h, w = depth.shape[:2]
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    xv, yv = np.meshgrid(xs, ys)
    z = depth[yv, xv]
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return np.zeros((0, 3)), valid.reshape(-1)
    zv = z[valid].astype(np.float64)
    u = xv[valid].astype(np.float64)
    v = yv[valid].astype(np.float64)
    pts = np.stack([(u - cx) * zv / fx, (v - cy) * zv / fy, zv], axis=1)
    return pts, valid.reshape(-1)


def fuse(
    colmap_dir: Path | str,
    frames_dir: Path | str,
    depth_dir: Path | str,
    out_ply: Path | str,
    voxel_frac: float = 0.004,
    voxel_size: Optional[float] = None,
    min_views: int = 2,
    stride: int = 2,
    min_sparse_points: int = 20,
    max_depth_factor: float = 3.0,
    keep_colors: bool = True,
) -> Path:
    """Fuse depth maps into `out_ply` using COLMAP poses for scale and alignment.

    Voxel size is specified as `voxel_frac`, a fraction of the sparse-cloud
    radius, because **COLMAP's world scale is arbitrary**: the same physical
    scene can come out with a radius of 1 or of 100 depending on the solve. An
    absolute voxel size is therefore meaningless as a default — measured on one
    synthetic scene of true radius ~1.0, COLMAP produced radius 9.6, so a
    nominal 0.02 voxel was 0.2% of the scene and merged almost nothing. Pass
    `voxel_size` to override with an absolute value when the scale is known.

    `max_depth_factor` bounds recovered depth at a multiple of the sparse-cloud
    radius, discarding the runaway values the affine inversion produces near its
    horizon.
    """
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)
    out_ply = Path(out_ply)

    model_dir = find_model_dir(colmap_dir)
    cameras, images, points3D = read_model(model_dir)
    if not points3D:
        raise RuntimeError(
            f"{model_dir} has no points3D.txt. The sparse points are required to "
            f"recover metric depth scale; re-export the COLMAP model with points."
        )

    extent = robust_scene_extent(points3D)
    max_depth = max_depth_factor * extent if np.isfinite(extent) else None

    if voxel_size is None:
        if not np.isfinite(extent) or extent <= 0:
            raise RuntimeError(
                "cannot derive a voxel size from a degenerate sparse cloud; "
                "pass an explicit voxel_size"
            )
        voxel_size = voxel_frac * extent
    print(
        f"Model {model_dir}: {len(images)} images, {len(points3D)} sparse points, "
        f"radius ~{extent:.3f}, depth cap {max_depth}, voxel {voxel_size:.5g} "
        f"({100.0 * voxel_size / extent:.3f}% of radius)"
    )

    meta_path = depth_dir / "depth_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("is_disparity", True):
            print(
                "WARNING: depth_meta.json reports metric depth, but this stage fits "
                "an affine disparity model. Results may be wrong."
            )

    # Streaming: one running entry per occupied voxel, so peak memory follows
    # scene extent rather than frame count (MPO-239).
    accum = VoxelAccumulator(voxel_size)
    n_used = 0
    skipped: Dict[str, int] = {"no_frame": 0, "no_depth": 0, "fit_failed": 0, "no_points": 0}
    fit_reasons: List[str] = []

    for img in sorted(images.values(), key=lambda im: im.name):
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue

        frame_path = _find_frame(frames_dir, img.name)
        if frame_path is None:
            skipped["no_frame"] += 1
            continue

        depth_path = _find_depth(depth_dir, frame_path.stem)
        if depth_path is None:
            skipped["no_depth"] += 1
            continue

        disparity = load_depth(depth_path).astype(np.float64)
        if disparity.ndim == 3 and disparity.shape[2] == 1:
            disparity = disparity[:, :, 0]
        if disparity.ndim != 2:
            skipped["no_depth"] += 1
            continue

        fx, fy, cx, cy, sx, sy = _scaled_intrinsics(cam, disparity.shape[:2])

        # The scale fit samples disparity at COLMAP feature coordinates, so those
        # must be expressed in depth-map pixels when the resolutions differ.
        img_for_fit = img
        if not (np.isclose(sx, 1.0) and np.isclose(sy, 1.0)):
            img_for_fit = replace(img, xys=img.xys * np.array([sx, sy]))

        metric, fit = fit_and_convert(
            disparity, img_for_fit, points3D,
            min_points=min_sparse_points, max_depth=max_depth,
        )
        if metric is None:
            skipped["fit_failed"] += 1
            if len(fit_reasons) < 5:
                fit_reasons.append(f"{frame_path.name}: {fit.reason}")
            continue

        pts_cam, valid = _unproject(metric, fx, fy, cx, cy, stride)
        if len(pts_cam) == 0:
            skipped["no_points"] += 1
            continue

        pts_world = img.camera_to_world(pts_cam)

        cols = None
        if keep_colors:
            try:
                rgb = np.array(PILImage.open(frame_path).convert("RGB"))
                if rgb.shape[:2] != metric.shape[:2]:
                    rgb = np.array(
                        PILImage.fromarray(rgb).resize(
                            (metric.shape[1], metric.shape[0]), PILImage.Resampling.BILINEAR
                        )
                    )
                h, w = metric.shape[:2]
                ys = np.arange(0, h, stride)
                xs = np.arange(0, w, stride)
                xv, yv = np.meshgrid(xs, ys)
                cols = rgb[yv, xv].reshape(-1, 3)[valid]
            except Exception as e:
                print(f"WARNING: no colors for {frame_path.name}: {e}")
                cols = np.zeros((len(pts_world), 3), dtype=np.uint8)

        # One observation per frame: the accumulator collapses this frame's
        # points onto the grid and credits each touched voxel with one view.
        touched = accum.add(pts_world, cols)
        if touched:
            n_used += 1

    if accum.n_voxels == 0:
        raise RuntimeError(
            "No frame produced usable points.\n"
            f"  skipped: {skipped}\n"
            + ("  example fit failures:\n    " + "\n    ".join(fit_reasons) if fit_reasons else "")
        )

    print(f"Fused {n_used} frames -> {accum.n_voxels} occupied voxels; skipped {skipped}")
    if fit_reasons:
        print("Example scale-fit failures:\n  " + "\n  ".join(fit_reasons))

    # Occupancy is a frame count, so min_views filters on cross-frame agreement.
    final_pts, final_cols = accum.result(min_views=min_views)
    print(f"After cross-frame merge (min_views={min_views}): {len(final_pts)} points")

    if len(final_pts) == 0:
        raise RuntimeError(
            f"Cross-frame merge left no points with min_views={min_views}. "
            f"Try --min-views 1 or a larger --voxel-size."
        )

    write_ply(out_ply, final_pts, final_cols)
    print(f"Wrote {out_ply} ({len(final_pts)} points)")
    return out_ply


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(
        description="Fuse depth maps into one point cloud using COLMAP poses."
    )
    p.add_argument("--colmap-dir", type=Path, required=True,
                   help="COLMAP model dir, or any ancestor of it.")
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--depth-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("cloud/fused_cloud.ply"))
    p.add_argument("--voxel-frac", type=float, default=0.004,
                   help="Voxel size as a fraction of the sparse-cloud radius. Preferred, "
                        "because COLMAP's world scale is arbitrary.")
    p.add_argument("--voxel-size", type=float, default=None,
                   help="Absolute voxel size in COLMAP world units; overrides --voxel-frac.")
    p.add_argument("--min-views", type=int, default=2,
                   help="Discard voxels seen by fewer than this many frames.")
    p.add_argument("--stride", type=int, default=2, help="Pixel stride when unprojecting.")
    p.add_argument("--min-sparse-points", type=int, default=20,
                   help="Minimum sparse observations required to fit a frame's depth scale.")
    p.add_argument("--max-depth-factor", type=float, default=3.0,
                   help="Cap depth at this multiple of the sparse-cloud radius.")
    p.add_argument("--no-color", dest="keep_colors", action="store_false")
    args = p.parse_args(argv)

    fuse(
        colmap_dir=args.colmap_dir,
        frames_dir=args.frames_dir,
        depth_dir=args.depth_dir,
        out_ply=args.out,
        voxel_frac=args.voxel_frac,
        voxel_size=args.voxel_size,
        min_views=args.min_views,
        stride=args.stride,
        min_sparse_points=args.min_sparse_points,
        max_depth_factor=args.max_depth_factor,
        keep_colors=args.keep_colors,
    )


if __name__ == "__main__":
    main()
