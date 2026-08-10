"""Geometry-based reconstruction: fuse per-frame depth maps into one point cloud.

This is the non-neural alternative to gaussian splatting. It unprojects each
frame's depth map using the COLMAP pose and intrinsics, then reduces the union
onto a voxel grid.

All COLMAP parsing now lives in `colmap_io`, and all point-cloud writing and
voxel reduction in `pointcloud_io`. The versions previously inlined here had
three defects (MPO-228):

* the quaternion was unpacked as `qx, qy, qz, qw` from COLMAP's `QW QX QY QZ`
  column order, rotating the components by one position — every pose was wrong;
* camera-to-world was computed as `R @ X_cam + t`, which is neither the COLMAP
  transform nor its inverse (correct is `R^T (X_cam - t)`);
* `SIMPLE_RADIAL` (`f, cx, cy, k`) intrinsics were read as `fx, fy, cx, cy`.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage

_workspace_root = Path(__file__).resolve().parents[1]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from colmap_io import find_model_dir, images_by_name, read_model  # noqa: E402
from pointcloud_io import voxel_downsample, write_ply  # noqa: E402
from structure_from_motion.sfm import list_frames, resolve_colmap_bin  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def backproject_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map to (N,3) camera-frame points.

    Returns the points plus a flat boolean mask over the strided sample grid,
    so a caller can select matching colors from the source image.
    """
    H, W = depth.shape[:2]
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xv, yv = np.meshgrid(xs, ys)
    z = depth[yv, xv]
    mask2d = np.isfinite(z) & (z > 0)
    if not mask2d.any():
        return np.empty((0, 3), dtype=float), mask2d.reshape(-1)
    z_valid = z[mask2d].astype(float)
    u = xv[mask2d].astype(float)
    v = yv[mask2d].astype(float)
    pts = np.stack([(u - cx) * z_valid / fx, (v - cy) * z_valid / fy, z_valid], axis=1)
    return pts, mask2d.reshape(-1)


def ensure_model_txt(
    sparse_dir: Path, model_txt_dir: Path, colmap_bin: Optional[str] = None
) -> Path:
    """Make sure `model_txt_dir` holds a text model, converting the binary one if needed."""
    model_txt_dir = Path(model_txt_dir).resolve()
    if (model_txt_dir / "cameras.txt").exists() and (model_txt_dir / "images.txt").exists():
        return model_txt_dir

    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
    if not all((sparse_dir / b).exists() for b in bin_files):
        logging.error("No text model in %s and no binary model in %s", model_txt_dir, sparse_dir)
        return model_txt_dir

    model_txt_dir.mkdir(parents=True, exist_ok=True)
    try:
        cb = resolve_colmap_bin(colmap_bin)
    except Exception as e:
        logging.error("COLMAP binary not found, cannot convert binary model to TXT: %s", e)
        return model_txt_dir

    cmd = [
        cb, "model_converter",
        "--input_path", str(sparse_dir),
        "--output_path", str(model_txt_dir),
        "--output_type", "TXT",
    ]
    logging.info("Converting binary COLMAP model to TXT: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("COLMAP model_converter failed: %s", e)
    return model_txt_dir


def _load_depth(depth_dir: Path, stem: str) -> Optional[np.ndarray]:
    for name in (f"{stem}_depth.npy", f"{stem}.npy"):
        p = depth_dir / name
        if p.exists():
            d = np.load(p)
            if d.ndim == 3 and d.shape[2] == 1:
                d = d[:, :, 0]
            if d.ndim != 2:
                logging.warning("Depth %s has unexpected shape %s, skipping.", p, d.shape)
                return None
            return d.astype(np.float32)
    return None


def reconstruct_point_cloud(
    frames_dir: Path,
    depth_dir: Path,
    colmap_model_txt_dir: Path,
    out_dir: Path,
    voxel_size: float = 0.05,
    stride: int = 4,
    min_voxel_points: int = 3,
    keep_colors: bool = True,
    out_name: str = "reconstructed_point_cloud.ply",
) -> Path:
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate the model, converting from binary if only that exists.
    try:
        model_dir = find_model_dir(colmap_model_txt_dir)
    except FileNotFoundError:
        model_dir = find_model_dir(
            ensure_model_txt(Path(colmap_model_txt_dir).parent, Path(colmap_model_txt_dir))
        )

    cameras, images, _ = read_model(model_dir)
    by_name = images_by_name(images)
    logging.info("COLMAP model %s: %d images, %d cameras", model_dir, len(images), len(cameras))

    frames = list_frames(frames_dir)
    logging.info("Found %d frames in %s", len(frames), frames_dir)

    all_points, all_cols = [], []
    skipped_no_pose = skipped_no_depth = 0

    for img_path in frames:
        img_meta = by_name.get(img_path.name)
        if img_meta is None:
            skipped_no_pose += 1
            continue

        cam = cameras.get(img_meta.camera_id)
        if cam is None:
            logging.warning("No camera %s for image %s", img_meta.camera_id, img_path.name)
            continue
        fx, fy, cx, cy = cam.intrinsics

        depth = _load_depth(depth_dir, img_path.stem)
        if depth is None:
            skipped_no_depth += 1
            continue

        pts_cam, mask = backproject_depth(depth, fx, fy, cx, cy, stride=stride)
        if pts_cam.shape[0] == 0:
            continue

        all_points.append(img_meta.camera_to_world(pts_cam))

        if keep_colors:
            try:
                rgb = np.array(PILImage.open(img_path).convert("RGB"))
                H, W = depth.shape[:2]
                ys = np.arange(0, H, stride)
                xs = np.arange(0, W, stride)
                xv, yv = np.meshgrid(xs, ys)
                sampled = rgb[yv, xv].reshape(-1, 3)[mask]
                all_cols.append(sampled)
            except Exception as e:
                logging.warning("Could not sample colors from %s: %s", img_path.name, e)
                all_cols.append(np.zeros((len(pts_cam), 3), dtype=np.uint8))

    if skipped_no_pose:
        logging.warning(
            "%d frames had no COLMAP pose (not registered, or filename mismatch).",
            skipped_no_pose,
        )
    if skipped_no_depth:
        logging.warning("%d frames had no depth map in %s.", skipped_no_depth, depth_dir)
    if not all_points:
        raise RuntimeError(
            "No points were reconstructed. Check that frame filenames match the COLMAP "
            "model and that depth maps exist."
        )

    pts = np.vstack(all_points)
    cols = np.vstack(all_cols) if (keep_colors and all_cols) else None
    logging.info("Points before voxel reduction: %d", len(pts))

    pts_clean, cols_clean = voxel_downsample(
        pts, cols, voxel_size=voxel_size, min_points_per_voxel=min_voxel_points
    )
    logging.info("Points after voxel reduction: %d", len(pts_clean))

    if len(pts_clean) == 0:
        logging.warning(
            "Empty cloud after filtering. Try a smaller --voxel-size or --min-voxel-points 1."
        )

    out_ply = out_dir / out_name
    write_ply(out_ply, pts_clean, cols_clean)
    logging.info("Wrote %s (%d points)", out_ply, len(pts_clean))
    return out_ply


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geometry-based 3D reconstruction from depth maps")
    parser.add_argument(
        "--frames", "--frames-dir", "--frames_dir",
        dest="frames_dir", type=Path, default=Path("frames"),
    )
    parser.add_argument(
        "--depth-maps", "--depth_maps", dest="depth_maps", type=Path, default=Path("depth_maps")
    )
    parser.add_argument(
        "--colmap-txt", "--colmap_txt",
        dest="colmap_txt", type=Path,
        default=Path("structure_from_motion/colmap_output/sparse/model_txt"),
        help="COLMAP model dir, or any ancestor of it.",
    )
    parser.add_argument("--out", type=Path, default=Path("cloud"))
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--stride", type=int, default=2, help="Subsample stride for backprojection.")
    parser.add_argument("--min-voxel-points", type=int, default=3)
    parser.add_argument("--no-color", dest="keep_color", action="store_false")
    args = parser.parse_args()

    reconstruct_point_cloud(
        frames_dir=args.frames_dir,
        depth_dir=args.depth_maps,
        colmap_model_txt_dir=args.colmap_txt,
        out_dir=args.out,
        voxel_size=args.voxel_size,
        stride=args.stride,
        min_voxel_points=args.min_voxel_points,
        keep_colors=args.keep_color,
    )
