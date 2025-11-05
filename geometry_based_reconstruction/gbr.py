import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import argparse
from PIL import Image
import sys
import subprocess
from typing import cast

try:
    from structure_from_motion.sfm import list_frames, resolve_colmap_bin
except Exception:
    _workspace_root = Path(__file__).resolve().parents[1]
    if str(_workspace_root) not in sys.path:
        sys.path.insert(0, str(_workspace_root))
    from structure_from_motion.sfm import list_frames, resolve_colmap_bin  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_cameras_txt(cameras_txt: Path) -> Dict[int, Dict[str, Any]]:
    cams: Dict[int, Dict] = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=float)
            cams[cid] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cams

def quat_to_rotmatrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )
    return R

def parse_images_txt(images_txt: Path) -> Dict[int, Dict[str, Any]]:
    imgs: Dict[int, Dict] = {}
    with images_txt.open('r') as f:
        lines = [line.rstrip("\n") for line in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        qx, qy, qz, qw = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]
        R = quat_to_rotmatrix(np.array([qw, qx, qy, qz], dtype=float))
        t = np.array([tx, ty, tz], dtype=float)
        meta = {
            "R": R,
            "t": t,
            "camera_id": camera_id
        }
        # store under several keys to be resilient to path / stem differences
        imgs[image_name] = meta
        try:
            p = Path(image_name)
            basename = p.name
            stem = p.stem
            imgs[basename] = meta
            imgs[stem] = meta
            # also store lowercase variants
            imgs[basename.lower()] = meta
            imgs[stem.lower()] = meta
            imgs[str(p.as_posix())] = meta
            imgs[str(p.as_posix()).lower()] = meta
            # if images.txt contains a 'frames' segment, store the path after it
            parts_p = list(p.parts)
            if "frames" in parts_p:
                idx = parts_p.index("frames")
                rel = "/".join(parts_p[idx+1:]) or basename
                imgs[rel] = meta
                imgs[rel.lower()] = meta
        except Exception:
            pass
        while i < len(lines) and lines[i].strip():
            i += 1
    return imgs

def list_model_image_names(images_txt: Path) -> List[str]:
    """Return a list of image names present in a COLMAP images.txt (metadata lines only)."""
    names: List[str] = []
    if not images_txt.exists():
        return names
    with images_txt.open('r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # metadata line begins with image_id (integer) and has at least 10 tokens
            if parts and parts[0].lstrip("-").isdigit() and len(parts) >= 10:
                names.append(parts[9])
    return names

def intrinsics_from_camera(cam: Dict) -> Tuple[float, float, float, float]:
    model = cam["model"]
    p = cam["params"]
    if "SIMPLE_PINHOLE" in model:
        fx = float(p[0])
        fy = fx
        cx = float(p[1])
        cy = float(p[2])
    elif "PINHOLE" in model or "OPENCV" in model or "SIMPLE_RADIAL" in model:
        fx = float(p[0])
        fy = float(p[1]) if p.size > 1 else fx
        cx = float(p[2]) if p.size > 2 else 0.5 * cam["width"]
        cy = float(p[3]) if p.size > 3 else 0.5 * cam["height"]
    else:
        fx = float(p[0])
        fy = float(p[1]) if p.size > 1 else fx
        cx = float(p[2]) if p.size > 2 else 0.5 * cam["width"]
        cy = float(p[3]) if p.size > 3 else 0.5 * cam["height"]
    return fx, fy, cx, cy

def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth.shape[:2]
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xv, yv = np.meshgrid(xs, ys)
    z = depth[yv, xv]
    mask2d = np.isfinite(z) & (z > 0)
    if not mask2d.any():
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=bool)
    z_valid = z[mask2d]
    u = xv[mask2d].astype(float)
    v = yv[mask2d].astype(float)
    x = (u - cx) * z_valid / fx
    y = (v - cy) * z_valid / fy
    pts = np.stack([x, y, z_valid], axis=1)
    # return flattened 1D mask that corresponds to the flattened sampled grid
    mask_flat = mask2d.flatten()
    return pts, mask_flat

def voxel_grid_downsample(points: np.ndarray, colors: Optional[np.ndarray], voxel_size: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.size == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    voxels: Dict[Tuple[int, int, int], List[int]] = {}
    for i, k in enumerate(keys):
        key = (int(k[0]), int(k[1]), int(k[2]))
        voxels.setdefault(key, []).append(i)
    out_pts = []
    out_cols = [] if colors is not None else None
    for inds in voxels.values():
        pts_block = points[inds]
        centroid = np.mean(pts_block, axis=0)
        out_pts.append(centroid)
        if colors is not None:
            out_cols.append(np.mean(colors[inds], axis=0))
    if not out_pts:
        return np.empty((0, 3), dtype=float), (np.empty((0, 3), dtype=float) if out_cols is not None else None)
    out_pts = np.vstack(out_pts)
    if out_cols is not None:
        out_cols = np.vstack(out_cols)
    return out_pts, out_cols

def remove_sparse_voxels(points: np.ndarray, colors: Optional[np.ndarray], voxel_size: float, min_points: int = 3) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.size == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    counts: Dict[Tuple[int, int, int], int] = {}
    key_list = [tuple(k) for k in keys]
    for k in key_list:
        counts[k] = counts.get(k, 0) + 1
    keep_mask = np.array([counts[k] >= min_points for k in key_list], dtype=bool)
    kept_points = points[keep_mask]
    kept_colors = colors[keep_mask] if colors is not None else None
    return kept_points, kept_colors

def write_ply_ascii(points: np.ndarray, colors: Optional[np.ndarray], out_path: Path) -> None:
    n = len(points)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    with out_path.open('w') as f:
        f.write("\n".join(header) + "\n")
        if colors is not None:
            cols_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            for p, c in zip(points, cols_u8):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

def ensure_model_txt(sparse_dir: Path, model_txt_dir: Path, colmap_bin: Optional[str] = None) -> Path:
    """
    Ensure model_txt_dir contains cameras.txt and images.txt.
    If not present but binary files exist under sparse_dir, try to run `colmap model_converter`
    to produce TXT files into model_txt_dir.
    Returns the (hopefully) created model_txt_dir.
    """
    model_txt_dir = model_txt_dir.resolve()
    cameras_txt = model_txt_dir / "cameras.txt"
    images_txt = model_txt_dir / "images.txt"
    if cameras_txt.exists() and images_txt.exists():
        return model_txt_dir

    # if model_txt missing but binary files exist in sparse_dir, attempt conversion
    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
    if all((sparse_dir / b).exists() for b in bin_files):
        model_txt_dir.mkdir(parents=True, exist_ok=True)
        try:
            cb = resolve_colmap_bin(colmap_bin or "colmap")
        except Exception as e:
            logging.error("COLMAP binary not found to convert binary model to TXT: %s", e)
            return model_txt_dir

        cmd = [cb, "model_converter", "--input_path", str(sparse_dir), "--output_path", str(model_txt_dir), "--output_type", "TXT"]
        logging.info("Converting binary COLMAP model to TXT: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            logging.info("Model conversion complete: %s", model_txt_dir)
        except subprocess.CalledProcessError as e:
            logging.error("COLMAP model_converter failed: %s", e)
    else:
        logging.error("No model_txt found and binary model not present in %s", sparse_dir)
    return model_txt_dir

def reconstruct_point_cloud(
        frames_dir: Path,
        depth_dir: Path,
        colmap_model_txt_dir: Path,
        out_dir: Path,
        voxel_size: float = 0.05,
        stride: int = 4,
        min_voxel_points: int = 3,
        keep_colors: bool = True,
):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras_txt = colmap_model_txt_dir / "cameras.txt"
    images_txt = colmap_model_txt_dir / "images.txt"
    if not (cameras_txt.exists() and images_txt.exists()):
        if colmap_model_txt_dir.exists() and colmap_model_txt_dir.is_dir():
            numeric_dirs = sorted([d for d in colmap_model_txt_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            if numeric_dirs:
                best_dir = None
                best_pts = -1
                for d in numeric_dirs:
                    cams = d / "cameras.txt"
                    imgs = d / "images.txt"
                    pts = d / "points3D.txt"
                    if cams.exists() and imgs.exists():
                        pts_count = 0
                        if pts.exists():
                            with pts.open('r') as pf:
                                for line in pf:
                                    if line.strip() and not line.startswith("#"):
                                        pts_count += 1
                        if pts_count > best_pts:
                            best_pts = pts_count
                            best_dir = d
                if best_dir is None:
                    # none had both txt files, pick first numeric folder and hope it contains txts
                    best_dir = numeric_dirs[0]
                logging.info("Selecting COLMAP model folder %s inside %s", best_dir.name, colmap_model_txt_dir)
                colmap_model_txt_dir = best_dir
                cameras_txt = colmap_model_txt_dir / "cameras.txt"
                images_txt = colmap_model_txt_dir / "images.txt"

        # 2) if still not found, 3) try converting from binary model in parent sparse folder
        if not (cameras_txt.exists() and images_txt.exists()):
            sparse_dir = colmap_model_txt_dir.parent
            logging.info("cameras.txt/images.txt missing under %s, attempting to convert binary model from %s", colmap_model_txt_dir, sparse_dir)
            model_txt_dir = ensure_model_txt(sparse_dir, colmap_model_txt_dir)
            cameras_txt = model_txt_dir / "cameras.txt"
            images_txt = model_txt_dir / "images.txt"
            if not (cameras_txt.exists() and images_txt.exists()):
                raise FileNotFoundError(f"COLMAP model files not found in {colmap_model_txt_dir}; tried numeric subfolders and converting from {sparse_dir} but failed.")

    cams = parse_cameras_txt(cameras_txt)
    imgs = parse_images_txt(images_txt)
    model_names = list_model_image_names(images_txt)
    if model_names:
        logging.info("COLMAP model contains %d image entries. sample: %s", len(model_names), model_names[:8])
    else:
        logging.info("COLMAP images.txt contains no image metadata lines (or file missing).")

    frames = list_frames(frames_dir)
    logging.info(f"Found {len(frames)} frames in {frames_dir}")

    all_points = []
    all_cols = []

    for img_path in frames:
        name = img_path.name
        stem = img_path.stem
        # try multiple matching variants
        meta = imgs.get(name)
        if meta is None:
            meta = imgs.get(Path(name).name)
        if meta is None:
            meta = imgs.get(stem)
        if meta is None:
            # try with common image extensions
            for ext in (".jpg", ".jpeg", ".png"):
                meta = imgs.get(f"{stem}{ext}")
                if meta is not None:
                    break
        if meta is None:
            # try lowercase variants
            meta = imgs.get(name.lower()) or imgs.get(stem.lower())
        if meta is None:
            logging.warning(f"No COLMAP metadata for image {name}, skipping. (Try passing a numeric subfolder inside your model_txt or check naming.)")
            continue

        cam = cams.get(meta["camera_id"])
        if cam is None:
            logging.warning(f"No COLMAP camera info for image {name}, skipping.")
            continue
        fx, fy, cx, cy = intrinsics_from_camera(cam)

        candidate = depth_dir / f"{stem}_depth.npy"
        if not candidate.exists():
            candidate = depth_dir / f"{stem}.npy"
            if not candidate.exists():
                logging.warning(f"No depth file for image {name}, skipping.")
                continue
        depth = np.load(candidate)
        pts_cam, mask = backproject_depth(depth, fx, fy, cx, cy, stride=stride)
        if pts_cam.shape[0] == 0:
            continue
        R = meta["R"]
        t = meta["t"]
        pts_world = (R @ pts_cam.T).T + t.reshape(1, 3)
        all_points.append(pts_world)

        if keep_colors:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                H, W = depth.shape[:2]
                ys = np.arange(0, H, stride)
                xs = np.arange(0, W, stride)
                xv, yv = np.meshgrid(xs, ys)
                sampled = img[yv, xv].reshape(-1, 3)[mask]
                all_cols.append(sampled.astype(np.float64) / 255.0)
            except Exception:
                pass

    if not all_points:
        raise RuntimeError("No points were reconstructed from the provided data.")
    
    pts = np.vstack(all_points)
    cols = np.vstack(all_cols) if (keep_colors and all_cols) else None

    logging.info(f"Total points before downsampling: {pts.shape[0]}")

    # First compute voxel occupancy counts from the original dense points.
    # Keep only original points that belong to voxels with at least min_voxel_points,
    # then perform voxel grid downsampling on that filtered set.
    logging.info("Filtering points by voxel occupancy (voxel_size=%f, min_points=%d)", voxel_size, min_voxel_points)
    orig_keys = np.floor(pts / voxel_size).astype(np.int64)
    # count points per voxel
    counts: Dict[Tuple[int, int, int], int] = {}
    for k in (tuple(k) for k in orig_keys):
        counts[k] = counts.get(k, 0) + 1
    keep_mask = np.array([counts[tuple(k)] >= min_voxel_points for k in orig_keys], dtype=bool)
    pts_filtered = pts[keep_mask]
    cols_filtered = cols[keep_mask] if cols is not None else None
    logging.info("Points remaining after occupancy filter: %d", pts_filtered.shape[0])

    logging.info("Voxel downsampling (voxel_size=%f)", voxel_size)
    pts_clean, cols_clean = voxel_grid_downsample(pts_filtered, cols_filtered, voxel_size)

    if pts_clean.shape[0] == 0:
        logging.warning("Final cleaned point cloud has 0 points. Try lowering --voxel-size (e.g. 0.01) or using --min-voxel-points 1 to keep more points.")
        logging.info("Wrote empty/filtered PLY output anyway.")

    out_ply = out_dir / "reconstructed_point_cloud.ply"
    write_ply_ascii(pts_clean, cols_clean, out_ply)
    logging.info(f"Reconstructed point cloud saved to {out_ply}, total points: {pts_clean.shape[0]}")

    return out_ply

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geometry-Based 3D Reconstruction from Depth Maps")
    parser.add_argument(
        "--frames", "--frames-dir", "--frames_dir",
        dest="frames_dir",
        type=Path,
        default=Path("frames"),
        help="Directory containing input frames/images.",
    )
    parser.add_argument(
        "--depth-maps", "--depth_maps",
        dest="depth_maps",
        type=Path,
        default=Path("depth_maps"),
        help="Directory containing depth maps.",
    )
    parser.add_argument(
        "--colmap-txt", "--colmap_txt",
        dest="colmap_txt",
        type=Path,
        # changed default to use the numeric model folder 0 as requested
        default=Path("structure_from_motion/colmap_output/sparse/0"),
        help="COLMAP model_txt dir (cameras.txt, images.txt)",
    )
    parser.add_argument("--out", type=Path, default=Path("gsplat_output/geometry"), help="output dir")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--stride", type=int, default=2, help="subsample stride for backprojection")
    parser.add_argument("--min-voxel-points", type=int, default=3, help="minimum points per voxel to keep")
    parser.add_argument("--no-color", dest="keep_color", action="store_false")
    args = parser.parse_args()

    out_ply = reconstruct_point_cloud(
        frames_dir=args.frames_dir,
        depth_dir=args.depth_maps,
        colmap_model_txt_dir=args.colmap_txt,
        out_dir=args.out,
        voxel_size=args.voxel_size,
        stride=args.stride,
        min_voxel_points=args.min_voxel_points,
        keep_colors=args.keep_color
    )

    logging.info(f"Reconstruction completed. Output saved to {out_ply}")