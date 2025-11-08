from pathlib import Path
import numpy as np
from PIL import Image
from typing import Optional, Tuple

import matplotlib.pyplot as plt

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
    """
    Save point cloud to ASCII PLY.
    """
    assert points.ndim == 2 and points.shape[1] == 3
    n = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]

    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        if colors is not None:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

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

def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    """Return Nx3 uint8 colors from a color array (handles 0..1 floats or 0..255 ints)."""
    if colors.dtype == np.float32 or colors.dtype == np.float64:
        # assume in [0,1]
        c = np.clip(colors, 0.0, 1.0) * 255.0
        return c.astype(np.uint8)
    return colors.astype(np.uint8)

def visualize_point_cloud(points: np.ndarray,
                          colors: Optional[np.ndarray] = None,
                          max_points: int = 200000,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize a point cloud using matplotlib 3D scatter.

    - points: (N,3) numpy array
    - colors: optional (N,3) uint8 or float [0,1]
    - max_points: downsample to this many points for plotting speed
    """
    assert points.ndim == 2 and points.shape[1] == 3
    n = points.shape[0]
    if n == 0:
        print("No points to visualize.")
        return

    idx = None
    if n > max_points:
        # random downsample
        idx = np.random.choice(n, size=max_points, replace=False)
        pts = points[idx]
        cols = colors[idx] if colors is not None else None
    else:
        pts = points
        cols = colors

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title or "Point Cloud")

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    if cols is not None:
        cols = _normalize_colors(cols)
        ax.scatter(x, y, z, c=cols / 255.0, s=0.5, linewidths=0)
    else:
        ax.scatter(x, y, z, c="k", s=0.5, linewidths=0)

    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())
    zlim = (z.min(), z.max())
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) + 1e-6
    x_mid = 0.5*(xlim[0]+xlim[1])
    y_mid = 0.5*(ylim[0]+ylim[1])
    z_mid = 0.5*(zlim[0]+zlim[1])
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

    plt.show()

def visualize_depth_file(depth_file: Path,
                         frames_dir: Path = Path("./frames"),
                         focal_factor: float = 0.5,
                         depth_scale: float = 1.0,
                         include_color: bool = True,
                         max_points: int = 200000) -> None:
    """
    Load a depth .npy file (as used elsewhere in this module), convert to points and visualize.
    Uses [`depth_to_point_cloud`](point_clouds/point_clouds_file.py).
    """
    if not depth_file.exists():
        print(f"Depth file not found: {depth_file}")
        return

    depth = np.load(str(depth_file))
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        print(f"Unexpected depth shape: {depth.shape}")
        return

    h, w = depth.shape
    fx = fy = float(w) * float(focal_factor)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    color = None
    if include_color:
        color = find_corresponding_color(depth_file, Path(frames_dir))

    pts, cols = depth_to_point_cloud(depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=depth_scale, color=color)
    visualize_point_cloud(pts, colors=cols, max_points=max_points, title=str(depth_file.name))

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
    # convert_all_depths(depth_dir="./depth_maps", out_dir="./point_clouds", frames_dir="./frames",
    #                    focal_factor=0.5, depth_scale=1.0, include_color=True)

    from pathlib import Path
    visualize_depth_file(Path("./depth_maps/frame_000001_depth.npy"), frames_dir="./frames",
                         focal_factor=0.5, depth_scale=1.0, include_color=True)