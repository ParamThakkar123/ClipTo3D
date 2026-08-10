"""Point cloud I/O and voxel reduction.

Consolidates what used to be three separate PLY writers (`3d_pc/3dpc.py`,
`point_clouds/point_clouds_file.py`, `geometry_based_reconstruction/gbr.py`)
and two separate voxel-downsample implementations.

All of the previous writers emitted one Python loop iteration per point — two
`f.write` calls per point in the binary case, and a formatted f-string per
point in the ASCII case. The implementations here are vectorized: the whole
cloud is packed into one structured array and written in a single call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_VERTEX_DTYPE_RGB = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
_VERTEX_DTYPE_XYZ = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])


def normalize_colors(colors: np.ndarray) -> np.ndarray:
    """Return (N,3) uint8 colors from either float [0,1] or integer [0,255]."""
    c = np.asarray(colors)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"colors must be (N,3), got {c.shape}")
    if np.issubdtype(c.dtype, np.floating):
        return np.clip(c * 255.0, 0, 255).astype(np.uint8)
    return np.clip(c, 0, 255).astype(np.uint8)


def write_ply(
    path: Path | str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    binary: bool = True,
) -> Path:
    """Write an (N,3) point cloud, optionally with (N,3) colors, to PLY.

    Binary little-endian by default. ASCII is available for debugging but is
    both slower to write and roughly 3x larger on disk.
    """
    path = Path(path)
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    n = len(pts)

    if colors is not None:
        cols = normalize_colors(colors)
        if len(cols) != n:
            raise ValueError(f"points/colors length mismatch: {n} vs {len(cols)}")
    else:
        cols = None

    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["ply"]
    header.append("format binary_little_endian 1.0" if binary else "format ascii 1.0")
    header.append(f"element vertex {n}")
    header += ["property float x", "property float y", "property float z"]
    if cols is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    if binary:
        dt = _VERTEX_DTYPE_RGB if cols is not None else _VERTEX_DTYPE_XYZ
        rec = np.empty(n, dtype=dt)
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        if cols is not None:
            rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
        with open(path, "wb") as f:
            f.write(header_bytes)
            f.write(rec.tobytes())
    else:
        if cols is not None:
            body = np.column_stack([pts, cols.astype(np.int64)])
            fmt = "%.6f %.6f %.6f %d %d %d"
        else:
            body = pts
            fmt = "%.6f %.6f %.6f"
        with open(path, "wb") as f:
            f.write(header_bytes)
            np.savetxt(f, body, fmt=fmt)

    return path


def read_ply(path: Path | str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a PLY into (points (N,3) float32, colors (N,3) uint8 or None)."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    names = v.dtype.names or ()
    if {"red", "green", "blue"} <= set(names):
        cols = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    else:
        cols = None
    return pts, cols


class VoxelAccumulator:
    """Streaming voxel grid (MPO-239).

    Fusion used to buffer every frame's points and `vstack` them at the end,
    so peak memory grew with the frame count — up to ~15M points for a
    300-frame clip, nearly all of them re-observations of the same surfaces.

    This holds one running entry per *occupied voxel* instead, so peak memory
    is bounded by scene extent and voxel size, flat in the number of frames.

    Each `add()` is treated as one observation: points landing in the same
    voxel within a single call are averaged and counted once. That is what
    makes the final occupancy count a count of *frames* that saw the voxel,
    which is a far better speckle signal than a raw point count — real
    surfaces are seen by several frames, isolated depth noise is not.
    """

    def __init__(self, voxel_size: float):
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        self.voxel_size = float(voxel_size)
        self._keys = np.zeros((0, 3), dtype=np.int64)
        self._sum_xyz = np.zeros((0, 3), dtype=np.float64)
        self._sum_rgb = np.zeros((0, 3), dtype=np.float64)
        self._counts = np.zeros((0,), dtype=np.int64)
        self._with_colors: Optional[bool] = None
        self.n_observations = 0

    @property
    def n_voxels(self) -> int:
        return len(self._keys)

    def add(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> int:
        """Merge one observation (typically one frame). Returns voxels touched."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            self.n_observations += 1
            return 0

        if self._with_colors is None:
            self._with_colors = colors is not None
        elif self._with_colors != (colors is not None):
            raise ValueError("colors must be supplied for either every observation or none")

        cols = None
        if colors is not None:
            cols = normalize_colors(colors).astype(np.float64)
            if len(cols) != len(pts):
                raise ValueError(f"points/colors length mismatch: {len(pts)} vs {len(cols)}")

        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            pts = pts[finite]
            if cols is not None:
                cols = cols[finite]
        if pts.size == 0:
            self.n_observations += 1
            return 0

        keys = np.floor(pts / self.voxel_size).astype(np.int64)

        # Collapse within this observation first, so the voxel is credited with
        # exactly one view no matter how many of its pixels landed there.
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        n_new = len(uniq)
        new_counts_in_batch = np.bincount(inverse, minlength=n_new).astype(np.float64)

        new_sum_xyz = np.empty((n_new, 3), dtype=np.float64)
        for d in range(3):
            new_sum_xyz[:, d] = np.bincount(inverse, weights=pts[:, d], minlength=n_new)
        new_sum_xyz /= new_counts_in_batch[:, None]

        if cols is not None:
            new_sum_rgb = np.empty((n_new, 3), dtype=np.float64)
            for d in range(3):
                new_sum_rgb[:, d] = np.bincount(inverse, weights=cols[:, d], minlength=n_new)
            new_sum_rgb /= new_counts_in_batch[:, None]
        else:
            new_sum_rgb = np.zeros((n_new, 3), dtype=np.float64)

        # Merge into the running grid. The temporary is 2x the occupied voxel
        # count, which is still independent of how many frames have been added.
        all_keys = np.concatenate([self._keys, uniq])
        all_xyz = np.concatenate([self._sum_xyz, new_sum_xyz])
        all_rgb = np.concatenate([self._sum_rgb, new_sum_rgb])
        all_counts = np.concatenate([self._counts, np.ones(n_new, dtype=np.int64)])

        merged, inv2 = np.unique(all_keys, axis=0, return_inverse=True)
        inv2 = inv2.ravel()
        m = len(merged)

        sum_xyz = np.empty((m, 3), dtype=np.float64)
        sum_rgb = np.empty((m, 3), dtype=np.float64)
        for d in range(3):
            sum_xyz[:, d] = np.bincount(inv2, weights=all_xyz[:, d], minlength=m)
            sum_rgb[:, d] = np.bincount(inv2, weights=all_rgb[:, d], minlength=m)

        self._keys = merged
        self._sum_xyz = sum_xyz
        self._sum_rgb = sum_rgb
        self._counts = np.bincount(inv2, weights=all_counts, minlength=m).astype(np.int64)
        self.n_observations += 1
        return n_new

    def result(self, min_views: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Voxel centroids, dropping those seen by fewer than `min_views`."""
        if len(self._keys) == 0:
            empty_c = np.zeros((0, 3), dtype=np.uint8) if self._with_colors else None
            return np.zeros((0, 3), dtype=np.float32), empty_c

        keep = self._counts >= max(1, min_views)
        counts = self._counts[keep][:, None].astype(np.float64)
        pts = (self._sum_xyz[keep] / counts).astype(np.float32)
        if self._with_colors:
            cols = np.clip(np.rint(self._sum_rgb[keep] / counts), 0, 255).astype(np.uint8)
        else:
            cols = None
        return pts, cols


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.01,
    min_points_per_voxel: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Collapse points onto a voxel grid, averaging position and color.

    This is what turns overlapping per-frame observations of one surface into a
    single set of points. Without it, N frames of the same wall produce N
    superimposed copies of that wall.

    `min_points_per_voxel` additionally drops voxels seen too few times, which
    removes most depth-estimation speckle: real surface points get observed by
    several frames, isolated noise does not.

    Returns cloud unchanged in dtype semantics: points float32, colors uint8.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        empty_c = np.zeros((0, 3), dtype=np.uint8) if colors is not None else None
        return np.zeros((0, 3), dtype=np.float32), empty_c
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    cols = normalize_colors(colors).astype(np.float64) if colors is not None else None
    if cols is not None and len(cols) != len(pts):
        raise ValueError(f"points/colors length mismatch: {len(pts)} vs {len(cols)}")

    # Drop non-finite points up front; they would poison the grid indices.
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pts = pts[finite]
        if cols is not None:
            cols = cols[finite]
        if pts.size == 0:
            empty_c = np.zeros((0, 3), dtype=np.uint8) if colors is not None else None
            return np.zeros((0, 3), dtype=np.float32), empty_c

    keys = np.floor(pts / voxel_size).astype(np.int64)
    _uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n_vox = len(counts)

    # Per-voxel centroid via bincount, which stays in C rather than looping in
    # Python over the voxel dictionary.
    centroids = np.empty((n_vox, 3), dtype=np.float64)
    for d in range(3):
        centroids[:, d] = np.bincount(inverse, weights=pts[:, d], minlength=n_vox)
    centroids /= counts[:, None]

    if cols is not None:
        mean_cols = np.empty((n_vox, 3), dtype=np.float64)
        for d in range(3):
            mean_cols[:, d] = np.bincount(inverse, weights=cols[:, d], minlength=n_vox)
        mean_cols /= counts[:, None]
    else:
        mean_cols = None

    if min_points_per_voxel > 1:
        keep = counts >= min_points_per_voxel
        centroids = centroids[keep]
        if mean_cols is not None:
            mean_cols = mean_cols[keep]

    out_pts = centroids.astype(np.float32)
    out_cols = (
        np.clip(np.rint(mean_cols), 0, 255).astype(np.uint8) if mean_cols is not None else None
    )
    return out_pts, out_cols
