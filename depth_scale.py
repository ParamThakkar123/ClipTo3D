"""Recover metric depth from relative-inverse-depth predictions (MPO-225).

MiDaS and Depth-Anything-v2 emit *relative inverse* depth (disparity). The
relation to true metric depth z is affine **in disparity space**:

    d_pred  ~=  a * (1/z)  +  b

with a and b unknown and different for every frame. That is the affine-invariant
formulation the MiDaS / Depth-Anything papers use for evaluation.

The previous fusion code did neither half of this. It fed disparity straight in
as Z, and it rescaled each frame independently to a fixed `GLOBAL_MAX_DEPTH`,
so every frame ended up on its own arbitrary scale and the per-frame clouds
could not align into one world however good the poses were.

Here (a, b) are fitted per frame by least squares against the COLMAP sparse
points visible in that frame, which are already in a single consistent world
scale. Frames without enough well-conditioned observations are rejected rather
than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from colmap_io import Image, Point3D


@dataclass
class DepthScaleFit:
    """Result of fitting `d = a/z + b` for one frame."""

    scale: float  # a
    shift: float  # b
    n_observations: int
    n_inliers: int
    rmse_disparity: float
    ok: bool
    reason: str = ""

    def to_metric(self, disparity: np.ndarray, max_depth: Optional[float] = None) -> np.ndarray:
        """Invert the fit: z = a / (d - b). Invalid pixels come back as NaN."""
        if not self.ok:
            raise ValueError(f"cannot apply a failed fit: {self.reason}")
        d = np.asarray(disparity, dtype=np.float64)
        denom = d - self.shift
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.scale / denom
        # Points at or behind the horizon implied by the fit, plus anything
        # non-finite, are not recoverable.
        z[~np.isfinite(z)] = np.nan
        z[denom <= 0] = np.nan
        z[z <= 0] = np.nan
        if max_depth is not None:
            z[z > max_depth] = np.nan
        return z


def sparse_observations(
    image: Image, points3D: Dict[int, Point3D]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xys (N,2) pixel coords, z_ref (N,) camera-frame depths) for one image.

    Only observations with a valid triangulated 3D point are returned. Depth is
    the Z component in this camera's frame, which is what a pinhole unprojection
    consumes.
    """
    ids = np.asarray(image.point3D_ids)
    xys = np.asarray(image.xys, dtype=np.float64).reshape(-1, 2)
    if ids.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))

    keep = []
    world = []
    for i, pid in enumerate(ids):
        pt = points3D.get(int(pid))
        if pt is None:
            continue
        keep.append(i)
        world.append(pt.xyz)

    if not keep:
        return np.zeros((0, 2)), np.zeros((0,))

    z_ref = image.world_to_camera(np.asarray(world))[:, 2]
    return xys[keep], z_ref


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D array at float pixel coordinates."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    x = np.clip(np.asarray(xs, dtype=np.float64), 0, w - 1)
    y = np.clip(np.asarray(ys, dtype=np.float64), 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    return (
        a[y0, x0] * (1 - wx) * (1 - wy)
        + a[y0, x1] * wx * (1 - wy)
        + a[y1, x0] * (1 - wx) * wy
        + a[y1, x1] * wx * wy
    )


def fit_depth_scale(
    disparity: np.ndarray,
    xys: np.ndarray,
    z_ref: np.ndarray,
    min_points: int = 20,
    trim_iters: int = 3,
    trim_sigma: float = 3.0,
    min_inv_depth_spread: float = 1e-3,
) -> DepthScaleFit:
    """Fit `d = a/z + b` for one frame against its sparse observations.

    Outliers (mismatched features, depth-model failures on thin structure) are
    removed by iterated sigma trimming on the disparity residual, using a
    median-absolute-deviation scale so a few gross outliers cannot inflate the
    threshold that is supposed to catch them.
    """
    xys = np.asarray(xys, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(z_ref, dtype=np.float64).reshape(-1)
    n_obs = len(z)

    if n_obs < min_points:
        return DepthScaleFit(
            np.nan, np.nan, n_obs, 0, np.nan, False,
            f"only {n_obs} sparse observations, need {min_points}",
        )

    d = sample_bilinear(disparity, xys[:, 0], xys[:, 1])

    # Points behind the camera carry no depth information.
    valid = np.isfinite(d) & np.isfinite(z) & (z > 0)
    if valid.sum() < min_points:
        return DepthScaleFit(
            np.nan, np.nan, n_obs, int(valid.sum()), np.nan, False,
            f"only {int(valid.sum())} usable observations, need {min_points}",
        )

    d = d[valid]
    inv_z = 1.0 / z[valid]

    mask = np.ones(len(d), dtype=bool)
    a = b = np.nan
    rmse = np.nan

    for _ in range(max(1, trim_iters)):
        if mask.sum() < min_points:
            break
        iz = inv_z[mask]
        # Without spread in 1/z the slope is unidentifiable — a fronto-parallel
        # view of a single plane, or a degenerate track set.
        if np.ptp(iz) < min_inv_depth_spread * max(1.0, float(np.abs(iz).max())):
            return DepthScaleFit(
                np.nan, np.nan, n_obs, int(mask.sum()), np.nan, False,
                "sparse depths span too little range to identify a scale",
            )

        A = np.column_stack([iz, np.ones(mask.sum())])
        sol, *_ = np.linalg.lstsq(A, d[mask], rcond=None)
        a, b = float(sol[0]), float(sol[1])

        resid_all = d - (a * inv_z + b)
        rmse = float(np.sqrt(np.mean(resid_all[mask] ** 2)))

        med = np.median(resid_all[mask])
        mad = np.median(np.abs(resid_all[mask] - med))
        if mad <= 0:
            break
        sigma = 1.4826 * mad  # MAD -> Gaussian-equivalent sigma
        new_mask = np.abs(resid_all - med) <= trim_sigma * sigma
        if new_mask.sum() < min_points or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    n_in = int(mask.sum())
    if not np.isfinite(a) or not np.isfinite(b):
        return DepthScaleFit(np.nan, np.nan, n_obs, n_in, np.nan, False, "least squares failed")
    if a <= 0:
        # Disparity must increase as 1/z increases. A non-positive slope means
        # the prediction is anti-correlated with geometry — unusable.
        return DepthScaleFit(
            a, b, n_obs, n_in, rmse, False,
            f"non-positive scale {a:.4g}: prediction anti-correlated with sparse depths",
        )

    return DepthScaleFit(a, b, n_obs, n_in, rmse, True)


def fit_and_convert(
    disparity: np.ndarray,
    image: Image,
    points3D: Dict[int, Point3D],
    min_points: int = 20,
    max_depth: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], DepthScaleFit]:
    """Convenience wrapper: fit against one image's sparse points and convert.

    Returns (metric_depth or None, fit).
    """
    xys, z_ref = sparse_observations(image, points3D)
    fit = fit_depth_scale(disparity, xys, z_ref, min_points=min_points)
    if not fit.ok:
        return None, fit
    return fit.to_metric(disparity, max_depth=max_depth), fit


def robust_scene_extent(points3D: Dict[int, Point3D], percentile: float = 99.0) -> float:
    """A robust scene radius from the sparse cloud, for depth sanity limits."""
    if not points3D:
        return float("inf")
    xyz = np.array([p.xyz for p in points3D.values()])
    centre = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - centre, axis=1)
    return float(np.percentile(dist, percentile))
