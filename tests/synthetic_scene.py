"""Generate a synthetic textured 3D scene that COLMAP can actually reconstruct.

Used by the integration test to validate the COLMAP readers and the pose
convention against *real* COLMAP output rather than hand-written fixtures.

The scene is three richly textured planes arranged as a corner, so it has
genuine 3D structure (a single plane is a degenerate SfM configuration). Views
are rendered by projecting each plane's corners and warping its texture through
the resulting homography, which yields real, SIFT-matchable imagery with exactly
known ground-truth camera poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

TEX = 512  # texture resolution


def make_texture(seed: int, size: int = TEX) -> np.ndarray:
    """High-frequency textured image with plenty of corners for SIFT."""
    rng = np.random.default_rng(seed)
    img = (rng.random((size // 8, size // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    for _ in range(60):
        x, y = rng.integers(0, size, 2)
        w, h = rng.integers(size // 24, size // 6, 2)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (min(x + w, size), min(y + h, size)), color, -1)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(0, size, 2))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(img, c, int(rng.integers(size // 40, size // 12)), color, -1)
    for _ in range(30):
        p1 = tuple(int(v) for v in rng.integers(0, size, 2))
        p2 = tuple(int(v) for v in rng.integers(0, size, 2))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.line(img, p1, p2, color, int(rng.integers(1, 4)))

    return cv2.GaussianBlur(img, (3, 3), 0)


@dataclass
class Plane:
    """A textured quad. `corners` are world-space, in texture order TL,TR,BR,BL."""

    corners: np.ndarray  # (4,3)
    texture: np.ndarray

    @property
    def centroid(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def default_planes() -> List[Plane]:
    """Three planes forming a corner around the origin."""
    return [
        # Back wall, in the z = 2 plane.
        Plane(
            np.array([[-1.2, -0.9, 2.0], [1.2, -0.9, 2.0], [1.2, 0.9, 2.0], [-1.2, 0.9, 2.0]]),
            make_texture(1),
        ),
        # Left wall, receding in z.
        Plane(
            np.array([[-1.2, -0.9, 0.4], [-1.2, -0.9, 2.0], [-1.2, 0.9, 2.0], [-1.2, 0.9, 0.4]]),
            make_texture(2),
        ),
        # Floor.
        Plane(
            np.array([[-1.2, 0.9, 0.4], [1.2, 0.9, 0.4], [1.2, 0.9, 2.0], [-1.2, 0.9, 2.0]]),
            make_texture(3),
        ),
    ]


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, -1.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Return COLMAP-convention (R, t) mapping world -> camera.

    Camera looks down +Z with +Y down, matching COLMAP/OpenCV.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up_v = np.asarray(up, dtype=float)
    right = np.cross(up_v, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    R = np.stack([right, true_up, forward], axis=0)  # world -> camera
    t = -R @ eye
    return R, t


def camera_arc(n: int = 14, radius: float = 1.9, height: float = 0.0) -> List[np.ndarray]:
    """Eye positions on an arc in front of the scene, giving real parallax."""
    eyes = []
    for i in range(n):
        a = np.deg2rad(-32.0 + 64.0 * i / max(1, n - 1))
        eyes.append(
            np.array([radius * np.sin(a), height + 0.12 * np.sin(2 * a), 1.2 - radius * np.cos(a)])
        )
    return eyes


def render_view(
    planes: List[Plane], R: np.ndarray, t: np.ndarray, K: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Render the planes from one camera by per-plane homography warping."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Painter's algorithm: draw far planes first.
    order = sorted(planes, key=lambda p: -float((R @ p.centroid + t)[2]))

    for plane in order:
        cam = (R @ plane.corners.T).T + t
        if np.any(cam[:, 2] <= 0.05):  # any corner behind/too near the camera
            continue
        proj = (K @ cam.T).T
        img_pts = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

        # Skip wildly off-screen quads, which make the homography ill-conditioned.
        if np.any(np.abs(img_pts) > 20 * max(width, height)):
            continue

        s = plane.texture.shape[0]
        src = np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32)
        try:
            H = cv2.getPerspectiveTransform(src, img_pts)
        except cv2.error:
            continue

        warped = cv2.warpPerspective(plane.texture, H, (width, height))
        mask = cv2.warpPerspective(
            np.full(plane.texture.shape[:2], 255, np.uint8), H, (width, height)
        )
        canvas[mask > 127] = warped[mask > 127]

    return canvas


def render_depth(
    planes: List[Plane], R: np.ndarray, t: np.ndarray, K: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Render a ground-truth metric depth map (camera-frame Z) by ray-plane intersection.

    Returns NaN where no plane is hit.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Ray directions in the camera frame, normalized so z == 1.
    dirs = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)

    best = np.full((height, width), np.inf)
    for plane in planes:
        cam = (R @ plane.corners.T).T + t  # (4,3) in camera frame
        origin = cam[0]
        e1 = cam[1] - cam[0]
        e2 = cam[3] - cam[0]
        normal = np.cross(e1, e2)
        nn = np.linalg.norm(normal)
        if nn < 1e-12:
            continue
        normal = normal / nn

        # Camera is at the origin, so t_ray = (n . origin) / (n . dir).
        denom = dirs @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            tray = (origin @ normal) / denom
        z = tray  # because dirs[...,2] == 1, t_ray *is* the depth

        hit = np.isfinite(z) & (z > 0.05)
        if not hit.any():
            continue

        # Restrict to the quad via barycentric-style coordinates in (e1, e2).
        p = dirs * z[..., None]
        rel = p - origin
        a11, a12, a22 = e1 @ e1, e1 @ e2, e2 @ e2
        b1 = rel @ e1
        b2 = rel @ e2
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            continue
        u = (b1 * a22 - b2 * a12) / det
        v = (b2 * a11 - b1 * a12) / det
        inside = hit & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)

        best = np.where(inside & (z < best), z, best)

    best[~np.isfinite(best)] = np.nan
    return best


def build_scene(
    out_dir: Path,
    n_views: int = 14,
    width: int = 640,
    height: int = 480,
    focal: float = 500.0,
    with_depth: bool = False,
) -> dict:
    """Render the scene to `out_dir` and return ground-truth camera metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planes = default_planes()
    K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])
    target = np.array([0.0, 0.0, 1.6])

    names, centres, rotations, depths = [], [], [], []
    for i, eye in enumerate(camera_arc(n_views)):
        R, t = look_at(eye, target)
        img = render_view(planes, R, t, K, width, height)

        # A frame that is mostly empty will not register; fail loudly instead.
        if (img.max(axis=2) > 0).mean() < 0.5:
            raise RuntimeError(f"view {i} is mostly empty; adjust the camera arc")

        name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out_dir / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        names.append(name)
        centres.append(eye)
        rotations.append(R)
        if with_depth:
            depths.append(render_depth(planes, R, t, K, width, height))

    return {
        "names": names,
        "centres": np.array(centres),
        "rotations": np.array(rotations),
        "K": K,
        "width": width,
        "height": height,
        "planes": planes,
        "depths": depths,
    }


def write_synthetic_disparity(
    scene: dict, depth_dir: Path, seed: int = 0
) -> List[Tuple[float, float]]:
    """Write per-frame *disparity* maps with a random affine scale per frame.

    This mimics what a monocular depth model emits: `d = a/z + b` with a and b
    arbitrary and different for every frame. Recovering consistent geometry from
    these is exactly what MPO-225 is about, so the fusion stage must undo it.

    Returns the (a, b) actually used, per frame.
    """
    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    used: List[Tuple[float, float]] = []

    for name, z in zip(scene["names"], scene["depths"]):
        a = float(rng.uniform(0.5, 50.0))
        b = float(rng.uniform(-5.0, 5.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = a / z + b
        # Unobserved pixels get the far-field value rather than NaN, since real
        # depth models always emit a full map.
        disp = np.where(np.isfinite(disp), disp, b)
        np.save(depth_dir / f"{Path(name).stem}_depth.npy", disp.astype(np.float32))
        used.append((a, b))

    (depth_dir / "depth_meta.json").write_text(
        '{"backend": "synthetic", "model": "affine-disparity", "is_disparity": true}',
        encoding="utf-8",
    )
    return used


def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Fit dst ~= s * R @ src + T (Umeyama). Returns (s, R, T, rmse).

    A COLMAP reconstruction is only determined up to a similarity transform, so
    recovered camera centres must be compared to ground truth through one.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    T = mu_d - s * R @ mu_s
    resid = dst - (s * (R @ src.T).T + T)
    rmse = float(np.sqrt((resid**2).sum(axis=1).mean()))
    return s, R, T, rmse
