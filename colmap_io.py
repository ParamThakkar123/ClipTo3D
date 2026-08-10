"""Shared reader for COLMAP text models.

This module exists because the repo previously had two independent COLMAP
parsers that were each wrong in a different way (MPO-228):

* `convert_colmap_to_gs.read_images_txt` filtered blank lines before pairing
  records two-at-a-time. A COLMAP image with zero 2D points writes an *empty*
  POINTS2D line, so that filter desynced every subsequent record — pose lines
  got parsed as POINTS2D lines and all later poses became garbage, silently.
  It also read camera intrinsics from a leaked loop variable after the loop
  had ended.
* `geometry_based_reconstruction.gbr.parse_images_txt` unpacked the quaternion
  as `qx, qy, qz, qw` from COLMAP's `QW QX QY QZ` column order, rotating the
  components by one position and scrambling every rotation matrix. Its
  `intrinsics_from_camera` also misread SIMPLE_RADIAL (`f, cx, cy, k`) as
  though it were `fx, fy, cx, cy`.

Both call sites now go through the functions here.

Reference for the text format:
https://colmap.github.io/format.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# COLMAP camera models, mapped to how many focal-length parameters they carry.
# Single-focal models are laid out `f, cx, cy, ...`; dual-focal models are
# `fx, fy, cx, cy, ...`. Anything not listed here is rejected rather than
# guessed at, because guessing is what produced silently wrong intrinsics.
_SINGLE_FOCAL = {
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}
_DUAL_FOCAL = {
    "PINHOLE",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "FOV",
    "THIN_PRISM_FISHEYE",
}


class ColmapFormatError(ValueError):
    """Raised when a COLMAP text file does not match the documented format."""


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) according to this camera's model."""
        p = np.asarray(self.params, dtype=float)
        if self.model in _SINGLE_FOCAL:
            if p.size < 3:
                raise ColmapFormatError(
                    f"camera {self.id} ({self.model}) needs >=3 params, got {p.size}"
                )
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        if self.model in _DUAL_FOCAL:
            if p.size < 4:
                raise ColmapFormatError(
                    f"camera {self.id} ({self.model}) needs >=4 params, got {p.size}"
                )
            return float(p[0]), float(p[1]), float(p[2]), float(p[3])
        raise ColmapFormatError(
            f"unsupported COLMAP camera model {self.model!r} for camera {self.id}. "
            f"Known models: {sorted(_SINGLE_FOCAL | _DUAL_FOCAL)}"
        )


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (4,) in COLMAP order: w, x, y, z
    tvec: np.ndarray  # (3,)
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    @property
    def R(self) -> np.ndarray:
        """World-to-camera rotation."""
        return qvec2rotmat(self.qvec)

    @property
    def camera_center(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.tvec

    def world_to_camera(self, pts_world: np.ndarray) -> np.ndarray:
        """Map (N,3) world points into this camera's frame: X_cam = R X_world + t."""
        pts = np.asarray(pts_world, dtype=float).reshape(-1, 3)
        return pts @ self.R.T + self.tvec

    def camera_to_world(self, pts_cam: np.ndarray) -> np.ndarray:
        """Map (N,3) camera-frame points into world: X_world = R^T (X_cam - t).

        Provided explicitly because the previous ad-hoc version in gbr.py
        computed `R @ X_cam + t`, which is neither this transform nor its
        inverse.
        """
        pts = np.asarray(pts_cam, dtype=float).reshape(-1, 3)
        return (pts - self.tvec) @ self.R

    def camera_to_world_matrix(self) -> np.ndarray:
        """4x4 camera-to-world transform (the `transform_matrix` convention)."""
        c2w = np.eye(4)
        c2w[:3, :3] = self.R.T
        c2w[:3, 3] = self.camera_center
        return c2w


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    point2D_idxs: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))


def qvec2rotmat(qvec: Iterable[float]) -> np.ndarray:
    """Convert a COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    q = np.asarray(list(qvec), dtype=float)
    if q.shape != (4,):
        raise ColmapFormatError(f"quaternion must have 4 components, got {q.shape}")
    n = float(q @ q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / np.sqrt(n)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _content_lines(path: Path) -> List[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    """Parse a COLMAP cameras.txt.

    Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    """
    cameras: Dict[int, Camera] = {}
    for raw in _content_lines(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        e = line.split()
        if len(e) < 4:
            raise ColmapFormatError(f"malformed cameras.txt line: {raw!r}")
        cam_id = int(e[0])
        cameras[cam_id] = Camera(
            id=cam_id,
            model=e[1],
            # width/height are pixel counts and must stay integral; the old
            # reader parsed them as floats, which leaked `1920.0` into the
            # exported transforms.json.
            width=int(float(e[2])),
            height=int(float(e[3])),
            params=np.array([float(x) for x in e[4:]], dtype=float),
        )
    if not cameras:
        raise ColmapFormatError(f"no cameras found in {path}")
    return cameras


def read_images_text(path: Path) -> Dict[int, Image]:
    """Parse a COLMAP images.txt.

    Each image occupies exactly two lines:

        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D[] as (X, Y, POINT3D_ID)

    The POINTS2D line is consumed unconditionally — including when it is
    empty, which is what happens for an image with no triangulated points.
    Filtering blank lines before pairing is what desynced the old parser.
    """
    lines = _content_lines(path)
    images: Dict[int, Image] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        e = line.split()
        if len(e) < 10:
            raise ColmapFormatError(
                f"malformed images.txt pose line (expected >=10 fields, got {len(e)}): {line!r}"
            )

        image_id = int(e[0])
        qvec = np.array([float(x) for x in e[1:5]], dtype=float)  # w, x, y, z
        tvec = np.array([float(x) for x in e[5:8]], dtype=float)
        camera_id = int(e[8])
        # NAME may legitimately contain spaces; rejoin the tail.
        name = " ".join(e[9:])

        # Exactly one line follows, whatever it contains.
        pts_raw = lines[i] if i < n else ""
        i += 1

        toks = pts_raw.split()
        if toks and len(toks) % 3 != 0:
            raise ColmapFormatError(
                f"POINTS2D for image {image_id} has {len(toks)} tokens, not a multiple of 3"
            )
        if toks:
            arr = np.array(toks, dtype=float).reshape(-1, 3)
            xys = arr[:, :2]
            point3D_ids = arr[:, 2].astype(np.int64)
        else:
            xys = np.zeros((0, 2), dtype=float)
            point3D_ids = np.zeros((0,), dtype=np.int64)

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys,
            point3D_ids=point3D_ids,
        )

    if not images:
        raise ColmapFormatError(f"no images found in {path}")
    return images


def read_points3D_text(path: Path) -> Dict[int, Point3D]:
    """Parse a COLMAP points3D.txt.

    Format: POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    points: Dict[int, Point3D] = {}
    for raw in _content_lines(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        e = line.split()
        if len(e) < 8:
            raise ColmapFormatError(f"malformed points3D.txt line: {raw!r}")
        pid = int(e[0])
        track = np.array([int(float(x)) for x in e[8:]], dtype=np.int64)
        if track.size % 2 != 0:
            raise ColmapFormatError(f"point {pid} has an odd-length track")
        track = track.reshape(-1, 2)
        points[pid] = Point3D(
            id=pid,
            xyz=np.array([float(x) for x in e[1:4]], dtype=float),
            rgb=np.array([int(float(x)) for x in e[4:7]], dtype=np.uint8),
            error=float(e[7]),
            image_ids=track[:, 0],
            point2D_idxs=track[:, 1],
        )
    return points


def find_model_dir(root: Path) -> Path:
    """Locate a directory holding cameras.txt + images.txt under `root`.

    COLMAP nests reconstructions in numbered subdirectories, and this repo
    additionally writes a `model_txt` level, so a model can sit at `root`,
    `root/0`, `root/model_txt/0`, and so on. When several candidates exist the
    one with the most 3D points wins, matching COLMAP's own notion of the
    largest reconstruction.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"COLMAP model root does not exist: {root}")

    candidates: List[Path] = []
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if (d / "cameras.txt").is_file() and (d / "images.txt").is_file():
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            f"no COLMAP text model (cameras.txt + images.txt) found under {root}"
        )

    def point_count(d: Path) -> int:
        p = d / "points3D.txt"
        if not p.is_file():
            return 0
        return sum(
            1
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.startswith("#")
        )

    return max(candidates, key=point_count)


def read_model(
    model_dir: Path,
) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """Read cameras/images/points3D from a COLMAP text model directory."""
    model_dir = Path(model_dir)
    cameras = read_cameras_text(model_dir / "cameras.txt")
    images = read_images_text(model_dir / "images.txt")
    pts_path = model_dir / "points3D.txt"
    points = read_points3D_text(pts_path) if pts_path.is_file() else {}
    return cameras, images, points


def images_by_name(images: Dict[int, Image]) -> Dict[str, Image]:
    """Index images by base filename, for matching against a frames directory."""
    out: Dict[str, Image] = {}
    for img in images.values():
        out[Path(img.name).name] = img
    return out


def sole_camera(cameras: Dict[int, Camera]) -> Camera:
    """Return the single camera in a model, or raise if it is not single-camera.

    Callers that emit one global set of intrinsics must go through this, so a
    multi-camera model fails loudly instead of silently adopting whichever
    camera happened to be visited last.
    """
    if len(cameras) != 1:
        raise ColmapFormatError(
            f"expected a single-camera model, found {len(cameras)} cameras "
            f"(ids: {sorted(cameras)}). Emit per-frame intrinsics instead."
        )
    return next(iter(cameras.values()))
