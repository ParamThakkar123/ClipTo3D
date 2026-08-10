"""TSDF fusion and surface extraction (MPO-239's alternative; unblocks MPO-235/248).

The voxel point cloud from `fusion.fuse` meets the fusion issue's criteria, but
a *point cloud* is the wrong deliverable for AR: Quick Look and Scene Viewer
expect a surface, and a cloud has no occlusion and no lighting response — it
reads as noise on a table. So the mobile export path needs a mesh, and that is
what this produces.

Truncated signed distance fusion, then marching cubes:

* Each frame's metric depth map is projected into a fixed voxel grid, and every
  voxel records a running weighted average of its signed distance to the
  nearest observed surface along the view ray.
* Truncation (`trunc`) bounds the band around the surface that a single frame
  is allowed to influence, which is what stops a distant depth reading from
  overwriting geometry it never actually saw.
* Averaging across frames is the point: it resolves *conflicting* depth
  observations rather than stacking them, which is exactly what a point cloud
  cannot do.

Marching cubes comes from scikit-image. Open3D would be the usual choice but
publishes no wheels for Python 3.13, and the marching-cubes step is the only
part of it this needs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from colmap_io import Camera, find_model_dir, read_model  # noqa: E402
from depth_io import find_depth, load_depth  # noqa: E402
from depth_scale import fit_and_convert, robust_scene_extent  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


@dataclass
class Mesh:
    vertices: np.ndarray          # (V,3) float32
    faces: np.ndarray             # (F,3) int32
    vertex_colors: Optional[np.ndarray] = None   # (V,3) uint8

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def describe(self) -> str:
        return f"{self.n_vertices:,} vertices, {self.n_faces:,} faces"


class TSDFVolume:
    """A fixed-resolution truncated signed distance grid.

    Memory is `resolution^3 * 8` bytes for the two float32 grids, so 256^3 is
    ~134MB and 512^3 is ~1.1GB. Resolution is capped rather than derived from
    the scene, because a runaway grid is the one failure mode here that takes
    the whole machine down rather than just the job.
    """

    def __init__(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        resolution: int = 192,
        trunc_voxels: float = 3.0,
    ):
        if resolution < 8 or resolution > 512:
            raise ValueError(f"resolution must be within 8..512, got {resolution}")
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        span = self.bounds_max - self.bounds_min
        if not np.all(span > 0):
            raise ValueError(f"degenerate bounds: {bounds_min} .. {bounds_max}")

        self.resolution = int(resolution)
        self.voxel_size = float(span.max() / self.resolution)
        self.trunc = trunc_voxels * self.voxel_size

        shape = (self.resolution,) * 3
        # Start at +1 (empty space) rather than 0: an untouched voxel must read
        # as "outside the surface", not as "exactly on it", or marching cubes
        # invents geometry in regions no camera ever observed.
        self.tsdf = np.ones(shape, dtype=np.float32)
        self.weight = np.zeros(shape, dtype=np.float32)
        self.color = np.zeros(shape + (3,), dtype=np.float32)

        # World coordinates of every voxel centre, built once.
        axes = [
            self.bounds_min[d] + (np.arange(self.resolution) + 0.5) * self.voxel_size
            for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        self._points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    def integrate(
        self,
        depth: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        colors: Optional[np.ndarray] = None,
    ) -> int:
        """Fold one posed depth map into the volume. Returns voxels updated."""
        h, w = depth.shape[:2]

        cam = self._points @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)
        z = cam[:, 2]
        in_front = z > 1e-6
        if not in_front.any():
            return 0

        u = np.full(len(cam), -1.0)
        v = np.full(len(cam), -1.0)
        u[in_front] = cam[in_front, 0] * fx / z[in_front] + cx
        v[in_front] = cam[in_front, 1] * fy / z[in_front] + cy

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        visible = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not visible.any():
            return 0

        idx = np.flatnonzero(visible)
        measured = depth[vi[idx], ui[idx]]
        good = np.isfinite(measured) & (measured > 0)
        idx = idx[good]
        measured = measured[good]
        if len(idx) == 0:
            return 0

        # Signed distance along the ray: positive in front of the surface.
        sdf = measured - z[idx]
        # Behind the surface by more than the truncation band means this voxel
        # is occluded from this view, and this frame knows nothing about it.
        keep = sdf >= -self.trunc
        idx = idx[keep]
        sdf = sdf[keep]
        if len(idx) == 0:
            return 0
        sdf = np.clip(sdf / self.trunc, -1.0, 1.0)

        flat = np.unravel_index(idx, self.tsdf.shape)
        w_old = self.weight[flat]
        w_new = w_old + 1.0
        self.tsdf[flat] = (self.tsdf[flat] * w_old + sdf.astype(np.float32)) / w_new
        if colors is not None:
            c = colors[vi[idx], ui[idx]].astype(np.float32)
            self.color[flat] = (self.color[flat] * w_old[:, None] + c) / w_new[:, None]
        self.weight[flat] = w_new
        return len(idx)

    def extract_mesh(self, level: float = 0.0, min_weight: float = 1.0) -> Mesh:
        """Marching cubes over the observed band of the volume."""
        try:
            from skimage import measure
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
            raise ModuleNotFoundError(
                "Surface extraction needs scikit-image, which is not in the base "
                "install: `uv sync --extra mesh`."
            ) from exc

        observed = self.weight >= min_weight
        if not observed.any():
            return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))

        # Unobserved voxels read as empty space rather than as zero, which
        # would otherwise sit exactly on the level set and invent geometry.
        field = np.where(observed, self.tsdf, 1.0).astype(np.float32)
        if field.min() > level or field.max() < level:
            return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))

        # The mask is what keeps the surface off the *edge of the observed
        # region*. Without it marching cubes also triangulates the boundary
        # between observed voxels and the +1 fill — producing a shell in the
        # shape of the camera frustum that is not a surface anyone saw. It
        # dominated the output: a single plane at z=1.9 extracted a median
        # vertex depth of 2.0, the centre of the volume, rather than 1.9.
        verts, faces, _normals, _vals = measure.marching_cubes(
            field, level=level, mask=observed
        )

        # Grid indices -> world. Voxel centres sit at +0.5, matching the grid.
        world = self.bounds_min + (verts + 0.5) * self.voxel_size

        cols = None
        if self.color.any():
            vi = np.clip(np.round(verts).astype(np.int64), 0, self.resolution - 1)
            sampled = self.color[vi[:, 0], vi[:, 1], vi[:, 2]]
            cols = np.clip(sampled, 0, 255).astype(np.uint8)

        return Mesh(world.astype(np.float32), faces.astype(np.int32), cols)


def mesh_from_job(
    colmap_dir: Path | str,
    frames_dir: Path | str,
    depth_dir: Path | str,
    resolution: int = 192,
    min_sparse_points: int = 20,
    max_depth_factor: float = 3.0,
    stride: int = 2,
    min_weight: float = 2.0,
) -> Mesh:
    """Build a surface from a job's COLMAP model and depth maps.

    Depth is converted to metric per frame by the same affine fit fusion uses
    (MPO-225) — fusing frames that do not share a scale just averages
    misaligned surfaces into mush.
    """
    from PIL import Image as PILImage

    frames_dir, depth_dir = Path(frames_dir), Path(depth_dir)
    model_dir = find_model_dir(colmap_dir)
    cameras, images, points3D = read_model(model_dir)
    if not points3D:
        raise RuntimeError(f"{model_dir} has no sparse points; depth scale cannot be recovered")

    extent = robust_scene_extent(points3D)
    max_depth = max_depth_factor * extent if np.isfinite(extent) else None

    # Bound the volume by the sparse cloud, which is the only thing that knows
    # the scene's actual extent before any depth has been converted.
    xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float64)
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    pad = 0.1 * (hi - lo).max()
    volume = TSDFVolume(lo - pad, hi + pad, resolution=resolution)

    used = 0
    skipped: Dict[str, int] = {"no_frame": 0, "no_depth": 0, "fit_failed": 0}
    for img in sorted(images.values(), key=lambda im: im.name):
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue
        frame_path = _find_frame(frames_dir, img.name)
        if frame_path is None:
            skipped["no_frame"] += 1
            continue
        depth_path = find_depth(depth_dir, frame_path.stem)
        if depth_path is None:
            skipped["no_depth"] += 1
            continue

        disparity = load_depth(depth_path).astype(np.float64)
        fx, fy, cx, cy, sx, sy = _scaled_intrinsics(cam, disparity.shape[:2])

        img_for_fit = img
        if not (np.isclose(sx, 1.0) and np.isclose(sy, 1.0)):
            from dataclasses import replace

            img_for_fit = replace(img, xys=img.xys * np.array([sx, sy]))

        metric, _fit = fit_and_convert(
            disparity, img_for_fit, points3D,
            min_points=min_sparse_points, max_depth=max_depth,
        )
        if metric is None:
            skipped["fit_failed"] += 1
            continue

        rgb = None
        try:
            arr = np.array(PILImage.open(frame_path).convert("RGB"))
            if arr.shape[:2] != metric.shape[:2]:
                arr = np.array(PILImage.fromarray(arr).resize(
                    (metric.shape[1], metric.shape[0]), PILImage.Resampling.BILINEAR))
            rgb = arr
        except Exception:
            rgb = None

        if stride > 1:
            metric = metric[::stride, ::stride]
            if rgb is not None:
                rgb = rgb[::stride, ::stride]
            fx, fy, cx, cy = fx / stride, fy / stride, cx / stride, cy / stride

        volume.integrate(metric, img.R, img.tvec, fx, fy, cx, cy, rgb)
        used += 1

    if used == 0:
        raise RuntimeError(f"no frame could be integrated; skipped {skipped}")

    mesh = volume.extract_mesh(min_weight=min_weight)
    print(f"TSDF: integrated {used} frames at {volume.resolution}^3 "
          f"(voxel {volume.voxel_size:.4g}); {mesh.describe()}; skipped {skipped}")
    return mesh


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


def _scaled_intrinsics(cam: Camera, depth_shape: Tuple[int, int]):
    dh, dw = depth_shape
    sx = dw / float(cam.width)
    sy = dh / float(cam.height)
    fx, fy, cx, cy = cam.intrinsics
    return fx * sx, fy * sy, cx * sx, cy * sy, sx, sy


def write_obj(mesh: Mesh, path: Path | str) -> Path:
    """A debug format that every tool on earth can open."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if mesh.vertex_colors is not None:
        for v, c in zip(mesh.vertices, mesh.vertex_colors):
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                         f"{c[0]/255:.4f} {c[1]/255:.4f} {c[2]/255:.4f}")
    else:
        for v in mesh.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in mesh.faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
