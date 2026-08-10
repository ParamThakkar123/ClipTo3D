"""Web and mobile export formats (MPO-235).

PLY was the only thing the pipeline produced, and no target platform consumes
it well: it has no compression and no streaming layout, so a multi-million
point cloud is tens of megabytes before it reaches a browser or a phone.

    .splat   trained gaussians, the format the web splat viewers read
    .glb     binary glTF, for the web and for Android Scene Viewer

Both writers are vectorized and dependency-free — glTF is JSON plus a binary
blob, and `.splat` is a flat 32-byte record per gaussian, so pulling in a
glTF library to emit two buffer views would cost more than it saves.

Every export reports its size against a budget, because the failure mode here
is silent: an export that quietly grows past what a phone will load is not
visible until someone tries it on a phone.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pointcloud_io import read_ply

# What a phone will fetch over mobile data without the user giving up.
MOBILE_BUDGET_BYTES = 20 * 1024 * 1024

# One record: position (3f), scale (3f), colour (4 u8), rotation (4 u8).
SPLAT_RECORD_BYTES = 32


@dataclass
class ExportResult:
    path: Path
    format: str
    n_primitives: int
    bytes: int
    budget_bytes: int = MOBILE_BUDGET_BYTES

    @property
    def within_budget(self) -> bool:
        return self.bytes <= self.budget_bytes

    def describe(self) -> str:
        mb = self.bytes / 1e6
        limit = self.budget_bytes / 1e6
        flag = "ok" if self.within_budget else f"OVER BUDGET (>{limit:.0f} MB)"
        return f"{self.format:6} {self.n_primitives:>9,} prims  {mb:7.2f} MB  {flag}"


# --- .splat ---------------------------------------------------------------

def _sh_dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    """Zeroth-order spherical harmonics -> 0..255 RGB."""
    C0 = 0.28209479177387814
    return np.clip((0.5 + C0 * dc) * 255.0, 0, 255).astype(np.uint8)


def write_splat(
    path: Path | str,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
) -> ExportResult:
    """Write the flat `.splat` format the web viewers read.

    Gaussians are sorted by a size/opacity importance so a viewer streaming the
    file progressively shows the visually significant ones first — the layout
    *is* the LOD story for this format.

    `scales` are expected in log space and `opacities` as logits, matching the
    3DGS PLY convention; both are activated here.
    """
    path = Path(path)
    means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    n = len(means)
    scales = np.exp(np.asarray(scales, dtype=np.float32).reshape(n, 3))
    rot = np.asarray(rotations, dtype=np.float32).reshape(n, 4)
    rot = rot / np.clip(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12, None)
    alpha = 1.0 / (1.0 + np.exp(-np.asarray(opacities, dtype=np.float32).reshape(n)))

    cols = np.asarray(colors)
    if cols.dtype != np.uint8:
        cols = _sh_dc_to_rgb(cols.reshape(n, 3))
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = cols.reshape(n, 3)
    # Round, don't truncate: a fully opaque gaussian (sigmoid ~0.9999) would
    # otherwise land on 254 and never be quite opaque.
    rgba[:, 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)

    # Bigger and more opaque first.
    importance = np.prod(scales, axis=1) * alpha
    order = np.argsort(-importance)

    rec = np.zeros(n, dtype=np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4),
    ]))
    rec["pos"] = means[order]
    rec["scale"] = scales[order]
    rec["rgba"] = rgba[order]
    # Quaternions pack into bytes as 128 + 128*q, which is what the format uses.
    rec["rot"] = np.clip(rot[order] * 128.0 + 128.0, 0, 255).astype(np.uint8)

    assert rec.itemsize == SPLAT_RECORD_BYTES, rec.itemsize
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rec.tobytes())
    return ExportResult(path, "splat", n, path.stat().st_size, budget_bytes)


def neighbour_spacing(
    points: np.ndarray,
    queries: int = 256,
    reference_cap: int = 250_000,
    seed: int = 0,
) -> float:
    """Median distance from a sampled point to its nearest neighbour.

    The queries are sampled but they are matched against the *whole* cloud.
    Measuring a sample against itself instead — which is what this used to do —
    reports the spacing of the sample, not of the cloud, and the error grows
    with the cloud: 4.7x too large at 40k points and 7.7x at 120k, measured on
    a sphere. Splat radius is derived from this, area goes as the square, and
    the viewer ended up drawing every gaussian ~20-60x too wide.

    Measured against an exact KD-tree it lands within a few percent (ratios
    0.97-1.02 across surface, volumetric and far-from-origin clouds of
    3k-500k points; 0.90-1.03 across seeds), at ~0.4s for 120k points.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    if n <= reference_cap:
        return _spacing_against(pts, queries, rng)

    # Too big to measure against every point — an exact pass over 1M points
    # takes ~30s. Subsampling the reference biases the answer high, but
    # predictably: spacing goes as m ** (-1/d) for a cloud of intrinsic
    # dimension d, so measuring at two reference sizes gives d, and d gives
    # the extrapolation back to the full cloud.
    big = pts[rng.choice(n, size=reference_cap, replace=False)]
    s_big = _spacing_against(big, queries, rng)
    s_small = _spacing_against(big[:reference_cap // 4], queries, rng)

    ratio = s_small / s_big if s_big > 0 else 0.0
    if not np.isfinite(ratio) or ratio <= 1.0:
        return s_big                       # no measurable trend; do not guess
    # 4x the reference, so s_small / s_big == 4 ** (1/d).
    d = float(np.clip(np.log(4.0) / np.log(ratio), 1.0, 3.0))
    return s_big * (reference_cap / n) ** (1.0 / d)


def _spacing_against(reference: np.ndarray, queries: int, rng) -> float:
    """Exact median nearest-neighbour distance within `reference`."""
    # Queries are sampled from the reference so each one's own row can be
    # excluded by index below.
    q_pos = rng.choice(len(reference), size=min(len(reference), queries), replace=False)
    q = reference[q_pos]

    # Chunked over the reference set: the full query x reference matrix is
    # gigabytes on a large cloud, and only the running minimum is needed.
    #
    # Squared distances via |a|^2 + |b|^2 - 2a.b, so the inner loop is one
    # GEMM. The obvious `norm(q[:, None] - block[None, :], axis=-1)` builds a
    # (queries, chunk, 3) temporary and runs ~15x slower.
    # Centred, and in float64. Both matter: |a|^2 + |b|^2 - 2a.b cancels, and
    # for the *closest* pairs — the only ones this function cares about — the
    # true d^2 is smaller than the float32 rounding error, so the subtraction
    # returns near-zero garbage. In float32 that corrupted 10% of the
    # neighbour distances, some by 200x, and dragged the median 9% low.
    centre = reference.mean(axis=0, dtype=np.float64)
    reference = reference.astype(np.float64) - centre
    q = q.astype(np.float64) - centre

    best = np.full(len(q), np.inf, dtype=np.float64)
    q_sq = np.einsum("ij,ij->i", q, q)[:, None]
    rows = np.arange(len(q))
    chunk = max(1, 4_000_000 // max(len(q), 1))
    for start in range(0, len(reference), chunk):
        block = reference[start:start + chunk]
        d2 = q_sq + np.einsum("ij,ij->i", block, block)[None, :] - 2.0 * (q @ block.T)

        # Exclude each query's own row *by index*. A magnitude threshold does
        # not work: |a|^2 + |a|^2 - 2a.a is not exactly 0 in floating point,
        # it lands near 2e-15, so a small-value filter lets self-matches
        # through and reports a spacing of ~0 for half the queries.
        here = (q_pos >= start) & (q_pos < start + len(block))
        d2[rows[here], q_pos[here] - start] = np.inf

        # Coincident points are degenerate rather than informative: they say
        # nothing about how far apart the surface samples are.
        d2[d2 <= 0.0] = np.inf
        np.minimum(best, d2.min(axis=1), out=best)

    finite = best[np.isfinite(best) & (best > 0)]
    return float(np.sqrt(np.median(finite))) if len(finite) else float("nan")


def splat_from_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    opacity: float = 0.9,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
    out_path: Path | str = "cloud.splat",
) -> ExportResult:
    """Turn a fused point cloud into isotropic gaussians.

    Not a substitute for training — these have no view-dependent appearance and
    no optimised covariance, so they will not look like a real 3DGS result.
    What they are is a *valid* `.splat` produced without a GPU, which makes the
    web viewer's splat path testable and gives it something to render before
    any training run exists.

    `radius` defaults to a few times the median nearest-neighbour spacing, so
    the splats just overlap into a surface rather than leaving gaps or turning
    the scene into soup.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        raise ValueError("cannot export an empty point cloud")

    if radius is None:
        radius = neighbour_spacing(pts) * 1.5
        if not np.isfinite(radius) or radius <= 0:
            span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            radius = max(span / max(n ** (1 / 3), 1.0), 1e-4)

    cols = (
        np.full((n, 3), 200, dtype=np.uint8) if colors is None
        else np.asarray(colors).reshape(n, 3)
    )
    # write_splat expects log-space scales and opacity logits.
    log_scale = np.full((n, 3), np.log(radius), dtype=np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    logit = np.full(n, np.log(opacity / (1.0 - opacity)), dtype=np.float32)
    return write_splat(out_path, pts, log_scale, quats, cols, logit, budget_bytes)


def splat_from_ply(
    ply_path: Path | str, out_path: Path | str, budget_bytes: int = MOBILE_BUDGET_BYTES
) -> ExportResult:
    """Convert a 3DGS PLY (as `gaussians.py` exports) into `.splat`."""
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"].data
    names = set(v.dtype.names or ())
    required = {"x", "y", "z", "scale_0", "rot_0", "opacity", "f_dc_0"}
    missing = required - names
    if missing:
        raise ValueError(
            f"{ply_path} does not look like a 3DGS PLY (missing {sorted(missing)}). "
            f"Use glb_from_ply for a plain point cloud."
        )

    means = np.column_stack([v["x"], v["y"], v["z"]])
    scales = np.column_stack([v[f"scale_{i}"] for i in range(3)])
    rots = np.column_stack([v[f"rot_{i}"] for i in range(4)])
    dc = np.column_stack([v[f"f_dc_{i}"] for i in range(3)])
    return write_splat(out_path, means, scales, rots, dc, np.asarray(v["opacity"]), budget_bytes)


# --- glTF / GLB -----------------------------------------------------------

def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(
    path: Path | str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
    quantize: bool = True,
) -> ExportResult:
    """Write a point cloud as binary glTF (mode 0 = POINTS).

    `quantize` stores positions as normalised int16 with the dequantisation
    folded into the node transform — half the bytes of float32, and glTF's own
    mechanism for it, so no extension is required and every loader handles it.
    Draco would compress further but needs a native encoder; this is the
    dependency-free win.
    """
    path = Path(path)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        raise ValueError("cannot export an empty point cloud")

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)

    buffers: List[bytes] = []
    accessors: List[dict] = []
    buffer_views: List[dict] = []
    attributes: Dict[str, int] = {}

    def add(data: bytes, target: int, accessor: dict, stride: Optional[int] = None) -> int:
        offset = sum(len(_pad4(b)) for b in buffers)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target}
        if stride:
            view["byteStride"] = stride
        buffer_views.append(view)
        buffers.append(data)
        accessor["bufferView"] = len(buffer_views) - 1
        accessors.append(accessor)
        return len(accessors) - 1

    if quantize:
        # Map into [-1, 1] as int16; the node scale/translation undoes it.
        norm = (pts - lo) / span            # 0..1
        q = np.clip(norm * 2.0 - 1.0, -1.0, 1.0)
        qi = np.round(q * 32767.0).astype("<i2")
        # 3 * int16 = 6 bytes, but glTF requires 4-byte aligned strides.
        padded = np.zeros((n, 4), dtype="<i2")
        padded[:, :3] = qi
        attributes["POSITION"] = add(
            padded.tobytes(), 34962,
            {
                "componentType": 5122,  # SHORT
                "normalized": True,
                "count": n,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0],
            },
            stride=8,
        )
        node = {
            "mesh": 0,
            # x_world = (q * 0.5 + 0.5) * span + lo
            "scale": [float(s * 0.5) for s in span],
            "translation": [float(lo[i] + span[i] * 0.5) for i in range(3)],
        }
    else:
        attributes["POSITION"] = add(
            pts.astype("<f4").tobytes(), 34962,
            {
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
                "min": [float(x) for x in lo],
                "max": [float(x) for x in hi],
            },
        )
        node = {"mesh": 0}

    if colors is not None:
        cols = np.asarray(colors)
        if cols.dtype != np.uint8:
            cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
        cols = cols.reshape(n, 3)
        rgba = np.full((n, 4), 255, dtype=np.uint8)
        rgba[:, :3] = cols
        attributes["COLOR_0"] = add(
            rgba.tobytes(), 34962,
            {"componentType": 5121, "normalized": True, "count": n, "type": "VEC4"},
        )

    bin_blob = b"".join(_pad4(b) for b in buffers)
    gltf = {
        "asset": {"version": "2.0", "generator": "clipto3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [node],
        "meshes": [{"primitives": [{"attributes": attributes, "mode": 0}]}],  # 0 = POINTS
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_blob = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_blob = _pad4(bin_blob)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_blob) + 8 + len(bin_blob)))
        fh.write(struct.pack("<II", len(json_blob), 0x4E4F534A))  # JSON chunk
        fh.write(json_blob)
        fh.write(struct.pack("<II", len(bin_blob), 0x004E4942))   # BIN chunk
        fh.write(bin_blob)

    return ExportResult(path, "glb", n, path.stat().st_size, budget_bytes)


def glb_from_ply(
    ply_path: Path | str,
    out_path: Path | str,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
    max_points: Optional[int] = None,
) -> ExportResult:
    """Convert a point-cloud PLY to GLB, optionally decimating to a budget."""
    pts, cols = read_ply(ply_path)
    if max_points is not None and len(pts) > max_points:
        # Uniform stride keeps spatial coverage; random sampling clumps.
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None
    return write_glb(out_path, pts, cols, budget_bytes)


# --- meshes: GLB (triangles) and USDZ -------------------------------------

def write_glb_mesh(
    path: Path | str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
) -> ExportResult:
    """Write a triangle mesh as binary glTF — the Android Scene Viewer asset.

    Same container as the point path, but `mode: 4` (TRIANGLES) with an index
    accessor. Indices are uint16 when the mesh is small enough, which halves
    the index buffer; glTF requires the accessor's componentType to match, so
    it is chosen from the vertex count rather than assumed.
    """
    path = Path(path)
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n, n_tri = len(verts), len(tris)
    if n == 0 or n_tri == 0:
        raise ValueError("cannot export an empty mesh")
    if tris.max() >= n:
        raise ValueError(f"face index {tris.max()} exceeds vertex count {n}")

    lo, hi = verts.min(axis=0), verts.max(axis=0)

    buffers: List[bytes] = []
    buffer_views: List[dict] = []
    accessors: List[dict] = []

    def add(data: bytes, target: Optional[int], accessor: dict) -> int:
        offset = sum(len(_pad4(b)) for b in buffers)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        buffers.append(data)
        accessor["bufferView"] = len(buffer_views) - 1
        accessors.append(accessor)
        return len(accessors) - 1

    pos_idx = add(
        verts.tobytes(), 34962,
        {"componentType": 5126, "count": n, "type": "VEC3",
         "min": [float(x) for x in lo], "max": [float(x) for x in hi]},
    )
    attributes = {"POSITION": pos_idx}

    if colors is not None:
        cols = np.asarray(colors)
        if cols.dtype != np.uint8:
            cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
        rgba = np.full((n, 4), 255, dtype=np.uint8)
        rgba[:, :3] = cols.reshape(n, 3)
        attributes["COLOR_0"] = add(
            rgba.tobytes(), 34962,
            {"componentType": 5121, "normalized": True, "count": n, "type": "VEC4"},
        )

    if n <= 65535:
        idx_data, comp = tris.astype("<u2").tobytes(), 5123
    else:
        idx_data, comp = tris.astype("<u4").tobytes(), 5125
    idx_acc = add(idx_data, 34963,
                  {"componentType": comp, "count": n_tri * 3, "type": "SCALAR"})

    bin_blob = _pad4(b"".join(_pad4(b) for b in buffers))
    gltf = {
        "asset": {"version": "2.0", "generator": "clipto3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [
            {"attributes": attributes, "indices": idx_acc, "mode": 4}
        ]}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_blob = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_blob) + 8 + len(bin_blob)))
        fh.write(struct.pack("<II", len(json_blob), 0x4E4F534A))
        fh.write(json_blob)
        fh.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        fh.write(bin_blob)

    return ExportResult(path, "glb-mesh", n_tri, path.stat().st_size, budget_bytes)


def write_usdz(
    path: Path | str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
) -> ExportResult:
    """Write a USDZ package — the iOS AR Quick Look asset.

    A point cloud is deliberately not accepted here: Quick Look renders a
    surface, and a cloud has no occlusion or lighting response, so it reads as
    noise on a table. USDZ therefore comes from the mesh path only.

    `UsdUtils.CreateNewUsdzPackage` is used rather than zipping by hand,
    because USDZ requires uncompressed entries at 64-byte alignment and
    getting that subtly wrong produces a file that opens everywhere except on
    a phone.
    """
    try:
        from pxr import Usd, UsdGeom, UsdUtils, Vt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
        raise ModuleNotFoundError(
            "USDZ export needs usd-core, which is not in the base install: "
            "`uv sync --extra ar`."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError("cannot export an empty mesh")

    work = path.with_suffix(".usdc")
    stage = Usd.Stage.CreateNew(str(work))
    # Quick Look expects Y-up metres; COLMAP/OpenCV is Y-down, so the mesh is
    # rotated rather than shipped upside down.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Scan")

    flipped = verts.copy()
    flipped[:, 1] *= -1.0        # Y-down -> Y-up
    flipped[:, 2] *= -1.0        # keep the winding order handed correctly

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(flipped))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).ravel()))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(len(tris), 3, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
        np.stack([flipped.min(axis=0), flipped.max(axis=0)]).astype(np.float32)))

    if colors is not None:
        cols = np.asarray(colors)
        if cols.dtype == np.uint8:
            cols = cols.astype(np.float32) / 255.0
        # The purpose-built API, rather than CreatePrimvar with a hand-built
        # type name — it gets the interpolation and value type right for us.
        pv = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(cols.reshape(-1, 3).astype(np.float32)))

    stage.GetRootLayer().Save()
    del stage

    if path.exists():
        path.unlink()
    if not UsdUtils.CreateNewUsdzPackage(str(work), str(path)):
        raise RuntimeError(f"USD failed to package {work} into {path}")
    work.unlink(missing_ok=True)

    return ExportResult(path, "usdz", len(tris), path.stat().st_size, budget_bytes)


def draco_compress_points(
    points: np.ndarray, colors: Optional[np.ndarray] = None, quantization_bits: int = 14
) -> Optional[bytes]:
    """Draco-encode a point cloud, or None when DracoPy is unavailable.

    Returned as raw bytes rather than wired into the GLB: embedding it needs
    the KHR_draco_mesh_compression extension, and a loader that does not
    implement it must still be able to read the file. Kept as a measurable
    size comparison until that trade-off is decided.
    """
    try:
        import DracoPy
    except ModuleNotFoundError:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    kwargs = {"quantization_bits": quantization_bits}
    if colors is not None:
        cols = np.asarray(colors)
        if cols.dtype != np.uint8:
            cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
        kwargs["colors"] = cols.reshape(-1, 3)
    try:
        return DracoPy.encode(pts, **kwargs)
    except Exception:
        return None


# --- LOD ------------------------------------------------------------------

def lod_levels(n_points: int, levels: int = 3) -> List[int]:
    """Point budgets per LOD, coarsest last. Each level is a quarter of the previous."""
    out = []
    n = n_points
    for _ in range(levels):
        out.append(max(1, int(n)))
        n = n // 4
    return out


def export_all(
    job_root: Path | str,
    budget_bytes: int = MOBILE_BUDGET_BYTES,
    lods: int = 3,
) -> List[ExportResult]:
    """Produce every export a job can support, from whatever it actually built."""
    from job_paths import JobPaths

    job = JobPaths(job_root)
    job.export.mkdir(parents=True, exist_ok=True)
    results: List[ExportResult] = []

    if job.fused_ply.is_file():
        pts, _cols = read_ply(job.fused_ply)
        for i, budget in enumerate(lod_levels(len(pts), lods)):
            name = "cloud.glb" if i == 0 else f"cloud_lod{i}.glb"
            results.append(
                glb_from_ply(job.fused_ply, job.export / name, budget_bytes, max_points=budget)
            )

    # A mesh unlocks the AR formats: Quick Look and Scene Viewer want a
    # surface, not a cloud.
    if job.mesh_npz.is_file():
        data = np.load(job.mesh_npz)
        verts, faces = data["vertices"], data["faces"]
        cols = data["colors"] if "colors" in data else None
        if len(verts) and len(faces):
            results.append(write_glb_mesh(job.export / "mesh.glb", verts, faces, cols,
                                          budget_bytes))
            try:
                results.append(write_usdz(job.export / "scene.usdz", verts, faces, cols,
                                          budget_bytes))
            except ModuleNotFoundError as exc:
                print(f"skipping USDZ: {exc}")

    splat_ply = job.splat / "splat.ply"
    if splat_ply.is_file():
        results.append(splat_from_ply(splat_ply, job.export / "scene.splat", budget_bytes))
    elif job.fused_ply.is_file():
        # No trained gaussians: derive isotropic ones from the cloud so the
        # viewer's splat path has a valid asset. Clearly not a 3DGS result.
        pts, cols = read_ply(job.fused_ply)
        results.append(splat_from_pointcloud(
            pts, cols, budget_bytes=budget_bytes,
            out_path=job.export / "cloud.splat"))

    return results


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Export a job's reconstruction for web and mobile.")
    p.add_argument("job", type=Path)
    p.add_argument("--budget-mb", type=float, default=MOBILE_BUDGET_BYTES / 1e6)
    p.add_argument("--lods", type=int, default=3)
    args = p.parse_args(argv)

    results = export_all(args.job, int(args.budget_mb * 1e6), args.lods)
    if not results:
        print("nothing to export: the job has no fused cloud and no trained splat")
        return 1
    for r in results:
        print(r.describe())
    over = [r for r in results if not r.within_budget]
    if over:
        print(f"\n{len(over)} export(s) over the {args.budget_mb:.0f} MB mobile budget")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
