"""Web and mobile export formats (MPO-235).

GLB is validated by parsing the container back per the glTF 2.0 spec rather
than by eyeballing it — chunk alignment and accessor bookkeeping are exactly
where hand-written glTF goes wrong, and a viewer's error message for a
malformed file is usually "failed to load".
"""

import json
import struct

import numpy as np
import pytest

from export import (
    MOBILE_BUDGET_BYTES,
    SPLAT_RECORD_BYTES,
    ExportResult,
    glb_from_ply,
    lod_levels,
    write_glb,
    write_splat,
)
from pointcloud_io import write_ply


def cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3)).astype(np.float32) * np.array([4.0, 2.0, 8.0], np.float32)
    cols = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    return pts, cols


def parse_glb(path):
    """Parse a GLB per the spec, returning (gltf_json, bin_chunk)."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "bad glTF magic"
    assert version == 2
    assert length == len(data), f"header length {length} != actual {len(data)}"

    offset = 12
    chunks = {}
    while offset < len(data):
        clen, ctype = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[ctype] = data[offset:offset + clen]
        assert clen % 4 == 0, "chunk length must be 4-byte aligned"
        offset += clen
    return json.loads(chunks[0x4E4F534A]), chunks.get(0x004E4942, b"")


# --- GLB ------------------------------------------------------------------

def test_glb_is_a_valid_container(tmp_path):
    pts, cols = cloud()
    res = write_glb(tmp_path / "c.glb", pts, cols)
    gltf, blob = parse_glb(res.path)

    assert gltf["asset"]["version"] == "2.0"
    assert gltf["meshes"][0]["primitives"][0]["mode"] == 0, "must be POINTS"
    assert gltf["buffers"][0]["byteLength"] == len(blob)
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] + view["byteLength"] <= len(blob)


def test_glb_positions_round_trip_through_quantisation(tmp_path):
    """Quantised int16 + node transform must reproduce the original points."""
    pts, cols = cloud()
    res = write_glb(tmp_path / "c.glb", pts, cols, quantize=True)
    gltf, blob = parse_glb(res.path)

    acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert acc["componentType"] == 5122 and acc["normalized"] is True
    view = gltf["bufferViews"][acc["bufferView"]]
    raw = np.frombuffer(blob, dtype="<i2", count=acc["count"] * 4,
                        offset=view["byteOffset"]).reshape(-1, 4)[:, :3]

    node = gltf["nodes"][0]
    decoded = (raw / 32767.0) * np.array(node["scale"]) + np.array(node["translation"])
    # int16 over the bounding box: error is bounded by half a quantisation step.
    span = pts.max(axis=0) - pts.min(axis=0)
    assert np.abs(decoded - pts).max() < (span / 65535.0 * 2).max()


def test_quantised_is_smaller_than_float(tmp_path):
    pts, cols = cloud(5000)
    q = write_glb(tmp_path / "q.glb", pts, cols, quantize=True)
    f = write_glb(tmp_path / "f.glb", pts, cols, quantize=False)
    assert q.bytes < f.bytes


def test_glb_colors_are_rgba_normalised(tmp_path):
    pts, cols = cloud(64)
    res = write_glb(tmp_path / "c.glb", pts, cols)
    gltf, blob = parse_glb(res.path)

    idx = gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]
    acc = gltf["accessors"][idx]
    assert acc["type"] == "VEC4" and acc["normalized"] is True
    view = gltf["bufferViews"][acc["bufferView"]]
    rgba = np.frombuffer(blob, dtype=np.uint8, count=64 * 4,
                         offset=view["byteOffset"]).reshape(-1, 4)
    np.testing.assert_array_equal(rgba[:, :3], cols)
    assert (rgba[:, 3] == 255).all()


def test_glb_without_colors(tmp_path):
    pts, _ = cloud(32)
    gltf, _ = parse_glb(write_glb(tmp_path / "c.glb", pts).path)
    assert "COLOR_0" not in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_empty_cloud_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_glb(tmp_path / "c.glb", np.zeros((0, 3)))


def test_glb_from_ply_round_trip(tmp_path):
    pts, cols = cloud(300)
    ply = write_ply(tmp_path / "c.ply", pts, cols)
    res = glb_from_ply(ply, tmp_path / "c.glb")
    assert res.n_primitives == 300
    parse_glb(res.path)


def test_decimation_preserves_spatial_extent(tmp_path):
    """Uniform stride, not random sampling — the cloud must not shrink."""
    pts, cols = cloud(4000)
    ply = write_ply(tmp_path / "c.ply", pts, cols)
    res = glb_from_ply(ply, tmp_path / "c.glb", max_points=500)
    assert res.n_primitives == 500

    gltf, _ = parse_glb(res.path)
    node = gltf["nodes"][0]
    # The node transform encodes the decimated bounding box; it should still
    # cover most of the original extent.
    span = np.array(node["scale"]) * 2.0
    assert (span > (pts.max(axis=0) - pts.min(axis=0)) * 0.9).all()


# --- .splat ---------------------------------------------------------------

def gaussians(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        means=rng.random((n, 3)).astype(np.float32),
        scales=rng.normal(-3, 0.5, (n, 3)).astype(np.float32),      # log space
        rotations=rng.normal(0, 1, (n, 4)).astype(np.float32),
        colors=rng.normal(0, 0.5, (n, 3)).astype(np.float32),        # SH DC
        opacities=rng.normal(2, 1, n).astype(np.float32),            # logits
    )


def test_splat_record_layout(tmp_path):
    g = gaussians(200)
    res = write_splat(tmp_path / "s.splat", **g)
    assert res.bytes == 200 * SPLAT_RECORD_BYTES
    assert res.n_primitives == 200


def test_splat_scales_are_activated_from_log_space(tmp_path):
    g = gaussians(50)
    res = write_splat(tmp_path / "s.splat", **g)
    rec = np.frombuffer(res.path.read_bytes(), dtype=np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4),
    ]))
    # exp() of a log-space scale is strictly positive; the raw values were not.
    assert (rec["scale"] > 0).all()
    assert np.allclose(np.sort(rec["scale"].ravel()),
                       np.sort(np.exp(g["scales"]).ravel()), rtol=1e-4)


def test_splat_is_ordered_by_visual_importance(tmp_path):
    """Progressive loading shows the significant gaussians first."""
    g = gaussians(300)
    res = write_splat(tmp_path / "s.splat", **g)
    rec = np.frombuffer(res.path.read_bytes(), dtype=np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4),
    ]))
    importance = np.prod(rec["scale"], axis=1) * (rec["rgba"][:, 3] / 255.0)
    assert importance[0] >= importance[-1]
    # Monotonically non-increasing apart from alpha quantisation noise.
    assert (np.diff(importance) <= 1e-6).mean() > 0.9


def test_splat_rotations_are_normalised_then_packed(tmp_path):
    g = gaussians(40)
    res = write_splat(tmp_path / "s.splat", **g)
    rec = np.frombuffer(res.path.read_bytes(), dtype=np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4),
    ]))
    q = (rec["rot"].astype(np.float32) - 128.0) / 128.0
    assert np.abs(np.linalg.norm(q, axis=1) - 1.0).max() < 0.05


def test_splat_opacity_becomes_alpha(tmp_path):
    n = 3
    res = write_splat(
        tmp_path / "s.splat",
        means=np.zeros((n, 3)), scales=np.zeros((n, 3)),
        rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
        colors=np.zeros((n, 3)),
        opacities=np.array([-10.0, 0.0, 10.0]),  # logits -> ~0, 0.5, ~1
    )
    rec = np.frombuffer(res.path.read_bytes(), dtype=np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4),
    ]))
    alphas = sorted(int(a) for a in rec["rgba"][:, 3])
    assert alphas[0] == 0 and alphas[2] == 255
    assert abs(alphas[1] - 128) <= 1, alphas


# --- budgets and LOD ------------------------------------------------------

def test_budget_flagging():
    assert ExportResult(pytest.importorskip("pathlib").Path("x"), "glb", 1, 1_000_000).within_budget
    assert not ExportResult(
        pytest.importorskip("pathlib").Path("x"), "glb", 1, MOBILE_BUDGET_BYTES + 1
    ).within_budget


def test_lod_levels_quarter_each_time():
    assert lod_levels(1000, 3) == [1000, 250, 62]
    assert lod_levels(1, 3) == [1, 1, 1], "must never produce an empty level"


def test_describe_reports_over_budget():
    r = ExportResult(pytest.importorskip("pathlib").Path("x"), "glb", 10, MOBILE_BUDGET_BYTES * 2)
    assert "OVER BUDGET" in r.describe()


# --- mesh formats: GLB triangles and USDZ (MPO-235 / MPO-248) -------------

def simple_mesh(n=40, seed=0):
    """A small triangulated grid — real topology, known bounds."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 2, n), np.linspace(0, 3, n))
    z = 0.2 * np.sin(xs * 3) + 0.05 * rng.standard_normal(xs.shape)
    verts = np.column_stack([xs.ravel(), ys.ravel(), z.ravel()]).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, c], [b, d, c]]
    cols = rng.integers(0, 256, (len(verts), 3), dtype=np.uint8)
    return verts, np.array(faces, dtype=np.int32), cols


def test_glb_mesh_is_a_valid_triangle_container(tmp_path):
    from export import write_glb_mesh

    v, f, c = simple_mesh()
    res = write_glb_mesh(tmp_path / "m.glb", v, f, c)
    gltf, blob = parse_glb(res.path)

    prim = gltf["meshes"][0]["primitives"][0]
    assert prim["mode"] == 4, "must be TRIANGLES"
    assert "indices" in prim
    idx_acc = gltf["accessors"][prim["indices"]]
    assert idx_acc["count"] == len(f) * 3
    assert gltf["buffers"][0]["byteLength"] == len(blob)


def test_glb_mesh_index_width_follows_vertex_count(tmp_path):
    """uint16 halves the index buffer, but only while it can address every vertex."""
    from export import write_glb_mesh

    v, f, c = simple_mesh(n=40)              # 1600 verts -> uint16
    gltf, _ = parse_glb(write_glb_mesh(tmp_path / "small.glb", v, f, c).path)
    assert gltf["accessors"][gltf["meshes"][0]["primitives"][0]["indices"]]["componentType"] == 5123

    big_v = np.tile(v, (45, 1))              # >65535 verts -> uint32
    big_f = np.concatenate([f + i * len(v) for i in range(45)])
    gltf, _ = parse_glb(write_glb_mesh(tmp_path / "big.glb", big_v, big_f).path)
    assert gltf["accessors"][gltf["meshes"][0]["primitives"][0]["indices"]]["componentType"] == 5125


def test_glb_mesh_rejects_out_of_range_faces(tmp_path):
    from export import write_glb_mesh

    v, _f, _c = simple_mesh(n=4)
    with pytest.raises(ValueError, match="exceeds vertex count"):
        write_glb_mesh(tmp_path / "m.glb", v, np.array([[0, 1, 9999]]))


def test_glb_mesh_rejects_empty(tmp_path):
    from export import write_glb_mesh

    with pytest.raises(ValueError, match="empty mesh"):
        write_glb_mesh(tmp_path / "m.glb", np.zeros((0, 3)), np.zeros((0, 3)))


def test_usdz_is_a_conformant_package(tmp_path):
    """USDZ demands uncompressed entries; a compressed one opens everywhere
    except on a phone."""
    pytest.importorskip("pxr", reason="needs the ar extra: uv sync --extra ar")
    import zipfile

    from export import write_usdz

    v, f, c = simple_mesh()
    res = write_usdz(tmp_path / "s.usdz", v, f, c)

    zf = zipfile.ZipFile(res.path)
    infos = zf.infolist()
    assert all(i.compress_type == zipfile.ZIP_STORED for i in infos), "USDZ must be stored, not deflated"
    assert infos[0].filename.endswith((".usdc", ".usda", ".usd")), "layer must come first"
    assert not (tmp_path / "s.usdc").exists(), "intermediate layer left behind"


def test_usdz_round_trips_the_geometry(tmp_path):
    pytest.importorskip("pxr", reason="needs the ar extra")
    from pxr import Usd, UsdGeom

    from export import write_usdz

    v, f, c = simple_mesh()
    res = write_usdz(tmp_path / "s.usdz", v, f, c)

    stage = Usd.Stage.Open(str(res.path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/Scan"))
    assert len(mesh.GetPointsAttr().Get()) == len(v)
    assert len(mesh.GetFaceVertexCountsAttr().Get()) == len(f)
    # Quick Look expects Y-up; COLMAP is Y-down, so this must be converted.
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert stage.GetDefaultPrim().IsValid(), "USDZ needs a default prim to open"


def test_usdz_flips_to_y_up(tmp_path):
    pytest.importorskip("pxr", reason="needs the ar extra")
    from pxr import Usd, UsdGeom

    from export import write_usdz

    v = np.array([[0, 1, 0], [1, 1, 0], [0, 2, 0]], dtype=np.float32)
    f = np.array([[0, 1, 2]], dtype=np.int32)
    res = write_usdz(tmp_path / "s.usdz", v, f)

    stage = Usd.Stage.Open(str(res.path))
    pts = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/Root/Scan")).GetPointsAttr().Get())
    # Y-down input becomes Y-up output.
    assert pts[:, 1].max() == -1.0 and pts[:, 1].min() == -2.0


def test_usdz_rejects_empty(tmp_path):
    pytest.importorskip("pxr", reason="needs the ar extra")
    from export import write_usdz

    with pytest.raises(ValueError, match="empty mesh"):
        write_usdz(tmp_path / "s.usdz", np.zeros((0, 3)), np.zeros((0, 3)))


def test_draco_compresses_or_reports_unavailable():
    from export import draco_compress_points

    pts, cols = cloud(5000)
    encoded = draco_compress_points(pts, cols)
    if encoded is None:
        pytest.skip("DracoPy not installed")
    raw = pts.nbytes + cols.nbytes
    assert len(encoded) < raw, f"draco grew the data: {len(encoded)} vs {raw}"


# --- splat radius ---------------------------------------------------------
#
# The radius came from nearest-neighbour distance measured *within a 2000
# point sample*, which reports the sample's spacing rather than the cloud's.
# It ran 4.7x too large at 40k points and 7.7x at 120k — and worsened as the
# cloud grew, so denser clouds got fatter splats. Area goes as the square, so
# the viewer was drawing every gaussian with ~20-60x the fill it needed.

def _sphere(n, seed=0):
    rng = np.random.default_rng(seed)
    u, v = rng.random(n) * np.pi, rng.random(n) * 2 * np.pi
    return np.stack([3 * np.sin(u) * np.cos(v), 3 * np.cos(u),
                     3 * np.sin(u) * np.sin(v)], 1).astype(np.float32)


def brute_force_spacing(pts):
    """Exact median nearest-neighbour distance. Only for small clouds."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def test_neighbour_spacing_matches_brute_force():
    from export import neighbour_spacing

    pts = _sphere(1500)
    exact = brute_force_spacing(pts)
    assert abs(neighbour_spacing(pts) / exact - 1) < 0.15


def test_neighbour_spacing_shrinks_as_the_cloud_gets_denser():
    """The regression, stated directly: more points means closer neighbours.

    The old sampled-against-itself estimate returned roughly the same number
    for 10k and 80k points, because it only ever measured 2000 of them.
    """
    from export import neighbour_spacing

    coarse = neighbour_spacing(_sphere(10_000))
    dense = neighbour_spacing(_sphere(80_000))
    # 8x the points on a surface: spacing should fall by about sqrt(8) ~ 2.8.
    assert 2.0 < coarse / dense < 4.0, (coarse, dense)


def test_neighbour_spacing_survives_a_cloud_far_from_the_origin():
    """|a|^2 + |b|^2 - 2a.b cancels badly without centring."""
    from export import neighbour_spacing

    pts = _sphere(3000)
    here = neighbour_spacing(pts)
    far = neighbour_spacing(pts + np.float32([12000, -9000, 40000]))
    assert abs(far / here - 1) < 0.05, (here, far)


def test_neighbour_spacing_handles_degenerate_clouds():
    from export import neighbour_spacing

    assert np.isnan(neighbour_spacing(np.zeros((1, 3), np.float32)))
    # Every point identical: no positive distance exists.
    assert np.isnan(neighbour_spacing(np.ones((50, 3), np.float32)))


def test_splat_radius_tracks_point_spacing(tmp_path):
    """Splats should overlap into a surface, not swamp it.

    A radius many times the point spacing is what made the viewer render a
    blurred blob at 0.2 fps instead of a surface.
    """
    from export import neighbour_spacing, splat_from_pointcloud

    pts = _sphere(40_000)
    spacing = neighbour_spacing(pts)
    res = splat_from_pointcloud(pts, out_path=tmp_path / "s.splat")
    rec = np.frombuffer(res.path.read_bytes(), dtype=np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4),
    ]))
    scale = float(np.median(rec["scale"]))
    assert 1.0 < scale / spacing < 2.5, f"radius {scale} vs spacing {spacing}"


@pytest.mark.parametrize("shape", ["surface", "volume"])
def test_neighbour_spacing_extrapolates_past_the_reference_cap(shape):
    """Large clouds measure a capped reference and extrapolate to full size.

    An exact pass over a million points takes ~30s, but simply subsampling
    biases the answer high. Spacing goes as m ** (-1/d), so two reference
    sizes give the cloud's intrinsic dimension and that gives the correction.
    """
    from export import neighbour_spacing

    rng = np.random.default_rng(3)
    n = 60_000
    if shape == "surface":
        u, v = rng.random(n) * np.pi, rng.random(n) * 2 * np.pi
        pts = np.stack([3 * np.sin(u) * np.cos(v), 3 * np.cos(u),
                        3 * np.sin(u) * np.sin(v)], 1).astype(np.float32)
    else:
        pts = (rng.random((n, 3)) * 4).astype(np.float32)

    exact = neighbour_spacing(pts, reference_cap=n)
    # A cap well below n forces the extrapolation path.
    capped = neighbour_spacing(pts, reference_cap=n // 8)
    assert 0.75 < capped / exact < 1.35, (exact, capped)

    # Without the correction, subsampling 8x would inflate the answer by
    # 8**(1/d) — at least 2x on a surface. Confirm we are nowhere near that.
    assert capped / exact < 1.6


def test_neighbour_spacing_is_not_fooled_by_duplicate_points():
    """Coincident points carry no spacing information."""
    from export import neighbour_spacing

    rng = np.random.default_rng(4)
    pts = (rng.random((4000, 3)) * 2).astype(np.float32)
    clean = neighbour_spacing(pts)
    # Half the cloud duplicated exactly on top of itself.
    doubled = np.concatenate([pts, pts[:2000]])
    assert abs(neighbour_spacing(doubled) / clean - 1) < 0.35
