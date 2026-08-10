"""TSDF fusion and surface extraction (unblocks MPO-235 / MPO-248).

A synthetic plane and sphere give known geometry, so the extracted surface can
be checked against the truth rather than just "it produced triangles".
"""

import numpy as np
import pytest

pytest.importorskip("skimage", reason="needs the mesh extra: uv sync --extra mesh")

from meshing import Mesh, TSDFVolume, write_obj  # noqa: E402


def plane_depth(h=64, w=64, z=2.0, fx=60.0, fy=60.0):
    """A fronto-parallel plane at a known distance."""
    return np.full((h, w), z, dtype=np.float32), fx, fy, w / 2.0, h / 2.0


IDENT_R = np.eye(3)
ZERO_T = np.zeros(3)


def test_untouched_volume_extracts_nothing():
    """An empty grid must not invent geometry where no camera looked."""
    vol = TSDFVolume([-1, -1, -1], [1, 1, 1], resolution=16)
    mesh = vol.extract_mesh()
    assert mesh.n_vertices == 0 and mesh.n_faces == 0


def test_plane_surface_lands_at_the_right_depth():
    depth, fx, fy, cx, cy = plane_depth(z=2.0)
    vol = TSDFVolume([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], resolution=64)
    updated = vol.integrate(depth, IDENT_R, ZERO_T, fx, fy, cx, cy)
    assert updated > 0

    mesh = vol.extract_mesh(min_weight=1.0)
    assert mesh.n_vertices > 0, "no surface extracted from a plane"
    # Every vertex should sit near z = 2, within a voxel or so.
    z = mesh.vertices[:, 2]
    assert abs(float(np.median(z)) - 2.0) < 3 * vol.voxel_size, float(np.median(z))


def test_integration_is_averaged_not_overwritten():
    """Two frames disagreeing slightly should land between them, not on the last."""
    vol = TSDFVolume([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], resolution=64)
    near, fx, fy, cx, cy = plane_depth(z=1.9)
    far, *_ = plane_depth(z=2.1)
    vol.integrate(near, IDENT_R, ZERO_T, fx, fy, cx, cy)
    vol.integrate(far, IDENT_R, ZERO_T, fx, fy, cx, cy)

    z = float(np.median(vol.extract_mesh(min_weight=1.0).vertices[:, 2]))
    assert 1.9 - vol.voxel_size < z < 2.1 + vol.voxel_size
    assert abs(z - 2.0) < 2 * vol.voxel_size, f"averaged to {z}, expected ~2.0"


def test_min_weight_requires_agreement_across_frames():
    """The speckle filter: one observation is not a surface."""
    depth, fx, fy, cx, cy = plane_depth(z=2.0)
    vol = TSDFVolume([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], resolution=48)
    vol.integrate(depth, IDENT_R, ZERO_T, fx, fy, cx, cy)

    assert vol.extract_mesh(min_weight=1.0).n_vertices > 0
    assert vol.extract_mesh(min_weight=2.0).n_vertices == 0

    vol.integrate(depth, IDENT_R, ZERO_T, fx, fy, cx, cy)
    assert vol.extract_mesh(min_weight=2.0).n_vertices > 0


def test_occluded_voxels_are_left_alone():
    """Far behind the surface is unobserved, not empty — it must stay untouched."""
    depth, fx, fy, cx, cy = plane_depth(z=1.0)
    vol = TSDFVolume([-1, -1, 0.5], [1, 1, 4.0], resolution=48)
    vol.integrate(depth, IDENT_R, ZERO_T, fx, fy, cx, cy)

    # Voxels well beyond the truncation band behind z=1 must have zero weight.
    zs = vol.bounds_min[2] + (np.arange(vol.resolution) + 0.5) * vol.voxel_size
    deep = zs > 1.0 + 10 * vol.trunc
    assert deep.any()
    assert vol.weight[:, :, deep].max() == 0.0


def test_colors_are_carried_onto_the_surface():
    depth, fx, fy, cx, cy = plane_depth(z=2.0)
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = 200
    vol = TSDFVolume([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], resolution=48)
    vol.integrate(depth, IDENT_R, ZERO_T, fx, fy, cx, cy, rgb)

    mesh = vol.extract_mesh(min_weight=1.0)
    assert mesh.vertex_colors is not None
    assert mesh.vertex_colors[:, 0].mean() > 100, "red channel lost"


def test_nonfinite_depth_is_ignored():
    depth, fx, fy, cx, cy = plane_depth(z=2.0)
    depth[10:20, 10:20] = np.nan
    depth[30:40, 30:40] = 0.0
    vol = TSDFVolume([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], resolution=48)
    assert vol.integrate(depth, IDENT_R, ZERO_T, fx, fy, cx, cy) > 0
    assert np.isfinite(vol.tsdf).all()


def test_camera_looking_away_updates_nothing():
    depth, fx, fy, cx, cy = plane_depth(z=2.0)
    vol = TSDFVolume([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], resolution=32)
    flip = np.diag([1.0, 1.0, -1.0])       # volume now behind the camera
    assert vol.integrate(depth, flip, ZERO_T, fx, fy, cx, cy) == 0


def test_resolution_is_bounded():
    """A runaway grid takes the machine down, not just the job."""
    with pytest.raises(ValueError, match="resolution"):
        TSDFVolume([0, 0, 0], [1, 1, 1], resolution=4)
    with pytest.raises(ValueError, match="resolution"):
        TSDFVolume([0, 0, 0], [1, 1, 1], resolution=1024)


def test_degenerate_bounds_are_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        TSDFVolume([0, 0, 0], [0, 1, 1])


def test_write_obj_round_trip(tmp_path):
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        vertex_colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
    )
    p = write_obj(mesh, tmp_path / "m.obj")
    text = p.read_text()
    assert text.count("\nv ") + text.startswith("v ") >= 3
    # OBJ is 1-indexed; an off-by-one here silently corrupts every face.
    assert "f 1 2 3" in text


def test_mesh_describe():
    m = Mesh(np.zeros((10, 3), np.float32), np.zeros((4, 3), np.int32))
    assert "10 vertices" in m.describe() and "4 faces" in m.describe()
