import numpy as np
import pytest

from pointcloud_io import normalize_colors, read_ply, voxel_downsample, write_ply


class TestPlyRoundTrip:
    def test_binary_with_colors(self, tmp_path):
        pts = np.array([[0.0, 1.0, 2.0], [-3.5, 4.25, 5.125]], dtype=np.float32)
        cols = np.array([[255, 0, 128], [1, 2, 3]], dtype=np.uint8)
        p = write_ply(tmp_path / "c.ply", pts, cols)

        back_pts, back_cols = read_ply(p)
        np.testing.assert_allclose(back_pts, pts)
        np.testing.assert_array_equal(back_cols, cols)

    def test_binary_without_colors(self, tmp_path):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        p = write_ply(tmp_path / "n.ply", pts)
        back_pts, back_cols = read_ply(p)
        np.testing.assert_allclose(back_pts, pts)
        assert back_cols is None

    def test_ascii_matches_binary(self, tmp_path):
        pts = np.array([[0.5, -1.25, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        cols = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        b_pts, b_cols = read_ply(write_ply(tmp_path / "b.ply", pts, cols, binary=True))
        a_pts, a_cols = read_ply(write_ply(tmp_path / "a.ply", pts, cols, binary=False))
        np.testing.assert_allclose(a_pts, b_pts, atol=1e-6)
        np.testing.assert_array_equal(a_cols, b_cols)

    def test_binary_is_smaller_than_ascii(self, tmp_path):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(5000, 3)).astype(np.float32)
        cols = rng.integers(0, 256, size=(5000, 3), dtype=np.uint8)
        b = write_ply(tmp_path / "b.ply", pts, cols, binary=True)
        a = write_ply(tmp_path / "a.ply", pts, cols, binary=False)
        assert b.stat().st_size < a.stat().st_size

    def test_header_declares_vertex_count(self, tmp_path):
        pts = np.zeros((7, 3), dtype=np.float32)
        p = write_ply(tmp_path / "h.ply", pts)
        head = p.read_bytes()[:200].decode("ascii", "replace")
        assert "element vertex 7" in head
        assert "binary_little_endian" in head

    def test_binary_payload_is_exactly_packed(self, tmp_path):
        """15 bytes/vertex (3 float32 + 3 uint8), no padding — PLY requires packed."""
        n = 11
        pts = np.zeros((n, 3), dtype=np.float32)
        cols = np.zeros((n, 3), dtype=np.uint8)
        p = write_ply(tmp_path / "p.ply", pts, cols)
        raw = p.read_bytes()
        payload = raw.split(b"end_header\n", 1)[1]
        assert len(payload) == n * 15

    def test_empty_cloud(self, tmp_path):
        p = write_ply(tmp_path / "e.ply", np.zeros((0, 3), dtype=np.float32))
        back, _ = read_ply(p)
        assert back.shape == (0, 3)

    def test_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="length mismatch"):
            write_ply(tmp_path / "x.ply", np.zeros((3, 3)), np.zeros((2, 3), dtype=np.uint8))

    def test_float_colors_are_scaled(self, tmp_path):
        pts = np.zeros((2, 3), dtype=np.float32)
        cols = np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 0.0]])
        _, back_cols = read_ply(write_ply(tmp_path / "f.ply", pts, cols))
        np.testing.assert_array_equal(back_cols[0], [0, 127, 255])
        np.testing.assert_array_equal(back_cols[1], [255, 255, 0])


class TestNormalizeColors:
    def test_uint8_passthrough(self):
        c = np.array([[1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(normalize_colors(c), c)

    def test_float_scaled_and_clipped(self):
        c = np.array([[-0.5, 0.5, 1.5]])
        np.testing.assert_array_equal(normalize_colors(c), [[0, 127, 255]])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_colors(np.zeros((4,)))


class TestVoxelDownsample:
    def test_collapses_duplicate_observations_of_one_surface(self):
        """The fusion case: many frames observing the same point."""
        # Deliberately not on a voxel boundary — see
        # test_points_on_a_voxel_boundary_may_split for why that matters.
        pt = np.array([[1.02, 2.03, 3.01]])
        pts = np.repeat(pt, 500, axis=0) + np.random.default_rng(1).normal(scale=1e-4, size=(500, 3))
        out, _ = voxel_downsample(pts, voxel_size=0.05)
        assert len(out) == 1
        np.testing.assert_allclose(out[0], pt[0], atol=1e-3)

    def test_points_on_a_voxel_boundary_may_split(self):
        """Documents inherent grid behaviour rather than a defect.

        A cluster centred exactly on a voxel corner straddles it, so jittered
        observations land in up to 8 adjacent voxels. Relevant when choosing a
        voxel size: it bounds how tightly a surface can be merged.
        """
        pt = np.array([[1.0, 2.0, 3.0]])  # 1.0/0.05 == 20.0 exactly
        pts = np.repeat(pt, 200, axis=0) + np.random.default_rng(1).normal(scale=1e-4, size=(200, 3))
        out, _ = voxel_downsample(pts, voxel_size=0.05)
        assert 1 < len(out) <= 8

    def test_distinct_voxels_are_preserved(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        out, _ = voxel_downsample(pts, voxel_size=0.5)
        assert len(out) == 3

    def test_output_count_is_independent_of_input_multiplicity(self):
        """Point count must be set by scene extent, not by frame count."""
        base = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        a, _ = voxel_downsample(np.repeat(base, 10, axis=0), voxel_size=0.1)
        b, _ = voxel_downsample(np.repeat(base, 1000, axis=0), voxel_size=0.1)
        assert len(a) == len(b) == 3

    def test_colors_are_averaged(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        cols = np.array([[0, 0, 0], [100, 200, 50]], dtype=np.uint8)
        out, out_cols = voxel_downsample(pts, cols, voxel_size=1.0)
        assert len(out) == 1
        np.testing.assert_array_equal(out_cols[0], [50, 100, 25])

    def test_min_points_per_voxel_removes_speckle(self):
        # Ten observations of a real surface point, one isolated noise point.
        real = np.zeros((10, 3))
        noise = np.array([[5.0, 5.0, 5.0]])
        pts = np.vstack([real, noise])
        out, _ = voxel_downsample(pts, voxel_size=0.1, min_points_per_voxel=3)
        assert len(out) == 1
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_non_finite_points_are_dropped(self):
        pts = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [np.inf, 1.0, 1.0]])
        out, _ = voxel_downsample(pts, voxel_size=0.5)
        assert len(out) == 1

    def test_empty_input(self):
        out, cols = voxel_downsample(np.zeros((0, 3)))
        assert out.shape == (0, 3)
        assert cols is None

    def test_empty_input_with_colors_returns_empty_colors(self):
        out, cols = voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
        assert out.shape == (0, 3)
        assert cols is not None and cols.shape == (0, 3)

    def test_bad_voxel_size_raises(self):
        with pytest.raises(ValueError):
            voxel_downsample(np.zeros((2, 3)), voxel_size=0.0)

    def test_dtypes(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        cols = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        out, out_cols = voxel_downsample(pts, cols, voxel_size=0.1)
        assert out.dtype == np.float32
        assert out_cols is not None and out_cols.dtype == np.uint8
