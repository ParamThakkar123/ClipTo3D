"""Streaming voxel fusion (MPO-239).

The bar in the issue is twofold: output point count set by scene extent and
voxel size rather than frame count, and **peak RSS flat in the number of
frames**. The second is the part the old two-pass implementation failed, and
it is what the memory tests below actually measure.
"""

import tracemalloc

import numpy as np
import pytest

from pointcloud_io import VoxelAccumulator, voxel_downsample


def surface_points(n=4000, seed=0):
    """Points on a fixed plane patch — the same 'surface' every frame sees."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 2.0
    z = 0.5 + 0.001 * rng.standard_normal(n)
    return np.column_stack([xy, z])


def test_output_size_is_set_by_extent_not_frame_count():
    """20 frames and 200 frames of the same surface must give the same cloud."""
    small = VoxelAccumulator(voxel_size=0.1)
    large = VoxelAccumulator(voxel_size=0.1)
    for i in range(20):
        small.add(surface_points(seed=i))
    for i in range(200):
        large.add(surface_points(seed=i))

    p_small, _ = small.result()
    p_large, _ = large.result()
    # Same scene, same voxel size -> same occupied voxels, regardless of frames.
    assert p_small.shape == p_large.shape
    assert small.n_voxels == large.n_voxels


def test_peak_memory_is_flat_in_frame_count():
    """The actual MPO-239 criterion."""
    def peak_bytes(n_frames):
        tracemalloc.start()
        acc = VoxelAccumulator(voxel_size=0.05)
        for i in range(n_frames):
            acc.add(surface_points(seed=i))
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    p20 = peak_bytes(20)
    p200 = peak_bytes(200)
    # 10x the frames must not mean materially more memory. The old
    # accumulate-then-vstack approach would be ~10x here.
    assert p200 < p20 * 2.0, f"peak grew with frames: {p20} -> {p200}"


def test_accumulator_matches_the_two_pass_result():
    """Streaming must not change the answer, only the memory profile."""
    frames = [surface_points(seed=i) for i in range(8)]
    voxel = 0.1

    acc = VoxelAccumulator(voxel_size=voxel)
    for f in frames:
        acc.add(f)
    streamed, _ = acc.result(min_views=1)

    # Old path: reduce each frame, stack, reduce again.
    chunks = [voxel_downsample(f, voxel_size=voxel)[0] for f in frames]
    two_pass, _ = voxel_downsample(np.vstack(chunks), voxel_size=voxel, min_points_per_voxel=1)

    assert streamed.shape == two_pass.shape

    def canonical(a):
        # Round before sorting: raw lexsort tie-breaks arbitrarily between
        # points whose leading coordinates agree to float precision.
        k = np.round(a, 5)
        return a[np.lexsort((k[:, 2], k[:, 1], k[:, 0]))]

    np.testing.assert_allclose(canonical(streamed), canonical(two_pass), atol=1e-5)


def test_min_views_counts_frames_not_points():
    """A voxel hit by 10,000 points in one frame is still a single view."""
    acc = VoxelAccumulator(voxel_size=1.0)
    dense = np.full((10_000, 3), 0.5)
    acc.add(dense)

    assert len(acc.result(min_views=1)[0]) == 1
    assert len(acc.result(min_views=2)[0]) == 0, "one frame must not satisfy min_views=2"

    acc.add(dense)
    assert len(acc.result(min_views=2)[0]) == 1


def test_min_views_filters_speckle_but_keeps_agreed_surface():
    acc = VoxelAccumulator(voxel_size=0.5)
    surface = np.array([[0.1, 0.1, 0.1]])
    for i in range(5):
        acc.add(np.vstack([surface, [[10.0 + i, 0.0, 0.0]]]))  # unique speckle per frame

    kept, _ = acc.result(min_views=2)
    assert len(kept) == 1
    np.testing.assert_allclose(kept[0], surface[0], atol=1e-5)


def test_colors_are_averaged_across_frames():
    acc = VoxelAccumulator(voxel_size=1.0)
    p = np.array([[0.5, 0.5, 0.5]])
    acc.add(p, np.array([[0, 0, 0]], dtype=np.uint8))
    acc.add(p, np.array([[255, 255, 255]], dtype=np.uint8))
    _pts, cols = acc.result()
    assert cols is not None
    np.testing.assert_allclose(cols[0], [128, 128, 128], atol=1)


def test_float_colors_are_normalised():
    acc = VoxelAccumulator(voxel_size=1.0)
    acc.add(np.array([[0.5, 0.5, 0.5]]), np.array([[1.0, 0.0, 0.5]]))
    _pts, cols = acc.result()
    assert cols is not None
    np.testing.assert_allclose(cols[0], [255, 0, 128], atol=1)


def test_mixing_coloured_and_uncoloured_observations_is_rejected():
    acc = VoxelAccumulator(voxel_size=1.0)
    acc.add(np.array([[0.0, 0.0, 0.0]]), np.array([[1, 2, 3]], dtype=np.uint8))
    with pytest.raises(ValueError, match="every observation or none"):
        acc.add(np.array([[0.0, 0.0, 0.0]]))


def test_non_finite_points_are_dropped_not_poisoning_the_grid():
    acc = VoxelAccumulator(voxel_size=1.0)
    pts = np.array([[0.5, 0.5, 0.5], [np.nan, 0.0, 0.0], [np.inf, 1.0, 1.0]])
    acc.add(pts)
    out, _ = acc.result()
    assert len(out) == 1
    assert np.isfinite(out).all()


def test_empty_observation_still_counts_as_a_frame():
    acc = VoxelAccumulator(voxel_size=1.0)
    acc.add(np.zeros((0, 3)))
    assert acc.n_voxels == 0
    assert acc.n_observations == 1
    pts, cols = acc.result()
    assert pts.shape == (0, 3) and cols is None


def test_rejects_non_positive_voxel_size():
    with pytest.raises(ValueError, match="voxel_size"):
        VoxelAccumulator(voxel_size=0.0)


def test_length_mismatch_is_rejected():
    acc = VoxelAccumulator(voxel_size=1.0)
    with pytest.raises(ValueError, match="length mismatch"):
        acc.add(np.zeros((5, 3)), np.zeros((3, 3), dtype=np.uint8))
