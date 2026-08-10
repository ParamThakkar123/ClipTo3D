"""Tests for metric-depth recovery from relative inverse depth (MPO-225)."""

import numpy as np
import pytest

from colmap_io import Image, Point3D
from depth_scale import (
    DepthScaleFit,
    fit_and_convert,
    fit_depth_scale,
    robust_scene_extent,
    sample_bilinear,
    sparse_observations,
)

H, W = 60, 80
TRUE_A, TRUE_B = 3.5, 0.25


def synthetic_disparity(z_map: np.ndarray, a: float = TRUE_A, b: float = TRUE_B) -> np.ndarray:
    """Build the disparity a depth model would emit for a known metric depth map."""
    return a / z_map + b


def depth_ramp(h: int = H, w: int = W, near: float = 1.0, far: float = 8.0) -> np.ndarray:
    """A depth map ramping from `near` to `far` down the image."""
    return np.linspace(near, far, h)[:, None] * np.ones((1, w))


class TestSampleBilinear:
    def test_exact_at_integer_coords(self):
        img = np.arange(12, dtype=float).reshape(3, 4)
        got = sample_bilinear(img, np.array([0, 3, 1]), np.array([0, 2, 1]))
        np.testing.assert_allclose(got, [0.0, 11.0, 5.0])

    def test_midpoint_is_the_average(self):
        img = np.array([[0.0, 10.0], [20.0, 30.0]])
        np.testing.assert_allclose(sample_bilinear(img, np.array([0.5]), np.array([0.5])), [15.0])

    def test_out_of_bounds_is_clamped(self):
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(sample_bilinear(img, np.array([-5.0]), np.array([-5.0])), [1.0])
        np.testing.assert_allclose(sample_bilinear(img, np.array([99.0]), np.array([99.0])), [4.0])


class TestFitRecovery:
    def _observations(self, z_map, n=200, seed=0):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0, W - 1, n)
        ys = rng.uniform(0, H - 1, n)
        # Sample the reference depth the same way the fit samples disparity.
        # Pairing bilinear disparity against nearest-neighbour depth injects a
        # systematic bias proportional to the local depth gradient — about 1% on
        # this ramp, which is easy to mistake for a broken fit.
        z = sample_bilinear(z_map, xs, ys)
        return np.column_stack([xs, ys]), z

    def test_recovers_known_scale_and_shift(self):
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)
        xys, z_ref = self._observations(z_map)

        fit = fit_depth_scale(disp, xys, z_ref)

        # Exact recovery is not expected: bilinear interpolation does not commute
        # with the nonlinear d = a/z + b, so sampling leaves a second-order
        # residual. 0.1% on this ramp.
        assert fit.ok, fit.reason
        assert fit.scale == pytest.approx(TRUE_A, rel=1e-2)
        assert fit.shift == pytest.approx(TRUE_B, rel=5e-2, abs=1e-2)

    def test_recovered_depth_matches_ground_truth(self):
        """The property that actually matters: inverting the fit returns metres."""
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)
        xys, z_ref = self._observations(z_map)

        fit = fit_depth_scale(disp, xys, z_ref)
        z_est = fit.to_metric(disp)

        np.testing.assert_allclose(z_est, z_map, rtol=1e-3)

    def test_two_frames_with_different_disparity_scales_agree_after_fitting(self):
        """The core regression: per-frame normalization used to destroy this.

        Two frames see the same surface but their depth models emit wildly
        different disparity scales. After fitting, both must recover the same
        metric depth.
        """
        z_map = depth_ramp()
        disp_a = synthetic_disparity(z_map, a=3.5, b=0.25)
        disp_b = synthetic_disparity(z_map, a=140.0, b=-9.0)

        xys, z_ref = self._observations(z_map)
        fit_a = fit_depth_scale(disp_a, xys, z_ref)
        fit_b = fit_depth_scale(disp_b, xys, z_ref)
        assert fit_a.ok and fit_b.ok

        z_a = fit_a.to_metric(disp_a)
        z_b = fit_b.to_metric(disp_b)

        np.testing.assert_allclose(z_a, z_b, rtol=1e-3)
        np.testing.assert_allclose(z_a, z_map, rtol=1e-3)

    def test_survives_outliers(self):
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)
        xys, z_ref = self._observations(z_map, n=300, seed=3)
        # Corrupt 12% of the reference depths, as a bad feature match would.
        rng = np.random.default_rng(7)
        idx = rng.choice(len(z_ref), size=len(z_ref) // 8, replace=False)
        z_ref = z_ref.copy()
        z_ref[idx] *= rng.uniform(4.0, 12.0, size=idx.shape)

        fit = fit_depth_scale(disp, xys, z_ref)

        assert fit.ok, fit.reason
        assert fit.n_inliers < fit.n_observations  # trimming actually removed some
        assert fit.scale == pytest.approx(TRUE_A, rel=0.05)

    def test_noise_in_reference_depth_is_tolerated(self):
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)
        xys, z_ref = self._observations(z_map, n=400, seed=11)
        rng = np.random.default_rng(5)
        z_ref = z_ref * (1.0 + rng.normal(scale=0.02, size=z_ref.shape))

        fit = fit_depth_scale(disp, xys, z_ref)
        assert fit.ok, fit.reason
        assert fit.scale == pytest.approx(TRUE_A, rel=0.1)


class TestFitRejection:
    def test_too_few_observations_is_rejected(self):
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)
        xys = np.array([[10.0, 10.0], [20.0, 20.0]])
        z_ref = np.array([2.0, 4.0])

        fit = fit_depth_scale(disp, xys, z_ref, min_points=20)
        assert not fit.ok
        assert "sparse observations" in fit.reason

    def test_no_observations_is_rejected(self):
        fit = fit_depth_scale(np.ones((H, W)), np.zeros((0, 2)), np.zeros((0,)))
        assert not fit.ok

    def test_constant_reference_depth_is_rejected_not_guessed(self):
        """A fronto-parallel view of one plane cannot identify the slope."""
        z_map = np.full((H, W), 3.0)
        disp = synthetic_disparity(z_map)
        rng = np.random.default_rng(0)
        xys = np.column_stack([rng.uniform(0, W - 1, 100), rng.uniform(0, H - 1, 100)])
        z_ref = np.full(100, 3.0)

        fit = fit_depth_scale(disp, xys, z_ref)
        assert not fit.ok
        assert "too little range" in fit.reason

    def test_anticorrelated_prediction_is_rejected(self):
        z_map = depth_ramp()
        # Disparity that *decreases* with 1/z — the sign error case.
        disp = -synthetic_disparity(z_map)
        rng = np.random.default_rng(1)
        xs = rng.uniform(0, W - 1, 200)
        ys = rng.uniform(0, H - 1, 200)
        z_ref = sample_bilinear(z_map, xs, ys)

        fit = fit_depth_scale(disp, np.column_stack([xs, ys]), z_ref)
        assert not fit.ok
        assert "non-positive scale" in fit.reason

    def test_negative_reference_depths_are_dropped(self):
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)
        rng = np.random.default_rng(2)
        xys = np.column_stack([rng.uniform(0, W - 1, 30), rng.uniform(0, H - 1, 30)])
        z_ref = np.full(30, -1.0)  # all behind the camera

        fit = fit_depth_scale(disp, xys, z_ref, min_points=20)
        assert not fit.ok
        assert "usable observations" in fit.reason

    def test_applying_a_failed_fit_raises(self):
        bad = DepthScaleFit(np.nan, np.nan, 0, 0, np.nan, False, "nope")
        with pytest.raises(ValueError, match="failed fit"):
            bad.to_metric(np.ones((4, 4)))


class TestToMetric:
    def test_pixels_beyond_the_fit_horizon_become_nan(self):
        fit = DepthScaleFit(scale=2.0, shift=1.0, n_observations=50, n_inliers=50,
                            rmse_disparity=0.0, ok=True)
        # d <= b implies z <= 0 or infinite.
        disp = np.array([[0.5, 1.0, 3.0]])
        z = fit.to_metric(disp)
        assert np.isnan(z[0, 0]) and np.isnan(z[0, 1])
        assert z[0, 2] == pytest.approx(1.0)

    def test_max_depth_clips_to_nan(self):
        fit = DepthScaleFit(2.0, 0.0, 50, 50, 0.0, True)
        disp = np.array([[2.0, 0.02]])  # z = 1.0 and z = 100.0
        z = fit.to_metric(disp, max_depth=10.0)
        assert z[0, 0] == pytest.approx(1.0)
        assert np.isnan(z[0, 1])

    def test_non_finite_disparity_becomes_nan(self):
        fit = DepthScaleFit(2.0, 0.0, 50, 50, 0.0, True)
        z = fit.to_metric(np.array([[np.nan, np.inf, 2.0]]))
        assert np.isnan(z[0, 0]) and np.isnan(z[0, 1])
        assert z[0, 2] == pytest.approx(1.0)


class TestSparseObservations:
    def _image(self, xys, ids):
        return Image(
            id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros(3),
            camera_id=1,
            name="f.jpg",
            xys=np.asarray(xys, dtype=float),
            point3D_ids=np.asarray(ids, dtype=np.int64),
        )

    def test_returns_camera_frame_z(self):
        # Identity pose, so camera Z equals world Z.
        pts = {
            10: Point3D(10, np.array([0.0, 0.0, 5.0]), np.zeros(3, np.uint8), 0.0),
            11: Point3D(11, np.array([1.0, 1.0, 2.0]), np.zeros(3, np.uint8), 0.0),
        }
        img = self._image([[1.0, 2.0], [3.0, 4.0]], [10, 11])
        xys, z = sparse_observations(img, pts)
        np.testing.assert_allclose(xys, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(z, [5.0, 2.0])

    def test_untriangulated_observations_are_dropped(self):
        pts = {10: Point3D(10, np.array([0.0, 0.0, 5.0]), np.zeros(3, np.uint8), 0.0)}
        img = self._image([[1.0, 2.0], [3.0, 4.0]], [10, -1])
        xys, z = sparse_observations(img, pts)
        assert len(xys) == 1 and len(z) == 1

    def test_no_observations(self):
        img = self._image(np.zeros((0, 2)), np.zeros((0,), dtype=np.int64))
        xys, z = sparse_observations(img, {})
        assert xys.shape == (0, 2) and z.shape == (0,)

    def test_pose_is_applied(self):
        """A translated camera must report translated depths."""
        pts = {10: Point3D(10, np.array([0.0, 0.0, 5.0]), np.zeros(3, np.uint8), 0.0)}
        img = self._image([[1.0, 2.0]], [10])
        img.tvec = np.array([0.0, 0.0, -2.0])  # camera moved forward by 2
        _, z = sparse_observations(img, pts)
        np.testing.assert_allclose(z, [3.0])


class TestFitAndConvert:
    def test_end_to_end_on_synthetic_frame(self):
        z_map = depth_ramp()
        disp = synthetic_disparity(z_map)

        rng = np.random.default_rng(4)
        n = 150
        xs = rng.uniform(0, W - 1, n)
        ys = rng.uniform(0, H - 1, n)
        zs = sample_bilinear(z_map, xs, ys)

        pts = {
            i: Point3D(i, np.array([0.0, 0.0, zs[i]]), np.zeros(3, np.uint8), 0.0)
            for i in range(n)
        }
        img = Image(
            id=1, qvec=np.array([1.0, 0, 0, 0]), tvec=np.zeros(3), camera_id=1, name="f.jpg",
            xys=np.column_stack([xs, ys]), point3D_ids=np.arange(n, dtype=np.int64),
        )

        metric, fit = fit_and_convert(disp, img, pts)
        assert fit.ok, fit.reason
        assert metric is not None
        np.testing.assert_allclose(metric, z_map, rtol=1e-3)

    def test_returns_none_when_fit_fails(self):
        img = Image(
            id=1, qvec=np.array([1.0, 0, 0, 0]), tvec=np.zeros(3), camera_id=1, name="f.jpg",
            xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), dtype=np.int64),
        )
        metric, fit = fit_and_convert(np.ones((H, W)), img, {})
        assert metric is None and not fit.ok


class TestSceneExtent:
    def test_radius_of_a_known_cloud(self):
        pts = {
            i: Point3D(i, np.array([float(i), 0.0, 0.0]), np.zeros(3, np.uint8), 0.0)
            for i in range(101)
        }
        # Median centre is x=50, so distances run 0..50.
        assert robust_scene_extent(pts, percentile=100.0) == pytest.approx(50.0)

    def test_empty_cloud_is_unbounded(self):
        assert robust_scene_extent({}) == float("inf")
