"""Tests for gaussian splat parameters, SSIM and 3DGS PLY export (MPO-223).

All CPU-only. The CUDA rasterizer is not exercised here; the training loop is
covered separately in test_trainer.py with an injected stub.
"""

import numpy as np
import pytest

# torch ships as an optional extra (MPO-231), so a base install has to be able
# to collect this file rather than erroring out.
torch = pytest.importorskip("torch", reason="needs a torch extra: uv sync --extra cpu")

from colmap_io import qvec2rotmat  # noqa: E402
from neural_reconstruction.gaussians import (  # noqa: E402
    C0,
    GaussianModel,
    inverse_sigmoid,
    knn_mean_distance,
    quat_to_rotmat,
    rgb_to_sh_dc,
    sh_dc_to_rgb,
    spherical_harmonic_degree_for_step,
    ssim,
)


def make_model(n=32, sh_degree=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * 2.0
    rgb = torch.rand(n, 3, generator=g)
    return GaussianModel(means, rgb, scene_scale=2.0, sh_degree=sh_degree, device="cpu")


class TestSHConversion:
    def test_round_trip(self):
        rgb = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]])
        torch.testing.assert_close(sh_dc_to_rgb(rgb_to_sh_dc(rgb)), rgb)

    def test_mid_grey_maps_to_zero(self):
        torch.testing.assert_close(
            rgb_to_sh_dc(torch.tensor([0.5])), torch.tensor([0.0])
        )

    def test_uses_the_standard_c0(self):
        assert C0 == pytest.approx(0.28209479177387814)


class TestInverseSigmoid:
    def test_is_the_inverse_of_sigmoid(self):
        x = torch.tensor([0.01, 0.1, 0.5, 0.9])
        torch.testing.assert_close(torch.sigmoid(inverse_sigmoid(x)), x)


class TestKnnMeanDistance:
    def test_unit_grid_spacing(self):
        # Points 1.0 apart on a line; nearest neighbours are at 1 and 2.
        pts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
        d = knn_mean_distance(pts, k=1)
        torch.testing.assert_close(d, torch.ones(4))

    def test_single_point_gets_a_fallback(self):
        d = knn_mean_distance(torch.zeros(1, 3))
        assert d.shape == (1,) and d[0] > 0

    def test_chunking_matches_unchunked(self):
        pts = torch.rand(200, 3, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(
            knn_mean_distance(pts, k=3, chunk=17), knn_mean_distance(pts, k=3, chunk=10_000)
        )

    def test_is_always_positive(self):
        # Duplicate points would give zero distance without the clamp.
        pts = torch.zeros(5, 3)
        assert (knn_mean_distance(pts) > 0).all()


class TestGaussianModelInit:
    def test_parameter_shapes(self):
        m = make_model(n=16, sh_degree=3)
        assert m.n == 16
        assert m.params["means"].shape == (16, 3)
        assert m.params["scales"].shape == (16, 3)
        assert m.params["quats"].shape == (16, 4)
        assert m.params["opacities"].shape == (16,)
        assert m.params["sh0"].shape == (16, 1, 3)
        assert m.params["shN"].shape == (16, (3 + 1) ** 2 - 1, 3)

    def test_quats_start_as_identity(self):
        m = make_model(n=8)
        torch.testing.assert_close(
            m.quats_act, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1)
        )

    def test_initial_opacity(self):
        m = make_model(n=8)
        torch.testing.assert_close(m.opacities_act, torch.full((8,), 0.1))

    def test_scales_are_positive_after_activation(self):
        m = make_model()
        assert (m.scales_act > 0).all()

    def test_sh_coeffs_truncation(self):
        m = make_model(n=4, sh_degree=3)
        assert m.sh_coeffs(0).shape == (4, 1, 3)
        assert m.sh_coeffs(1).shape == (4, 4, 3)
        assert m.sh_coeffs(3).shape == (4, 16, 3)

    def test_means_lr_scales_with_scene(self):
        small = GaussianModel(torch.rand(4, 3), torch.rand(4, 3), 1.0, device="cpu")
        big = GaussianModel(torch.rand(4, 3), torch.rand(4, 3), 100.0, device="cpu")
        assert big.lrs["means"] == pytest.approx(100.0 * small.lrs["means"])


class TestParameterSurgery:
    """Adding/removing gaussians must keep Adam's state aligned per gaussian."""

    def _step_once(self, m):
        loss = (m.params["means"] ** 2).sum() + m.params["opacities"].sum()
        m.optimizer.zero_grad()
        loss.backward()
        m.optimizer.step()

    def test_prune_reduces_every_parameter(self):
        m = make_model(n=10)
        keep = torch.zeros(10, dtype=torch.bool)
        keep[[1, 3, 5]] = True
        m.prune(keep)
        assert m.n == 3
        for name, p in m.params.items():
            assert p.shape[0] == 3, name

    def test_prune_keeps_the_right_rows(self):
        m = make_model(n=6)
        original = m.params["means"].detach().clone()
        keep = torch.tensor([False, True, False, True, False, True])
        m.prune(keep)
        torch.testing.assert_close(m.params["means"].detach(), original[keep])

    def test_prune_remaps_optimizer_state(self):
        m = make_model(n=6)
        self._step_once(m)
        state_before = m.optimizer.state[m.params["means"]]["exp_avg"].detach().clone()

        keep = torch.tensor([False, True, True, False, True, False])
        m.prune(keep)

        state_after = m.optimizer.state[m.params["means"]]["exp_avg"]
        assert state_after.shape[0] == 3
        torch.testing.assert_close(state_after, state_before[keep])

    def test_append_grows_every_parameter(self):
        m = make_model(n=5, sh_degree=2)
        extra = {k: v.detach()[:2].clone() for k, v in m.params.items()}
        m.append(extra)
        assert m.n == 7
        for name, p in m.params.items():
            assert p.shape[0] == 7, name

    def test_append_zero_initializes_new_momentum(self):
        m = make_model(n=5)
        self._step_once(m)
        before = m.optimizer.state[m.params["means"]]["exp_avg"].detach().clone()

        extra = {k: v.detach()[:2].clone() for k, v in m.params.items()}
        m.append(extra)

        after = m.optimizer.state[m.params["means"]]["exp_avg"]
        assert after.shape[0] == 7
        torch.testing.assert_close(after[:5], before)
        torch.testing.assert_close(after[5:], torch.zeros(2, 3))

    def test_optimizer_still_steps_after_surgery(self):
        m = make_model(n=8)
        self._step_once(m)
        m.prune(torch.tensor([True] * 4 + [False] * 4))
        m.append({k: v.detach()[:1].clone() for k, v in m.params.items()})
        before = m.params["means"].detach().clone()
        self._step_once(m)
        # Every gaussian, old and new, must actually move.
        assert not torch.allclose(m.params["means"].detach(), before)

    def test_param_groups_track_the_live_tensors(self):
        m = make_model(n=6)
        m.prune(torch.tensor([True, True, True, False, False, False]))
        for group in m.optimizer.param_groups:
            assert group["params"][0] is m.params[group["name"]]

    def test_reset_opacity_clamps_down_only(self):
        m = make_model(n=4)
        with torch.no_grad():
            m.params["opacities"].fill_(float(inverse_sigmoid(torch.tensor(0.9))))
        m.reset_opacity(0.01)
        assert torch.all(m.opacities_act <= 0.0101)

    def test_reset_opacity_does_not_raise_low_values(self):
        m = make_model(n=4)
        with torch.no_grad():
            m.params["opacities"].fill_(float(inverse_sigmoid(torch.tensor(0.001))))
        m.reset_opacity(0.01)
        assert torch.all(m.opacities_act <= 0.0011)


class TestSSIM:
    def test_identical_images_score_one(self):
        x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        assert float(ssim(x, x)) == pytest.approx(1.0, abs=1e-5)

    def test_different_images_score_below_one(self):
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1, 3, 32, 32, generator=g)
        y = torch.rand(1, 3, 32, 32, generator=g)
        assert float(ssim(x, y)) < 0.5

    def test_is_symmetric(self):
        g = torch.Generator().manual_seed(1)
        x = torch.rand(1, 3, 24, 24, generator=g)
        y = torch.rand(1, 3, 24, 24, generator=g)
        assert float(ssim(x, y)) == pytest.approx(float(ssim(y, x)), abs=1e-6)

    def test_small_perturbation_scores_high(self):
        x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(2))
        y = (x + 0.01).clamp(0, 1)
        assert float(ssim(x, y)) > 0.9

    def test_is_differentiable(self):
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        y = torch.rand(1, 3, 16, 16)
        (1 - ssim(x, y)).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ssim(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 9, 9))

    def test_preserves_batch_and_channel_count(self):
        # Single-channel input must work too (grayscale).
        x = torch.rand(2, 1, 20, 20)
        assert 0.0 <= float(ssim(x, x)) <= 1.0 + 1e-6


class TestQuatToRotmat:
    def test_identity(self):
        R = quat_to_rotmat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        torch.testing.assert_close(R[0], torch.eye(3))

    def test_matches_the_numpy_colmap_implementation(self):
        """Guards against the two diverging, which is how MPO-228 happened."""
        g = torch.Generator().manual_seed(3)
        q = torch.randn(10, 4, generator=g)
        got = quat_to_rotmat(q).numpy()
        for i in range(10):
            np.testing.assert_allclose(got[i], qvec2rotmat(q[i].numpy()), atol=1e-6)

    def test_output_is_a_rotation(self):
        q = torch.randn(5, 4, generator=torch.Generator().manual_seed(4))
        R = quat_to_rotmat(q)
        torch.testing.assert_close(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-6, rtol=1e-6)


class TestSHSchedule:
    def test_starts_at_zero_and_climbs(self):
        assert spherical_harmonic_degree_for_step(1, 1000, 3) == 0
        assert spherical_harmonic_degree_for_step(1000, 1000, 3) == 1
        assert spherical_harmonic_degree_for_step(3000, 1000, 3) == 3

    def test_is_capped_at_max_degree(self):
        assert spherical_harmonic_degree_for_step(100_000, 1000, 3) == 3


class TestSavePly:
    def test_field_layout_matches_the_3dgs_convention(self, tmp_path):
        m = make_model(n=5, sh_degree=3)
        p = m.save_ply(tmp_path / "splat.ply")
        head = p.read_bytes().split(b"end_header")[0].decode("ascii")

        for f in ("x", "y", "z", "nx", "ny", "nz", "opacity"):
            assert f"property float {f}\n" in head
        for i in range(3):
            assert f"property float f_dc_{i}\n" in head
            assert f"property float scale_{i}\n" in head
        for i in range(4):
            assert f"property float rot_{i}\n" in head
        # degree 3 -> 16 SH coefficients -> 15 rest, times 3 channels
        assert "property float f_rest_44\n" in head
        assert "property float f_rest_45\n" not in head

    def test_payload_size_is_consistent(self, tmp_path):
        n, sh = 7, 2
        m = make_model(n=n, sh_degree=sh)
        p = m.save_ply(tmp_path / "s.ply")
        raw = p.read_bytes()
        head, payload = raw.split(b"end_header\n", 1)
        n_fields = head.decode("ascii").count("property float ")
        assert len(payload) == n * n_fields * 4

    def test_positions_round_trip_via_plyfile(self, tmp_path):
        from plyfile import PlyData

        m = make_model(n=6, sh_degree=1)
        expected = m.params["means"].detach().numpy()
        p = m.save_ply(tmp_path / "s.ply")

        v = PlyData.read(str(p))["vertex"].data
        got = np.column_stack([v["x"], v["y"], v["z"]])
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_dc_term_encodes_the_input_colour(self, tmp_path):
        from plyfile import PlyData

        rgb = torch.tensor([[0.2, 0.4, 0.6], [1.0, 0.0, 0.5]])
        m = GaussianModel(torch.rand(2, 3), rgb, scene_scale=1.0, sh_degree=0, device="cpu")
        p = m.save_ply(tmp_path / "s.ply")

        v = PlyData.read(str(p))["vertex"].data
        dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]])
        np.testing.assert_allclose(dc * C0 + 0.5, rgb.numpy(), atol=1e-6)

    def test_f_rest_is_channel_major(self, tmp_path):
        """3DGS stores all R coefficients, then all G, then all B."""
        from plyfile import PlyData

        m = make_model(n=1, sh_degree=1)  # 4 coeffs -> 3 rest per channel
        with torch.no_grad():
            # Distinct values so the layout is unambiguous.
            m.params["shN"][0] = torch.tensor(
                [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]]
            )
        p = m.save_ply(tmp_path / "s.ply")
        v = PlyData.read(str(p))["vertex"].data
        rest = [float(v[f"f_rest_{i}"][0]) for i in range(9)]
        assert rest == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]

    def test_quaternions_are_written_normalized(self, tmp_path):
        from plyfile import PlyData

        m = make_model(n=4)
        with torch.no_grad():
            m.params["quats"].mul_(7.0)  # denormalize
        p = m.save_ply(tmp_path / "s.ply")
        v = PlyData.read(str(p))["vertex"].data
        q = np.column_stack([v[f"rot_{i}"] for i in range(4)])
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)

    def test_creates_parent_directories(self, tmp_path):
        m = make_model(n=2)
        p = m.save_ply(tmp_path / "deep" / "nested" / "s.ply")
        assert p.is_file()
