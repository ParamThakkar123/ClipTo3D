"""Training-loop tests for the splat trainer (MPO-223).

gsplat's rasterizer is CUDA-only, so the loop is exercised with an injected
CPU stub. The stub is a real (if crude) differentiable additive splat renderer:
gaussians are projected with the true pinhole model and accumulated as isotropic
screen-space blobs. That is enough to verify the parts this repo owns — loss
computation, gradient flow to positions, densification bookkeeping, the SH
schedule, and export — without asserting anything about gsplat's own kernels.

What is NOT covered here: gsplat's actual rasterization and its exact `info`
tensor shapes.
"""

import json

import numpy as np
import pytest

# torch ships as an optional extra (MPO-231), so a base install has to be able
# to collect this file rather than erroring out.
torch = pytest.importorskip("torch", reason="needs a torch extra: uv sync --extra cpu")

from neural_reconstruction.gaussians import C0, GaussianModel  # noqa: E402
from neural_reconstruction.trainer import (  # noqa: E402
    SplatTrainer,
    TrainConfig,
    TrainView,
    load_colmap_views,
    train_from_colmap,
)


def stub_rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    sh_degree=None, packed=False, **kwargs,
):
    """Differentiable additive splat renderer, CPU-friendly.

    Returns the same (colors, alphas, info) contract the trainer consumes.
    """
    n = means.shape[0]
    R, t = viewmats[0, :3, :3], viewmats[0, :3, 3]
    cam = means @ R.T + t
    z = cam[:, 2].clamp(min=1e-4)
    fx, fy = Ks[0, 0, 0], Ks[0, 1, 1]
    cx, cy = Ks[0, 0, 2], Ks[0, 1, 2]
    px = cam[:, 0] / z * fx + cx
    py = cam[:, 1] / z * fy + cy
    means2d = torch.stack([px, py], dim=-1).unsqueeze(0)  # [1,N,2]

    sigma = (scales.mean(dim=-1) / z * fx).clamp(0.5, float(max(width, height)))
    rgb = colors[:, 0, :] * C0 + 0.5 if colors.dim() == 3 else colors

    yy = torch.arange(height, dtype=means.dtype).view(1, height, 1)
    xx = torch.arange(width, dtype=means.dtype).view(1, 1, width)
    # Read the projected positions back out of means2d so the rendered image
    # genuinely depends on it. gsplat differentiates through means2d, and the
    # trainer's densification reads means2d.grad, so a stub that bypassed it
    # would leave that path untested.
    d2 = (xx - means2d[0, :, 0].view(n, 1, 1)) ** 2 + (yy - means2d[0, :, 1].view(n, 1, 1)) ** 2
    w = torch.exp(-d2 / (2 * sigma.view(n, 1, 1) ** 2)) * opacities.view(n, 1, 1)
    w = w * (z > 0).to(w.dtype).view(n, 1, 1)

    num = (w.unsqueeze(-1) * rgb.view(n, 1, 1, 3)).sum(0)
    den = w.sum(0).unsqueeze(-1)
    img = num / (den + 1e-6)
    alpha = (1.0 - torch.exp(-den)).clamp(0.0, 1.0)
    radii = (sigma * 3.0).unsqueeze(0)
    return img.unsqueeze(0), alpha.unsqueeze(0), {"means2d": means2d, "radii": radii}


def make_views(n_views=3, size=24, device="cpu"):
    """A few cameras looking at the origin from slightly different places."""
    views = []
    g = torch.Generator().manual_seed(0)
    for i in range(n_views):
        img = torch.rand(3, size, size, generator=g)
        viewmat = torch.eye(4)
        viewmat[0, 3] = -0.1 * i
        viewmat[2, 3] = 3.0  # camera 3 units back along +Z
        K = torch.tensor(
            [[float(size), 0.0, size / 2.0], [0.0, float(size), size / 2.0], [0.0, 0.0, 1.0]]
        )
        views.append(
            TrainView(image=img.to(device), viewmat=viewmat.to(device), K=K.to(device),
                      width=size, height=size, name=f"v{i}.png")
        )
    return views


def make_fittable_views(gt_model, n_views=3, size=24, sh_degree=0):
    """Views whose target images are renders of a known gaussian configuration.

    Random per-pixel noise is not a fittable target for a handful of blobs — the
    best attainable loss is roughly the loss at initialization, so a convergence
    assertion against noise measures nothing. Rendering the target from a real
    configuration makes the optimum reachable.
    """
    views = make_views(n_views, size)
    with torch.no_grad():
        for v in views:
            img, alpha, _ = stub_rasterization(
                gt_model.params["means"], gt_model.quats_act, gt_model.scales_act,
                gt_model.opacities_act, gt_model.sh_coeffs(sh_degree),
                v.viewmat[None], v.K[None], v.width, v.height, sh_degree=sh_degree,
            )
            bg = torch.zeros(1, 1, 1, 3)
            target = (img + (1.0 - alpha) * bg).squeeze(0).permute(2, 0, 1).clamp(0, 1)
            v.image = target.contiguous()
    return views


def make_model(n=40, sh_degree=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * 1.5
    rgb = torch.rand(n, 3, generator=g)
    return GaussianModel(means, rgb, scene_scale=1.0, sh_degree=sh_degree, device="cpu")


class TestStubSanity:
    """If the stub is broken, everything below is meaningless."""

    def test_renders_expected_shapes(self):
        m = make_model(n=10)
        v = make_views(1)[0]
        img, alpha, info = stub_rasterization(
            m.params["means"], m.quats_act, m.scales_act, m.opacities_act,
            m.sh_coeffs(0), v.viewmat[None], v.K[None], v.width, v.height, sh_degree=0,
        )
        assert img.shape == (1, v.height, v.width, 3)
        assert alpha.shape == (1, v.height, v.width, 1)
        assert info["means2d"].shape == (1, 10, 2)

    def test_gradients_reach_means(self):
        m = make_model(n=10)
        img, _, info = stub_rasterization(
            m.params["means"], m.quats_act, m.scales_act, m.opacities_act,
            m.sh_coeffs(0), make_views(1)[0].viewmat[None], make_views(1)[0].K[None], 24, 24,
            sh_degree=0,
        )
        info["means2d"].retain_grad()
        img.sum().backward()
        assert m.params["means"].grad is not None
        assert info["means2d"].grad is not None


class TestTrainingLoop:
    def _trainer(self, steps=60, **kw):
        cfg = TrainConfig(
            max_steps=steps, sh_degree=1, sh_increase_every=20,
            refine_start=10, refine_every=10, refine_stop_frac=0.8,
            log_every=10, reset_opacity_every=0, **kw,
        )
        return SplatTrainer(make_model(), make_views(), cfg, rasterize_fn=stub_rasterization)

    def test_loss_decreases_on_a_fittable_target(self):
        """Convergence test: fit a perturbed model back towards a known one."""
        gt = make_model(n=30, sh_degree=0, seed=5)
        views = make_fittable_views(gt, sh_degree=0)

        start = make_model(n=30, sh_degree=0, seed=5)
        with torch.no_grad():  # perturb away from the optimum
            start.params["means"] += torch.randn_like(start.params["means"]) * 0.15
            start.params["sh0"] += torch.randn_like(start.params["sh0"]) * 0.3

        cfg = TrainConfig(
            max_steps=300, sh_degree=0, refine_start=10**9,  # isolate the optimizer
            log_every=25, reset_opacity_every=0,
        )
        t = SplatTrainer(start, views, cfg, rasterize_fn=stub_rasterization)
        result = t.train()

        losses = [h["loss"] for h in result.history]
        assert len(losses) >= 3
        # Mean of the last few vs the first, so one noisy step cannot decide it.
        assert np.mean(losses[-3:]) < 0.7 * losses[0], losses

    def test_loss_still_decreases_with_densification_enabled(self):
        gt = make_model(n=30, sh_degree=0, seed=5)
        views = make_fittable_views(gt, sh_degree=0)

        start = make_model(n=30, sh_degree=0, seed=5)
        with torch.no_grad():
            start.params["means"] += torch.randn_like(start.params["means"]) * 0.15

        cfg = TrainConfig(
            max_steps=300, sh_degree=0, refine_start=50, refine_every=50,
            refine_stop_frac=0.7, log_every=25, reset_opacity_every=0,
            max_gaussians=2_000,
            # This 30-gaussian init is coarse relative to the scene, so the
            # default percent_dense would classify every gaussian as "large" and
            # make refinement split-only. A real reconstruction has orders of
            # magnitude more sparse points and sits in the clone-dominated
            # regime, which is what this exercises.
            percent_dense=0.5,
        )
        t = SplatTrainer(start, views, cfg, rasterize_fn=stub_rasterization)
        result = t.train()

        losses = [h["loss"] for h in result.history]
        assert np.mean(losses[-3:]) < losses[0], losses

    def test_coarse_init_is_warned_about(self, caplog):
        """A split-dominated configuration must not fail silently."""
        cfg = TrainConfig(max_steps=1, sh_degree=0, percent_dense=1e-6, log_every=100)
        with caplog.at_level("WARNING"):
            SplatTrainer(make_model(sh_degree=0), make_views(), cfg,
                         rasterize_fn=stub_rasterization)
        assert any("split-dominated" in r.message for r in caplog.records)

    def test_oversized_prune_is_gated_behind_warmup(self):
        """Ungated, this drove a 30-gaussian model down to 4 (see prune_large_after)."""
        cfg = TrainConfig(
            max_steps=200, sh_degree=0, refine_start=20, refine_every=20,
            refine_stop_frac=0.9, log_every=50, reset_opacity_every=0,
            percent_dense=1e-6,  # force the split path
            prune_large_after=10**9,  # warmup never elapses
        )
        t = SplatTrainer(make_model(n=30, sh_degree=0), make_views(), cfg,
                         rasterize_fn=stub_rasterization)
        t.train()
        # Without the gate the population collapses; with it, it survives.
        assert t.model.n >= 15, f"model collapsed to {t.model.n} gaussians"

    def test_reports_finite_final_loss(self):
        result = self._trainer(steps=30).train()
        assert np.isfinite(result.final_loss)
        assert result.steps == 30

    def test_parameters_actually_move(self):
        t = self._trainer(steps=30)
        before = t.model.params["means"].detach().clone()
        t.train()
        n = min(len(before), t.model.n)
        assert not torch.allclose(before[:n], t.model.params["means"].detach()[:n])

    def test_sh_degree_climbs_over_training(self):
        result = self._trainer(steps=80).train()
        degrees = [h["sh_degree"] for h in result.history]
        assert degrees[0] == 0
        assert max(degrees) == 1  # capped at cfg.sh_degree
        assert degrees == sorted(degrees)

    def test_gaussian_count_stays_positive_and_bounded(self):
        t = self._trainer(steps=80, max_gaussians=500)
        result = t.train()
        assert 0 < result.n_gaussians <= 500
        for name, p in t.model.params.items():
            assert p.shape[0] == result.n_gaussians, name

    def test_all_parameters_stay_aligned_through_densification(self):
        """The failure mode this guards: parameter tensors drifting out of sync."""
        t = self._trainer(steps=100, grad_threshold=0.0)  # force aggressive refinement
        t.train()
        counts = {name: p.shape[0] for name, p in t.model.params.items()}
        assert len(set(counts.values())) == 1, counts
        # Optimizer state must match too.
        for group in t.model.optimizer.param_groups:
            p = group["params"][0]
            assert p is t.model.params[group["name"]]
            state = t.model.optimizer.state.get(p)
            if state and "exp_avg" in state:
                assert state["exp_avg"].shape[0] == p.shape[0], group["name"]

    def test_densification_can_grow_the_model(self):
        t = self._trainer(steps=60, grad_threshold=0.0)
        n_before = t.model.n
        t.train()
        # With a zero gradient threshold every visible gaussian is a candidate.
        assert t.model.n != n_before

    def test_no_nan_parameters_after_training(self):
        t = self._trainer(steps=60, grad_threshold=0.0)
        t.train()
        for name, p in t.model.params.items():
            assert torch.isfinite(p.detach()).all(), name

    def test_opacity_reset_is_applied(self):
        cfg = TrainConfig(
            max_steps=20, sh_degree=0, refine_start=1000, log_every=100,
            reset_opacity_every=10,
        )
        t = SplatTrainer(make_model(sh_degree=0), make_views(), cfg,
                         rasterize_fn=stub_rasterization)
        t.train()
        assert float(t.model.opacities_act.detach().max()) <= 0.02


class TestOutputs:
    def test_writes_ply_and_log(self, tmp_path):
        cfg = TrainConfig(max_steps=20, sh_degree=1, refine_start=1000,
                          log_every=10, reset_opacity_every=0)
        t = SplatTrainer(make_model(sh_degree=1), make_views(), cfg,
                         rasterize_fn=stub_rasterization)
        result = t.train(tmp_path)

        assert result.ply_path is not None and result.ply_path.is_file()
        log = json.loads((tmp_path / "train_log.json").read_text())
        assert log["config"]["max_steps"] == 20
        assert len(log["history"]) >= 2
        assert np.isfinite(log["final_loss"])

    def test_exported_ply_is_readable(self, tmp_path):
        from plyfile import PlyData

        cfg = TrainConfig(max_steps=10, sh_degree=1, refine_start=1000,
                          log_every=10, reset_opacity_every=0)
        t = SplatTrainer(make_model(sh_degree=1), make_views(), cfg,
                         rasterize_fn=stub_rasterization)
        result = t.train(tmp_path)

        v = PlyData.read(str(result.ply_path))["vertex"].data
        assert len(v) == t.model.n
        assert np.isfinite(np.column_stack([v["x"], v["y"], v["z"]])).all()

    def test_intermediate_checkpoints(self, tmp_path):
        cfg = TrainConfig(max_steps=20, sh_degree=0, refine_start=1000,
                          log_every=100, reset_opacity_every=0, save_every=10)
        t = SplatTrainer(make_model(sh_degree=0), make_views(), cfg,
                         rasterize_fn=stub_rasterization)
        t.train(tmp_path)
        assert (tmp_path / "splat_000010.ply").is_file()
        assert (tmp_path / "splat_000020.ply").is_file()


class TestGuards:
    def test_cpu_without_a_stub_fails_with_a_clear_message(self, tmp_path):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            train_from_colmap(tmp_path, tmp_path, tmp_path, device="cpu")

    def test_default_rasterizer_error_names_the_extra(self, monkeypatch):
        """When gsplat is absent the message must say how to install it."""
        import builtins

        import neural_reconstruction.trainer as tr

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "gsplat":
                raise ImportError("no gsplat")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="--extra splat"):
            tr._default_rasterizer()


CAMERAS_TXT = "1 PINHOLE 32 24 30.0 30.0 16.0 12.0\n"
IMAGES_TXT = (
    "1 1.0 0.0 0.0 0.0 0.0 0.0 3.0 1 a.png\n"
    "1.0 2.0 1\n"
    "2 1.0 0.0 0.0 0.0 0.5 0.0 3.0 1 b.png\n"
    "\n"  # deliberately empty POINTS2D, the MPO-228 case
)
POINTS3D_TXT = (
    "1 0.0 0.0 0.0 255 0 0 0.1 1 0\n"
    "2 0.2 0.1 0.3 0 255 0 0.1 1 0\n"
    "3 -0.2 -0.1 0.1 0 0 255 0.1 1 0\n"
)


class TestLoadColmapViews:
    def _model(self, tmp_path):
        from PIL import Image as PILImage

        model = tmp_path / "colmap" / "sparse" / "0"
        model.mkdir(parents=True)
        (model / "cameras.txt").write_text(CAMERAS_TXT)
        (model / "images.txt").write_text(IMAGES_TXT)
        (model / "points3D.txt").write_text(POINTS3D_TXT)

        images = tmp_path / "frames"
        images.mkdir()
        for name in ("a.png", "b.png"):
            PILImage.fromarray(
                (np.random.default_rng(0).random((24, 32, 3)) * 255).astype(np.uint8)
            ).save(images / name)
        return tmp_path / "colmap", images

    def test_loads_views_and_sparse_points(self, tmp_path):
        colmap_dir, images = self._model(tmp_path)
        views, xyz, rgb, radius = load_colmap_views(colmap_dir, images, device="cpu")

        assert len(views) == 2
        assert xyz.shape == (3, 3) and rgb.shape == (3, 3)
        assert radius > 0
        assert views[0].name == "a.png"
        assert views[0].image.shape == (3, 24, 32)

    def test_colours_are_normalized(self, tmp_path):
        colmap_dir, images = self._model(tmp_path)
        _, _, rgb, _ = load_colmap_views(colmap_dir, images, device="cpu")
        assert float(rgb.max()) <= 1.0
        np.testing.assert_allclose(rgb[0].numpy(), [1.0, 0.0, 0.0], atol=1e-6)

    def test_viewmat_is_world_to_camera(self, tmp_path):
        colmap_dir, images = self._model(tmp_path)
        views, _, _, _ = load_colmap_views(colmap_dir, images, device="cpu")
        # Identity rotation with tvec (0,0,3): the world origin sits 3 in front.
        vm = views[0].viewmat
        torch.testing.assert_close(vm[:3, :3], torch.eye(3))
        torch.testing.assert_close(vm[:3, 3], torch.tensor([0.0, 0.0, 3.0]))

    def test_downscaling_rescales_intrinsics(self, tmp_path):
        colmap_dir, images = self._model(tmp_path)
        full, _, _, _ = load_colmap_views(colmap_dir, images, device="cpu")
        small, _, _, _ = load_colmap_views(
            colmap_dir, images, max_image_side=16, device="cpu"
        )
        assert small[0].width == 16
        ratio = small[0].width / full[0].width
        assert float(small[0].K[0, 0]) == pytest.approx(float(full[0].K[0, 0]) * ratio, rel=1e-5)
        assert float(small[0].K[0, 2]) == pytest.approx(float(full[0].K[0, 2]) * ratio, rel=1e-5)

    def test_missing_points3D_raises(self, tmp_path):
        colmap_dir, images = self._model(tmp_path)
        (colmap_dir / "sparse" / "0" / "points3D.txt").write_text("# none\n")
        with pytest.raises(RuntimeError, match="no points3D"):
            load_colmap_views(colmap_dir, images, device="cpu")

    def test_missing_images_raises(self, tmp_path):
        colmap_dir, _ = self._model(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="No COLMAP image"):
            load_colmap_views(colmap_dir, empty, device="cpu")

    def test_end_to_end_train_from_a_colmap_model(self, tmp_path):
        colmap_dir, images = self._model(tmp_path)
        cfg = TrainConfig(max_steps=15, sh_degree=0, refine_start=1000,
                          log_every=5, reset_opacity_every=0, max_image_side=32)
        result = train_from_colmap(
            colmap_dir, images, tmp_path / "out", config=cfg,
            device="cpu", rasterize_fn=stub_rasterization,
        )
        assert result.ply_path is not None and result.ply_path.is_file()
        assert np.isfinite(result.final_loss)
        assert result.n_gaussians == 3
