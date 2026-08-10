"""Self-contained 3D gaussian splatting trainer (MPO-223).

Replaces the previous dead code path, which probed for `gsplat.train()` and a
`gsplat` console script. Neither exists — gsplat ships rasterization kernels,
not a trainer; its training example lives in the repo's `examples/` directory
and is not part of the wheel. So `train_from_colmap` could only ever raise.

What is implemented here is the standard 3DGS recipe on top of
`gsplat.rasterization`:

* initialize gaussians at the COLMAP sparse points, sized by nearest-neighbour
  spacing and coloured from the sparse point colours;
* optimize with L1 + D-SSIM against the training views;
* adaptive density control (clone / split / prune / periodic opacity reset);
* progressive spherical-harmonic band activation;
* export the standard 3DGS PLY.

The rasterizer is injected (`rasterize_fn`) so the training loop can be
exercised without CUDA. The default binds `gsplat.rasterization` lazily, which
keeps this module importable in a CPU-only environment.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    # torch is an optional extra so the base install stays CUDA-free (MPO-231).
    # Splat training additionally needs the CUDA build for gsplat's kernels.
    raise ModuleNotFoundError(
        "Splat training needs torch, which is not in the base install. On an "
        "NVIDIA worker: `uv sync --extra cuda --extra splat`."
    ) from exc

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from colmap_io import find_model_dir, read_model  # noqa: E402
from neural_reconstruction.gaussians import (  # noqa: E402
    GaussianModel,
    quat_multiply,
    quat_to_rotmat,
    spherical_harmonic_degree_for_step,
    ssim,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    max_steps: int = 7_000
    # Loss
    ssim_lambda: float = 0.2
    # SH schedule
    sh_degree: int = 3
    sh_increase_every: int = 1_000
    # Densification
    refine_start: int = 500
    refine_stop_frac: float = 0.5  # stop refining after this fraction of max_steps
    refine_every: int = 100
    grad_threshold: float = 2e-4
    # A gaussian smaller than this fraction of the scene radius is cloned;
    # larger ones are split.
    percent_dense: float = 0.01
    prune_opacity: float = 5e-3
    prune_scale_frac: float = 0.1  # prune gaussians wider than this * scene radius
    # The oversized-gaussian prune is gated behind a warmup, matching reference
    # 3DGS. Applying it from step 0 is destructive when the sparse init is
    # coarse: every gaussian starts "large", so every refinement splits
    # everything, shrinking scales until the prune wipes the model out. Measured
    # on a 30-point init, ungated pruning drove 30 gaussians down to 4 and the
    # loss up by 25x.
    prune_large_after: int = 3_000
    reset_opacity_every: int = 3_000
    split_samples: int = 2
    max_gaussians: int = 1_500_000
    # Data
    max_image_side: int = 1_600
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Reporting
    log_every: int = 100
    save_every: int = 0  # 0 disables intermediate checkpoints
    seed: int = 0


@dataclass
class TrainView:
    """One training camera."""

    image: torch.Tensor  # [3,H,W] float in [0,1]
    viewmat: torch.Tensor  # [4,4] world-to-camera
    K: torch.Tensor  # [3,3]
    width: int
    height: int
    name: str


@dataclass
class TrainResult:
    steps: int
    final_loss: float
    n_gaussians: int
    ply_path: Optional[Path] = None
    history: List[Dict] = field(default_factory=list)
    seconds: float = 0.0


def _default_rasterizer() -> Callable:
    """Bind gsplat's rasterization lazily, with an actionable error."""
    try:
        from gsplat import rasterization  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Gaussian splat training needs the optional 'splat' dependency:\n"
            "  uv sync --extra splat\n"
            "It also requires a CUDA-capable GPU and a CUDA toolchain, because "
            "gsplat compiles its kernels on first use."
        ) from e
    return rasterization


def load_colmap_views(
    colmap_dir: Path | str,
    images_dir: Path | str,
    max_image_side: int = 1_600,
    device: str | torch.device = "cuda",
) -> Tuple[List[TrainView], torch.Tensor, torch.Tensor, float]:
    """Load training views and sparse-point initialization from a COLMAP model.

    Returns (views, sparse_xyz, sparse_rgb01, scene_radius).
    """
    device = torch.device(device)
    images_dir = Path(images_dir)
    model_dir = find_model_dir(colmap_dir)
    cameras, images, points3D = read_model(model_dir)

    if not points3D:
        raise RuntimeError(
            f"{model_dir} contains no points3D.txt. Sparse points are needed to "
            f"initialize the gaussians."
        )

    views: List[TrainView] = []
    for img in sorted(images.values(), key=lambda im: im.name):
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue
        path = images_dir / Path(img.name).name
        if not path.is_file():
            continue

        pil = PILImage.open(path).convert("RGB")
        w, h = pil.size
        fx, fy, cx, cy = cam.intrinsics
        # Intrinsics are recorded for the camera's resolution; rescale if the
        # image on disk differs, and again if we downscale for VRAM.
        sx, sy = w / float(cam.width), h / float(cam.height)

        if max(w, h) > max_image_side:
            f = max_image_side / float(max(w, h))
            new_w, new_h = max(1, int(round(w * f))), max(1, int(round(h * f)))
            pil = pil.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
            sx *= new_w / w
            sy *= new_h / h
            w, h = new_w, new_h

        arr = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)

        K = torch.tensor(
            [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        viewmat = torch.eye(4, dtype=torch.float32)
        viewmat[:3, :3] = torch.from_numpy(img.R).float()
        viewmat[:3, 3] = torch.from_numpy(img.tvec).float()

        views.append(
            TrainView(
                image=arr.to(device), viewmat=viewmat.to(device), K=K.to(device),
                width=w, height=h, name=Path(img.name).name,
            )
        )

    if not views:
        raise RuntimeError(
            f"No COLMAP image was found in {images_dir}. The model references names "
            f"like {next(iter(images.values())).name!r}."
        )

    xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
    rgb = np.array([p.rgb for p in points3D.values()], dtype=np.float32) / 255.0

    # Scene radius from camera positions, which is what the 3DGS learning-rate
    # and pruning heuristics are calibrated against.
    centres = np.array(
        [(-v.viewmat[:3, :3].T @ v.viewmat[:3, 3]).cpu().numpy() for v in views]
    )
    radius = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max())
    radius = max(radius, 1e-3)

    logger.info(
        "Loaded %d views, %d sparse points, scene radius %.4f", len(views), len(xyz), radius
    )
    return views, torch.from_numpy(xyz), torch.from_numpy(rgb), radius


class SplatTrainer:
    def __init__(
        self,
        model: GaussianModel,
        views: List[TrainView],
        config: Optional[TrainConfig] = None,
        rasterize_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.views = views
        self.cfg = config or TrainConfig()
        self._rasterize = rasterize_fn  # resolved lazily
        self.device = model.device

        # Densification statistics, kept per gaussian and resized alongside.
        n = model.n
        self.grad_accum = torch.zeros(n, device=self.device)
        self.grad_count = torch.zeros(n, device=self.device)
        self.max_radii = torch.zeros(n, device=self.device)
        self._warned_no_grad = False

        # `percent_dense` decides clone vs split. If most gaussians start above
        # the threshold the split path dominates, which scatters children by the
        # parent's (large) extent and can destabilize training. That happens when
        # the sparse cloud is too coarse for the scene, so surface it rather than
        # letting it silently degrade quality.
        with torch.no_grad():
            largest = model.scales_act.detach().amax(dim=-1)
            frac_large = float((largest > self.cfg.percent_dense * model.scene_scale).float().mean())
        if frac_large > 0.5:
            logger.warning(
                "%.0f%% of initial gaussians exceed percent_dense * scene_scale "
                "(%.4g); densification will be split-dominated and may be unstable. "
                "The sparse cloud is likely too coarse for this scene — consider "
                "more COLMAP points or a larger percent_dense.",
                100 * frac_large, self.cfg.percent_dense * model.scene_scale,
            )

    @property
    def rasterize(self) -> Callable:
        if self._rasterize is None:
            self._rasterize = _default_rasterizer()
        return self._rasterize

    # -- rendering ---------------------------------------------------------
    def render(self, view: TrainView, active_sh: int):
        m = self.model
        colors = m.sh_coeffs(active_sh)
        out, alphas, info = self.rasterize(
            m.params["means"],
            m.quats_act,
            m.scales_act,
            m.opacities_act,
            colors,
            view.viewmat[None],
            view.K[None],
            view.width,
            view.height,
            sh_degree=active_sh,
            packed=False,
        )
        bg = torch.tensor(self.cfg.background, device=self.device).view(1, 1, 1, 3)
        rendered = out + (1.0 - alphas) * bg
        return rendered, info

    # -- densification -----------------------------------------------------
    def _accumulate_stats(self, info: Dict) -> None:
        means2d = info.get("means2d")
        if means2d is None or means2d.grad is None:
            # Densification is driven entirely by means2d gradients, so without
            # them it silently becomes a no-op. Warn once rather than quietly
            # training a model that never densifies.
            if not self._warned_no_grad:
                logger.warning(
                    "rasterizer info has no means2d gradient; densification is "
                    "disabled. The rasterizer must expose a differentiable "
                    "'means2d' entry."
                )
                self._warned_no_grad = True
            return
        grad = means2d.grad.detach()
        if grad.dim() == 3:  # [C,N,2] unpacked
            grad = grad.squeeze(0)
        norm = grad.norm(dim=-1)

        # `radii` has varied in shape across gsplat versions ([C,N] and [C,N,2]
        # have both existed), so normalize defensively to [N].
        radii = info.get("radii")
        r = None
        if radii is not None:
            r = radii.detach()
            while r.dim() > 1 and r.shape[0] == 1:
                r = r.squeeze(0)
            if r.dim() > 1:
                r = r.amax(dim=-1)
        visible = (r > 0) if r is not None and r.shape == norm.shape else (norm > 0)

        n = norm.shape[0]
        if n != self.grad_accum.shape[0]:
            # Should not happen, but never silently misalign statistics.
            logger.warning(
                "densification stats size %d != gaussian count %d; resetting",
                self.grad_accum.shape[0], n,
            )
            self.grad_accum = torch.zeros(n, device=self.device)
            self.grad_count = torch.zeros(n, device=self.device)
            self.max_radii = torch.zeros(n, device=self.device)

        self.grad_accum[visible] += norm[visible]
        self.grad_count[visible] += 1
        if r is not None and r.shape == norm.shape:
            self.max_radii[visible] = torch.maximum(
                self.max_radii[visible], r[visible].float()
            )

    def _reset_stats(self, n: int) -> None:
        self.grad_accum = torch.zeros(n, device=self.device)
        self.grad_count = torch.zeros(n, device=self.device)
        self.max_radii = torch.zeros(n, device=self.device)

    def densify_and_prune(self, step: int = 0) -> Dict[str, int]:
        m = self.model
        cfg = self.cfg
        scene_r = m.scene_scale

        avg_grad = self.grad_accum / self.grad_count.clamp_min(1.0)
        high_grad = avg_grad > cfg.grad_threshold

        scales = m.scales_act.detach()
        max_scale = scales.amax(dim=-1)
        small = max_scale <= cfg.percent_dense * scene_r

        headroom = max(0, cfg.max_gaussians - m.n)
        stats = {"cloned": 0, "split": 0, "pruned": 0}

        if headroom > 0:
            extras: Dict[str, List[torch.Tensor]] = {k: [] for k in m.params}
            n_before = m.n

            # Clone: small, high-gradient gaussians are duplicated in place and
            # the parent is kept.
            clone_idx = (high_grad & small).nonzero(as_tuple=True)[0][:headroom]
            if len(clone_idx):
                for k, p in m.params.items():
                    extras[k].append(p.detach()[clone_idx].clone())
                stats["cloned"] = len(clone_idx)
                headroom -= len(clone_idx)

            # Split: large, high-gradient gaussians are replaced by K smaller
            # children, so the parents must be removed afterwards.
            k_split = max(1, cfg.split_samples)
            split_idx = (high_grad & (~small)).nonzero(as_tuple=True)[0][: headroom // k_split]
            if len(split_idx):
                s = scales[split_idx]
                R = quat_to_rotmat(m.quats_act.detach()[split_idx])
                for _ in range(k_split):
                    noise = torch.randn_like(s) * s
                    offset = torch.einsum("mij,mj->mi", R, noise)
                    extras["means"].append(m.params["means"].detach()[split_idx] + offset)
                    # Shrink children so total coverage is roughly preserved.
                    extras["scales"].append(torch.log(s / (0.8 * k_split)).clamp(min=-20.0))
                    for k in ("quats", "opacities", "sh0", "shN"):
                        extras[k].append(m.params[k].detach()[split_idx].clone())
                stats["split"] = len(split_idx) * k_split

            merged = {k: torch.cat(v, dim=0) for k, v in extras.items() if v}
            if merged:
                m.append(merged)
                if len(split_idx):
                    # New gaussians were appended, so indices below n_before
                    # still address the originals. Drop exactly the parents we
                    # actually split — not every candidate, since the split set
                    # was budget-limited.
                    keep = torch.ones(m.n, dtype=torch.bool, device=self.device)
                    keep[split_idx] = False
                    assert keep.shape[0] >= n_before
                    m.prune(keep)
                    stats["pruned"] += int(len(split_idx))

        # Opacity pruning always applies; the oversized-gaussian prune only
        # after warmup (see TrainConfig.prune_large_after).
        opac = m.opacities_act.detach().reshape(-1)
        drop = opac < cfg.prune_opacity
        if step > cfg.prune_large_after:
            drop = drop | (m.scales_act.detach().amax(dim=-1) > cfg.prune_scale_frac * scene_r)
        if drop.any() and (~drop).sum() > 0:
            m.prune(~drop)
            stats["pruned"] += int(drop.sum().item())

        self._reset_stats(m.n)
        return stats

    # -- main loop ---------------------------------------------------------
    def train(self, out_dir: Optional[Path] = None) -> TrainResult:
        cfg = self.cfg
        m = self.model
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        refine_stop = int(cfg.max_steps * cfg.refine_stop_frac)
        history: List[Dict] = []
        started = time.time()
        loss_val = float("nan")

        for step in range(1, cfg.max_steps + 1):
            view = self.views[int(torch.randint(len(self.views), (1,), generator=g).item())]
            active_sh = spherical_harmonic_degree_for_step(
                step, cfg.sh_increase_every, cfg.sh_degree
            )

            rendered, info = self.render(view, active_sh)
            if info.get("means2d") is not None and info["means2d"].requires_grad:
                info["means2d"].retain_grad()

            # rasterization returns [C,H,W,3]; compare as NCHW.
            pred = rendered.squeeze(0).permute(2, 0, 1).clamp(0.0, 1.0)
            gt = view.image

            l1 = F.l1_loss(pred, gt)
            d_ssim = 1.0 - ssim(pred[None], gt[None])
            loss = (1.0 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * d_ssim

            m.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            loss_val = float(loss.detach())

            if step <= refine_stop:
                self._accumulate_stats(info)

            m.optimizer.step()

            if (
                cfg.refine_start < step <= refine_stop
                and step % cfg.refine_every == 0
            ):
                stats = self.densify_and_prune(step)
                logger.debug("step %d densify %s -> %d gaussians", step, stats, m.n)

            if cfg.reset_opacity_every and step % cfg.reset_opacity_every == 0:
                m.reset_opacity()

            if step % cfg.log_every == 0 or step == 1:
                entry = {"step": step, "loss": loss_val, "l1": float(l1.detach()),
                         "n_gaussians": m.n, "sh_degree": active_sh}
                history.append(entry)
                logger.info(
                    "step %d/%d loss %.5f (l1 %.5f) gaussians %d sh %d",
                    step, cfg.max_steps, loss_val, entry["l1"], m.n, active_sh,
                )

            if out_dir and cfg.save_every and step % cfg.save_every == 0:
                m.save_ply(Path(out_dir) / f"splat_{step:06d}.ply")

        result = TrainResult(
            steps=cfg.max_steps, final_loss=loss_val, n_gaussians=m.n,
            history=history, seconds=time.time() - started,
        )

        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            result.ply_path = m.save_ply(out_dir / "splat.ply")
            (out_dir / "train_log.json").write_text(
                json.dumps(
                    {"config": asdict(cfg), "history": history,
                     "final_loss": loss_val, "n_gaussians": m.n,
                     "seconds": result.seconds},
                    indent=2, default=str,
                ),
                encoding="utf-8",
            )
        return result


def train_from_colmap(
    colmap_dir: Path | str,
    images_dir: Path | str,
    out_dir: Path | str,
    config: Optional[TrainConfig] = None,
    device: Optional[str] = None,
    rasterize_fn: Optional[Callable] = None,
) -> TrainResult:
    """Train a gaussian splat model from a COLMAP model plus its images."""
    cfg = config or TrainConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and rasterize_fn is None:
        raise RuntimeError(
            "Gaussian splat training requires CUDA — gsplat's rasterizer has no CPU "
            "backend. No CUDA device was detected. Install a CUDA build of torch and "
            "run on a GPU host."
        )

    torch.manual_seed(cfg.seed)
    views, xyz, rgb, radius = load_colmap_views(
        colmap_dir, images_dir, max_image_side=cfg.max_image_side, device=device
    )
    model = GaussianModel(
        xyz, rgb, scene_scale=radius, sh_degree=cfg.sh_degree, device=device
    )
    trainer = SplatTrainer(model, views, cfg, rasterize_fn=rasterize_fn)
    return trainer.train(Path(out_dir))


__all__ = [
    "SplatTrainer",
    "TrainConfig",
    "TrainResult",
    "TrainView",
    "load_colmap_views",
    "train_from_colmap",
]
