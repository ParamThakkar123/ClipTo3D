"""Gaussian splat parameter container, SH helpers, SSIM, and 3DGS PLY export.

Split out from the trainer so the parameter bookkeeping — in particular keeping
Adam's state aligned when gaussians are added or removed during densification —
is testable on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

# Zeroth-order spherical harmonic. Converting RGB in [0,1] to an SH DC term:
#   dc = (rgb - 0.5) / C0
C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> zeroth-order SH coefficient."""
    return (rgb - 0.5) / C0


def sh_dc_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    """Inverse of `rgb_to_sh_dc`."""
    return dc * C0 + 0.5


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def knn_mean_distance(points: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance to the k nearest other points, used to size initial gaussians.

    Chunked so a large sparse cloud does not need an N x N distance matrix.
    """
    n = len(points)
    if n <= 1:
        return torch.full((n,), 0.01, device=points.device)
    k = min(k, n - 1)
    out = torch.empty(n, device=points.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(points[start:end], points)
        # Exclude self (distance 0) by taking k+1 and dropping the first.
        vals, _ = torch.topk(d, k + 1, largest=False)
        out[start:end] = vals[:, 1:].mean(dim=1)
    return out.clamp_min(1e-7)


class GaussianModel:
    """Raw (pre-activation) gaussian parameters plus their optimizer.

    Activations: `scales = exp(raw)`, `opacities = sigmoid(raw)`, and quats are
    L2-normalized at use time.
    """

    def __init__(
        self,
        means: torch.Tensor,
        rgb: torch.Tensor,
        scene_scale: float,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.sh_degree = sh_degree
        self.scene_scale = float(scene_scale)

        means = means.to(self.device, torch.float32)
        rgb = rgb.to(self.device, torch.float32)
        n = len(means)

        dist = knn_mean_distance(means)
        scales = torch.log(dist)[:, None].repeat(1, 3)
        quats = torch.zeros(n, 4, device=self.device)
        quats[:, 0] = 1.0
        opacities = inverse_sigmoid(
            torch.full((n,), init_opacity, device=self.device)
        )

        n_sh = (sh_degree + 1) ** 2
        sh0 = rgb_to_sh_dc(rgb)[:, None, :]  # [N,1,3]
        shN = torch.zeros(n, n_sh - 1, 3, device=self.device)

        self.params: Dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }

        # Means move in world units, so their learning rate is scaled by scene
        # size; the rest are scale-free.
        self.lrs = {
            "means": 1.6e-4 * self.scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20.0,
        }
        self.optimizer = torch.optim.Adam(
            [{"params": [p], "lr": self.lrs[k], "name": k} for k, p in self.params.items()],
            eps=1e-15,
        )

    # -- derived quantities ------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.params["means"])

    @property
    def scales_act(self) -> torch.Tensor:
        return torch.exp(self.params["scales"])

    @property
    def opacities_act(self) -> torch.Tensor:
        return torch.sigmoid(self.params["opacities"])

    @property
    def quats_act(self) -> torch.Tensor:
        return F.normalize(self.params["quats"], dim=-1)

    def sh_coeffs(self, active_degree: int) -> torch.Tensor:
        """SH coefficients [N,K,3] truncated to `active_degree`."""
        k = (active_degree + 1) ** 2
        return torch.cat([self.params["sh0"], self.params["shN"]], dim=1)[:, :k, :]

    # -- parameter set surgery --------------------------------------------
    def _replace(self, new_tensors: Dict[str, torch.Tensor], state_op) -> None:
        """Swap in a new parameter set, remapping Adam state with `state_op`.

        Adding or removing gaussians changes the leading dimension of every
        parameter. Adam keeps `exp_avg`/`exp_avg_sq` buffers of matching shape,
        so they must be transformed identically or the optimizer silently
        applies one gaussian's momentum to another.
        """
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            new = torch.nn.Parameter(new_tensors[name].contiguous())

            state = self.optimizer.state.pop(old, None)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key] = state_op(state[key])
                self.optimizer.state[new] = state

            group["params"] = [new]
            self.params[name] = new

    def prune(self, keep: torch.Tensor) -> None:
        """Keep only gaussians where `keep` is True."""
        self._replace(
            {k: v.detach()[keep] for k, v in self.params.items()},
            lambda buf: buf[keep],
        )

    def append(self, extra: Dict[str, torch.Tensor]) -> None:
        """Append new gaussians, zero-initializing their optimizer momentum."""
        n_new = len(extra["means"])
        self._replace(
            {k: torch.cat([v.detach(), extra[k]], dim=0) for k, v in self.params.items()},
            lambda buf: torch.cat(
                [buf, torch.zeros((n_new, *buf.shape[1:]), device=buf.device, dtype=buf.dtype)],
                dim=0,
            ),
        )

    def reset_opacity(self, value: float = 0.01) -> None:
        """Clamp opacities down, the standard 3DGS floater-suppression step."""
        with torch.no_grad():
            target = inverse_sigmoid(torch.tensor(value, device=self.device))
            self.params["opacities"].clamp_(max=float(target))

    # -- export ------------------------------------------------------------
    def save_ply(self, path: Path | str) -> Path:
        """Write the standard 3DGS PLY that splat viewers expect."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            means = self.params["means"].detach().cpu().numpy()
            sh0 = self.params["sh0"].detach().cpu().numpy()  # [N,1,3]
            shN = self.params["shN"].detach().cpu().numpy()  # [N,K-1,3]
            opac = self.params["opacities"].detach().cpu().numpy()
            scales = self.params["scales"].detach().cpu().numpy()
            quats = F.normalize(self.params["quats"], dim=-1).detach().cpu().numpy()

        n = len(means)
        f_dc = sh0.reshape(n, 3)
        # 3DGS stores f_rest channel-major: all R coefficients, then G, then B.
        f_rest = shN.transpose(0, 2, 1).reshape(n, -1)

        fields = ["x", "y", "z", "nx", "ny", "nz"]
        fields += [f"f_dc_{i}" for i in range(3)]
        fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        fields += ["opacity", "scale_0", "scale_1", "scale_2"]
        fields += ["rot_0", "rot_1", "rot_2", "rot_3"]

        dtype = np.dtype([(f, "<f4") for f in fields])
        rec = np.empty(n, dtype=dtype)
        rec["x"], rec["y"], rec["z"] = means[:, 0], means[:, 1], means[:, 2]
        rec["nx"] = rec["ny"] = rec["nz"] = 0.0
        for i in range(3):
            rec[f"f_dc_{i}"] = f_dc[:, i]
        for i in range(f_rest.shape[1]):
            rec[f"f_rest_{i}"] = f_rest[:, i]
        rec["opacity"] = opac
        for i in range(3):
            rec[f"scale_{i}"] = scales[:, i]
        for i in range(4):
            rec[f"rot_{i}"] = quats[:, i]

        header = "\n".join(
            ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
            + [f"property float {f}" for f in fields]
            + ["end_header", ""]
        )
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(rec.tobytes())
        return path


def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable mean SSIM over NCHW tensors.

    Implemented locally rather than pulling in torchmetrics, which would add a
    dependency for one function.
    """
    if x.shape != y.shape:
        raise ValueError(f"ssim shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    c = x.shape[1]
    win1d = _gaussian_window(window_size, sigma, x.device)
    win_x = win1d.view(1, 1, 1, -1).expand(c, 1, 1, window_size)
    win_y = win1d.view(1, 1, -1, 1).expand(c, 1, window_size, 1)
    pad = window_size // 2

    def blur(t: torch.Tensor) -> torch.Tensor:
        t = F.conv2d(t, win_x, padding=(0, pad), groups=c)
        return F.conv2d(t, win_y, padding=(pad, 0), groups=c)

    mu_x, mu_y = blur(x), blur(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = blur(x * x) - mu_xx
    sigma_yy = blur(y * y) - mu_yy
    sigma_xy = blur(x * y) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return (num / den).mean()


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (w,x,y,z) quaternions, batched."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Batched (w,x,y,z) quaternion to 3x3 rotation matrices."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def spherical_harmonic_degree_for_step(
    step: int, interval: int, max_degree: int
) -> int:
    """SH bands are introduced gradually, as in the original 3DGS schedule."""
    return min(max_degree, step // max(1, interval))


__all__ = [
    "C0",
    "GaussianModel",
    "inverse_sigmoid",
    "knn_mean_distance",
    "quat_multiply",
    "quat_to_rotmat",
    "rgb_to_sh_dc",
    "sh_dc_to_rgb",
    "spherical_harmonic_degree_for_step",
    "ssim",
]
