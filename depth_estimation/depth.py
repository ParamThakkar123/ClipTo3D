"""Monocular depth estimation over a directory of frames.

Two backends: MiDaS (via torch.hub) and Depth-Anything-v2 (local checkpoint).

IMPORTANT — what these models output. Both produce *relative inverse* depth
(disparity), not metric depth. Larger values mean nearer, and the scale and
offset are arbitrary and differ per frame. Consumers must not treat the saved
values as Z. `fit_depth_scale` in `depth_scale.py` recovers metric depth by
fitting an affine transform in disparity space against COLMAP sparse points.

To make that contract explicit, this module writes a `depth_meta.json`
alongside the maps recording the backend and `"is_disparity": true`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    # torch is an optional extra so the base install stays CUDA-free (MPO-231).
    raise ModuleNotFoundError(
        "Depth estimation needs torch, which is not in the base install. "
        "Install a backend: `uv sync --extra cpu` (portable, also Apple "
        "Silicon), `--extra cuda` (NVIDIA worker), or `--extra mps` (macOS)."
    ) from exc

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import checkpoints  # noqa: E402  (needs the repo root on sys.path first)
import depth_io  # noqa: E402

DepthAnythingV2 = None
_DA_V2_TRANSFORMS = None
# Why the failure is kept: this used to swallow the exception entirely, which
# hid a missing torchvision dependency behind a misleading "expected it at
# .../dpt.py" message. The backend could not run at all and nothing said so.
_DA_V2_IMPORT_ERROR: Optional[BaseException] = None
try:
    from .depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    from .depth_anything_v2.util.transform import (  # type: ignore
        NormalizeImage,
        PrepareForNet,
        Resize,
    )

    _DA_V2_TRANSFORMS = (Resize, NormalizeImage, PrepareForNet)
except Exception as _relative_exc:
    _DA_V2_IMPORT_ERROR = _relative_exc
    # Running as a script rather than a package member: the vendored tree has
    # to go on sys.path before the flat imports below can resolve.
    _pkg_root = Path(__file__).resolve().parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        from depth_anything_v2.util.transform import (  # type: ignore
            NormalizeImage,
            PrepareForNet,
            Resize,
        )

        _DA_V2_TRANSFORMS = (Resize, NormalizeImage, PrepareForNet)
        _DA_V2_IMPORT_ERROR = None
    except Exception as _flat_exc:
        _DA_V2_IMPORT_ERROR = _flat_exc
        DepthAnythingV2 = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Encoder tiers for Depth-Anything-v2. Only vitl was previously reachable, so
# there was no cheaper option for previews or low-VRAM hosts.
DA_V2_ENCODERS: Dict[str, Dict] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def resolve_device(use_cuda: bool = True) -> torch.device:
    """Pick the best available accelerator, falling back to CPU.

    `use_cuda=False` means "force CPU" and disables MPS too — it is the
    `--cpu` escape hatch, not a CUDA-only switch.

    The MPS branch matters because the macOS wheel is Metal-enabled: without
    it, Macs took the slowest possible route even on hardware that works.
    `torch.backends.mps` is missing from some slim builds, hence the getattr.
    """
    if not use_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(frames_dir: Path) -> list[Path]:
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"frames directory does not exist: {frames_dir}")
    return sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _write_meta(
    out_path: Path,
    backend: str,
    model: str,
    count: int,
    weights: Optional[Dict[str, object]] = None,
) -> None:
    meta = {
        "backend": backend,
        "model": model,
        "frames": count,
        # Both backends emit relative inverse depth. Downstream code must fit
        # scale/shift before treating these values as distances.
        "is_disparity": True,
        "units": "relative_inverse_depth",
        # Which weights produced these maps, pinned by digest, so a run can be
        # reproduced rather than merely re-attempted (MPO-233).
        "weights": weights or {},
    }
    (out_path / "depth_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save(
    out_path: Path, stem: str, depth: np.ndarray, save_preview: bool,
    depth_format: str = depth_io.DEFAULT_FORMAT,
) -> None:
    depth_io.save_depth(out_path, stem, depth, fmt=depth_format)
    if save_preview:
        # Debug artifact, off by default (MPO-240) — it used to double the
        # per-frame storage. Named _preview so it cannot collide with the
        # png16 depth format.
        lo, hi = float(np.nanmin(depth)), float(np.nanmax(depth))
        norm = (depth - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(depth)
        Image.fromarray((np.nan_to_num(norm) * 255).astype(np.uint8)).save(
            out_path / f"{stem}_preview.png"
        )


# --- Depth-Anything-v2 batched inference (MPO-241) -------------------------

# Loaded models are cached so processing several clips in one worker does not
# rebuild the network and re-read the checkpoint each time.
_MODEL_CACHE: Dict[tuple, Any] = {}


def _load_da2_model(checkpoint: Path, encoder: str, device: torch.device):
    key = (str(checkpoint.resolve()), encoder, str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model = DepthAnythingV2(**DA_V2_ENCODERS[encoder])
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys when loading checkpoint, e.g. {missing[:3]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _da2_transform(input_size: int = 518):
    """The model's own preprocessing, built once rather than per image.

    `DepthAnythingV2.image2tensor` rebuilds this Compose on every call and —
    worse — hardcodes its own device selection, so it moved tensors to CUDA
    even when the caller asked for CPU. Bypassing it fixes both and is what
    makes batching possible.
    """
    from torchvision.transforms import Compose

    if _DA_V2_TRANSFORMS is None:  # pragma: no cover - same cause as the model import
        raise ImportError(
            f"Depth-Anything-v2 preprocessing is unavailable: {_DA_V2_IMPORT_ERROR!r}"
        )
    Resize, NormalizeImage, PrepareForNet = _DA_V2_TRANSFORMS

    return Compose([
        Resize(
            width=input_size, height=input_size, resize_target=False,
            keep_aspect_ratio=True, ensure_multiple_of=14,
            resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def _infer_batch(model, transform, bgr_images: list, device: torch.device, use_fp16: bool):
    """Run one batch, returning depth maps at each image's original size."""
    tensors = []
    sizes = []
    for bgr in bgr_images:
        h, w = bgr.shape[:2]
        sizes.append((h, w))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
        tensors.append(torch.from_numpy(transform({"image": rgb})["image"]))

    batch = torch.stack(tensors).to(device)
    # fp16 only helps on CUDA; CPU/MPS stay fp32 where half is often slower.
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16)
    with torch.inference_mode(), autocast:
        pred = model.forward(batch)

    pred = pred.float()
    out = []
    for i, (h, w) in enumerate(sizes):
        d = torch.nn.functional.interpolate(
            pred[i][None, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]
        out.append(d.cpu().numpy())
    return out


def estimate_depths_midas(
    frames_dir: Path | str,
    out_dir: Path | str,
    model_type: str = "DPT_Large",
    use_cuda: bool = True,
    save_preview: bool = False,
    depth_format: str = depth_io.DEFAULT_FORMAT,
) -> Path:
    """MiDaS backend.

    Pinned to an immutable upstream tag rather than the default branch, so a
    run is at least reproducible. It still needs network access on first use:
    torch.hub supplies the MiDaS *architecture*, not only its weights, so this
    backend cannot go fully offline without vendoring the upstream model code.
    Use the depthanythingv2 backend — whose model code is vendored here — for
    offline and air-gapped operation (MPO-233).
    """
    # Fail before doing any work, not after loading the frame list.
    if checkpoints.offline():
        raise RuntimeError(
            f"The MiDaS backend fetches its model code from torch.hub "
            f"({checkpoints.MIDAS_HUB_REF}) and cannot run with "
            f"{checkpoints.OFFLINE_ENV_VAR} set. Use --backend depthanythingv2, "
            f"whose weights are pinned and whose model code is vendored."
        )

    frames_path = Path(frames_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    device = resolve_device(use_cuda)
    print(f"MiDaS {model_type} on {device}")

    # torch.hub.load is typed as returning `object`; the modules it hands back
    # are only known at runtime.
    midas: Any = torch.hub.load(checkpoints.MIDAS_HUB_REF, model_type, trust_repo=True)  # type: ignore[arg-type]
    midas.to(device).eval()
    transforms: Any = torch.hub.load(checkpoints.MIDAS_HUB_REF, "transforms", trust_repo=True)  # type: ignore[arg-type]
    transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.dpt_transform

    images = list_images(frames_path)
    for img_file in images:
        img = Image.open(img_file).convert("RGB")
        orig_w, orig_h = img.size

        batch = transform(np.asarray(img))
        if isinstance(batch, dict):
            batch = batch["image"]
        batch = batch.to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)

        with torch.inference_mode():
            pred = midas(batch)
            pred = pred.squeeze(0) if pred.dim() == 3 else pred.squeeze()
            pred = torch.nn.functional.interpolate(
                pred[None, None], size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze()
            depth = pred.float().cpu().numpy()

        _save(out_path, img_file.stem, depth, save_preview, depth_format)

    _write_meta(out_path, "midas", model_type, len(images), {"torch_hub_ref": checkpoints.MIDAS_HUB_REF})
    print(f"Wrote {len(images)} depth maps to {out_path}")
    return out_path


def estimate_depths_depthanything(
    frames_dir: Path | str,
    out_dir: Path | str,
    checkpoint: Optional[Path | str] = None,
    encoder: str = "vitl",
    use_cuda: bool = True,
    save_preview: bool = False,
    batch_size: int = 8,
    fp16: bool = True,
    input_size: int = 518,
    depth_format: str = depth_io.DEFAULT_FORMAT,
) -> Path:
    """Depth-Anything-v2 backend.

    `checkpoint=None` resolves the pinned checkpoint for `encoder` from the
    registry, downloading it once into the cache and verifying its SHA256
    (MPO-233). Passing an explicit path bypasses the registry — the file is
    used as given and recorded in the metadata as unpinned.
    """
    if DepthAnythingV2 is None:
        raise ImportError(
            f"Could not import DepthAnythingV2 from "
            f"depth_estimation/depth_anything_v2/dpt.py: {_DA_V2_IMPORT_ERROR!r}. "
            f"Its dependencies ship with the torch extras — try "
            f"`uv sync --extra cpu` (or --extra cuda / --extra mps)."
        ) from _DA_V2_IMPORT_ERROR
    if encoder not in DA_V2_ENCODERS:
        raise ValueError(f"encoder must be one of {sorted(DA_V2_ENCODERS)}, got {encoder!r}")

    frames_path = Path(frames_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        # Pinned by URL + SHA256, fetched once into the cache, verified here.
        pinned = checkpoints.for_encoder(encoder)
        ckpt = checkpoints.ensure(pinned.name)
        weights = pinned.metadata()
    else:
        ckpt = Path(checkpoint).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Depth-Anything-v2 checkpoint not found: {ckpt}. Omit --checkpoint "
                f"to use the pinned {encoder} weights, or run "
                f"`python checkpoints.py --fetch {checkpoints.ENCODER_TO_CHECKPOINT[encoder]}`."
            )
        # Operator-supplied file: record what it actually is rather than
        # claiming the pinned provenance.
        weights = {
            "name": f"local:{ckpt.name}",
            "path": str(ckpt),
            "sha256": checkpoints.sha256_file(ckpt),
            "size_bytes": ckpt.stat().st_size,
            "pinned": False,
        }

    device = resolve_device(use_cuda)
    use_fp16 = fp16 and device.type == "cuda"
    print(
        f"Depth-Anything-v2 {encoder} on {device} "
        f"(batch {batch_size}, {'fp16' if use_fp16 else 'fp32'}, store {depth_format})"
    )

    model = _load_da2_model(ckpt, encoder, device)
    transform = _da2_transform(input_size)

    images = list_images(frames_path)
    written = 0
    # Frames from one clip share a resolution, so a whole batch is one shape.
    for start in range(0, len(images), max(1, batch_size)):
        chunk = images[start:start + max(1, batch_size)]
        loaded, kept = [], []
        for img_file in chunk:
            raw_bgr = cv2.imread(str(img_file))
            if raw_bgr is None:
                print(f"WARNING: could not read {img_file}, skipping.")
                continue
            loaded.append(raw_bgr)
            kept.append(img_file)
        if not loaded:
            continue

        # Mixed resolutions cannot share a batch tensor; fall back per image.
        shapes = {img.shape[:2] for img in loaded}
        groups = [(loaded, kept)] if len(shapes) == 1 else [([i], [f]) for i, f in zip(loaded, kept)]

        for imgs, files in groups:
            depths = _infer_batch(model, transform, imgs, device, use_fp16)
            for img_file, raw_bgr, depth in zip(files, imgs, depths):
                depth = np.asarray(depth).squeeze()
                orig_h, orig_w = raw_bgr.shape[:2]
                if depth.shape[:2] != (orig_h, orig_w):
                    depth = cv2.resize(
                        depth.astype(np.float32), (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                _save(out_path, img_file.stem, depth, save_preview, depth_format)
                written += 1

    _write_meta(out_path, "depthanythingv2", encoder, written, weights)
    print(f"Wrote {written} depth maps to {out_path}")
    return out_path


def estimate_depths(
    frames_dir: Path | str,
    out_dir: Path | str,
    model_backend: str = "depthanythingv2",
    model_type: str = "DPT_Large",
    depthanything_ckpt: Optional[Path | str] = None,
    encoder: str = "vitl",
    use_cuda: bool = True,
    save_preview: bool = False,
    batch_size: int = 8,
    fp16: bool = True,
    depth_format: str = depth_io.DEFAULT_FORMAT,
) -> Path:
    """Dispatch to a depth backend.

    Defaults to depthanythingv2: its model code is vendored here and its
    weights are pinned by digest, so it is the backend that runs offline. The
    midas backend still pulls its architecture from torch.hub (MPO-233).

    Paths are used exactly as given — previously they were re-rooted at the
    repo parent via `Path(x).relative_to(".")`, which both crashed on absolute
    paths and silently ignored where the caller pointed.
    """
    backend = model_backend.lower()
    if backend == "midas":
        return estimate_depths_midas(
            frames_dir, out_dir, model_type=model_type, use_cuda=use_cuda,
            save_preview=save_preview, depth_format=depth_format,
        )
    if backend in ("depthanythingv2", "depth_anything_v2", "da2"):
        # None is now valid: it means "use the pinned checkpoint for `encoder`".
        return estimate_depths_depthanything(
            frames_dir, out_dir, depthanything_ckpt, encoder=encoder,
            use_cuda=use_cuda, save_preview=save_preview,
            batch_size=batch_size, fp16=fp16, depth_format=depth_format,
        )
    raise ValueError(f"Unsupported model_backend {model_backend!r}. Use 'midas' or 'depthanythingv2'.")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Estimate relative depth maps for a directory of frames.")
    # Repo-relative defaults belong here, in the CLI, not inside the functions.
    p.add_argument("--frames-dir", type=Path, default=repo / "frames")
    p.add_argument("--out-dir", type=Path, default=repo / "depth")
    p.add_argument("--backend", default="depthanythingv2", choices=["midas", "depthanythingv2"],
                   help="depthanythingv2 runs offline; midas needs torch.hub.")
    p.add_argument("--model-type", default="DPT_Large", help="MiDaS model type.")
    p.add_argument("--checkpoint", type=Path,
                   help="Depth-Anything-v2 .pth. Omit to use the pinned, hash-verified "
                        "checkpoint for --encoder.")
    p.add_argument("--encoder", default="vitl", choices=sorted(DA_V2_ENCODERS))
    p.add_argument("--cpu", dest="use_cuda", action="store_false", help="Force CPU.")
    p.add_argument("--save-preview", action="store_true",
                   help="Also write 8-bit PNG previews (debug only; doubles storage).")
    p.add_argument("--batch-size", type=int, default=8, help="Frames per forward pass.")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="Disable mixed precision on CUDA.")
    p.add_argument("--depth-format", default=depth_io.DEFAULT_FORMAT, choices=depth_io.FORMATS,
                   help="fp16 halves storage; png16 is ~10x smaller.")
    args = p.parse_args()

    estimate_depths(
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        model_backend=args.backend,
        model_type=args.model_type,
        depthanything_ckpt=args.checkpoint,
        encoder=args.encoder,
        use_cuda=args.use_cuda,
        save_preview=args.save_preview,
        batch_size=args.batch_size,
        fp16=args.fp16,
        depth_format=args.depth_format,
    )
