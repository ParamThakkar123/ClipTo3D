"""Depth-map storage (MPO-240).

Depth was written as full-resolution uncompressed float32 `.npy`: ~8MB per
1080p frame, so ~2.4GB of intermediate per 300-frame job. On a multi-tenant
service that is the dominant storage cost, and on network-backed storage the
dominant inter-stage I/O cost.

Three formats, all round-tripped through `load_depth` so consumers do not care
which was used:

* ``fp16`` (default) — `.npy` at half precision. Exactly half the bytes, and
  the values are *relative* disparity whose precision never justified fp32.
* ``png16`` — 16-bit PNG plus a per-frame scale/offset in a JSON sidecar.
  Roughly 10x smaller because depth is smooth and compresses well. Lossy at the
  1/65535-of-range level, which is far below the noise of any monocular model.
* ``fp32`` — the original, kept for exactness when someone wants it.

The quantisation is per frame, so a frame's own dynamic range is preserved
regardless of what the rest of the clip looks like.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

FORMATS = ("fp16", "png16", "fp32")
DEFAULT_FORMAT = "fp16"

_DTYPE = {"fp16": np.float16, "fp32": np.float32}


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def save_depth(out_dir: Path | str, stem: str, depth: np.ndarray, fmt: str = DEFAULT_FORMAT) -> Path:
    """Write one depth map. Returns the path written."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown depth format {fmt!r}; expected one of {FORMATS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth, dtype=np.float64)

    if fmt in ("fp16", "fp32"):
        path = out_dir / f"{stem}_depth.npy"
        np.save(path, depth.astype(_DTYPE[fmt]))
        return path

    # png16: normalise this frame to [0, 65535] and record the affine mapping.
    import cv2

    path = out_dir / f"{stem}_depth.png"
    finite = np.isfinite(depth)
    if not finite.any():
        lo, hi = 0.0, 1.0
    else:
        lo = float(depth[finite].min())
        hi = float(depth[finite].max())
    span = hi - lo
    if span <= 0:
        span = 1.0
    q = np.clip((depth - lo) / span, 0.0, 1.0)
    # Non-finite pixels would otherwise quantise to 0, which reads back as a
    # real measurement at the near plane. Record them and restore as NaN.
    q = np.where(finite, q, 0.0)
    cv2.imwrite(str(path), (q * 65535.0).round().astype(np.uint16))
    _sidecar(path).write_text(
        json.dumps({"format": "png16", "lo": lo, "span": span,
                    "has_nonfinite": bool((~finite).any())}),
        encoding="utf-8",
    )
    return path


def load_depth(path: Path | str) -> np.ndarray:
    """Read a depth map back as float32, whatever format it was written in."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr.astype(np.float32)

    if path.suffix == ".png":
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"could not read depth PNG {path}")
        side = _sidecar(path)
        if not side.is_file():
            raise FileNotFoundError(
                f"{path} has no {side.name} sidecar; the scale/offset needed to "
                f"decode it is stored there."
            )
        meta = json.loads(side.read_text(encoding="utf-8"))
        lo = float(meta["lo"])
        span = float(meta["span"])
        return (raw.astype(np.float32) / 65535.0) * span + lo

    raise ValueError(f"unrecognised depth file {path}")


def find_depth(depth_dir: Path | str, stem: str) -> Optional[Path]:
    """Locate the depth map for a frame stem, in any supported format."""
    depth_dir = Path(depth_dir)
    for name in (f"{stem}_depth.npy", f"{stem}.npy", f"{stem}_depth.png"):
        p = depth_dir / name
        if p.is_file():
            return p
    return None


def depth_bytes(depth_dir: Path | str) -> Tuple[int, int]:
    """(total bytes, file count) of depth artifacts — for reporting savings."""
    depth_dir = Path(depth_dir)
    total = 0
    count = 0
    for p in depth_dir.iterdir():
        if p.is_file() and (p.name.endswith("_depth.npy") or p.name.endswith("_depth.png")
                            or p.name.endswith("_depth.json")):
            total += p.stat().st_size
            count += 1
    return total, count
