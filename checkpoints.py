"""Pinned, hash-verified model checkpoints (MPO-233).

The depth stage used to call `torch.hub.load("intel-isl/MiDaS", ...)` on every
invocation. That meant the pipeline needed network access at runtime, tracked
the upstream default branch with no pinned revision, and had no way to
reproduce a past run's weights.

This module is the replacement: every checkpoint is pinned by immutable URL
(an upstream commit, not a branch) plus its SHA256 and byte size. Weights are
downloaded once into a cache directory — at image build time for the worker —
and verified against the pinned digest on load.

    python checkpoints.py --list
    python checkpoints.py --fetch depth-anything-v2-vitl     # image build step
    python checkpoints.py --verify                           # audit the cache

The cache directory is `$CLIPTO3D_CHECKPOINT_DIR`, falling back to
`~/.cache/clipto3d/checkpoints`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

CACHE_ENV_VAR = "CLIPTO3D_CHECKPOINT_DIR"
OFFLINE_ENV_VAR = "CLIPTO3D_OFFLINE"
TRUST_CACHE_ENV_VAR = "CLIPTO3D_TRUST_CHECKPOINT_CACHE"

DEFAULT_CACHE = Path.home() / ".cache" / "clipto3d" / "checkpoints"
_CHUNK = 1 << 20  # 1 MiB

# torch.hub still supplies the MiDaS *architecture*, not just its weights, so
# that backend cannot be made fully offline without vendoring the upstream
# model code. Pinning the ref at least makes it reproducible and stops it
# tracking the default branch. Depth-Anything-v2 is the offline path: its model
# code is already vendored under depth_estimation/depth_anything_v2/.
MIDAS_HUB_REF = "intel-isl/MiDaS:v3_1"


class ChecksumMismatch(RuntimeError):
    """A checkpoint on disk does not match its pinned digest."""


@dataclass(frozen=True)
class Checkpoint:
    name: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    revision: str
    description: str

    def metadata(self) -> Dict[str, object]:
        """The record written into job metadata, so a run is reproducible."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "url": self.url,
            "revision": self.revision,
        }


def _hf(repo: str, revision: str, filename: str) -> str:
    # /resolve/<commit>/ rather than /resolve/main/ — an immutable URL.
    return f"https://huggingface.co/depth-anything/{repo}/resolve/{revision}/{filename}"


REGISTRY: Dict[str, Checkpoint] = {
    "depth-anything-v2-vits": Checkpoint(
        name="depth-anything-v2-vits",
        filename="depth_anything_v2_vits.pth",
        url=_hf("Depth-Anything-V2-Small", "03876f8651c73a60fe4c2c48294e09fcb6838fcf",
                "depth_anything_v2_vits.pth"),
        sha256="715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
        size_bytes=99_218_434,
        revision="03876f8651c73a60fe4c2c48294e09fcb6838fcf",
        description="Depth-Anything-v2 Small (24.8M params) — previews, low-VRAM hosts.",
    ),
    "depth-anything-v2-vitb": Checkpoint(
        name="depth-anything-v2-vitb",
        filename="depth_anything_v2_vitb.pth",
        url=_hf("Depth-Anything-V2-Base", "a4e71a6c2ce52fe50df0f212066b0d4a87be9b5e",
                "depth_anything_v2_vitb.pth"),
        sha256="0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328",
        size_bytes=389_961_218,
        revision="a4e71a6c2ce52fe50df0f212066b0d4a87be9b5e",
        description="Depth-Anything-v2 Base (97.5M params).",
    ),
    "depth-anything-v2-vitl": Checkpoint(
        name="depth-anything-v2-vitl",
        filename="depth_anything_v2_vitl.pth",
        url=_hf("Depth-Anything-V2-Large", "cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9",
                "depth_anything_v2_vitl.pth"),
        sha256="a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345",
        size_bytes=1_341_395_338,
        revision="cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9",
        description="Depth-Anything-v2 Large (335.3M params) — default.",
    ),
}

# depth.py takes an encoder tier; this maps it onto the registry.
ENCODER_TO_CHECKPOINT = {
    "vits": "depth-anything-v2-vits",
    "vitb": "depth-anything-v2-vitb",
    "vitl": "depth-anything-v2-vitl",
}


def cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_CACHE


def offline() -> bool:
    return os.environ.get(OFFLINE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _trust_cache() -> bool:
    return os.environ.get(TRUST_CACHE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def get(name: str) -> Checkpoint:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown checkpoint {name!r}; known: {sorted(REGISTRY)}") from None


def for_encoder(encoder: str) -> Checkpoint:
    try:
        return REGISTRY[ENCODER_TO_CHECKPOINT[encoder]]
    except KeyError:
        raise KeyError(
            f"no pinned checkpoint for encoder {encoder!r}; known: {sorted(ENCODER_TO_CHECKPOINT)}"
        ) from None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, ckpt: Checkpoint) -> None:
    """Raise unless `path` matches the pinned size and digest."""
    actual_size = path.stat().st_size
    if actual_size != ckpt.size_bytes:
        raise ChecksumMismatch(
            f"{path} is {actual_size} bytes, expected {ckpt.size_bytes} for "
            f"{ckpt.name}. Delete it and re-fetch."
        )
    actual = sha256_file(path)
    if actual != ckpt.sha256:
        raise ChecksumMismatch(
            f"{path} has sha256 {actual}, expected {ckpt.sha256} for {ckpt.name}. "
            f"Delete it and re-fetch."
        )


def download(
    ckpt: Checkpoint,
    dest_dir: Optional[Path] = None,
    *,
    force: bool = False,
    retries: int = 3,
) -> Path:
    """Download and verify one checkpoint. Returns its path.

    Writes to a `.part` file and only renames after the digest matches, so an
    interrupted download can never be mistaken for a complete one.

    Retries on truncation and transport errors. These files are 99MB–1.3GB and
    a mid-transfer cut is routine; without retries a truncated read fails the
    whole job — or the whole image build, which is how this was found: 16MB of
    a 99MB checkpoint arrived and the size check correctly rejected it.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / ckpt.filename

    if target.exists() and not force:
        verify(target, ckpt)
        return target

    if offline():
        raise FileNotFoundError(
            f"{ckpt.name} is not in {dest_dir} and {OFFLINE_ENV_VAR} is set. "
            f"Pre-fetch it with `python checkpoints.py --fetch {ckpt.name}`."
        )

    part = dest_dir / f"{ckpt.filename}.part"
    last: Optional[Exception] = None

    for attempt in range(1, max(1, retries) + 1):
        suffix = "" if attempt == 1 else f" (attempt {attempt}/{retries})"
        print(f"Downloading {ckpt.name} ({ckpt.size_bytes / 1e6:.0f} MB) from {ckpt.url}{suffix}")
        # Always start clean: a partial file from the previous attempt would
        # otherwise be appended to or mistaken for progress.
        part.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(ckpt.url) as response, open(part, "wb") as fh:
                shutil.copyfileobj(response, fh, _CHUNK)
            verify(part, ckpt)
        except (urllib.error.URLError, ChecksumMismatch, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                delay = 2 ** (attempt - 1)
                print(f"  {type(exc).__name__}: {exc}\n  retrying in {delay}s")
                time.sleep(delay)
            continue

        part.replace(target)
        print(f"Verified {ckpt.name} -> {target}")
        return target

    raise RuntimeError(
        f"failed to download {ckpt.name} from {ckpt.url} after {retries} attempts: {last}"
    ) from last


def ensure(name: str, *, dest_dir: Optional[Path] = None, allow_download: bool = True) -> Path:
    """Return a verified local path to `name`, downloading it if permitted.

    The digest is checked on every load. On the 1.3GB vitl checkpoint that
    costs a couple of seconds against a job measured in minutes; set
    CLIPTO3D_TRUST_CHECKPOINT_CACHE=1 to check only the size instead.
    """
    ckpt = get(name)
    dest_dir = Path(dest_dir) if dest_dir is not None else cache_dir()
    target = dest_dir / ckpt.filename

    if target.exists():
        if _trust_cache():
            actual_size = target.stat().st_size
            if actual_size != ckpt.size_bytes:
                raise ChecksumMismatch(
                    f"{target} is {actual_size} bytes, expected {ckpt.size_bytes} "
                    f"for {ckpt.name}. Delete it and re-fetch."
                )
        else:
            verify(target, ckpt)
        return target

    if not allow_download or offline():
        raise FileNotFoundError(
            f"Checkpoint {ckpt.name} not found at {target}. Fetch it with "
            f"`python checkpoints.py --fetch {ckpt.name}`, or point "
            f"{CACHE_ENV_VAR} at a directory that already has it."
        )
    return download(ckpt, dest_dir)


# --- CLI ------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List pinned checkpoints.")
    p.add_argument("--fetch", nargs="*", metavar="NAME",
                   help="Download checkpoints (no names = all).")
    p.add_argument("--verify", action="store_true", help="Re-verify everything already cached.")
    p.add_argument("--cache-dir", type=Path, default=None, help=f"Overrides ${CACHE_ENV_VAR}.")
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = p.parse_args(argv)

    dest = args.cache_dir or cache_dir()

    if args.list or not (args.fetch is not None or args.verify):
        print(f"cache: {dest}")
        for ckpt in REGISTRY.values():
            path = dest / ckpt.filename
            state = "cached" if path.exists() else "absent"
            print(f"  {ckpt.name:26} {ckpt.size_bytes / 1e6:8.0f} MB  {state:7} {ckpt.description}")
        return 0

    if args.verify:
        failures = 0
        for ckpt in REGISTRY.values():
            path = dest / ckpt.filename
            if not path.exists():
                print(f"  {ckpt.name:26} absent")
                continue
            try:
                verify(path, ckpt)
                print(f"  {ckpt.name:26} OK")
            except ChecksumMismatch as exc:
                failures += 1
                print(f"  {ckpt.name:26} FAILED: {exc}")
        return 1 if failures else 0

    names: Iterable[str] = args.fetch or list(REGISTRY)
    for name in names:
        download(get(name), dest, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
