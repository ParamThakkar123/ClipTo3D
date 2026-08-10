"""Keyframe selection between frame extraction and everything downstream (MPO-237).

`frames.py` extracts at a fixed fps, and every extracted frame was then fed to
COLMAP, depth estimation and fusion. Consecutive video frames are ~90%
redundant, so most of that compute bought nothing. Because this sits upstream
of the three most expensive stages, cutting the frame count cuts all of them by
roughly the same factor.

It is not purely an efficiency win: blurry frames actively corrupt COLMAP
poses, so dropping them tends to *improve* the reconstruction.

Two signals, combined:

* **Sharpness** — variance of the Laplacian. Cheap, and the standard proxy for
  motion blur. Used both as an absolute floor and to pick the best frame out of
  a set of otherwise-equivalent candidates.
* **Parallax** — sparse Lucas-Kanade optical flow against the last *kept*
  frame, as median displacement normalised by image width. A frame is only
  worth keeping once the view has actually moved; this is what collapses slow
  pans and hold-stills, which fixed-fps extraction cannot do.

Selection walks forward, so cost is O(frames) with one flow computation per
frame, all on downscaled greyscale.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger("keyframes")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Frames are analysed at this width. Sharpness and flow are both scale
# sensitive, so fixing the analysis width keeps thresholds meaningful across
# 720p/1080p/4K input instead of silently changing behaviour with resolution.
ANALYSIS_WIDTH = 640


@dataclass
class FrameStats:
    path: Path
    index: int
    sharpness: float
    # Normalised median parallax against the previously kept frame. None for
    # the first frame, which has nothing to compare against.
    motion: Optional[float] = None
    kept: bool = False
    reason: str = ""


@dataclass
class Selection:
    kept: List[Path] = field(default_factory=list)
    stats: List[FrameStats] = field(default_factory=list)
    n_extracted: int = 0
    blur_floor: float = 0.0

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def reduction(self) -> float:
        """Extracted / kept, e.g. 4.0 means a 4x cut."""
        return (self.n_extracted / self.n_kept) if self.n_kept else 0.0

    def summary(self) -> str:
        dropped = self.n_extracted - self.n_kept
        blur = sum(1 for s in self.stats if s.reason == "blurry")
        dup = sum(1 for s in self.stats if s.reason == "redundant")
        sharper = sum(1 for s in self.stats if s.reason == "sharper-neighbour")
        budget = sum(1 for s in self.stats if s.reason == "over-budget")
        return (
            f"kept {self.n_kept}/{self.n_extracted} frames ({self.reduction:.1f}x reduction); "
            f"dropped {dropped} — {dup} redundant, {sharper} outshone by a sharper "
            f"neighbour, {blur} below the blur floor, {budget} over budget"
        )

    def manifest(self) -> Dict[str, object]:
        return {
            "n_extracted": self.n_extracted,
            "n_kept": self.n_kept,
            "reduction": round(self.reduction, 3),
            "blur_floor": round(self.blur_floor, 4),
            "kept": [p.name for p in self.kept],
            "frames": [
                {
                    "name": s.path.name,
                    "index": s.index,
                    "sharpness": round(s.sharpness, 4),
                    "motion": None if s.motion is None else round(s.motion, 5),
                    "kept": s.kept,
                    "reason": s.reason,
                }
                for s in self.stats
            ],
        }


def list_images(frames_dir: Path | str) -> List[Path]:
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"frames directory does not exist: {frames_dir}")
    return sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_analysis_gray(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > ANALYSIS_WIDTH:
        scale = ANALYSIS_WIDTH / float(w)
        img = cv2.resize(img, (ANALYSIS_WIDTH, max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    return img


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Sharpness proxy: high for crisp edges, low for motion blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def median_parallax(prev_gray: np.ndarray, curr_gray: np.ndarray, max_corners: int = 200) -> float:
    """Median feature displacement between two frames, normalised by width.

    Returns 0.0 when the view has not moved and there is nothing to gain from
    the second frame. Falls back to a whole-frame difference when the scene is
    too textureless to track — a blank wall yields no corners, and returning 0
    there would stall selection forever.
    """
    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8
    )
    width = float(prev_gray.shape[1])

    if corners is None or len(corners) < 8:
        # Untrackable: use normalised mean absolute difference as a coarse
        # stand-in so a textureless pan still eventually registers as motion.
        diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()
        return float(diff / 255.0)

    # nextPts=None is the documented "estimate it for me" form; the stub types
    # it as required.
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        prev_gray, curr_gray, corners, None
    )
    if nxt is None or status is None:
        return 0.0
    ok = status.ravel() == 1
    if ok.sum() < 4:
        # Nearly everything lost tracking, which itself means a large change.
        return 1.0
    disp = np.linalg.norm(nxt[ok].reshape(-1, 2) - corners[ok].reshape(-1, 2), axis=1)
    return float(np.median(disp) / width)


def select_keyframes(
    frames_dir: Path | str,
    min_motion: float = 0.012,
    blur_percentile: float = 20.0,
    sharpness_window: int = 3,
    min_frames: int = 12,
    max_frames: Optional[int] = 300,
) -> Selection:
    """Choose a subset of `frames_dir` worth sending downstream.

    `min_motion` is normalised median parallax — 0.012 means "the view has
    shifted by ~1.2% of the image width since the last kept frame".

    `blur_percentile` sets an absolute sharpness floor from the clip's own
    distribution, so it adapts to footage that is uniformly soft rather than
    throwing away everything.

    `sharpness_window` is how many frames past the motion threshold may be
    considered together; the sharpest of them is kept. This is what stops a
    blurry frame being chosen purely because it happened to land on the
    threshold first.

    `min_frames` / `max_frames` bound the result: a long clip stays tractable,
    and a short one still yields enough views for COLMAP to solve.
    """
    frames = list_images(frames_dir)
    sel = Selection(n_extracted=len(frames))
    if not frames:
        return sel

    grays: List[Optional[np.ndarray]] = []
    for i, p in enumerate(frames):
        g = _load_analysis_gray(p)
        grays.append(g)
        sharp = variance_of_laplacian(g) if g is not None else 0.0
        sel.stats.append(FrameStats(path=p, index=i, sharpness=sharp))

    readable = [s for s, g in zip(sel.stats, grays) if g is not None]
    for s, g in zip(sel.stats, grays):
        if g is None:
            s.reason = "unreadable"
    if not readable:
        return sel

    # Absolute blur floor from this clip's own distribution.
    sharp_values = np.array([s.sharpness for s in readable], dtype=np.float64)
    sel.blur_floor = float(np.percentile(sharp_values, blur_percentile))

    # Always keep the first readable frame — it anchors the sequence.
    first = readable[0]
    first.kept = True
    first.reason = "first"
    sel.kept.append(first.path)
    last_kept_gray = grays[first.index]

    i = first.index + 1
    n = len(frames)
    while i < n:
        g = grays[i]
        if g is None:
            i += 1
            continue

        assert last_kept_gray is not None
        motion = median_parallax(last_kept_gray, g)
        sel.stats[i].motion = motion

        if motion < min_motion:
            sel.stats[i].reason = "redundant"
            i += 1
            continue

        # Past the threshold: consider a short window and take the sharpest,
        # so the choice is not hostage to whichever frame crossed first.
        window = [i]
        for j in range(i + 1, min(i + sharpness_window, n)):
            if grays[j] is not None:
                window.append(j)
        best = max(window, key=lambda k: sel.stats[k].sharpness)

        if sel.stats[best].sharpness < sel.blur_floor:
            # Every candidate here is blurrier than the clip's floor.
            for k in window:
                if not sel.stats[k].reason:
                    sel.stats[k].reason = "blurry"
            i = window[-1] + 1
            continue

        sel.stats[best].kept = True
        sel.stats[best].reason = "keyframe"
        sel.kept.append(sel.stats[best].path)
        last_kept_gray = grays[best]
        for k in window:
            if k != best and not sel.stats[k].reason:
                # Distinct from "redundant": these cleared the motion bar but
                # lost to a sharper neighbour. That is where most blurred
                # frames actually die — the "blurry" floor only fires when the
                # entire window is below it — so counting them separately keeps
                # the job log honest about why frames were dropped.
                sel.stats[k].reason = "sharper-neighbour"
        i = window[-1] + 1

    _apply_budget(sel, grays, min_frames, max_frames)
    sel.kept = [s.path for s in sel.stats if s.kept]
    return sel


def _apply_budget(
    sel: Selection,
    grays: Sequence[Optional[np.ndarray]],
    min_frames: int,
    max_frames: Optional[int],
) -> None:
    """Bound the selection at both ends.

    Trimming subsamples uniformly rather than cutting the tail, so coverage of
    the whole clip is preserved. Backfilling adds the sharpest of the dropped
    frames, since a too-small set fails COLMAP outright.
    """
    kept_idx = [s.index for s in sel.stats if s.kept]

    if max_frames is not None and len(kept_idx) > max_frames:
        keep_positions = set(np.linspace(0, len(kept_idx) - 1, max_frames).round().astype(int))
        for pos, idx in enumerate(kept_idx):
            if pos not in keep_positions:
                sel.stats[idx].kept = False
                sel.stats[idx].reason = "over-budget"
        kept_idx = [s.index for s in sel.stats if s.kept]

    available = [
        s for s in sel.stats
        if not s.kept and s.reason != "unreadable" and grays[s.index] is not None
    ]
    if len(kept_idx) < min_frames and available:
        need = min(min_frames - len(kept_idx), len(available))
        # Sharpest first: if we must take frames we previously rejected, take
        # the ones least likely to corrupt the solve.
        for s in sorted(available, key=lambda s: -s.sharpness)[:need]:
            s.kept = True
            s.reason = "budget-backfill"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_selection(sel: Selection, out_dir: Path | str) -> Path:
    """Materialise the kept frames into `out_dir` plus a `keyframes.json`.

    Hardlinks where the filesystem allows it, so selecting keyframes on a
    multi-GB frame directory costs no extra disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sel.kept:
        _link_or_copy(p, out_dir / p.name)
    manifest = out_dir / "keyframes.json"
    manifest.write_text(json.dumps(sel.manifest(), indent=2), encoding="utf-8")
    return out_dir


def select_and_write(
    frames_dir: Path | str,
    out_dir: Path | str,
    **kwargs,
) -> Selection:
    sel = select_keyframes(frames_dir, **kwargs)
    write_selection(sel, out_dir)
    logger.info("%s", sel.summary())
    return sel


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Select keyframes from an extracted frame directory (MPO-237)."
    )
    p.add_argument("frames_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--min-motion", type=float, default=0.012,
                   help="Normalised median parallax needed to keep a frame.")
    p.add_argument("--blur-percentile", type=float, default=20.0)
    p.add_argument("--sharpness-window", type=int, default=3)
    p.add_argument("--min-frames", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args(argv)

    sel = select_and_write(
        args.frames_dir, args.out_dir,
        min_motion=args.min_motion,
        blur_percentile=args.blur_percentile,
        sharpness_window=args.sharpness_window,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    print(sel.summary())


if __name__ == "__main__":
    main()
