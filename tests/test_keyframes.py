"""Keyframe selection (MPO-237).

Frames are synthesised so the right answer is known: a textured scene is
panned by a controlled number of pixels per frame, with specific frames
deliberately blurred or held still.
"""

import json

import cv2
import numpy as np
import pytest

import keyframes
from keyframes import (
    Selection,
    median_parallax,
    select_keyframes,
    variance_of_laplacian,
    write_selection,
)


def textured_scene(h=400, w=1400, seed=0):
    """High-frequency texture so corner detection has something to track."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    # Blocky structure on top of the noise, so features are locally distinctive.
    for _ in range(120):
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        cv2.rectangle(img, (x, y), (x + 50, y + 50),
                      tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
    return img


def write_pan(tmp_path, n=20, step=12, blur_at=(), hold_from=None, crop_w=640, seed=0):
    """Emit `n` frames panning across a scene `step` px at a time.

    blur_at:   indices to Gaussian-blur (simulating motion blur)
    hold_from: index after which the camera stops moving (duplicate frames)
    """
    scene = textured_scene(seed=seed, w=crop_w + n * step + 50)
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    for i in range(n):
        off = i * step if (hold_from is None or i < hold_from) else hold_from * step
        crop = scene[:, off:off + crop_w].copy()
        if i in blur_at:
            crop = cv2.GaussianBlur(crop, (21, 21), 8)
        cv2.imwrite(str(d / f"frame_{i:04d}.png"), crop)
    return d


# --- primitives -----------------------------------------------------------

def test_variance_of_laplacian_ranks_blur():
    sharp = cv2.cvtColor(textured_scene(200, 200), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 8)
    assert variance_of_laplacian(sharp) > variance_of_laplacian(blurred) * 5


def test_parallax_is_zero_for_identical_frames():
    g = cv2.cvtColor(textured_scene(200, 400), cv2.COLOR_BGR2GRAY)
    assert median_parallax(g, g.copy()) == pytest.approx(0.0, abs=1e-6)


def test_parallax_grows_with_displacement():
    scene = cv2.cvtColor(textured_scene(200, 900), cv2.COLOR_BGR2GRAY)
    a = scene[:, 0:600]
    small = median_parallax(a, scene[:, 10:610])
    large = median_parallax(a, scene[:, 60:660])
    assert 0 < small < large
    # Normalised by width: a 60px shift on a 600px-wide frame is ~0.1.
    assert large == pytest.approx(0.1, abs=0.03)


def test_parallax_falls_back_when_untrackable():
    """A blank wall yields no corners; returning 0 would stall selection."""
    flat = np.full((200, 400), 128, dtype=np.uint8)
    other = np.full((200, 400), 200, dtype=np.uint8)
    assert median_parallax(flat, other) > 0


# --- selection ------------------------------------------------------------

def test_drops_redundant_frames_during_a_hold(tmp_path):
    """Frames 10..19 are identical — a hold-still. They must not all survive."""
    d = write_pan(tmp_path, n=20, step=14, hold_from=10)
    sel = select_keyframes(d, min_frames=0, max_frames=None)

    kept_idx = [s.index for s in sel.stats if s.kept]
    held = [i for i in kept_idx if i > 10]
    assert len(held) <= 1, f"kept {len(held)} frames from a static hold: {kept_idx}"
    assert sel.n_kept < 20


def test_blurry_frames_are_not_preferred(tmp_path):
    blur_at = {3, 4, 9, 10}
    d = write_pan(tmp_path, n=18, step=14, blur_at=blur_at)
    sel = select_keyframes(d, min_frames=0, max_frames=None)

    kept = {s.index for s in sel.stats if s.kept}
    assert not (kept & blur_at), f"kept blurred frames {sorted(kept & blur_at)}"


def test_blur_rejection_is_attributed_not_hidden(tmp_path):
    """Most blurred frames die by losing to a sharper neighbour, not by the
    blur floor. The reason has to say so, or the job log implies blur
    rejection never fired."""
    d = write_pan(tmp_path, n=18, step=14, blur_at={3, 4, 9, 10})
    sel = select_keyframes(d, min_frames=0, max_frames=None)

    reasons = {s.index: s.reason for s in sel.stats if not s.kept}
    assert all(reasons[i] in {"sharper-neighbour", "redundant", "blurry"} for i in (3, 4, 9, 10))
    assert any(r == "sharper-neighbour" for r in reasons.values())
    assert "outshone by a sharper neighbour" in sel.summary()


def test_reduction_is_reported(tmp_path):
    d = write_pan(tmp_path, n=24, step=6)
    sel = select_keyframes(d, min_frames=0, max_frames=None)
    assert sel.n_extracted == 24
    assert 0 < sel.n_kept < 24
    assert sel.reduction > 1.0
    assert "reduction" in sel.summary() and f"{sel.n_kept}/24" in sel.summary()


def test_higher_min_motion_keeps_fewer(tmp_path):
    d = write_pan(tmp_path, n=24, step=8)
    loose = select_keyframes(d, min_motion=0.005, min_frames=0, max_frames=None)
    tight = select_keyframes(d, min_motion=0.05, min_frames=0, max_frames=None)
    assert tight.n_kept < loose.n_kept


def test_max_frames_budget_is_enforced_and_spans_the_clip(tmp_path):
    d = write_pan(tmp_path, n=30, step=20)
    sel = select_keyframes(d, min_motion=0.0, min_frames=0, max_frames=5)
    kept = [s.index for s in sel.stats if s.kept]
    assert len(kept) == 5
    # Uniform subsample, not a truncation: the tail of the clip is represented.
    assert kept[-1] > 20, f"budget trimming dropped the end of the clip: {kept}"
    assert any(s.reason == "over-budget" for s in sel.stats)


def test_min_frames_backfills(tmp_path):
    """A static clip would otherwise yield 1 frame, which COLMAP cannot solve."""
    d = write_pan(tmp_path, n=15, step=0)
    sel = select_keyframes(d, min_frames=8, max_frames=None)
    assert sel.n_kept >= 8
    assert any(s.reason == "budget-backfill" for s in sel.stats)


def test_first_frame_always_anchors(tmp_path):
    d = write_pan(tmp_path, n=10, step=15)
    sel = select_keyframes(d, min_frames=0, max_frames=None)
    assert sel.stats[0].kept and sel.stats[0].reason == "first"


def test_empty_directory(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    sel = select_keyframes(d)
    assert sel.n_kept == 0 and sel.n_extracted == 0 and sel.reduction == 0.0


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        select_keyframes(tmp_path / "nope")


def test_unreadable_files_are_skipped_not_fatal(tmp_path):
    d = write_pan(tmp_path, n=8, step=15)
    (d / "frame_9999.png").write_bytes(b"not an image")
    sel = select_keyframes(d, min_frames=0, max_frames=None)
    assert any(s.reason == "unreadable" for s in sel.stats)
    assert sel.n_kept > 0


# --- output ---------------------------------------------------------------

def test_write_selection_materialises_frames_and_manifest(tmp_path):
    d = write_pan(tmp_path, n=16, step=14)
    sel = select_keyframes(d, min_frames=0, max_frames=None)
    out = write_selection(sel, tmp_path / "keyframes")

    images = sorted(p.name for p in out.glob("*.png"))
    assert images == sorted(p.name for p in sel.kept)

    manifest = json.loads((out / "keyframes.json").read_text())
    assert manifest["n_extracted"] == 16
    assert manifest["n_kept"] == sel.n_kept
    assert manifest["kept"] == [p.name for p in sel.kept]
    assert len(manifest["frames"]) == 16
    # Every frame is accounted for with a reason.
    assert all(f["reason"] for f in manifest["frames"])


def test_write_selection_does_not_duplicate_bytes(tmp_path):
    """Hardlinks where possible — selecting on a multi-GB frame dir is free."""
    d = write_pan(tmp_path, n=6, step=20)
    sel = select_keyframes(d, min_frames=0, max_frames=None)
    out = write_selection(sel, tmp_path / "keyframes")

    src = sel.kept[0]
    dst = out / src.name
    assert dst.stat().st_size == src.stat().st_size
    if hasattr(dst.stat(), "st_ino") and dst.stat().st_ino != 0:
        assert dst.stat().st_ino == src.stat().st_ino or dst.read_bytes() == src.read_bytes()


def test_select_and_write_is_idempotent(tmp_path):
    d = write_pan(tmp_path, n=10, step=15)
    a = keyframes.select_and_write(d, tmp_path / "kf", min_frames=0, max_frames=None)
    b = keyframes.select_and_write(d, tmp_path / "kf", min_frames=0, max_frames=None)
    assert [p.name for p in a.kept] == [p.name for p in b.kept]


def test_summary_of_empty_selection_does_not_divide_by_zero():
    assert "0/0" in Selection(n_extracted=0).summary()
