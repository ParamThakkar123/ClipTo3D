"""Depth-map storage formats (MPO-240).

Depth was full-resolution uncompressed float32 — ~8MB per 1080p frame, ~2.4GB
per 300-frame job, plus a duplicate 8-bit PNG. These tests pin the size
reduction and, more importantly, that every format still round-trips through
one loader so consumers never need to know which was used.
"""

import numpy as np
import pytest

from depth_io import (
    DEFAULT_FORMAT,
    FORMATS,
    depth_bytes,
    find_depth,
    load_depth,
    save_depth,
)


def disparity_map(h=270, w=480, seed=0):
    """Smooth relative-inverse-depth, like a real model emits."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = 30.0 / (1.0 + (xx / w) * 3.0) + 5.0 * np.sin(yy / 40.0)
    rng = np.random.default_rng(seed)
    return (d + rng.normal(0, 0.05, d.shape)).astype(np.float32)


def test_default_is_fp16():
    assert DEFAULT_FORMAT == "fp16"


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_through_one_loader(tmp_path, fmt):
    d = disparity_map()
    path = save_depth(tmp_path, "frame_0001", d, fmt=fmt)
    back = load_depth(path)

    assert back.shape == d.shape
    assert back.dtype == np.float32
    span = float(d.max() - d.min())
    # png16 quantises to 1/65535 of range; fp16 to ~3 decimal digits.
    tol = span * 1e-3 if fmt == "png16" else span * 1e-2
    assert np.abs(back - d).max() < tol, f"{fmt} round-trip drifted too far"


@pytest.mark.parametrize("fmt", FORMATS)
def test_find_depth_locates_every_format(tmp_path, fmt):
    save_depth(tmp_path, "frame_0007", disparity_map(), fmt=fmt)
    found = find_depth(tmp_path, "frame_0007")
    assert found is not None
    assert load_depth(found).shape == (270, 480)


def test_find_depth_returns_none_when_absent(tmp_path):
    assert find_depth(tmp_path, "nope") is None


def test_storage_sizes_shrink_as_advertised(tmp_path):
    """The actual point of the issue."""
    d = disparity_map(1080, 1920)
    sizes = {}
    for fmt in FORMATS:
        sub = tmp_path / fmt
        save_depth(sub, "f", d, fmt=fmt)
        sizes[fmt], _ = depth_bytes(sub)

    assert sizes["fp16"] < sizes["fp32"] * 0.55, sizes      # ~half
    assert sizes["png16"] < sizes["fp32"] * 0.5, sizes      # substantially smaller
    # A 1080p fp32 map is ~8MB; that is the number the issue is about.
    assert sizes["fp32"] > 7_000_000


def test_png16_preserves_non_finite_as_nan(tmp_path):
    """Quantising NaN to 0 would read back as a real near-plane measurement."""
    d = disparity_map(64, 64)
    d[10:20, 10:20] = np.nan
    path = save_depth(tmp_path, "f", d, fmt="png16")
    back = load_depth(path)
    # The masked region must not come back as a plausible depth value.
    assert not np.allclose(back[10:20, 10:20], d[~np.isnan(d)].mean(), atol=1.0)


def test_png16_sidecar_is_required_to_decode(tmp_path):
    path = save_depth(tmp_path, "f", disparity_map(32, 32), fmt="png16")
    path.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_depth(path)


def test_png16_scale_is_per_frame(tmp_path):
    """A dim frame must not lose its dynamic range to a bright one."""
    bright = disparity_map(64, 64) * 100.0
    dim = disparity_map(64, 64, seed=1) * 0.01
    for name, arr in (("bright", bright), ("dim", dim)):
        p = save_depth(tmp_path, name, arr, fmt="png16")
        back = load_depth(p)
        rel = np.abs(back - arr).max() / float(arr.max() - arr.min())
        assert rel < 1e-3, f"{name} lost range: {rel}"


def test_constant_depth_does_not_divide_by_zero(tmp_path):
    flat = np.full((32, 32), 3.5, dtype=np.float32)
    back = load_depth(save_depth(tmp_path, "f", flat, fmt="png16"))
    np.testing.assert_allclose(back, flat, atol=1e-4)


def test_all_nan_map_is_survivable(tmp_path):
    allnan = np.full((16, 16), np.nan, dtype=np.float32)
    back = load_depth(save_depth(tmp_path, "f", allnan, fmt="png16"))
    assert back.shape == (16, 16)


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown depth format"):
        save_depth(tmp_path, "f", disparity_map(8, 8), fmt="jpeg2000")


def test_previews_are_not_counted_as_depth(tmp_path):
    """The preview PNG is a debug artifact and must not be mistaken for depth."""
    save_depth(tmp_path, "frame_1", disparity_map(16, 16), fmt="fp16")
    (tmp_path / "frame_1_preview.png").write_bytes(b"not depth")
    assert find_depth(tmp_path, "frame_1") == tmp_path / "frame_1_depth.npy"
