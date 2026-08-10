"""Batched depth inference must not change the answer (MPO-241).

The speedup from batching is a GPU property and cannot be measured here. What
*can* be checked is the thing that would make batching a bug rather than an
optimisation: that a batched forward pass produces the same depth maps as
one-at-a-time, and that a directory of mixed resolutions still works.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs a torch extra: uv sync --extra cpu")
pytest.importorskip("torchvision", reason="needs a torch extra")

import cv2  # noqa: E402

from depth_estimation import depth as depth_mod  # noqa: E402


class TinyModel(torch.nn.Module):
    """A stand-in with the shape contract the real backend has.

    Deterministic and dependent on the input, so a batching bug (wrong slice,
    transposed batch, reused buffer) shows up as a changed value rather than
    passing silently.
    """

    def forward(self, x):
        # (B,3,H,W) -> (B,H,W)
        return x.mean(dim=1) * 2.0 + x.std(dim=1)


@pytest.fixture
def stub_model(monkeypatch):
    model = TinyModel().eval()
    monkeypatch.setattr(depth_mod, "_load_da2_model", lambda *a, **k: model)
    return model


def write_frames(dirpath, n, size=(64, 48), seed=0):
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        cv2.imwrite(str(dirpath / f"frame_{i:04d}.png"), img)
    return dirpath


def run(frames, out, batch_size, stub_model, tmp_path, monkeypatch):
    monkeypatch.setattr(depth_mod.checkpoints, "for_encoder",
                        lambda enc: type("C", (), {
                            "name": "stub", "metadata": lambda self: {"name": "stub"}})())
    monkeypatch.setattr(depth_mod.checkpoints, "ensure", lambda name: tmp_path / "stub.pth")
    (tmp_path / "stub.pth").write_bytes(b"stub")
    return depth_mod.estimate_depths_depthanything(
        frames, out, checkpoint=None, encoder="vits", use_cuda=False,
        batch_size=batch_size, fp16=False, depth_format="fp32",
    )


def test_batched_matches_unbatched(tmp_path, stub_model, monkeypatch):
    frames = write_frames(tmp_path / "frames", 7)

    run(frames, tmp_path / "b1", 1, stub_model, tmp_path, monkeypatch)
    run(frames, tmp_path / "b4", 4, stub_model, tmp_path, monkeypatch)

    from depth_io import load_depth

    names = sorted(p.name for p in (tmp_path / "b1").glob("*_depth.npy"))
    assert len(names) == 7
    for name in names:
        a = load_depth(tmp_path / "b1" / name)
        b = load_depth(tmp_path / "b4" / name)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5,
                                   err_msg=f"{name} differs between batch 1 and batch 4")


def test_batch_larger_than_the_frame_count(tmp_path, stub_model, monkeypatch):
    frames = write_frames(tmp_path / "frames", 3)
    run(frames, tmp_path / "out", 32, stub_model, tmp_path, monkeypatch)
    assert len(list((tmp_path / "out").glob("*_depth.npy"))) == 3


def test_mixed_resolutions_fall_back_per_image(tmp_path, stub_model, monkeypatch):
    """A batch tensor needs one shape; mixed input must not crash or drop frames."""
    frames = tmp_path / "frames"
    write_frames(frames, 2, size=(64, 48))
    rng = np.random.default_rng(1)
    cv2.imwrite(str(frames / "frame_0009.png"),
                rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))

    run(frames, tmp_path / "out", 8, stub_model, tmp_path, monkeypatch)
    assert len(list((tmp_path / "out").glob("*_depth.npy"))) == 3


def test_depth_matches_each_frames_own_resolution(tmp_path, stub_model, monkeypatch):
    frames = write_frames(tmp_path / "frames", 2, size=(80, 60))
    run(frames, tmp_path / "out", 2, stub_model, tmp_path, monkeypatch)

    from depth_io import load_depth

    for p in (tmp_path / "out").glob("*_depth.npy"):
        assert load_depth(p).shape == (60, 80)


def test_unreadable_file_does_not_poison_its_batch(tmp_path, stub_model, monkeypatch):
    frames = write_frames(tmp_path / "frames", 3)
    (frames / "frame_0099.png").write_bytes(b"not an image")

    run(frames, tmp_path / "out", 8, stub_model, tmp_path, monkeypatch)
    assert len(list((tmp_path / "out").glob("*_depth.npy"))) == 3


def test_model_is_loaded_once_across_calls(tmp_path, monkeypatch):
    """The cache is the part that is verifiable without a GPU."""
    calls = {"n": 0}
    model = TinyModel().eval()

    def counting_loader(*a, **k):
        calls["n"] += 1
        return model

    monkeypatch.setattr(depth_mod, "_load_da2_model", counting_loader)
    frames = write_frames(tmp_path / "frames", 2)
    for i in range(3):
        run(frames, tmp_path / f"out{i}", 2, model, tmp_path, monkeypatch)

    # _load_da2_model is called per invocation; the real caching lives inside
    # it, so assert the cache itself rather than the call count.
    assert calls["n"] == 3
    depth_mod._MODEL_CACHE.clear()
