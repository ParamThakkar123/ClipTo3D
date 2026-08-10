"""Pinned checkpoint resolution and hash verification (MPO-233).

No network: every test either works against a tmp cache seeded with known
bytes, or stubs the download. The registry's real digests are checked for
shape only — verifying them for real would mean pulling 1.8GB.
"""

import hashlib

import pytest

import checkpoints
from checkpoints import Checkpoint, ChecksumMismatch


def make_ckpt(payload: bytes, name="test-ckpt", filename="test.pth") -> Checkpoint:
    return Checkpoint(
        name=name,
        filename=filename,
        url="https://example.invalid/test.pth",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        revision="deadbeef",
        description="fixture",
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoints.CACHE_ENV_VAR, str(tmp_path))
    monkeypatch.delenv(checkpoints.OFFLINE_ENV_VAR, raising=False)
    monkeypatch.delenv(checkpoints.TRUST_CACHE_ENV_VAR, raising=False)
    return tmp_path


# --- registry -------------------------------------------------------------

def test_registry_entries_are_pinned_to_immutable_urls():
    """A branch name in the URL would reintroduce the drift this issue is about."""
    for ckpt in checkpoints.REGISTRY.values():
        assert len(ckpt.sha256) == 64, ckpt.name
        assert int(ckpt.sha256, 16) >= 0, ckpt.name
        assert ckpt.size_bytes > 0, ckpt.name
        assert ckpt.revision in ckpt.url, ckpt.name
        assert "/resolve/main/" not in ckpt.url, ckpt.name


def test_every_encoder_tier_has_a_checkpoint():
    from depth_estimation import depth  # imported lazily: needs torch

    assert set(checkpoints.ENCODER_TO_CHECKPOINT) == set(depth.DA_V2_ENCODERS)
    for name in checkpoints.ENCODER_TO_CHECKPOINT.values():
        assert name in checkpoints.REGISTRY


def test_unknown_names_report_what_is_available():
    with pytest.raises(KeyError, match="depth-anything-v2-vitl"):
        checkpoints.get("nope")
    with pytest.raises(KeyError, match="vits"):
        checkpoints.for_encoder("vitxl")


def test_metadata_records_provenance():
    meta = checkpoints.get("depth-anything-v2-vits").metadata()
    assert set(meta) == {"name", "sha256", "size_bytes", "url", "revision"}


# --- cache dir ------------------------------------------------------------

def test_cache_dir_honours_env(cache):
    assert checkpoints.cache_dir() == cache


def test_cache_dir_default(monkeypatch):
    monkeypatch.delenv(checkpoints.CACHE_ENV_VAR, raising=False)
    assert checkpoints.cache_dir() == checkpoints.DEFAULT_CACHE


# --- verification ---------------------------------------------------------

def test_ensure_returns_verified_file(cache, monkeypatch):
    payload = b"weights" * 100
    ckpt = make_ckpt(payload)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)
    (cache / ckpt.filename).write_bytes(payload)

    assert checkpoints.ensure(ckpt.name) == cache / ckpt.filename


def test_ensure_rejects_corrupted_file(cache, monkeypatch):
    """Same length, different bytes — the size check alone would pass this."""
    payload = b"weights" * 100
    ckpt = make_ckpt(payload)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)
    corrupt = bytearray(payload)
    corrupt[0] ^= 0xFF
    (cache / ckpt.filename).write_bytes(bytes(corrupt))

    with pytest.raises(ChecksumMismatch, match="sha256"):
        checkpoints.ensure(ckpt.name)


def test_ensure_rejects_truncated_file(cache, monkeypatch):
    payload = b"weights" * 100
    ckpt = make_ckpt(payload)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)
    (cache / ckpt.filename).write_bytes(payload[:-10])

    with pytest.raises(ChecksumMismatch, match="bytes"):
        checkpoints.ensure(ckpt.name)


def test_trust_cache_skips_the_digest_but_not_the_size(cache, monkeypatch):
    payload = b"weights" * 100
    ckpt = make_ckpt(payload)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)
    monkeypatch.setenv(checkpoints.TRUST_CACHE_ENV_VAR, "1")

    corrupt = bytearray(payload)
    corrupt[0] ^= 0xFF
    (cache / ckpt.filename).write_bytes(bytes(corrupt))
    assert checkpoints.ensure(ckpt.name) == cache / ckpt.filename  # digest not checked

    (cache / ckpt.filename).write_bytes(payload[:-1])
    with pytest.raises(ChecksumMismatch, match="bytes"):
        checkpoints.ensure(ckpt.name)


# --- offline / missing ----------------------------------------------------

def test_missing_checkpoint_names_the_fetch_command(cache, monkeypatch):
    ckpt = make_ckpt(b"x" * 10)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)

    with pytest.raises(FileNotFoundError, match="checkpoints.py --fetch"):
        checkpoints.ensure(ckpt.name, allow_download=False)


def test_offline_env_blocks_download(cache, monkeypatch):
    ckpt = make_ckpt(b"x" * 10)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)
    monkeypatch.setenv(checkpoints.OFFLINE_ENV_VAR, "1")

    with pytest.raises(FileNotFoundError):
        checkpoints.ensure(ckpt.name)


def test_offline_flag_parsing(monkeypatch):
    for value, expected in [("1", True), ("true", True), ("YES", True),
                            ("0", False), ("", False), ("no", False)]:
        monkeypatch.setenv(checkpoints.OFFLINE_ENV_VAR, value)
        assert checkpoints.offline() is expected, value


# --- download atomicity ---------------------------------------------------

def test_download_is_atomic_on_checksum_failure(cache, monkeypatch):
    """A bad download must not leave anything that a later run would trust.

    Downloads retry now, so a persistently corrupt source surfaces as a
    RuntimeError naming the attempts rather than the raw ChecksumMismatch.
    """
    ckpt = make_ckpt(b"expected" * 10)
    monkeypatch.setitem(checkpoints.REGISTRY, ckpt.name, ckpt)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)

    class FakeResponse:
        def read(self, *a):
            return b""

        def __enter__(self):
            import io

            return io.BytesIO(b"corrupted payload")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="attempts"):
        checkpoints.download(ckpt, cache, retries=2)

    assert not (cache / ckpt.filename).exists()
    assert not (cache / f"{ckpt.filename}.part").exists()


def test_download_writes_and_verifies(cache, monkeypatch):
    payload = b"expected" * 10
    ckpt = make_ckpt(payload)

    class FakeResponse:
        def __enter__(self):
            import io

            return io.BytesIO(payload)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    path = checkpoints.download(ckpt, cache)
    assert path.read_bytes() == payload
    assert not (cache / f"{ckpt.filename}.part").exists()


def test_download_cleans_up_on_network_error(cache, monkeypatch):
    ckpt = make_ckpt(b"expected" * 10)

    def boom(*a, **k):
        raise checkpoints.urllib.error.URLError("no route to host")

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", boom)

    with pytest.raises(RuntimeError, match="failed to download"):
        checkpoints.download(ckpt, cache)
    assert not (cache / f"{ckpt.filename}.part").exists()


# --- retry on truncated downloads ----------------------------------------
#
# Found by the worker image build: 16MB of a 99MB checkpoint arrived, the size
# check correctly rejected it, and the whole build failed. These files are
# 99MB-1.3GB, so a mid-transfer cut is routine rather than exceptional.

def test_truncated_download_is_retried(cache, monkeypatch):
    payload = b"weights" * 1000
    ckpt = make_ckpt(payload)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)

    attempts = {"n": 0}

    class Resp:
        def __enter__(self):
            import io

            attempts["n"] += 1
            # Truncated twice, then complete.
            return io.BytesIO(payload if attempts["n"] >= 3 else payload[:100])

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *a, **k: Resp())

    path = checkpoints.download(ckpt, cache, retries=3)
    assert attempts["n"] == 3
    assert path.read_bytes() == payload
    assert not (cache / f"{ckpt.filename}.part").exists()


def test_retries_are_bounded_and_report_the_cause(cache, monkeypatch):
    ckpt = make_ckpt(b"expected" * 100)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)

    class Resp:
        def __enter__(self):
            import io

            return io.BytesIO(b"truncated")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *a, **k: Resp())

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        checkpoints.download(ckpt, cache, retries=3)
    assert not (cache / f"{ckpt.filename}.part").exists()
    assert not (cache / ckpt.filename).exists()


def test_transport_errors_are_retried_too(cache, monkeypatch):
    payload = b"weights" * 100
    ckpt = make_ckpt(payload)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def flaky(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise checkpoints.urllib.error.URLError("connection reset")

        class Resp:
            def __enter__(self):
                import io

                return io.BytesIO(payload)

            def __exit__(self, *a):
                return False

        return Resp()

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", flaky)
    assert checkpoints.download(ckpt, cache, retries=3).read_bytes() == payload


def test_a_stale_part_file_does_not_corrupt_the_retry(cache, monkeypatch):
    """Appending to a leftover .part would produce a plausible-sized garbage file."""
    payload = b"weights" * 100
    ckpt = make_ckpt(payload)
    (cache / f"{ckpt.filename}.part").write_bytes(b"junk from a previous run")

    class Resp:
        def __enter__(self):
            import io

            return io.BytesIO(payload)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(checkpoints.urllib.request, "urlopen", lambda *a, **k: Resp())
    assert checkpoints.download(ckpt, cache).read_bytes() == payload
