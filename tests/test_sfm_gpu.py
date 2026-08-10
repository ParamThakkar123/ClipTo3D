"""GPU SIFT detection and CPU fallback for the COLMAP stage (MPO-232).

`use_gpu` used to default to True and was never checked, so a headless host
went straight into a raw subprocess failure. These tests cover the three paths
that replaced it: detect, fall back when the guess was wrong, and fail loudly
when the GPU was explicitly required.

No COLMAP binary is involved — `_run_cmd` is stubbed, so this runs anywhere.
"""

import subprocess
import sys

import pytest

from structure_from_motion import sfm


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


TWO_GPUS = "GPU 0: NVIDIA A10G (UUID: GPU-aaa)\nGPU 1: NVIDIA A10G (UUID: GPU-bbb)\n"


# --- detect_gpu_sift ------------------------------------------------------

def test_detect_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(sfm.shutil, "which", lambda _: None)
    ok, why = sfm.detect_gpu_sift()
    assert ok is False
    assert "nvidia-smi" in why


def test_detect_lists_gpus(monkeypatch):
    monkeypatch.setattr(sfm.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sfm.subprocess, "run", lambda *a, **k: FakeCompleted(0, TWO_GPUS))
    ok, why = sfm.detect_gpu_sift()
    assert ok is True
    assert "2 NVIDIA GPU" in why


def test_detect_driver_present_but_no_device(monkeypatch):
    """The container-without-`--gpus` case: the binary exists, the device does not."""
    monkeypatch.setattr(sfm.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sfm.subprocess, "run", lambda *a, **k: FakeCompleted(0, "\n"))
    ok, why = sfm.detect_gpu_sift()
    assert ok is False
    assert "no GPUs" in why


def test_detect_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(sfm.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        sfm.subprocess, "run",
        lambda *a, **k: FakeCompleted(9, "", "Failed to initialize NVML: Driver/library mismatch"),
    )
    ok, why = sfm.detect_gpu_sift()
    assert ok is False
    assert "NVML" in why


def test_detect_survives_timeout(monkeypatch):
    """A hung nvidia-smi must not take the whole job down."""
    monkeypatch.setattr(sfm.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=30)

    monkeypatch.setattr(sfm.subprocess, "run", boom)
    ok, why = sfm.detect_gpu_sift()
    assert ok is False
    assert "TimeoutExpired" in why


# --- _run_sift_stage ------------------------------------------------------

def _cmd(gpu: bool):
    return ["colmap", "feature_extractor", "--use_gpu", "1" if gpu else "0"]


def test_stage_gpu_success_stays_on_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(sfm, "_run_cmd", lambda cmd, cancel=None: calls.append(cmd) or [])
    assert sfm._run_sift_stage("feature_extractor", _cmd, True, False) is True
    assert calls == [_cmd(True)]


def test_stage_autodetected_gpu_falls_back_to_cpu(monkeypatch):
    """Detection can only see the driver, not COLMAP's build. Failure must recover."""
    calls = []

    def fake_run(cmd, cancel=None):
        calls.append(cmd)
        if "1" in cmd:
            raise RuntimeError("Command failed (exit 1): ... no CUDA-capable device")
        return []

    monkeypatch.setattr(sfm, "_run_cmd", fake_run)
    assert sfm._run_sift_stage("feature_extractor", _cmd, True, False) is False
    assert calls == [_cmd(True), _cmd(False)]


def test_stage_explicit_gpu_failure_raises_with_hint(monkeypatch):
    def fake_run(cmd, cancel=None):
        raise RuntimeError("Command failed (exit 1): ... no CUDA-capable device")

    monkeypatch.setattr(sfm, "_run_cmd", fake_run)
    with pytest.raises(RuntimeError) as excinfo:
        sfm._run_sift_stage("feature_extractor", _cmd, True, True)
    msg = str(excinfo.value)
    assert "CUDA_ENABLED" in msg and "--no-colmap-gpu" in msg
    # The underlying subprocess error is preserved, not swallowed by the hint.
    assert "no CUDA-capable device" in str(excinfo.value.__cause__)


def test_stage_cpu_failure_is_not_retried(monkeypatch):
    calls = []

    def fake_run(cmd, cancel=None):
        calls.append(cmd)
        raise RuntimeError("Command failed (exit 1): COLMAP is unhappy")

    monkeypatch.setattr(sfm, "_run_cmd", fake_run)
    with pytest.raises(RuntimeError, match="COLMAP is unhappy"):
        sfm._run_sift_stage("feature_extractor", _cmd, False, False)
    assert len(calls) == 1


# --- option-name resolution across COLMAP versions ------------------------

COLMAP_313_HELP = """
  --ImageReader.single_camera arg (=0)
  --FeatureExtraction.num_threads arg (=-1)
  --FeatureExtraction.use_gpu arg (=1)
  --SiftExtraction.max_num_features arg (=8192)
"""

COLMAP_39_HELP = """
  --ImageReader.single_camera arg (=0)
  --SiftExtraction.num_threads arg (=-1)
  --SiftExtraction.use_gpu arg (=1)
  --SiftExtraction.max_num_features arg (=8192)
"""


def _fake_help(text):
    return lambda *a, **k: FakeCompleted(0, text)


def test_supported_options_parses_dotted_names(monkeypatch):
    sfm.supported_options.cache_clear()
    monkeypatch.setattr(sfm.subprocess, "run", _fake_help(COLMAP_313_HELP))
    opts = sfm.supported_options("colmap", "feature_extractor")
    assert "FeatureExtraction.use_gpu" in opts
    assert "SiftExtraction.max_num_features" in opts
    assert "SiftExtraction.use_gpu" not in opts


def test_pick_option_prefers_the_modern_name():
    modern = frozenset({"FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"})
    assert sfm.pick_option(modern, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu") == \
        "FeatureExtraction.use_gpu"


def test_pick_option_falls_back_to_the_legacy_name():
    """COLMAP 3.9 and older only know SiftExtraction.*."""
    sfm.supported_options.cache_clear()
    legacy = frozenset({"SiftExtraction.use_gpu"})
    assert sfm.pick_option(legacy, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu") == \
        "SiftExtraction.use_gpu"


def test_pick_option_reports_through_colmap_when_nothing_matches():
    """Unknown binary: emit the first candidate so COLMAP prints its own error
    rather than us silently dropping the flag."""
    assert sfm.pick_option(frozenset(), "A.b", "C.d") == "A.b"


def test_supported_options_survives_a_missing_binary(monkeypatch):
    sfm.supported_options.cache_clear()

    def boom(*a, **k):
        raise OSError("no such file")

    monkeypatch.setattr(sfm.subprocess, "run", boom)
    assert sfm.supported_options("nope", "feature_extractor") == frozenset()


# --- _run_cmd error detail ------------------------------------------------

def test_run_cmd_error_includes_output_tail():
    """COLMAP reports the real cause on stderr then exits non-zero silently."""
    script = "import sys; print('ERROR: no CUDA-capable device is detected'); sys.exit(1)"
    with pytest.raises(RuntimeError) as excinfo:
        sfm._run_cmd([sys.executable, "-c", script])
    assert "no CUDA-capable device" in str(excinfo.value)
    assert "exit 1" in str(excinfo.value)


def test_run_cmd_returns_output_on_success():
    lines = sfm._run_cmd([sys.executable, "-c", "print('hello'); print('world')"])
    assert lines == ["hello", "world"]


# --- cooperative cancellation (MPO-243) -----------------------------------

def test_run_cmd_terminates_a_running_process_on_cancel():
    """The blocking stdout read used to make a runaway job uninterruptible."""
    import time

    # Emits forever; only cancellation can end it.
    script = "import sys, time\nwhile True:\n    print('working'); sys.stdout.flush(); time.sleep(0.05)"
    calls = {"n": 0}

    def cancel_after_a_few_lines():
        calls["n"] += 1
        return calls["n"] >= 3

    started = time.time()
    with pytest.raises(sfm.ColmapCancelled, match="cancelled during"):
        sfm._run_cmd([sys.executable, "-u", "-c", script], cancel_after_a_few_lines)
    # Must stop promptly rather than running to completion (it never completes).
    assert time.time() - started < 30


def test_run_cmd_without_a_canceller_is_unaffected():
    lines = sfm._run_cmd([sys.executable, "-c", "print('one')"], None)
    assert lines == ["one"]


def test_cancellation_propagates_through_the_sift_stage(monkeypatch):
    """A cancel must not be mistaken for a GPU failure and retried on CPU."""
    calls = []

    def fake_run(cmd, cancel=None):
        calls.append(cmd)
        raise sfm.ColmapCancelled("cancelled during: colmap feature_extractor")

    monkeypatch.setattr(sfm, "_run_cmd", fake_run)
    with pytest.raises(sfm.ColmapCancelled):
        sfm._run_sift_stage("feature_extractor", _cmd, True, False, lambda: True)
    assert len(calls) == 1, "cancellation was retried as if it were a GPU failure"
