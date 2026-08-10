"""Local reconstruction capability and mode selection (MPO-249).

The desktop shell has to answer one question before it does anything: can this
machine reconstruct, or does it need the hosted service? Getting that wrong in
the optimistic direction fails *during* a long reconstruction, which is much
worse than declining up front — so the tests here lean on the negative cases.

Everything is probed through injected fakes rather than the real machine, so
the answers do not depend on whatever happens to be installed on the runner.
"""

import subprocess

import pytest

import local_runtime as lr
from local_runtime import MODE_LOCAL, MODE_LOCAL_CPU, MODE_SERVICE, Capability, Tool


def tool(name, ok=True, version="1.0"):
    return Tool(name, path=f"/usr/bin/{name}" if ok else None,
                version=version if ok else "", problem="" if ok else "not on PATH")


def capability(ffmpeg=True, colmap=True, gpus=("RTX 4090",)):
    return Capability(
        tools={"ffmpeg": tool("ffmpeg", ffmpeg), "colmap": tool("colmap", colmap)},
        gpus=list(gpus),
        gpu_detail="" if gpus else "nvidia-smi not on PATH (no NVIDIA driver)",
    )


# --- mode selection -------------------------------------------------------

def test_a_gpu_machine_with_both_tools_runs_locally():
    assert capability().mode() == MODE_LOCAL


def test_a_machine_without_a_gpu_is_offered_local_cpu_not_forced_to_the_service():
    """Slow is a decision for the user; it is not the same as impossible."""
    cap = capability(gpus=())
    assert cap.mode() == MODE_LOCAL_CPU
    assert cap.can_run_locally() is True


@pytest.mark.parametrize("missing", ["ffmpeg", "colmap"])
def test_a_missing_tool_forces_the_service(missing):
    cap = capability(**{missing: False})
    assert cap.mode() == MODE_SERVICE
    assert cap.missing() == [missing]
    assert cap.can_run_locally() is False


def test_a_gpu_does_not_rescue_a_missing_tool():
    """The GPU is irrelevant if COLMAP is not there to use it."""
    assert capability(colmap=False, gpus=("RTX 4090",)).mode() == MODE_SERVICE


def test_every_mode_has_a_summary():
    """The shell prints this; a missing key would be a KeyError at runtime."""
    for mode in (MODE_LOCAL, MODE_LOCAL_CPU, MODE_SERVICE):
        assert lr.MODE_SUMMARY[mode]


# --- explaining -----------------------------------------------------------

def test_a_missing_tool_is_explained_with_how_to_install_it(monkeypatch):
    monkeypatch.setattr(lr.platform, "system", lambda: "Linux")
    text = "\n".join(capability(colmap=False).explain())
    assert "colmap" in text and "MISSING" in text
    # "not found" alone is not actionable.
    assert "apt install colmap" in text


def test_no_gpu_is_explained_rather_than_left_blank():
    text = "\n".join(capability(gpus=()).explain())
    assert "none detected" in text and "nvidia-smi" in text


def test_a_gpu_torch_cannot_use_is_called_out():
    """The exact state this machine was in: a real GPU and a CPU-only torch."""
    cap = capability()
    cap.torch_cuda = False
    assert any("torch cannot use it" in line for line in cap.explain())


def test_a_cpu_only_colmap_is_called_out_but_does_not_block_local_mode():
    cap = capability()
    cap.colmap_cuda = False
    assert any("no CUDA support" in line for line in cap.explain())
    assert cap.mode() == MODE_LOCAL


def test_report_names_the_mode_and_what_it_means():
    text = lr.report(capability(colmap=False))
    assert f"mode: {MODE_SERVICE}" in text
    assert "cannot reconstruct" in text
    assert "missing: colmap" in text


def test_as_dict_is_serialisable_and_complete():
    import json

    cap = capability()
    cap.torch_cuda, cap.colmap_cuda = True, True
    d = lr.as_dict(cap)
    json.dumps(d)          # the shell reads this over a pipe
    for key in ("mode", "can_run_locally", "has_gpu", "gpus", "missing", "tools",
                "platform", "torch_cuda", "colmap_cuda"):
        assert key in d, key
    assert d["tools"]["ffmpeg"]["available"] is True


# --- probing --------------------------------------------------------------

def fake_run(mapping):
    """Replace subprocess.run with scripted output keyed by the executable."""
    def run(cmd, **kwargs):
        for key, (code, out) in mapping.items():
            if key in cmd[0] or (len(cmd) > 1 and key in " ".join(cmd)):
                return subprocess.CompletedProcess(cmd, code, out, "")
        return subprocess.CompletedProcess(cmd, 127, "", "not found")
    return run


def test_ffmpeg_version_is_parsed(monkeypatch):
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/ffmpeg")
    monkeypatch.setattr(lr.subprocess, "run",
                        fake_run({"ffmpeg": (0, "ffmpeg version 6.1.1 Copyright (c)")}))
    t = lr.find_ffmpeg()
    assert t.available and t.version == "6.1.1"


def test_a_tool_that_is_present_but_broken_is_not_available(monkeypatch):
    """Found on PATH is not the same as working — a broken install is worse.

    This is the case that matters: reporting it as present means failing
    forty minutes into a reconstruction instead of at startup.
    """
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/ffmpeg")
    monkeypatch.setattr(lr.subprocess, "run", fake_run({"ffmpeg": (1, "")}))
    t = lr.find_ffmpeg()
    assert not t.available and "would not run" in t.problem


def test_a_tool_that_cannot_be_executed_at_all_is_not_available(monkeypatch):
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/colmap")

    def boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(lr.subprocess, "run", boom)
    assert not lr.find_colmap().available


def test_colmap_version_is_parsed_from_the_banner(monkeypatch):
    """COLMAP exits non-zero on `-h` in some builds; the banner is the signal."""
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/colmap")
    monkeypatch.setattr(lr.subprocess, "run",
                        fake_run({"colmap": (1, "COLMAP 3.11.1 -- Structure-from-Motion")}))
    t = lr.find_colmap()
    assert t.available and t.version == "3.11.1"


def test_colmap_producing_no_output_is_treated_as_broken(monkeypatch):
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/colmap")
    monkeypatch.setattr(lr.subprocess, "run", fake_run({"colmap": (0, "   ")}))
    assert not lr.find_colmap().available


@pytest.mark.parametrize("help_text,expected", [
    ("--SiftExtraction.use_gpu arg", True),
    ("--FeatureExtraction.use_gpu arg", True),
    ("--SiftExtraction.max_num_features arg", False),
])
def test_colmap_cuda_support_is_read_from_its_own_help(monkeypatch, help_text, expected):
    monkeypatch.setattr(lr.subprocess, "run", fake_run({"colmap": (0, help_text)}))
    assert lr.colmap_has_cuda("/usr/bin/colmap") is expected


def test_colmap_cuda_is_unknown_rather_than_false_when_it_cannot_be_told(monkeypatch):
    monkeypatch.setattr(lr.subprocess, "run", fake_run({"colmap": (0, "")}))
    assert lr.colmap_has_cuda("/usr/bin/colmap") is None


def test_gpus_are_listed_from_nvidia_smi(monkeypatch):
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(lr.subprocess, "run", fake_run(
        {"nvidia-smi": (0, "NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB\n")}))
    gpus, detail = lr.detect_gpus()
    assert gpus == ["NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB"] and detail == ""


def test_no_driver_is_reported_with_a_reason(monkeypatch):
    monkeypatch.setattr(lr.shutil, "which", lambda n: None)
    gpus, detail = lr.detect_gpus()
    assert gpus == [] and "no NVIDIA driver" in detail


def test_a_failing_nvidia_smi_is_reported_with_its_error(monkeypatch):
    """A driver/library mismatch is common and must not read as 'no GPU'."""
    monkeypatch.setattr(lr.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(lr.subprocess, "run", fake_run(
        {"nvidia-smi": (9, "Failed to initialize NVML: Driver/library version mismatch")}))
    gpus, detail = lr.detect_gpus()
    assert gpus == [] and "exited 9" in detail


def test_probe_prefers_explicit_paths_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIPTO3D_FFMPEG", "/env/ffmpeg")
    monkeypatch.setattr(lr.subprocess, "run", fake_run({"ffmpeg": (0, "ffmpeg version 7.0")}))
    monkeypatch.setattr(lr.shutil, "which", lambda n: None)
    assert lr.find_ffmpeg("/explicit/ffmpeg").path == "/explicit/ffmpeg"


def test_probe_falls_back_to_the_environment_variable(monkeypatch):
    monkeypatch.setenv("CLIPTO3D_COLMAP", "/env/colmap")
    monkeypatch.setattr(lr.shutil, "which", lambda n: None)
    monkeypatch.setattr(lr.subprocess, "run", fake_run({"colmap": (0, "COLMAP 3.10")}))
    cap = lr.probe()
    assert cap.tools["colmap"].path == "/env/colmap"


def test_torch_absence_is_not_a_failure(monkeypatch):
    """A machine can have ffmpeg and COLMAP and no torch; that is not fatal here."""
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert lr.torch_sees_cuda() is None


# --- the real machine -----------------------------------------------------

def test_probing_this_machine_does_not_raise():
    """Whatever is installed, the probe must return an answer, not crash."""
    cap = lr.probe()
    assert cap.mode() in (MODE_LOCAL, MODE_LOCAL_CPU, MODE_SERVICE)
    assert isinstance(lr.report(cap), str)
