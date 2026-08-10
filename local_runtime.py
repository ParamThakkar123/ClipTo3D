"""What this machine can actually reconstruct, and where a job should run (MPO-249).

The desktop app's one genuine advantage over the web client is running the
pipeline locally instead of uploading gigabytes to a GPU somewhere else. That
advantage only exists on a machine that can *actually* run it — which means
ffmpeg, COLMAP and, realistically, an NVIDIA GPU. On any other machine the app
has to fall back to the hosted service or it simply does not work.

So the desktop shell needs one question answered before it does anything:
**can this machine do the work, or does it need the service?** That question is
Python, not Rust, and it is the same answer the CLI wants when a user asks why
a stage was skipped. It lives here rather than inside the shell so both get it,
and so it can be tested without a desktop app at all.

Detection is deliberately conservative. Claiming a capability that turns out to
be missing fails *during* a long reconstruction, which is far worse than
declining it up front and uploading instead.

    python -m local_runtime          # or: clipto3d doctor
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Where each tool comes from when it is missing. Printed rather than left to
# the user to search for, because "colmap not found" is not actionable on its
# own and the answer differs per platform.
INSTALL_HINTS = {
    "ffmpeg": {
        "Windows": "winget install Gyan.FFmpeg  (or https://ffmpeg.org/download.html)",
        "Darwin": "brew install ffmpeg",
        "Linux": "apt install ffmpeg  (or your distribution's package)",
    },
    "colmap": {
        "Windows": "https://github.com/colmap/colmap/releases  (CUDA build)",
        "Darwin": "brew install colmap",
        "Linux": "apt install colmap  (CUDA support varies; build from source for GPU SIFT)",
    },
}

MODE_LOCAL = "local"
MODE_LOCAL_CPU = "local-cpu"
MODE_SERVICE = "service"


@dataclass
class Tool:
    name: str
    path: Optional[str] = None
    version: str = ""
    problem: str = ""

    @property
    def available(self) -> bool:
        return self.path is not None and not self.problem

    def hint(self) -> str:
        return INSTALL_HINTS.get(self.name, {}).get(platform.system(), "")


@dataclass
class Capability:
    """What this machine can do, and what it therefore should do."""

    tools: Dict[str, Tool] = field(default_factory=dict)
    gpus: List[str] = field(default_factory=list)
    gpu_detail: str = ""
    torch_cuda: Optional[bool] = None
    colmap_cuda: Optional[bool] = None

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def missing(self) -> List[str]:
        return [n for n, t in self.tools.items() if not t.available]

    def can_run_locally(self) -> bool:
        """Every *required* tool is present. GPU is a separate question."""
        return not self.missing()

    def mode(self) -> str:
        """Where a job should run on this machine.

        Three answers, not two. A machine with the tools but no GPU can run the
        pipeline — it will just be slow — and that is a decision for the user
        rather than something to silently take away. The shell offers it; it
        does not pick it by default.
        """
        if not self.can_run_locally():
            return MODE_SERVICE
        return MODE_LOCAL if self.has_gpu else MODE_LOCAL_CPU

    def explain(self) -> List[str]:
        """Human-readable reasons, in the order a user needs them."""
        lines = []
        for name, tool in self.tools.items():
            if tool.available:
                lines.append(f"  {name:8} {tool.version or 'found'}  ({tool.path})")
            else:
                detail = tool.problem or "not found"
                hint = tool.hint()
                lines.append(f"  {name:8} MISSING - {detail}" + (f"\n           {hint}" if hint else ""))
        if self.gpus:
            for g in self.gpus:
                lines.append(f"  gpu      {g}")
        else:
            lines.append(f"  gpu      none detected - {self.gpu_detail}")
        if self.colmap_cuda is False:
            lines.append("  note     COLMAP has no CUDA support; feature extraction will use the CPU")
        if self.torch_cuda is False and self.gpus:
            lines.append("  note     a GPU is present but torch cannot use it; install the cuda extra")
        return lines


# --- probes ---------------------------------------------------------------

def _run(cmd: List[str], timeout: int = 30) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def find_ffmpeg(override: Optional[str] = None) -> Tool:
    path = override or shutil.which("ffmpeg")
    if not path:
        return Tool("ffmpeg", problem="not on PATH")
    proc = _run([path, "-version"])
    if proc is None or proc.returncode != 0:
        return Tool("ffmpeg", path=path, problem="found but would not run")
    first = (proc.stdout or "").splitlines()[:1]
    m = re.search(r"ffmpeg version (\S+)", first[0]) if first else None
    return Tool("ffmpeg", path=path, version=m.group(1) if m else "unknown")


def find_colmap(override: Optional[str] = None) -> Tool:
    path = override or shutil.which("colmap")
    if not path:
        return Tool("colmap", problem="not on PATH")
    # `colmap -h` exits non-zero on some builds, so the version banner is the
    # signal, not the exit status.
    proc = _run([path, "-h"], timeout=60)
    if proc is None:
        return Tool("colmap", path=path, problem="found but would not run")
    text = (proc.stdout or "") + (proc.stderr or "")
    if not text.strip():
        return Tool("colmap", path=path, problem="produced no output; the install may be broken")
    m = re.search(r"COLMAP\s+(\d+\.\d+(?:\.\d+)?)", text)
    return Tool("colmap", path=path, version=m.group(1) if m else "unknown")


def colmap_has_cuda(colmap_path: str) -> Optional[bool]:
    """Whether this COLMAP was built with CUDA, or None if it cannot be told.

    A CPU-only COLMAP still reconstructs, just far more slowly — so this
    changes the estimate shown to the user, not whether local mode is offered.
    """
    proc = _run([colmap_path, "feature_extractor", "-h"], timeout=60)
    if proc is None:
        return None
    text = ((proc.stdout or "") + (proc.stderr or "")).lower()
    if not text.strip():
        return None
    # Every CUDA-enabled build exposes a use_gpu option somewhere in this help.
    return "use_gpu" in text


def detect_gpus() -> tuple[List[str], str]:
    """Names of visible NVIDIA GPUs, plus why the list is empty if it is."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return [], "nvidia-smi not on PATH (no NVIDIA driver)"
    proc = _run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if proc is None:
        return [], "nvidia-smi could not be run"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return [], f"nvidia-smi exited {proc.returncode}: {detail[0] if detail else 'no output'}"
    names = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return names, "" if names else "nvidia-smi reported no devices"


def torch_sees_cuda() -> Optional[bool]:
    """None when torch is not installed at all — which is not a failure here."""
    try:
        import torch
    except Exception:      # noqa: BLE001 - a broken torch install is also "no"
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:      # noqa: BLE001
        return False


def probe(ffmpeg: Optional[str] = None, colmap: Optional[str] = None) -> Capability:
    """Everything the desktop shell needs to decide where a job runs."""
    cap = Capability()
    cap.tools["ffmpeg"] = find_ffmpeg(ffmpeg or os.environ.get("CLIPTO3D_FFMPEG"))
    cap.tools["colmap"] = find_colmap(colmap or os.environ.get("CLIPTO3D_COLMAP"))
    cap.gpus, cap.gpu_detail = detect_gpus()
    cap.torch_cuda = torch_sees_cuda()
    if cap.tools["colmap"].available:
        cap.colmap_cuda = colmap_has_cuda(cap.tools["colmap"].path)
    return cap


# --- reporting ------------------------------------------------------------

MODE_SUMMARY = {
    MODE_LOCAL: "reconstruct locally on the GPU",
    MODE_LOCAL_CPU: "reconstruct locally on the CPU (slow) or use the service",
    MODE_SERVICE: "upload to the job service - this machine cannot reconstruct",
}


def report(cap: Optional[Capability] = None) -> str:
    cap = cap or probe()
    lines = [f"ClipTo3D on {platform.system()} {platform.machine()}", ""]
    lines += cap.explain()
    lines += ["", f"mode: {cap.mode()} - {MODE_SUMMARY[cap.mode()]}"]
    if cap.missing():
        lines.append(f"missing: {', '.join(cap.missing())}")
    return "\n".join(lines)


def as_dict(cap: Optional[Capability] = None) -> Dict[str, object]:
    """The same answer as JSON, for the desktop shell to consume."""
    cap = cap or probe()
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "mode": cap.mode(),
        "can_run_locally": cap.can_run_locally(),
        "has_gpu": cap.has_gpu,
        "gpus": cap.gpus,
        "gpu_detail": cap.gpu_detail,
        "torch_cuda": cap.torch_cuda,
        "colmap_cuda": cap.colmap_cuda,
        "missing": cap.missing(),
        "tools": {
            name: {"path": t.path, "version": t.version, "problem": t.problem,
                   "available": t.available, "hint": t.hint()}
            for name, t in cap.tools.items()
        },
    }


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    print(report())
