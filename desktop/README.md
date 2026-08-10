# Desktop shell (MPO-249)

**Status: design only. There is no shell code here yet — this file is the
whole directory.** No Rust toolchain was available on the machine this was
written on, and an uncompiled `src-tauri/` written blind would look like
progress while being unverified in every line. The Python half the shell
depends on *is* built and tested, and is described below.

## Why a desktop app at all

One reason, and it is in the issue: **running the pipeline on the user's own
GPU instead of uploading gigabytes of video**. Wrapping the web viewer in a
window is not worth shipping on its own — the browser already does that. The
local pipeline is the whole point.

That means the shell's first job is to answer *can this machine reconstruct?*
and behave accordingly. That question is answered by `local_runtime.py`, which
is tested and works today:

```console
$ clipto3d doctor
ClipTo3D on Windows AMD64

  ffmpeg   N-121583-g4348bde2d2  (E:\ffmpeg\bin\ffmpeg.EXE)
  colmap   3.12.6                (E:\colmap-x64-windows-cuda\bin\colmap.EXE)
  gpu      NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB

mode: local - reconstruct locally on the GPU
```

`clipto3d doctor --json` is the machine-readable form the shell consumes. It
exits non-zero when the machine cannot reconstruct on its own, so the
installer or a launch script can branch on it without parsing anything.

Three modes, not two:

| mode | meaning |
| --- | --- |
| `local` | tools present, GPU present — run here |
| `local-cpu` | tools present, no GPU — offer it, do not default to it |
| `service` | a tool is missing — upload to the hosted service |

`local-cpu` exists because "slow" is the user's decision to make, not something
to quietly take away.

## Bundle vs detect

**Detect, do not bundle.** The issue asks for this decision early, and it is
not close:

* **ffmpeg's licence follows its build flags.** A build configured with
  `--enable-gpl` or `--enable-nonfree` changes what may be redistributed and
  under what terms. Shipping a binary means owning that decision for every
  user; detecting one they already installed does not.
* **COLMAP with CUDA is large** and its useful build is GPU- and
  driver-specific. Bundling one build for everyone means shipping the wrong
  one for most.
* Together they would dominate installer size for a shell whose own code is
  a few hundred kilobytes.

So the app detects, and when a tool is missing it says *which* tool and *how to
install it on this platform* — `INSTALL_HINTS` in `local_runtime.py` — rather
than reporting "colmap not found", which is not actionable. Falling back to the
hosted service is always available, so a machine with neither tool still works.

## What the shell needs to do

1. On launch, run `clipto3d doctor --json` and pick a mode.
2. Native file dialog for choosing a video, and for choosing where exports go.
3. In local mode, run `clipto3d reconstruct` as a child process and stream its
   `events.jsonl` progress into the viewer's existing progress UI — the same
   events the job service serves, so the frontend needs no second code path.
4. In service mode, do nothing special: the web viewer's upload panel already
   talks to the API.
5. Host `viewer/index.html` unchanged. It is a single self-contained file with
   no build step, which is exactly why it was written that way.

Point 3 is the only real work, and it is small because `job_state.py` already
writes machine-readable progress that something other than the process itself
can read.

## Not done

* **None of the five points above are written.** No `src-tauri/`, no
  `Cargo.toml`, no `tauri.conf.json`.
* Not packaged, not signed. Windows and macOS signing needs certificates and is
  a distribution decision rather than a coding one.

What *is* done is everything the shell would call into: mode detection
(`local_runtime.py`, `clipto3d doctor --json`), the pipeline itself
(`clipto3d reconstruct`), machine-readable progress a separate process can tail
(`job_state.py` → `events.jsonl`), and the viewer that renders it.
