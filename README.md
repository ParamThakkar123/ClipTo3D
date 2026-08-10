# ClipTo3D

Reconstruct 3D scenes from video clips: frames → depth → SfM → gaussian splats / fused point cloud.

## Install

The base install is **torch-free**: frame and audio extraction, COLMAP parsing,
dataset conversion, depth fusion, PLY export and viewing. Everything that needs
torch sits behind an extra, so no machine downloads a CUDA stack it cannot use.

Pick exactly one torch backend — they are declared as conflicting extras, so uv
rejects combining them:

| Command | Torch build | Use for |
| --- | --- | --- |
| `uv sync` | none | frame extraction, format conversion, export, viewing |
| `uv sync --extra cpu` | `+cpu` wheels | dev machines, ARM Linux, CI; portable |
| `uv sync --extra mps` | macOS Metal wheel | Apple Silicon — everything but splat training |
| `uv sync --extra cuda` | `+cu130` wheels | NVIDIA reconstruction workers |
| `uv sync --extra cuda --extra splat` | `+cu130` + gsplat | gaussian splat training |
| `uv sync --extra mesh` | — | TSDF surface extraction (scikit-image) |
| `uv sync --extra ar` | — | USDZ + Draco (`usd-core`, `DracoPy`) |
| `uv sync --extra service` | — | the job API (FastAPI) |
| `uv sync --extra s3` | — | S3-compatible artifact storage |

Notes:

- **`mps` and `cpu` install the same wheel on macOS.** The macOS build has Metal
  compiled in; `--extra mps` exists to make that intent explicit. `mps` is scoped
  to darwin and `cuda` to Linux/Windows, so requesting the wrong one on the wrong
  host resolves to nothing rather than breaking the lock.
- **`splat` needs `cuda` alongside it.** gsplat JIT-compiles CUDA kernels via
  ninja at import time and therefore needs a host `nvcc`. MPO-232 bakes the
  compiled kernels into the worker image so the runtime host needs no toolchain.
- **`--extra cuda` currently fails on Windows.** Not our bug: the PyTorch cu130
  index publishes no `#sha256=` for its five `torchvision …win_amd64.whl` files,
  so uv has no hash to lock and rejects the download. The ten Linux cu130 wheels
  do carry hashes and verify correctly, so the worker container is unaffected.
  On Windows use `--extra cpu` for development and the container for CUDA work.
- **Running the tests** needs a torch backend plus the dev group:
  `uv sync --extra cpu --group dev && uv run pytest`. Without torch the suite
  still collects — the two splat test modules skip.
- **SAM2** is opt-in (`--extra sam2`) and not part of the pipeline; nothing
  downstream consumes its output (MPO-230).

Non-Python prerequisites, not installed by uv: **ffmpeg** (frame extraction) and
**COLMAP** (structure from motion).

## Quickstart

```bash
uv sync --extra cpu
uv run clipto3d reconstruct clip.mp4 --out ./result
```

That runs the whole pipeline: frame extraction → keyframe selection → depth →
COLMAP → dataset conversion → fusion. The depth checkpoint is fetched and
hash-verified on first use.

```bash
clipto3d reconstruct clip.mp4 --out ./result --quality preview   # fast look
clipto3d reconstruct clip.mp4 --out ./result --quality final     # best
clipto3d reconstruct clip.mp4 --out ./result --splat             # + gaussians (CUDA)

clipto3d status ./result --tail 20     # stage status and progress
clipto3d status ./result --json        # for a service to poll
clipto3d cancel ./result               # stop at the next checkpoint
clipto3d clean  ./result --dry-run     # drop intermediates
clipto3d presets                       # what the quality levels do
clipto3d doctor                        # what this machine can reconstruct
```

`doctor` reports whether ffmpeg, COLMAP and a GPU are present and picks a mode:
`local`, `local-cpu` (runs here, slowly) or `service` (a tool is missing — upload
instead). It exits non-zero in `service` mode so a script can branch on it, and
`--json` is the form the desktop shell consumes. Missing tools come with the
install command for your platform.

Note that a present GPU and a CUDA-capable torch are separate questions, and
`doctor` reports them separately — `torch.cuda.is_available()` answers the
second, not the first.

### Quality presets

The knobs interact — a small encoder with a huge keyframe budget is incoherent
— so presets set them together.

| | fps | encoder | depth store | keyframes | SIFT features | splat steps |
| --- | --- | --- | --- | --- | --- | --- |
| `preview` | 2 | vits | png16 | ≤60 | 1024 | 2k |
| `balanced` *(default)* | 4 | vitb | fp16 | ≤150 | 2048 | 7k |
| `final` | 6 | vitl | fp16 | ≤300 | 4096 | 30k |

Any explicit flag overrides the preset.

### Resume, progress and cancellation

Each stage records a fingerprint of its parameters and inputs, chained to its
upstream stages. A re-run with nothing changed is a no-op; a job that died
mid-COLMAP resumes at COLMAP rather than re-extracting frames; changing one
knob re-runs only the stages it actually affects.

State lives in the job directory — `state.json`, append-only `events.jsonl`,
and a `CANCEL` sentinel — so a service that did not start the job can still
poll it and stop it. Cancellation is cooperative: the job stops between stages
(or between COLMAP output lines), leaving every artifact on disk complete.

Running a single stage is still possible and still useful:

```bash
python keyframes.py runs/j/frames runs/j/keyframes
python fusion/fuse.py --colmap-dir runs/j/colmap --frames-dir runs/j/keyframes \
    --depth-dir runs/j/depth --out runs/j/cloud/fused.ply
```

## Keyframe selection

Consecutive video frames are ~90% redundant, and blurry ones actively corrupt
COLMAP poses. The `keyframe` stage sits between extraction and everything
downstream, so cutting the frame count cuts COLMAP, depth and fusion together.

Two signals: variance-of-Laplacian sharpness, and sparse Lucas-Kanade parallax
against the last *kept* frame (median displacement as a fraction of image
width). A frame is kept once the view has actually moved — which is what
collapses slow pans and hold-stills that fixed-fps extraction cannot.

```bash
python keyframes.py runs/my-clip/frames runs/my-clip/keyframes
```

Tuning, via `pipeline.py`:

| Flag | Default | Effect |
| --- | --- | --- |
| `--keyframe-min-motion` | `0.012` | Parallax needed to keep a frame. Raise to cut harder |
| `--keyframe-blur-percentile` | `20` | Sharpness floor, taken from the clip's own distribution |
| `--keyframe-min-frames` | `12` | Backfills the sharpest rejects rather than starving COLMAP |
| `--keyframe-max-frames` | `300` | Uniform subsample, so the whole clip stays represented |

The kept/dropped ratio is logged per job and written to
`keyframes/keyframes.json` with a per-frame reason. Selected frames are
hardlinked, so this costs no extra disk. Drop `keyframe` from `--stages` to
feed every extracted frame downstream.

## Model checkpoints

Depth weights are pinned by immutable URL + SHA256 in `checkpoints.py`, cached
on disk, and verified on load. Nothing tracks an upstream default branch, so a
past run's weights can be reproduced.

```bash
python checkpoints.py --list                            # what's pinned, what's cached
python checkpoints.py --fetch depth-anything-v2-vits    # ~99 MB, preview tier
python checkpoints.py --verify                          # audit the cache
```

The cache lives at `$CLIPTO3D_CHECKPOINT_DIR`, defaulting to
`~/.cache/clipto3d/checkpoints`. Set `CLIPTO3D_OFFLINE=1` to forbid downloads —
a missing checkpoint then fails with the command to pre-fetch it rather than
reaching for the network.

The `depthanythingv2` backend is the default and the only offline-capable one:
its model code is vendored under `depth_estimation/depth_anything_v2/`. The
`midas` backend loads its *architecture* from `torch.hub` (pinned to
`intel-isl/MiDaS:v3_1`, not the default branch), so it still needs network on
first use and refuses to run under `CLIPTO3D_OFFLINE=1`.

Which weights produced a set of depth maps is recorded in `depth_meta.json`
alongside them, digest included.

## Job service

```bash
uv sync --extra cpu --extra service
uvicorn service:app --port 8000
```

| | |
| --- | --- |
| `POST /jobs` | upload a clip → job id |
| `GET /jobs` | this caller's jobs |
| `GET /jobs/{id}` | status, per-stage progress, timings |
| `GET /jobs/{id}/events` | progress event tail |
| `GET /jobs/{id}/artifacts` | result list, with URLs |
| `GET /jobs/{id}/artifacts/{name}` | download one artifact |
| `POST /jobs/{id}/cancel` | cooperative cancel |
| `DELETE /jobs/{id}` | delete job and artifacts |
| `POST /uploads/resumable` | begin a resumable upload |
| `GET /uploads/{id}` | where to resume from |
| `PATCH /uploads/{id}?offset=` | append a chunk |
| `POST /uploads/{id}/job` | turn a finished upload into a job |
| `GET /viewer` | the web viewer, same origin as the API |
| `GET /capture` | the guided capture client |

Resumable upload is append-only against an offset the **server** owns: the
client asks where it got to and continues, so a resume can neither duplicate
nor skip bytes. A chunk at the wrong offset is refused with `409` and the
correct offset, rather than corrupting the file.

Artifact URLs are presigned when `CLIPTO3D_STORAGE` points at a bucket, and
otherwise resolve to the download route above — so the viewer works against a
locally-run service with no object storage configured.

A job's export directory is written once and never rewritten (a changed
reconstruction is a new job id), so artifacts are served `immutable` with a
one-year max-age and answer `If-None-Match` with a `304`. That matters more
than it sounds: the viewer's progressive path fetches three LODs, and without
it every revisit and every share-link open re-downloaded all three.

### Keys

```bash
export BOOT=$CLIPTO3D_BOOTSTRAP_KEY
curl -XPOST localhost:8000/keys?label=admin -H "X-Bootstrap-Key: $BOOT"  # key one
curl -XPOST localhost:8000/keys -H "X-API-Key: $KEY"   # after that, a key is required
curl localhost:8000/keys -H "X-API-Key: $KEY"          # digest prefixes only
curl -XDELETE localhost:8000/keys/<prefix> -H "X-API-Key: $KEY"   # revoke
```

Only a SHA-256 digest is stored, so the plaintext exists exactly once — at
issue time. Revocation is immediate and **keeps the jobs**, so turning off a
leaked key doesn't destroy its history. The listing exposes a 12-character
digest prefix, not the full digest, since the full digest is enough to revoke
someone else's key.

**The bootstrap secret is what closes the land grab.** `CLIPTO3D_AUTH=strict`
(the default) rejects unknown keys even before any key exists, and mints key
one only against `CLIPTO3D_BOOTSTRAP_KEY`. Without that, there is a race
between the process becoming reachable and you issuing key one — and on a
public address you do not reliably win it. The secret mints exactly one key;
once a key exists it stops working, so a leak is not a permanent skeleton key.

With no secret configured the process generates one and logs it at startup, so
an unconfigured deployment is still usable without leaving the window open.
Set `CLIPTO3D_BOOTSTRAP_KEY` to keep it across restarts.

`CLIPTO3D_AUTH=open` restores the old behaviour — no keys, no auth. That is
the right mode for `uvicorn service:app` on a laptop and the wrong one for
anything reachable, which is why it has to be asked for by name. `/health`
reports which mode is live.

### Self-service signup

`CLIPTO3D_AUTH=public` is the mode for "anyone can upload a video": a visitor
`POST /keys` with no credential and gets their own key back. Isolation and
quotas are already keyed on the key digest, so each visitor lands in their own
namespace with their own budget at no extra cost.

Two things make that safe rather than an open door. Self-service keys are
**non-admin** — they cannot issue further keys, so one visitor stays one
caller and cannot mint their way past `CLIPTO3D_QUOTA_TOTAL`. And signups are
capped per address (`CLIPTO3D_KEYS_PER_ADDRESS`, default 5 per day), because a
per-caller quota only bites if a fresh caller identity costs something. That
cap is in-memory abuse friction, not an access control — it resets on restart
and does nothing against a botnet. Put a real WAF in front if you need one.

You keep an admin key by passing the bootstrap secret, which in public mode
keeps working rather than retiring after the first key.

| | `strict` | `public` | `open` |
| --- | --- | --- | --- |
| unknown key | rejected | rejected | accepted |
| who mints keys | you, via the secret | anyone, rate-limited | nobody needed |
| new keys can issue keys | yes | no | — |
| use for | private deployment | self-serve site | localhost |

### Share links

A finished reconstruction is something people want to send to someone, and
that someone must not be handed an API key — a key is a write credential that
spends GPU time.

```bash
curl -XPOST localhost:8000/jobs/<id>/share -H "X-API-Key: $KEY"
# -> {"token": "...", "viewer_url": "/viewer?job=/shared/<token>/artifacts"}
```

`viewer_url` opens in a browser with **no key at all**. The token reads that
one job's artifacts and status and does nothing else: it cannot list jobs,
upload, or cancel. Pass `ttl_seconds` (or set `CLIPTO3D_SHARE_TTL_SECONDS`)
to expire it; `DELETE /jobs/<id>/share` revokes every link for a job, and
deleting the job takes its links with it.

Tokens are stored as digests, like API keys, so a leaked database yields no
working links. Unknown, expired and revoked all answer 404 alike, so a token
cannot be probed for which it is.

### Isolation and state

Callers are separated by an `X-API-Key` header; another caller's job is a 404,
not a 403, so existence doesn't leak. Quotas are per caller
(`CLIPTO3D_QUOTA_ACTIVE`, `CLIPTO3D_QUOTA_TOTAL`) because GPU time is the
expensive resource. Concurrency is `CLIPTO3D_WORKERS`, which should equal your
GPU count. Set `CLIPTO3D_STORAGE=s3://bucket/prefix` to push artifacts to
object storage and get signed URLs back.

Job state and the work queue live in SQLite (`<jobs-root>/jobs.db`, WAL mode),
so both survive a restart. **The row is the queue**: `POST /jobs` writes it and
returns, and a worker claims it from the table under a **lease**, heartbeating
while it works. That is what makes a restart survivable — a job that was
queued or mid-flight when the process died is requeued at startup and picked
up where it left off, since every stage checkpoints. A worker that dies
without the process has its lease lapse and is swept back by the reaper within
`CLIPTO3D_REAPER_SECONDS`. `/health` reports `queue_depth` and `expired_leases`.

Because the queue is in the database rather than in a process, workers can run
somewhere else entirely:

```bash
uvicorn service:app --port 8000        # CLIPTO3D_WORKERS=0: API only
python worker.py --jobs-root ./runs    # one or more, alongside it
```

That is the shape `docker-compose.yml` deploys, and the reason for it is that
a reconstruction OOM should not take the API down with it.

`POST /uploads` returns a presigned PUT so large videos go straight to storage
instead of streaming through the API (needs `CLIPTO3D_STORAGE`). Clips posted
to `/jobs` are streamed to disk in chunks rather than buffered, so concurrent
uploads cost disk rather than resident memory.

**Retention.** Quotas cap the job *count* per caller; they do not cap bytes,
and a job directory is gigabytes. `CLIPTO3D_JOB_RETENTION_DAYS` (30) drops
finished jobs, and `CLIPTO3D_UPLOAD_RETENTION_HOURS` (24) collects resumable
uploads that were started and abandoned. Unfinished jobs are never touched;
`0` disables either sweep.

Still **single-node**: SQLite coordinates threads and processes on one host,
not across machines. A multi-host deployment needs a broker behind the same
`claim`/`heartbeat`/`complete` interface — see the MPO-244 comment.

## Hosting it

```bash
cp .env.example .env          # set CLIPTO3D_BOOTSTRAP_KEY at minimum
docker compose up -d api worker               # HTTP on 127.0.0.1:8000
docker compose --profile tls up -d            # + Caddy, needs CLIPTO3D_DOMAIN
```

Two images, on purpose. `docker/Dockerfile.service` is the API: base install
plus FastAPI, no torch and no COLMAP, non-root, **676 MB** measured, and it
starts in about a second. `docker/Dockerfile` is the worker: CUDA, COLMAP and
the checkpoints, 4.98 GB. They share `/data` and coordinate through the lease
queue, so scaling reconstruction means adding worker containers rather than
API replicas.

676 MB is more than an API needs, and it is worth knowing where it goes: of a
336 MB virtualenv, OpenCV is 188 MB and numpy 69 MB. The base install pulls
them in for frame extraction and keyframe selection, which the API never runs
— it only enqueues. Carving a genuinely API-only dependency set out of the
base extra would cut the image by roughly half, and is not done here.

The API port binds to loopback by default (`CLIPTO3D_BIND`), because a
`compose up` on a public VPS should not put unproxied HTTP on the internet.
The `tls` profile puts Caddy in front, which gets and renews a certificate on
its own. **That is a functional requirement, not hygiene**: `getUserMedia`
needs a secure context, so `/capture` does not work over plain HTTP anywhere
except localhost.

Before exposing it, the three that actually matter:

| | |
| --- | --- |
| `CLIPTO3D_BOOTSTRAP_KEY` | set it, or key one is a race you might lose |
| `CLIPTO3D_TRUST_PROXY` | `1` **only** with a proxy in front — otherwise `X-Forwarded-For` is client-controlled and the rate limiter is bypassed by forging it. Do set it with the `tls` profile: it also gates uvicorn's own proxy-header handling (`docker/api-entrypoint.sh`) and is what lets the API know a request arrived over TLS, so HSTS depends on it |
| `CLIPTO3D_BIND` | leave on loopback unless something else terminates TLS |

Everything tunable is in `.env.example` with what it costs to get it wrong.

Not covered here, and worth knowing before you rely on it: there is no
multi-host story (SQLite is per-machine), no metrics endpoint, and no backup
of `jobs.db` — it is a single file in the data volume, so snapshot the volume
if the job history matters.

## Export formats

The `export` stage writes everything a client can consume, each checked against
a mobile size budget (default 20 MB) and flagged if it exceeds it. Measured on
a 12-keyframe reconstruction:

| file | what | size |
| --- | --- | --- |
| `cloud.glb` + `cloud_lod1/2.glb` | point cloud, 3 LODs, int16-quantised | 1.26 / 0.31 / 0.08 MB |
| `mesh.glb` | triangle mesh — Android Scene Viewer | 2.93 MB |
| `scene.usdz` | iOS AR Quick Look | 2.29 MB |
| `cloud.splat` / `scene.splat` | gaussians for the web viewer | 3.35 MB |

The `mesh` stage builds the surface (TSDF + marching cubes) that the AR formats
need — a point cloud in AR has no occlusion or lighting response and reads as
noise. Without `--extra mesh` the stage is skipped and the point-cloud formats
are still produced.

`.splat` is written from trained gaussians when a splat run exists, and
otherwise derived from the fused cloud as isotropic gaussians — enough to
exercise the viewer's splat path, not a substitute for training.

## Viewer

`viewer/index.html` — self-contained, no build step, no CDN. Drop a `.glb` or
`.splat` on it, serve it next to a job's `export/` directory, or reach it at
`/viewer` when the job service is running:

```bash
python -m http.server -d runs/my-clip/export 8080   # then open the viewer
```

| | |
| --- | --- |
| `?job=<base>` | progressive: `cloud_lod2` → `lod1` → `cloud`, coarsest first |
| `?src=<url>` | open one asset directly — a shareable link to a result |

WebGL2, two render paths. `.glb` point clouds draw as perspective-sized round
sprites. `.splat` gaussians go through a real EWA rasteriser: the 3D covariance
is built from scale and rotation, pushed through the projection Jacobian, and
inverted into a 2D conic the fragment shader evaluates — so splats are oriented
ellipses, depth-sorted back-to-front (65,536-bucket counting sort) and
alpha-composited with depth testing off.

Also: device-capability tiering (picks a splat budget rather than attempting the
full asset on a weak phone), mouse + touch controls, and an upload panel that
posts a video to the job service, shows per-stage progress, and loads the result
when it finishes.

`tests/test_viewer.py` runs the page's own JavaScript under Node against real
`export.py` output. The GLSL is not covered there — it cannot run headless.

### AR handoff

On a phone the viewer offers the result to the OS: **AR Quick Look** on iOS
(`scene.usdz`) and **Scene Viewer** on Android (`mesh.glb`), both needing the
mesh export — neither accepts splats or lights a bare point cloud.

Scene Viewer fetches the file from another app, so it cannot reach `localhost`
or a private address; the viewer detects that and says so rather than opening a
viewer that fails silently.

## Capture

`viewer/capture.html`, served at `/capture` — guided recording in the browser,
because reconstruction quality is decided at capture time and unguided clips
are what COLMAP cannot solve.

It coaches live against the three failure modes that actually kill a
reconstruction:

| | how it's detected |
| --- | --- |
| motion blur | variance of the Laplacian — the same metric `keyframes.py` selects on |
| **no parallax** | gyroscope: high rotation rate with motion the rotation alone explains |
| too few viewpoints | compass heading bucketed into 36 sectors of an orbit |

The parallax one is the point. Spinning on the spot produces plenty of
frame-to-frame motion and *zero* depth — it is the common user instinct, and a
camera alone cannot distinguish it from orbiting. The gyroscope can.

After recording it says plainly whether the clip is likely to reconstruct, and
uploads resumably so a dropped mobile connection costs one chunk rather than
the whole file. The clip is always offered as a download too — a capture the
network ate should not be lost.

Sensors are optional: without a gyroscope or compass it still coaches on blur
and movement, and simply does not claim what it cannot measure.

## Worker container

`docker/Dockerfile` builds the reconstruction worker: ffmpeg, COLMAP built with
CUDA SIFT and without GUI/OpenGL (so it works headless), the `cuda` + `splat`
extras, and the depth checkpoints baked in.

```bash
docker build -f docker/Dockerfile -t clipto3d-worker .
docker run --gpus all -v "$PWD/runs:/runs" clipto3d-worker /runs/clip.mp4 --job /runs/job1
```

Build args worth knowing:

| Arg | Default | Why you'd change it |
| --- | --- | --- |
| `CUDA_ARCHS` / `TORCH_CUDA_ARCH_LIST` | Turing→Hopper | Narrow to your fleet's SM to cut build time substantially |
| `COLMAP_REF` | `3.13.0` | Last 3.x; 4.x additionally pulls in ONNX |
| `BAKE_CHECKPOINTS` | `depth-anything-v2-vitl` | `…-vits` for a much smaller image, or empty to mount a cache at `/opt/checkpoints` |
| `BUILD_JOBS` | all cores | **Set to ~4 on a low-RAM machine.** ninja at full width across COLMAP's 349 C++/CUDA targets can OOM the builder VM, which surfaces confusingly as `buildkit … rpc error: EOF` |
| `RUNTIME_FLAVOR` | `runtime` | The runtime image ships **without `nvcc`** — gsplat's kernels are precompiled at build time. Use `devel` (with `REQUIRE_PRECOMPILE=0`) to keep a JIT fallback |
| `REQUIRE_PRECOMPILE` | `1` | A precompile miss is a hard build failure, because there is no toolchain at runtime to fall back on |

The image installs `cuda`, `splat`, `mesh` and `ar`, so the worker can produce
every export format. Measured: **4.98 GB** content on the `runtime` base versus
7.3 GB on `devel`.

Building the heavy stage on its own first (`--target colmap-builder`) is worth
it on a constrained machine: it caches, so a later crash doesn't repeat it.

COLMAP's GPU SIFT is now detected rather than assumed: `pipeline.py` probes for
a device and falls back to CPU SIFT if the attempt fails. Pass `--colmap-gpu` to
make it binding — the run then fails with a diagnosis instead of silently taking
the slow path.
