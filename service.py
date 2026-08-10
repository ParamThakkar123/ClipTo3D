"""Reconstruction job service (MPO-244).

COLMAP and 3DGS cannot run on a phone or in a browser, so reconstruction is
server-side and the clients are thin. This is the API and the worker loop.

    uvicorn service:app --host 0.0.0.0 --port 8000

    POST   /jobs                 upload a clip, get a job id
    GET    /jobs                 list this caller's jobs
    GET    /jobs/{id}            status, per-stage progress, timings
    GET    /jobs/{id}/events     progress event stream (tail)
    GET    /jobs/{id}/artifacts  result URLs
    POST   /jobs/{id}/cancel     cooperative cancel
    DELETE /jobs/{id}            delete a job and its artifacts

**Scope, stated plainly.** This is a correct *single-node* service: workers
that claim from a durable lease queue (`job_registry.py` + `worker.py`), job
state on the filesystem (`job_state.py`), artifacts through the storage
abstraction (`storage.py`), API-key isolation and per-caller quotas.

Work is dispatched through the registry, not through an in-process future.
That is what makes a restart survivable: the row *is* the queue, so a job that
was queued or mid-flight when the process died is requeued and picked up
rather than stranded. Workers can therefore also live in their own containers
— see `worker.main()` and docker-compose.yml — which is the deployment that
keeps a reconstruction OOM from taking the API down with it.

It is deliberately NOT a distributed system. SQLite coordinates threads and
processes on one host, not across machines; there is no Redis/Arq/Celery
queue, no Postgres and no identity provider, because each of those is a real
deployment decision rather than something to guess at. The seams are `claim`/
`heartbeat`/`complete` and the `Storage` interface, so swapping the queue for
a broker is a class, not a rewrite.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from job_paths import JobPaths
from job_registry import (
    JobRecord, JobRegistry, KeyStore, ShareStore, hash_key, new_api_key,
)
from job_state import JobState
from storage import Storage, from_uri
from worker import Reaper, WorkerPool, pipeline_runner

log = logging.getLogger("clipto3d.service")


def configure_logging(level: Optional[str] = None) -> None:
    """Give the `clipto3d` loggers somewhere to write.

    uvicorn configures its own `uvicorn.*` loggers and leaves the root logger
    without a handler, so without this every INFO record here is swallowed by
    the last-resort handler's WARNING threshold — the service would run in
    production with its request log silently discarded.

    Idempotent, and it never touches the root logger, so pytest's caplog and
    anything embedding this app keep their own configuration.
    """
    parent = logging.getLogger("clipto3d")
    parent.setLevel((level or os.environ.get("CLIPTO3D_LOG_LEVEL", "INFO")).upper())
    if not parent.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        parent.addHandler(handler)

# Imported at module scope, not inside build_app(): `from __future__ import
# annotations` makes every annotation a string, and FastAPI resolves those
# against the *module* namespace. Function-local imports leave it unable to
# see UploadFile/Request, and every request 422s with "field required".
try:
    from fastapi import (
        Depends, FastAPI, File, Header, HTTPException, Request, UploadFile,
    )
    from fastapi.responses import (
        FileResponse, HTMLResponse, JSONResponse, Response,
    )

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on install extras
    FASTAPI_AVAILABLE = False

# Browsers refuse to decode a .glb served as text/plain, and the viewer's
# fetch of a .splat should not be sniffed as HTML.
MEDIA_TYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".splat": "application/octet-stream",
    ".ply": "application/octet-stream",
    ".obj": "text/plain",
    ".usdz": "model/vnd.usdz+zip",
    ".npz": "application/octet-stream",
    ".json": "application/json",
}

# Artifacts are write-once. The pipeline builds a job's export directory and
# never rewrites it; a changed reconstruction is a new job id, so the bytes
# behind one artifact URL cannot change. That is exactly what `immutable` is
# for, and it matters here more than it usually would: the viewer's
# progressive path fetches three LODs and a revisit re-fetched every one of
# them, because Starlette's FileResponse sets an ETag but answers no
# conditional request. A warm cache cost the same as a cold one.
ARTIFACT_CACHE_CONTROL = "public, max-age=31536000, immutable"


def if_none_match(header: str, etag: str) -> bool:
    """Does a client's If-None-Match cover `etag`?

    Split out and given a name because the header is a *list*, and the weak
    prefix has to be ignored on comparison — a naive `header == etag` silently
    never matches for any client that sends more than one validator.
    """
    header = header.strip()
    if not header or not etag:
        return False
    if header == "*":
        return True
    for candidate in header.split(","):
        candidate = candidate.strip()
        if candidate.startswith("W/"):
            candidate = candidate[2:]
        if candidate == etag:
            return True
    return False


def artifact_response(request: "Request", target: Path):
    """Serve one artifact, cacheably.

    `stat_result` is passed explicitly because FileResponse only fills in the
    ETag when it is — left to stat the file itself, it does so inside the ASGI
    call, far too late to compare against the request.
    """
    stat_result = target.stat()
    response = FileResponse(
        target, filename=target.name, stat_result=stat_result,
        media_type=MEDIA_TYPES.get(target.suffix, "application/octet-stream"))
    response.headers["Cache-Control"] = ARTIFACT_CACHE_CONTROL

    etag = response.headers.get("etag", "")
    if if_none_match(request.headers.get("if-none-match", ""), etag):
        # 304 carries no body, and must repeat the validators so the cache
        # entry it refreshes keeps them.
        return Response(status_code=304, headers={
            "ETag": etag,
            "Cache-Control": ARTIFACT_CACHE_CONTROL,
            "Last-Modified": response.headers.get("last-modified", ""),
        })
    return response

# --- configuration --------------------------------------------------------

JOBS_ROOT = Path(os.environ.get("CLIPTO3D_JOBS_ROOT", "./runs")).expanduser()
STORAGE_URI = os.environ.get("CLIPTO3D_STORAGE", "")
# GPU count bounds concurrency: reconstruction is GPU-bound, so running more
# jobs than GPUs makes every one of them slower without finishing any sooner.
MAX_CONCURRENT = int(os.environ.get("CLIPTO3D_WORKERS", "1"))

MAX_UPLOAD_BYTES = int(os.environ.get("CLIPTO3D_MAX_UPLOAD_MB", "500")) * 1024 * 1024
# Suggested chunk for resumable uploads. Small enough that a dropped mobile
# connection loses little, large enough not to pay a round trip per megabyte.
RESUMABLE_CHUNK_BYTES = int(os.environ.get("CLIPTO3D_CHUNK_MB", "4")) * 1024 * 1024
QUOTA_ACTIVE_JOBS = int(os.environ.get("CLIPTO3D_QUOTA_ACTIVE", "2"))
QUOTA_TOTAL_JOBS = int(os.environ.get("CLIPTO3D_QUOTA_TOTAL", "50"))
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

# How long a worker's claim on a job stays valid without a heartbeat. It has to
# comfortably exceed the gap between heartbeats, not the job's runtime — a live
# worker keeps refreshing it (see worker.WorkerPool).
LEASE_SECONDS = int(os.environ.get("CLIPTO3D_LEASE_SECONDS", "3600"))
REAPER_SECONDS = float(os.environ.get("CLIPTO3D_REAPER_SECONDS", "60"))

# Three modes, in increasing order of who can get in:
#
#   strict  (default) an unknown key is rejected even before any key exists,
#           and minting the first key needs the bootstrap secret. Every key
#           after that is issued by you.
#   public  anyone may mint a key for themselves, rate-limited per address.
#           Self-service signup: each visitor gets their own namespace and
#           their own quota, and cannot issue further keys. This is the mode
#           for "anyone can upload a video".
#   open    no keys, no auth. Right for `uvicorn service:app` on a laptop,
#           wrong for anything with a public address.
AUTH_MODE = os.environ.get("CLIPTO3D_AUTH", "strict").strip().lower()
BOOTSTRAP_KEY = os.environ.get("CLIPTO3D_BOOTSTRAP_KEY", "")

# Public mode only. Caps self-service signups from one address, because the
# per-caller job quota is only a limit if getting a new caller identity costs
# something. In memory, so it resets on restart — this is abuse friction, not
# an access control.
KEYS_PER_ADDRESS = int(os.environ.get("CLIPTO3D_KEYS_PER_ADDRESS", "5"))
KEYS_WINDOW_SECONDS = float(os.environ.get("CLIPTO3D_KEYS_WINDOW_SECONDS", "86400"))

# How long a share link stays valid. 0 means it does not expire.
SHARE_TTL_SECONDS = float(os.environ.get("CLIPTO3D_SHARE_TTL_SECONDS", "0"))

# Retention. Quotas cap the job *count* per caller; these cap how long the
# bytes live, which is the axis that actually fills a disk.
UPLOAD_RETENTION_HOURS = float(os.environ.get("CLIPTO3D_UPLOAD_RETENTION_HOURS", "24"))
JOB_RETENTION_DAYS = float(os.environ.get("CLIPTO3D_JOB_RETENTION_DAYS", "30"))
RETENTION_SWEEP_SECONDS = float(os.environ.get("CLIPTO3D_RETENTION_SWEEP_SECONDS", "3600"))

# Requests per minute per caller (or per source address when unauthenticated).
RATE_LIMIT_PER_MINUTE = int(os.environ.get("CLIPTO3D_RATE_LIMIT", "120"))

# The viewer is served from this origin, so CORS is not needed by default.
# Set this only when a front end is hosted somewhere else.
CORS_ORIGINS = [o.strip() for o in os.environ.get("CLIPTO3D_CORS_ORIGINS", "").split(",")
                if o.strip()]

# Behind a reverse proxy the peer address is the proxy's. Only honour
# X-Forwarded-For when the deployment says a trusted proxy is in front,
# because otherwise a client can spoof its way around the rate limiter.
TRUST_PROXY = os.environ.get("CLIPTO3D_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _all_key_hashes(registry) -> List[str]:
    rows = registry._cx.execute("SELECT key_hash FROM api_keys").fetchall()
    return [r["key_hash"] for r in rows]


def validate_upload(filename: str, size: int) -> Optional[str]:
    """Reject what cannot reconstruct, before spending a GPU slot on it."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return f"unsupported container {suffix!r}; expected one of {sorted(ALLOWED_SUFFIXES)}"
    if size <= 0:
        return "empty upload"
    if size > MAX_UPLOAD_BYTES:
        return f"upload is {size / 1e6:.0f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB"
    return None


def safe_artifact_path(export: Path, name: str) -> Optional[Path]:
    """Resolve `name` inside `export`, or None if it escapes or is missing.

    A module-level function rather than an inline check so it can be tested
    directly. Going through HTTP is not enough on its own: httpx collapses
    `../..` in a URL path before it is ever sent, so an end-to-end test of the
    obvious attack string never reaches this code and passes vacuously.

    Rejects absolute paths and drive letters too — `Path("/etc/passwd")` and
    `Path("C:/Windows/...")` both *replace* the left operand of `/` rather
    than joining onto it.
    """
    if not name or "\0" in name:
        return None
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or name.startswith(("/", "\\")):
        return None
    base = export.resolve()
    try:
        target = (base / candidate).resolve()
    except OSError:  # pragma: no cover - malformed names on some platforms
        return None
    if target == base or base not in target.parents:
        return None
    return target if target.is_file() else None


class RateLimiter:
    """Fixed-window request cap, per identity, in memory.

    In memory because this is a single-node service — the same reason the
    queue is SQLite. It is not a defence against a distributed flood (that
    belongs at the proxy); it is there so one misbehaving client, or someone
    grinding at `/keys`, cannot monopolise the process.
    """

    def __init__(self, per_minute: int, window: float = 60.0):
        self.per_minute = per_minute
        self.window = window
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, identity: str, now: Optional[float] = None) -> Optional[float]:
        """None when allowed, else the seconds to wait before retrying."""
        if self.per_minute <= 0:          # 0 disables the limiter entirely
            return None
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.setdefault(identity, deque())
            cutoff = now - self.window
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(0.0, q[0] + self.window - now)
            q.append(now)
            # Bound the table: an idle identity should not be remembered
            # forever just because it once made a request.
            if len(self._hits) > 10_000:
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)
            return None


def sweep_retention(
    root: Path,
    registry: JobRegistry,
    upload_retention_hours: float = UPLOAD_RETENTION_HOURS,
    job_retention_days: float = JOB_RETENTION_DAYS,
    now: Optional[float] = None,
) -> Dict[str, List[str]]:
    """Delete what has aged out. Returns what went, for logging and tests.

    Two separate leaks, both unbounded before this existed:

    * A resumable upload that is started and never finished leaves its bytes
      under `_uploads/` with nothing to ever collect them. A phone that loses
      the network mid-capture does exactly this.
    * A finished job keeps its whole directory — frames, depth maps, COLMAP
      workspace — which is gigabytes. The per-caller job *count* quota does
      not bound that, because 50 jobs of 4 GB is 200 GB.

    Running jobs and unfinished-but-recent uploads are never touched.
    """
    now = time.time() if now is None else now
    removed: Dict[str, List[str]] = {"uploads": [], "jobs": []}

    uploads_root = root / "_uploads"
    if upload_retention_hours > 0 and uploads_root.is_dir():
        cutoff = now - upload_retention_hours * 3600
        for owner_dir in uploads_root.iterdir():
            if not owner_dir.is_dir():
                continue
            for up in owner_dir.iterdir():
                meta = up / "meta.json"
                try:
                    created = json.loads(meta.read_text(encoding="utf-8"))["created_at"]
                except (OSError, ValueError, KeyError):
                    # Unreadable metadata means we cannot date it; fall back to
                    # the directory's own mtime rather than keeping it forever.
                    try:
                        created = up.stat().st_mtime
                    except OSError:
                        continue
                if created < cutoff:
                    shutil.rmtree(up, ignore_errors=True)
                    removed["uploads"].append(up.name)
            # Tidy the owner directory once its last upload is gone.
            try:
                next(owner_dir.iterdir())
            except StopIteration:
                owner_dir.rmdir()
            except OSError:
                pass

    if job_retention_days > 0:
        cutoff = now - job_retention_days * 86400
        for rec in registry.list(limit=100_000):
            if rec.status not in ("done", "failed", "cancelled"):
                continue
            ended = rec.finished_at or rec.created_at
            if ended < cutoff:
                registry.remove(rec.id)
                shutil.rmtree(root / rec.id, ignore_errors=True)
                removed["jobs"].append(rec.id)

    return removed


class _RetentionSweeper:
    """Runs `sweep_retention` on a timer for the life of the app."""

    def __init__(self, root: Path, registry: JobRegistry,
                 interval: float = RETENTION_SWEEP_SECONDS):
        self.root = root
        self.registry = registry
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "_RetentionSweeper":
        if self._thread is not None or self.interval <= 0:
            return self
        self._thread = threading.Thread(target=self._loop, name="retention", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        # Sweep once at startup: a process that was down for a week should not
        # wait another hour before reclaiming the disk.
        first = True
        while first or not self._stop.wait(self.interval):
            first = False
            try:
                gone = sweep_retention(self.root, self.registry)
            except Exception:
                log.exception("retention sweep failed")
                continue
            if gone["uploads"] or gone["jobs"]:
                log.info("retention: removed %d abandoned upload(s), %d expired job(s)",
                         len(gone["uploads"]), len(gone["jobs"]))
            if self._stop.is_set():
                return


def build_app(
    jobs_root: Optional[Path] = None,
    storage: Optional[Storage] = None,
    max_workers: Optional[int] = None,
    runner: Optional[Callable[..., None]] = None,
    auth_mode: Optional[str] = None,
    bootstrap_key: Optional[str] = None,
    rate_limit_per_minute: Optional[int] = None,
    reaper_seconds: Optional[float] = None,
):
    """Construct the FastAPI app.

    A factory rather than a module-level app so tests get an isolated root and
    their own worker pool.

    `runner` is the seam the worker pool calls. Injecting it is how the API is
    tested without running COLMAP — and it is also where a distributed queue
    would slot in, since submitting to a broker is the same signature. The pool
    owns the job's *status*; the runner owns the *work*.

    `auth_mode` is `strict` or `open`; see AUTH_MODE above for what each costs.
    """
    if not FASTAPI_AVAILABLE:  # pragma: no cover - depends on install extras
        raise ModuleNotFoundError(
            "The job service needs FastAPI, which is not in the base install: "
            "`uv sync --extra service`."
        )

    configure_logging()
    mode = (auth_mode or AUTH_MODE).strip().lower()
    if mode not in ("strict", "public", "open"):
        raise ValueError(
            f"CLIPTO3D_AUTH must be 'strict', 'public' or 'open', got {mode!r}")
    # Both closed modes verify keys the same way; they differ only in who is
    # allowed to mint one.
    strict = mode in ("strict", "public")
    self_service = mode == "public"

    root = Path(jobs_root or JOBS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    registry = JobRegistry(root / 'jobs.db', lease_seconds=LEASE_SECONDS)
    keys = KeyStore(registry, open_when_unconfigured=not strict)
    store = storage if storage is not None else (from_uri(STORAGE_URI) if STORAGE_URI else None)
    limiter = RateLimiter(RATE_LIMIT_PER_MINUTE if rate_limit_per_minute is None
                          else rate_limit_per_minute)

    # In strict mode the first key needs a secret that only the operator has.
    # Without one there is a race between the process becoming reachable and
    # the operator issuing key one, and whoever wins it owns the deployment.
    # Generating and logging a secret when none is configured keeps an
    # unconfigured deployment usable without leaving that race open — the
    # bootstrap value is then in the logs, which only the operator can read.
    boot = bootstrap_key if bootstrap_key is not None else BOOTSTRAP_KEY
    generated_boot = False
    if mode == "strict" and not boot and not keys.any_issued():
        boot = new_api_key("boot")
        generated_boot = True

    # Separate budget from the request limiter: signing up is rare and
    # expensive to abuse, so it gets its own much tighter window.
    key_limiter = RateLimiter(KEYS_PER_ADDRESS, window=KEYS_WINDOW_SECONDS)
    shares = ShareStore(registry)

    @asynccontextmanager
    async def lifespan(_app):
        # Startup recovery. A job that was queued or mid-flight when the last
        # process died is unowned now; hand it back to the queue so it is
        # picked up instead of sitting in `running` forever. Stages checkpoint,
        # so a resumed job restarts at the stage it died in.
        stale = registry.requeue_stale()
        if stale:
            log.warning("requeued %d job(s) stranded by the previous process: %s",
                        len(stale), ", ".join(stale))
        if generated_boot:
            log.warning(
                "no CLIPTO3D_BOOTSTRAP_KEY configured; generated one for this "
                "process only: %s\n"
                "    Issue the first API key with:  "
                "curl -XPOST <url>/keys -H 'X-Bootstrap-Key: %s'\n"
                "    Set CLIPTO3D_BOOTSTRAP_KEY to keep it across restarts.",
                boot, boot,
            )
        elif not strict:
            log.warning(
                "auth is OPEN: any request without a configured key is accepted. "
                "Set CLIPTO3D_AUTH=strict before exposing this to a network.")
        pool.start()
        reaper.start()
        retention.start()
        try:
            yield
        finally:
            reaper.stop()
            retention.stop()
            # Let in-flight jobs finish rather than leaving half-written
            # exports; anything still going keeps its lease and is reclaimed.
            pool.shutdown(wait=True, timeout=30)

    app = FastAPI(title="ClipTo3D", version="0.1.0", lifespan=lifespan)

    pool = WorkerPool(
        registry=registry,
        root=root,
        # The same runner a standalone worker process uses, so a split
        # deployment and an all-in-one behave identically.
        runner=runner or pipeline_runner(root, store, registry),
        max_workers=MAX_CONCURRENT if max_workers is None else max_workers,
    )
    reaper = Reaper(
        registry,
        interval=REAPER_SECONDS if reaper_seconds is None else reaper_seconds,
        on_requeue=pool.notify,
    )
    retention = _RetentionSweeper(root, registry)

    app.state.registry = registry
    app.state.keys = keys
    app.state.pool = pool
    app.state.root = root
    app.state.storage = store
    app.state.limiter = limiter
    app.state.auth_mode = mode
    app.state.bootstrap_key = boot

    # Only when a front end is hosted off-origin. The bundled viewer and
    # capture pages are served from here, so the default is no CORS at all
    # rather than a permissive wildcard nobody remembers to tighten.
    if CORS_ORIGINS:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=False,      # auth is a header, not a cookie
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    # The viewer is self-contained by design — no CDN, no external fetches —
    # and this is what keeps it that way at runtime. `unsafe-inline` is
    # unavoidable while the pages carry their own script and style inline;
    # what the policy still buys is that an injected `<script src=…>` pointing
    # at another origin will not load. `connect-src https:` is deliberately
    # open: artifact URLs are presigned against whatever bucket is configured.
    CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob: data:; "
        "worker-src 'self' blob:; "
        "connect-src 'self' https:; "
        "frame-ancestors 'self'; "
        "base-uri 'self'"
    )

    @app.middleware("http")
    async def observe_and_guard(request: "Request", call_next):
        started = time.monotonic()
        path = request.url.path

        # /health is what a load balancer polls; rate-limiting it would take
        # the service out of rotation under exactly the load it should survive.
        if path != "/health":
            retry_after = limiter.check(client_identity(request))
            if retry_after is not None:
                log.warning("rate limited %s on %s", client_identity(request), path)
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )

        response = await call_next(request)

        elapsed = (time.monotonic() - started) * 1000
        # No query string and no key: the log should be useful without being a
        # place credentials accumulate.
        log.info('%s %s -> %d (%.0fms)', request.method, path, response.status_code, elapsed)

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Content-Security-Policy", CSP)
        if TRUST_PROXY or request.url.scheme == "https":
            # Only once TLS is actually in front; asserting HSTS over plain
            # HTTP on localhost just makes the service unreachable later.
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    def client_identity(request: "Request") -> str:
        """Who to charge a request to, before auth has run.

        Prefers the API key so one caller behind a shared NAT cannot spend
        another's budget, and falls back to the peer address. X-Forwarded-For
        is only believed when a trusted proxy is declared — otherwise it is
        client-controlled and the limiter is trivially bypassed.
        """
        key = request.headers.get("x-api-key")
        if key:
            return "k:" + hash_key(key)[:16]
        if TRUST_PROXY:
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return "ip:" + fwd.split(",")[0].strip()
        return "ip:" + (request.client.host if request.client else "unknown")

    def caller(request: "Request", x_api_key: str = Header(default="")) -> str:
        """Per-caller isolation.

        An API key, not an identity system: it is enough to keep one caller's
        jobs invisible to another, which is the property the clients need. Real
        auth (OIDC, per-user accounts) stays a deployment decision.
        """
        if not x_api_key:
            raise HTTPException(status_code=401, detail="X-API-Key header required")
        # The registry only ever sees a digest, so a leaked jobs database does
        # not hand over working credentials.
        digest = hash_key(x_api_key)
        if not keys.verify(digest):
            # Revoked or unknown. Deliberately the same message for both: which
            # one it is tells an attacker whether a key ever existed.
            log.info("rejected key %s… from %s", digest[:12], client_identity(request))
            raise HTTPException(status_code=401, detail="invalid or revoked API key")
        return digest

    def owned(job_id: str, owner: str) -> JobRecord:
        rec = registry.get(job_id)
        # 404 rather than 403 for someone else's job: existence itself should
        # not leak across callers.
        if rec is None or rec.owner != owner:
            raise HTTPException(status_code=404, detail="job not found")
        return rec

    # --- key management ---------------------------------------------------
    # Issuing a key is an admin action, gated on an existing valid key once any
    # key exists. The first key is the interesting case: in strict mode it
    # needs the bootstrap secret, and in open mode anyone can mint it — which
    # is precisely why open mode is not the default.

    @app.post("/keys", status_code=201)
    def create_key(
        request: "Request",
        label: str = "",
        x_api_key: str = Header(default=""),
        x_bootstrap_key: str = Header(default=""),
    ):
        # An operator minting an admin key with the bootstrap secret. Checked
        # first so it works in every mode, including public.
        if boot and x_bootstrap_key and hmac.compare_digest(x_bootstrap_key, boot):
            if keys.any_issued() and not self_service:
                # In strict mode the secret mints key one and then retires, so
                # a leaked bootstrap value is not a permanent skeleton key.
                raise HTTPException(
                    401, "the bootstrap secret is spent; use an existing admin key")
            key = keys.issue(label, admin=True)
            log.info("issued ADMIN key %s… (label=%r)", hash_key(key)[:12], label)
            return {"api_key": key, "label": label, "admin": True,
                    "note": "store this now; only its digest is kept"}

        # An existing key issuing another. Checked before self-service, or an
        # operator's admin key in public mode would fall through to the signup
        # branch and be handed a *non-admin* key back — silently demoting the
        # one credential that can administer the deployment.
        if x_api_key:
            digest = hash_key(x_api_key)
            if not keys.verify(digest):
                raise HTTPException(401, "invalid or revoked API key")
            if not keys.is_admin(digest):
                raise HTTPException(403, "this key is not permitted to issue keys")
            key = keys.issue(label, admin=True)
            log.info("issued ADMIN key %s… (label=%r)", hash_key(key)[:12], label)
            return {"api_key": key, "label": label, "admin": True,
                    "note": "store this now; only its digest is kept and it "
                            "cannot be shown again"}

        if self_service:
            # Self-service signup. The key is non-admin, so it cannot mint
            # more of itself — otherwise one visitor turns into a hundred
            # callers and the per-caller quota stops meaning anything.
            who = client_identity(request)
            wait = key_limiter.check("signup:" + who)
            if wait is not None:
                log.warning("signup rate limited for %s", who)
                raise HTTPException(
                    429, f"too many keys from this address; try again in "
                         f"{int(wait / 60) + 1} minutes",
                    headers={"Retry-After": str(int(wait) + 1)},
                )
            key = keys.issue(label or "self-service", admin=False)
            log.info("self-service key %s… issued to %s", hash_key(key)[:12], who)
            return {
                "api_key": key,
                "label": label or "self-service",
                "admin": False,
                "note": "store this now; only its digest is kept and it cannot be shown again",
            }

        # No key presented. In strict mode that is only allowed for the very
        # first one, and only against the bootstrap secret — which the branch
        # above already checked and rejected if it was wrong.
        if keys.any_issued():
            raise HTTPException(401, "a valid API key is required to issue another")
        if strict:
            raise HTTPException(
                401,
                "the first key needs the bootstrap secret in X-Bootstrap-Key "
                "(set CLIPTO3D_BOOTSTRAP_KEY, or read the one this process "
                "logged at startup)",
            )
        key = keys.issue(label)
        log.info("issued API key %s… (label=%r)", hash_key(key)[:12], label)
        return {
            "api_key": key,
            "label": label,
            # Said once, because it is true: this is the only time the
            # plaintext exists anywhere.
            "note": "store this now; only its digest is kept and it cannot be shown again",
        }

    @app.get("/keys")
    def list_keys(_owner: str = Depends(caller)):
        return {"keys": keys.list()}

    @app.delete("/keys/{key_hash_prefix}", status_code=200)
    def revoke_key(key_hash_prefix: str, owner: str = Depends(caller)):
        matches = [k for k in keys.list() if k["key_hash_prefix"] == key_hash_prefix]
        if not matches:
            raise HTTPException(404, "no such key")
        full = next(
            (h for h in _all_key_hashes(registry) if h.startswith(key_hash_prefix)), None)
        if full is None or not keys.revoke(full):
            raise HTTPException(409, "key is already revoked")
        return {"revoked": key_hash_prefix,
                "self": full == owner,
                "note": "jobs created by this key are retained and still readable by an admin key"}

    def _page(name: str) -> "HTMLResponse":
        """Serve a client page from the same origin as the API.

        Same origin so the browser's fetches carry no CORS problem, and so a
        deployment is one process rather than an API plus a static host. It
        also matters for capture specifically: `getUserMedia` needs a secure
        context, and sharing the API's origin means sharing its TLS.
        """
        page = Path(__file__).resolve().parent / "viewer" / name
        if not page.is_file():  # pragma: no cover - only if the file is missing
            raise HTTPException(404, f"{name} not installed")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/viewer", response_class=HTMLResponse)
    def viewer():
        return _page("index.html")

    @app.get("/capture", response_class=HTMLResponse)
    def capture():
        return _page("capture.html")

    @app.get("/health")
    def health():
        stale = registry.reap_expired()
        return {
            "status": "ok",
            "workers": pool.max_workers,
            "queue_depth": registry.queue_depth(),
            # A job whose worker died holds an expired (or absent) lease.
            # The reaper returns these to the queue; surfacing the count is
            # what makes a worker that keeps dying visible rather than silent.
            "expired_leases": len(stale),
            "auth": mode,
        }

    @app.post("/jobs", status_code=201)
    async def create_job(
        request: "Request",
        video: UploadFile,
        quality: str = "balanced",
        owner: str = Depends(caller),
    ):
        from cli import PRESETS

        if quality not in PRESETS:
            raise HTTPException(400, f"unknown quality {quality!r}; expected {sorted(PRESETS)}")

        # Quotas first: GPU time is the expensive resource, so a caller must
        # not be able to fill the queue before anything is validated.
        if registry.active_for(owner) >= QUOTA_ACTIVE_JOBS:
            raise HTTPException(429, f"quota: at most {QUOTA_ACTIVE_JOBS} active jobs")
        if registry.count_for(owner) >= QUOTA_TOTAL_JOBS:
            raise HTTPException(429, f"quota: at most {QUOTA_TOTAL_JOBS} jobs retained")

        filename = Path(video.filename or "clip.mp4").name
        # Container check before a single byte is read: rejecting a .pdf should
        # not cost the bandwidth of receiving it.
        problem = validate_upload(filename, 1)
        if problem:
            raise HTTPException(400, problem)
        # And refuse an over-budget body on its declared length, so the
        # 500 MB case is turned away at the header rather than after we have
        # already taken it. A client can lie here; the streaming cap below is
        # what actually enforces the limit.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                400, f"upload is {int(declared) / 1e6:.0f} MB; "
                     f"the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB")

        job_id = uuid.uuid4().hex[:16]
        job = JobPaths(root / job_id).ensure()
        dest = job.input_dir / filename

        # Stream to disk in chunks. `await video.read()` materialised the whole
        # clip as one bytes object, so N concurrent 500 MB uploads were N x
        # 500 MB of resident memory — and the size check only ran once all of
        # it had already been accepted.
        written = 0
        try:
            with open(dest, "wb") as fh:
                while chunk := await video.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            400, f"upload exceeds the {MAX_UPLOAD_BYTES / 1e6:.0f} MB limit")
                    fh.write(chunk)
            if written <= 0:
                raise HTTPException(400, "empty upload")
        except BaseException:
            # Rejected, or the disk filled up mid-stream. Either way the
            # directory exists but no registry row does, so nothing would ever
            # look at it again — remove it rather than leaking a job-shaped
            # orphan the retention sweep cannot identify.
            shutil.rmtree(root / job_id, ignore_errors=True)
            raise

        rec = JobRecord(id=job_id, owner=owner, created_at=time.time(),
                        video_name=dest.name, quality=quality)
        registry.add(rec)
        # The row is the queue. A worker picks it up from there, so a crash
        # between here and the first stage loses nothing.
        pool.notify()
        log.info("job %s queued (%s, %.1f MB) for %s…",
                 job_id, quality, written / 1e6, owner[:12])
        return {"id": job_id, "status": rec.status, "quality": quality}

    @app.post("/uploads", status_code=201)
    def create_upload(filename: str, owner: str = Depends(caller)):
        """Hand back a presigned PUT so large videos never stream through here.

        The issue asks for direct-to-storage upload, and this is the half that
        does not depend on which bucket you run: the API mints a key inside the
        caller's namespace and signs it. Without a storage backend configured
        there is nowhere to upload *to*, so it says so rather than pretending.
        """
        if store is None:
            raise HTTPException(
                503,
                "direct upload needs a storage backend; set CLIPTO3D_STORAGE "
                "(e.g. s3://bucket/prefix), or POST the file to /jobs instead",
            )
        problem = validate_upload(filename, 1)
        if problem:
            raise HTTPException(400, problem)

        upload_id = uuid.uuid4().hex[:16]
        key = f"uploads/{upload_id}/{Path(filename).name}"
        url = store.job(owner[:16]).signed_url(key, expires=3600)
        if url is None:
            raise HTTPException(
                503, "the configured storage backend cannot issue presigned URLs")
        return {"upload_id": upload_id, "key": key, "url": url, "method": "PUT",
                "expires_in": 3600}

    # --- resumable upload (MPO-247) ---------------------------------------
    #
    # A phone loses the network mid-upload, or gets backgrounded and has its
    # connection torn down. Re-sending a 200 MB clip from zero is the
    # difference between a client that works on mobile and one that does not.
    #
    # Deliberately append-only against a byte offset the server owns: the
    # client asks where it got to and continues, so a resume can never
    # interleave or duplicate. Same shape as a tus/GCS resumable PUT, without
    # taking the dependency.

    def upload_dir(owner: str, upload_id: str) -> Path:
        d = root / "_uploads" / owner[:16] / upload_id
        return d

    def upload_meta(owner: str, upload_id: str) -> Dict[str, Any]:
        meta = upload_dir(owner, upload_id) / "meta.json"
        if not meta.is_file():
            raise HTTPException(404, "unknown upload")
        return json.loads(meta.read_text(encoding="utf-8"))

    @app.post("/uploads/resumable", status_code=201)
    def start_resumable(filename: str, total: int, owner: str = Depends(caller)):
        # Validate the declared size up front: there is no point accepting
        # 400 MB of chunks only to reject the assembled file.
        problem = validate_upload(filename, total)
        if problem:
            raise HTTPException(400, problem)
        if registry.active_for(owner) >= QUOTA_ACTIVE_JOBS:
            raise HTTPException(429, f"quota: at most {QUOTA_ACTIVE_JOBS} active jobs")

        upload_id = uuid.uuid4().hex[:16]
        d = upload_dir(owner, upload_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({
            "id": upload_id, "filename": Path(filename).name,
            "total": int(total), "created_at": time.time(),
        }), encoding="utf-8")
        (d / "part").write_bytes(b"")
        return {"upload_id": upload_id, "offset": 0, "total": int(total),
                "chunk_size": RESUMABLE_CHUNK_BYTES}

    @app.get("/uploads/{upload_id}")
    def resumable_status(upload_id: str, owner: str = Depends(caller)):
        """Where to resume from. The client asks this after any failure."""
        meta = upload_meta(owner, upload_id)
        part = upload_dir(owner, upload_id) / "part"
        return {"upload_id": upload_id, "offset": part.stat().st_size if part.is_file() else 0,
                "total": meta["total"], "filename": meta["filename"]}

    @app.patch("/uploads/{upload_id}")
    async def resumable_append(
        upload_id: str,
        request: Request,
        offset: int = 0,
        owner: str = Depends(caller),
    ):
        meta = upload_meta(owner, upload_id)
        part = upload_dir(owner, upload_id) / "part"
        have = part.stat().st_size if part.is_file() else 0

        # A chunk that starts before where we are is a retry of data already
        # stored — acknowledge it rather than corrupting the file by appending
        # it twice. A chunk that starts *after* would leave a hole.
        if offset != have:
            return JSONResponse(
                {"upload_id": upload_id, "offset": have, "total": meta["total"],
                 "detail": f"expected offset {have}, got {offset}"},
                status_code=409,
            )

        body = await request.body()
        if have + len(body) > meta["total"]:
            raise HTTPException(400, "chunk would exceed the declared total")
        with open(part, "ab") as fh:
            fh.write(body)

        now = part.stat().st_size
        return {"upload_id": upload_id, "offset": now, "total": meta["total"],
                "complete": now >= meta["total"]}

    @app.post("/uploads/{upload_id}/job", status_code=201)
    def job_from_upload(
        upload_id: str,
        quality: str = "balanced",
        owner: str = Depends(caller),
    ):
        """Turn a completed resumable upload into a job."""
        from cli import PRESETS

        if quality not in PRESETS:
            raise HTTPException(400, f"unknown quality {quality!r}; expected {sorted(PRESETS)}")
        meta = upload_meta(owner, upload_id)
        part = upload_dir(owner, upload_id) / "part"
        have = part.stat().st_size if part.is_file() else 0
        if have != meta["total"]:
            raise HTTPException(
                409, f"upload is incomplete: {have} of {meta['total']} bytes")

        if registry.active_for(owner) >= QUOTA_ACTIVE_JOBS:
            raise HTTPException(429, f"quota: at most {QUOTA_ACTIVE_JOBS} active jobs")
        if registry.count_for(owner) >= QUOTA_TOTAL_JOBS:
            raise HTTPException(429, f"quota: at most {QUOTA_TOTAL_JOBS} jobs retained")

        job_id = uuid.uuid4().hex[:16]
        job = JobPaths(root / job_id).ensure()
        dest = job.input_dir / meta["filename"]
        # Move rather than copy: the bytes are already on this filesystem and
        # a 400 MB copy for no reason is a real cost.
        shutil.move(str(part), str(dest))
        shutil.rmtree(upload_dir(owner, upload_id), ignore_errors=True)

        rec = JobRecord(id=job_id, owner=owner, created_at=time.time(),
                        video_name=dest.name, quality=quality)
        registry.add(rec)
        pool.notify()
        log.info("job %s queued from resumable upload %s", job_id, upload_id)
        return {"id": job_id, "status": rec.status, "quality": quality}

    @app.delete("/uploads/{upload_id}", status_code=204)
    def abandon_upload(upload_id: str, owner: str = Depends(caller)):
        upload_meta(owner, upload_id)          # 404s if it is not the caller's
        shutil.rmtree(upload_dir(owner, upload_id), ignore_errors=True)
        return JSONResponse(status_code=204, content=None)

    @app.get("/jobs")
    def list_jobs(owner: str = Depends(caller)):
        return {"jobs": [
            {"id": r.id, "status": r.status, "created_at": _iso(r.created_at),
             "quality": r.quality, "video": r.video_name}
            for r in registry.list(owner)
        ]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, owner: str = Depends(caller)):
        rec = owned(job_id, owner)
        state = JobState.load(root / job_id)
        return {
            "id": rec.id,
            "status": rec.status,
            "quality": rec.quality,
            "created_at": _iso(rec.created_at),
            "finished_at": _iso(rec.finished_at),
            "error": rec.error,
            # Straight from the pipeline's own state, not a second copy.
            "progress": state.progress(),
            "stages": {
                name: {"status": s.status, "seconds": s.seconds, "message": s.message,
                       "error": s.error}
                for name, s in state.stages.items()
            },
        }

    @app.get("/jobs/{job_id}/events")
    def get_events(job_id: str, limit: int = 100, owner: str = Depends(caller)):
        owned(job_id, owner)
        return {"events": JobState.load(root / job_id).events(limit=limit)}

    @app.get("/jobs/{job_id}/artifacts")
    def get_artifacts(job_id: str, owner: str = Depends(caller)):
        rec = owned(job_id, owner)
        job = JobPaths(root / job_id)
        out = []
        if job.export.is_dir():
            for p in sorted(job.export.iterdir()):
                if not p.is_file():
                    continue
                entry = {"name": p.name, "bytes": p.stat().st_size,
                         # Always resolvable. A signed URL is better when there
                         # is a bucket, but without one the viewer still needs
                         # somewhere to fetch the result from.
                         "url": f"/jobs/{job_id}/artifacts/{p.name}"}
                if store is not None:
                    url = store.job(job_id).signed_url(p.name)
                    if url:
                        entry["url"] = url
                out.append(entry)
        return {"id": rec.id, "status": rec.status, "artifacts": out}

    @app.get("/jobs/{job_id}/artifacts/{name}")
    def get_artifact(request: "Request", job_id: str, name: str,
                     owner: str = Depends(caller)):
        """Serve one artifact.

        Without this a locally-run service can list results but not hand them
        over, so the viewer has nothing to load unless S3 is configured.
        """
        owned(job_id, owner)
        target = safe_artifact_path(JobPaths(root / job_id).export, name)
        if target is None:
            raise HTTPException(404, "artifact not found")
        return artifact_response(request, target)

    # --- share links ------------------------------------------------------
    #
    # A finished reconstruction is something people want to send to someone,
    # and that someone has no API key and must not be given one — a key is a
    # write credential that spends GPU time. A share token is the narrow thing
    # instead: it reads one job's artifacts and does nothing else. No listing
    # jobs, no uploading, no cancelling.
    #
    # `/viewer?job=/shared/<token>/artifacts` works with no key at all, which
    # is the whole point: the viewer's relative-URL path already sends an
    # empty key header, and these routes do not ask for one.

    def shared_job(token: str) -> JobRecord:
        job_id = shares.resolve(token)
        rec = registry.get(job_id) if job_id else None
        # One indistinguishable 404 for unknown, expired and deleted, so a
        # token cannot be probed for which of those it is.
        if rec is None:
            raise HTTPException(404, "no such share link")
        return rec

    @app.post("/jobs/{job_id}/share", status_code=201)
    def create_share(job_id: str, ttl_seconds: float = 0, owner: str = Depends(caller)):
        owned(job_id, owner)
        ttl = ttl_seconds or SHARE_TTL_SECONDS
        token = shares.create(job_id, ttl or None)
        log.info("share link created for job %s (ttl=%s)", job_id, ttl or "none")
        return {
            "token": token,
            "url": f"/shared/{token}",
            "viewer_url": f"/viewer?job=/shared/{token}/artifacts",
            "expires_in": ttl or None,
            "note": "anyone with this link can read this job's results",
        }

    @app.get("/jobs/{job_id}/share")
    def list_shares(job_id: str, owner: str = Depends(caller)):
        owned(job_id, owner)
        return {"shares": shares.list(job_id)}

    @app.delete("/jobs/{job_id}/share", status_code=200)
    def revoke_shares(job_id: str, owner: str = Depends(caller)):
        owned(job_id, owner)
        return {"revoked": shares.revoke_all(job_id)}

    @app.get("/shared/{token}")
    def shared_status(token: str):
        """Enough for the viewer to show progress, and nothing more.

        Deliberately not the owner's view: no error strings, no timings, no
        job id. A share link is for looking at a result, not for auditing how
        it was produced.
        """
        rec = shared_job(token)
        state = JobState.load(root / rec.id)
        return {"status": rec.status, "quality": rec.quality,
                "progress": state.progress()}

    @app.get("/shared/{token}/artifacts")
    def shared_artifacts(token: str):
        rec = shared_job(token)
        job = JobPaths(root / rec.id)
        out = []
        if job.export.is_dir():
            for p in sorted(job.export.iterdir()):
                if not p.is_file():
                    continue
                entry = {"name": p.name, "bytes": p.stat().st_size,
                         "url": f"/shared/{token}/artifacts/{p.name}"}
                if store is not None:
                    url = store.job(rec.id).signed_url(p.name)
                    if url:
                        entry["url"] = url
                out.append(entry)
        return {"status": rec.status, "artifacts": out}

    @app.get("/shared/{token}/artifacts/{name}")
    def shared_artifact(request: "Request", token: str, name: str):
        rec = shared_job(token)
        target = safe_artifact_path(JobPaths(root / rec.id).export, name)
        if target is None:
            raise HTTPException(404, "artifact not found")
        return artifact_response(request, target)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, owner: str = Depends(caller)):
        rec = owned(job_id, owner)
        if rec.status in ("done", "failed", "cancelled"):
            return JSONResponse({"id": job_id, "status": rec.status, "cancelled": False},
                                status_code=409)
        JobState(root=root / job_id).request_cancel(f"requested by {owner}")
        return {"id": job_id, "cancelled": True,
                "note": "the job stops at its next checkpoint; artifacts already written are kept"}

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, owner: str = Depends(caller)):
        rec = owned(job_id, owner)
        if rec.status == "running":
            raise HTTPException(409, "cancel the job before deleting it")
        # Deregister first: a worker that starts between these two lines sees
        # the job is gone and declines to recreate its directory.
        registry.remove(job_id)
        shutil.rmtree(root / job_id, ignore_errors=True)
        return JSONResponse(status_code=204, content=None)

    return app


# uvicorn service:app
def _default_app():  # pragma: no cover - import-time convenience
    return build_app()


# uvicorn imports this. None when FastAPI is absent — `build_app()` then
# raises with the extra to install.
app = build_app() if FASTAPI_AVAILABLE else None  # pragma: no cover
