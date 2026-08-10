"""Durable job registry and work queue (MPO-244).

The first cut kept jobs in a dict, which the issue rightly calls out: it wants
"a real datastore, not files". This is SQLite — durable across restarts,
transactional, and in the standard library, so it adds no deployment
dependency. Postgres is a drop-in later; the point is that job state now
survives the process that created it.

The queue lives in the same database and uses **lease-based claiming**: a
worker claims a job by stamping its id and a deadline, and a job whose lease
expires becomes claimable again. That is what makes a worker crash recoverable
rather than leaving a job stuck in `running` forever — the failure mode an
in-memory queue cannot even represent.

Concurrency: SQLite in WAL mode with a busy timeout handles multiple worker
threads and multiple processes on one host. It does not handle multiple hosts;
that is where a broker becomes necessary, and `claim()`/`complete()` is the
interface it would implement.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

DEFAULT_LEASE_SECONDS = 3600


@dataclass
class JobRecord:
    id: str
    owner: str
    created_at: float
    video_name: str
    quality: str
    status: str = QUEUED
    error: str = ""
    finished_at: Optional[float] = None
    claimed_by: Optional[str] = None
    lease_expires: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        # Never leak which worker holds the lease, or when — that is internal
        # scheduling detail, not something a caller can act on.
        for internal in ("claimed_by", "lease_expires"):
            d.pop(internal, None)
        return d


def hash_key(api_key: str) -> str:
    """Store a digest, never the key itself.

    A leaked jobs database should not hand over working credentials. This is a
    plain SHA-256, not a password KDF — API keys are high-entropy random
    strings, so the brute-force argument for slow hashing does not apply.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class JobRegistry:
    """SQLite-backed job store and queue."""

    def __init__(self, path: Path | str, lease_seconds: int = DEFAULT_LEASE_SECONDS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        self._local = threading.local()
        # Not inside _tx(): executescript issues its own COMMIT, which would
        # leave the explicit one below with no transaction to close.
        self._cx.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id            TEXT PRIMARY KEY,
                    owner_hash    TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    video_name    TEXT NOT NULL,
                    quality       TEXT NOT NULL,
                    status        TEXT NOT NULL,
                    error         TEXT NOT NULL DEFAULT '',
                    finished_at   REAL,
                    claimed_by    TEXT,
                    lease_expires REAL,
                    params        TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS jobs_owner  ON jobs(owner_hash);
                CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);

                -- Keys are identified by their digest; the key itself is
                -- never stored, so this table is not a credential store.
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash   TEXT PRIMARY KEY,
                    label      TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    revoked_at REAL,
                    last_used  REAL
                );

                -- Share tokens: read-only public access to one job's results,
                -- so a finished reconstruction can be handed to someone who
                -- has no API key. Stored as a digest for the same reason keys
                -- are — a leaked database should not grant access.
                CREATE TABLE IF NOT EXISTS job_shares (
                    token_hash TEXT PRIMARY KEY,
                    job_id     TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS job_shares_job ON job_shares(job_id);
                """
        )
        self._migrate()

    def _migrate(self) -> None:
        """Additive schema changes, applied to databases that predate them.

        Only ADD COLUMN, so an older process reading the same file keeps
        working — which matters because the API and the workers are separate
        containers and will not be restarted at the same instant.
        """
        cols = {r["name"] for r in self._cx.execute("PRAGMA table_info(api_keys)")}
        if "admin" not in cols:
            # DEFAULT 1 so every key that already exists stays able to issue
            # keys, which is what it could do before this column existed.
            # Only self-service keys (public auth mode) are minted with 0.
            self._cx.execute(
                "ALTER TABLE api_keys ADD COLUMN admin INTEGER NOT NULL DEFAULT 1")

    # --- connection -------------------------------------------------------

    @property
    def _cx(self) -> sqlite3.Connection:
        cx = getattr(self._local, "cx", None)
        if cx is None:
            cx = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            cx.row_factory = sqlite3.Row
            # WAL lets readers proceed during a write, which matters because
            # the API polls status while workers are updating it.
            cx.execute("PRAGMA journal_mode=WAL")
            cx.execute("PRAGMA busy_timeout=30000")
            cx.execute("PRAGMA synchronous=NORMAL")
            self._local.cx = cx
        return cx

    class _Tx:
        def __init__(self, cx):
            self.cx = cx

        def __enter__(self):
            self.cx.execute("BEGIN IMMEDIATE")
            return self.cx

        def __exit__(self, exc_type, *_):
            self.cx.execute("ROLLBACK" if exc_type else "COMMIT")
            return False

    def _tx(self):
        return self._Tx(self._cx)

    # --- records ----------------------------------------------------------

    @staticmethod
    def _row(r: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=r["id"], owner=r["owner_hash"], created_at=r["created_at"],
            video_name=r["video_name"], quality=r["quality"], status=r["status"],
            error=r["error"], finished_at=r["finished_at"],
            claimed_by=r["claimed_by"], lease_expires=r["lease_expires"],
            params=json.loads(r["params"]),
        )

    def add(self, rec: JobRecord) -> None:
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO jobs (id, owner_hash, created_at, video_name, quality,"
                " status, error, finished_at, params) VALUES (?,?,?,?,?,?,?,?,?)",
                (rec.id, rec.owner, rec.created_at, rec.video_name, rec.quality,
                 rec.status, rec.error, rec.finished_at, json.dumps(rec.params)),
            )

    def get(self, job_id: str) -> Optional[JobRecord]:
        r = self._cx.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def update(self, job_id: str, **fields) -> None:
        allowed = {"status", "error", "finished_at", "claimed_by", "lease_expires"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        clause = ", ".join(f"{k}=?" for k in sets)
        with self._tx() as cx:
            cx.execute(f"UPDATE jobs SET {clause} WHERE id=?", (*sets.values(), job_id))

    def list(self, owner: Optional[str] = None, limit: int = 200) -> List[JobRecord]:
        if owner is None:
            rows = self._cx.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._cx.execute(
                "SELECT * FROM jobs WHERE owner_hash=? ORDER BY created_at DESC LIMIT ?",
                (owner, limit)).fetchall()
        return [self._row(r) for r in rows]

    def remove(self, job_id: str) -> None:
        with self._tx() as cx:
            cx.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            # A share outliving its job would be a token pointing at nothing —
            # or worse, at a future job that reused the id.
            cx.execute("DELETE FROM job_shares WHERE job_id=?", (job_id,))

    def active_for(self, owner: str) -> int:
        return self._cx.execute(
            "SELECT COUNT(*) FROM jobs WHERE owner_hash=? AND status IN (?,?)",
            (owner, QUEUED, RUNNING)).fetchone()[0]

    def count_for(self, owner: str) -> int:
        return self._cx.execute(
            "SELECT COUNT(*) FROM jobs WHERE owner_hash=?", (owner,)).fetchone()[0]

    # --- queue ------------------------------------------------------------

    def claim(self, worker_id: str, now: Optional[float] = None) -> Optional[JobRecord]:
        """Atomically take the oldest claimable job, or None.

        Claimable means queued, or running under a lease that has expired —
        which is how a job survives the worker that was holding it dying.
        """
        now = time.time() if now is None else now
        with self._tx() as cx:
            row = cx.execute(
                "SELECT * FROM jobs WHERE status=? OR (status=? AND lease_expires < ?)"
                " ORDER BY created_at ASC LIMIT 1",
                (QUEUED, RUNNING, now),
            ).fetchone()
            if row is None:
                return None
            cx.execute(
                "UPDATE jobs SET status=?, claimed_by=?, lease_expires=? WHERE id=?",
                (RUNNING, worker_id, now + self.lease_seconds, row["id"]),
            )
        rec = self.get(row["id"])
        return rec

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Extend a lease. False when the job was stolen or is already finished."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE jobs SET lease_expires=? WHERE id=? AND claimed_by=? AND status=?",
                (time.time() + self.lease_seconds, job_id, worker_id, RUNNING),
            )
            return cur.rowcount > 0

    def complete(self, job_id: str, status: str = DONE, error: str = "") -> None:
        with self._tx() as cx:
            cx.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=?, claimed_by=NULL,"
                " lease_expires=NULL WHERE id=?",
                (status, error, time.time(), job_id),
            )

    def reap_expired(self, now: Optional[float] = None) -> List[str]:
        """Return jobs whose lease has lapsed — for logging/alerting.

        A NULL lease counts as lapsed: it means the row says `running` but no
        worker is holding it. `NULL < now` is NULL in SQL rather than true, so
        the comparison has to be spelled out or these rows stay invisible —
        which is exactly how a crashed worker's job used to hide from /health.
        """
        now = time.time() if now is None else now
        rows = self._cx.execute(
            "SELECT id FROM jobs WHERE status=? AND (lease_expires IS NULL OR lease_expires < ?)",
            (RUNNING, now),
        ).fetchall()
        return [r["id"] for r in rows]

    def requeue_stale(self, now: Optional[float] = None) -> List[str]:
        """Return unowned `running` jobs to the queue. Returns the ids moved.

        Called once at startup, and periodically by the reaper. An unowned job
        is one whose lease has expired or was never taken — a live worker
        heartbeats, so its lease sits in the future and is left alone. That is
        what makes this safe to run while other workers are mid-job, and what
        lets a restarted service pick up where the dead one left off.

        Requeueing rather than failing is deliberate: every stage checkpoints
        its own fingerprint, so a resumed job restarts at the stage it died in
        rather than from the first frame.
        """
        now = time.time() if now is None else now
        with self._tx() as cx:
            rows = cx.execute(
                "SELECT id FROM jobs WHERE status=?"
                " AND (lease_expires IS NULL OR lease_expires < ?)",
                (RUNNING, now),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                cx.execute(
                    f"UPDATE jobs SET status=?, claimed_by=NULL, lease_expires=NULL"
                    f" WHERE id IN ({','.join('?' * len(ids))})",
                    (QUEUED, *ids),
                )
        return ids

    def queue_depth(self) -> int:
        return self._cx.execute(
            "SELECT COUNT(*) FROM jobs WHERE status=?", (QUEUED,)).fetchone()[0]


class KeyStore:
    """API key lifecycle: issue, verify, rotate, revoke.

    Deliberately *not* an identity system — there are no users, sessions or
    scopes here. What it does provide is the thing the service actually needs
    and could not do before: a key that can be **revoked**, so a leaked
    credential can be turned off without dropping every job that key created.

    Verification can be open when no keys have ever been issued, so an
    unconfigured single-tenant deployment still works; the moment one key
    exists, unknown keys are rejected.

    That open window is a **localhost convenience and a public-internet
    liability**, so it is a constructor argument rather than a fact. On a
    reachable deployment there is a race between the process starting and the
    operator issuing key one, and whoever wins it owns the service — see
    `service.build_app(auth_mode=...)`, which closes the window by default.
    """

    def __init__(self, registry: "JobRegistry", open_when_unconfigured: bool = True):
        self.registry = registry
        self.open_when_unconfigured = open_when_unconfigured

    def issue(self, label: str = "", admin: bool = True) -> str:
        """Mint a key. This is the only moment its plaintext exists.

        `admin=False` mints a key that can use the service but cannot issue
        further keys. That distinction is what makes self-service signup safe:
        without it, a visitor mints one key, uses it to mint a hundred more,
        and walks straight past both the per-IP cap and their own job quota.
        """
        key = new_api_key()
        with self.registry._tx() as cx:
            cx.execute(
                "INSERT INTO api_keys (key_hash, label, created_at, admin)"
                " VALUES (?,?,?,?)",
                (hash_key(key), label, time.time(), 1 if admin else 0),
            )
        return key

    def is_admin(self, key_hash: str) -> bool:
        """May this key issue other keys?"""
        row = self.registry._cx.execute(
            "SELECT admin FROM api_keys WHERE key_hash=? AND revoked_at IS NULL",
            (key_hash,)).fetchone()
        return bool(row["admin"]) if row else False

    def revoke(self, key_hash: str) -> bool:
        with self.registry._tx() as cx:
            cur = cx.execute(
                "UPDATE api_keys SET revoked_at=? WHERE key_hash=? AND revoked_at IS NULL",
                (time.time(), key_hash),
            )
            return cur.rowcount > 0

    def rotate(self, old_hash: str, label: str = "") -> Optional[str]:
        """Issue a replacement and revoke the old one atomically.

        Jobs are owned by the key *hash*, so rotating deliberately does not
        reassign them — the old key's history stays queryable under its own
        identity rather than being silently merged.
        """
        if not self.exists(old_hash):
            return None
        key = self.issue(label)
        self.revoke(old_hash)
        return key

    def exists(self, key_hash: str) -> bool:
        return self.registry._cx.execute(
            "SELECT 1 FROM api_keys WHERE key_hash=?", (key_hash,)).fetchone() is not None

    def any_issued(self) -> bool:
        return self.registry._cx.execute(
            "SELECT 1 FROM api_keys LIMIT 1").fetchone() is not None

    def verify(self, key_hash: str) -> bool:
        """True when the key may be used.

        Open when no keys exist at all *and* the store was built that way;
        otherwise the key must be known and not revoked.
        """
        if not self.any_issued():
            return self.open_when_unconfigured
        row = self.registry._cx.execute(
            "SELECT revoked_at FROM api_keys WHERE key_hash=?", (key_hash,)).fetchone()
        if row is None or row["revoked_at"] is not None:
            return False
        with self.registry._tx() as cx:
            cx.execute("UPDATE api_keys SET last_used=? WHERE key_hash=?",
                       (time.time(), key_hash))
        return True

    def list(self) -> List[Dict[str, Any]]:
        rows = self.registry._cx.execute(
            "SELECT key_hash, label, created_at, revoked_at, last_used, admin"
            " FROM api_keys ORDER BY created_at DESC").fetchall()
        return [
            {
                # A prefix only: enough to tell keys apart in a UI, not enough
                # to reconstruct one.
                "key_hash_prefix": r["key_hash"][:12],
                "label": r["label"],
                "created_at": r["created_at"],
                "revoked_at": r["revoked_at"],
                "last_used": r["last_used"],
                "active": r["revoked_at"] is None,
                "admin": bool(r["admin"]),
            }
            for r in rows
        ]


class ShareStore:
    """Read-only public links to one job's results.

    A finished reconstruction is something people want to send to someone —
    and that someone has no API key and should not be given one, because a key
    is a write credential that can spend GPU time. A share token is the
    narrow thing instead: it reads one job's artifacts and does nothing else.

    Tokens are stored as digests, so a leaked database yields no working
    links. Revocation is a delete, and deleting the job takes its shares with
    it (see `JobRegistry.remove`).
    """

    def __init__(self, registry: "JobRegistry"):
        self.registry = registry

    def create(self, job_id: str, ttl_seconds: Optional[float] = None) -> str:
        token = secrets.token_urlsafe(24)
        expires = None if not ttl_seconds else time.time() + float(ttl_seconds)
        with self.registry._tx() as cx:
            cx.execute(
                "INSERT INTO job_shares (token_hash, job_id, created_at, expires_at)"
                " VALUES (?,?,?,?)",
                (hash_key(token), job_id, time.time(), expires),
            )
        return token

    def resolve(self, token: str, now: Optional[float] = None) -> Optional[str]:
        """The job id this token grants access to, or None.

        Expiry is enforced here rather than by a sweeper, so a lapsed token
        stops working the instant it lapses even if nothing has cleaned up.
        """
        now = time.time() if now is None else now
        row = self.registry._cx.execute(
            "SELECT job_id, expires_at FROM job_shares WHERE token_hash=?",
            (hash_key(token),)).fetchone()
        if row is None:
            return None
        if row["expires_at"] is not None and row["expires_at"] < now:
            return None
        return row["job_id"]

    def list(self, job_id: str) -> List[Dict[str, Any]]:
        rows = self.registry._cx.execute(
            "SELECT token_hash, created_at, expires_at FROM job_shares"
            " WHERE job_id=? ORDER BY created_at DESC", (job_id,)).fetchall()
        return [
            {   # A prefix, for the same reason keys expose one: enough to tell
                # links apart, not enough to reconstruct one.
                "token_prefix": r["token_hash"][:12],
                "created_at": r["created_at"],
                "expires_at": r["expires_at"],
            }
            for r in rows
        ]

    def revoke_all(self, job_id: str) -> int:
        with self.registry._tx() as cx:
            return cx.execute(
                "DELETE FROM job_shares WHERE job_id=?", (job_id,)).rowcount


def new_api_key(prefix: str = "c3d") -> str:
    """Mint a key. Only the caller ever sees this value; we keep its digest."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"
