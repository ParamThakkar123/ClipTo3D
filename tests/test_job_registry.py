"""Durable job registry and lease-based queue (MPO-244).

The properties worth testing are the ones an in-memory dict could not have:
survival across a restart, atomic claiming under concurrency, and recovery of
a job whose worker died mid-run.
"""

import threading
import time

import pytest

from job_registry import (
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    JobRecord,
    JobRegistry,
    hash_key,
    new_api_key,
)


@pytest.fixture
def reg(tmp_path):
    return JobRegistry(tmp_path / "jobs.db", lease_seconds=60)


def make(reg, job_id="j1", owner="owner-hash", status=QUEUED, created=None):
    rec = JobRecord(id=job_id, owner=owner, created_at=created or time.time(),
                    video_name="clip.mp4", quality="balanced", status=status)
    reg.add(rec)
    return rec


# --- durability -----------------------------------------------------------

def test_jobs_survive_a_restart(tmp_path):
    """The point of using a datastore rather than a dict."""
    path = tmp_path / "jobs.db"
    r1 = JobRegistry(path)
    make(r1, "j1", owner="alice")
    r1.complete("j1", status=DONE)

    r2 = JobRegistry(path)                 # a fresh process would do exactly this
    rec = r2.get("j1")
    assert rec is not None
    assert rec.status == DONE
    assert rec.video_name == "clip.mp4"


def test_listing_is_scoped_to_the_owner(reg):
    make(reg, "a", owner="alice")
    make(reg, "b", owner="bob")
    assert [r.id for r in reg.list("alice")] == ["a"]
    assert reg.count_for("alice") == 1


def test_quota_counters(reg):
    make(reg, "a", owner="alice", status=QUEUED)
    make(reg, "b", owner="alice", status=RUNNING)
    make(reg, "c", owner="alice", status=DONE)
    assert reg.active_for("alice") == 2
    assert reg.count_for("alice") == 3


def test_update_ignores_unknown_columns(reg):
    """A typo'd field must not become a silent SQL injection surface."""
    make(reg, "j1")
    reg.update("j1", status=FAILED, bogus="x'; DROP TABLE jobs; --")
    assert reg.get("j1").status == FAILED


# --- the queue ------------------------------------------------------------

def test_claim_takes_the_oldest_first(reg):
    make(reg, "old", created=1000.0)
    make(reg, "new", created=2000.0)
    assert reg.claim("worker-1").id == "old"
    assert reg.claim("worker-1").id == "new"
    assert reg.claim("worker-1") is None


def test_claim_marks_running_and_stamps_a_lease(reg):
    make(reg, "j1")
    rec = reg.claim("worker-7")
    assert rec.status == RUNNING
    assert rec.claimed_by == "worker-7"
    assert rec.lease_expires > time.time()


def test_a_claimed_job_is_not_claimed_twice(reg):
    make(reg, "j1")
    assert reg.claim("w1") is not None
    assert reg.claim("w2") is None, "two workers took the same job"


def test_expired_lease_makes_a_job_claimable_again(reg):
    """A worker that died must not strand its job in `running` forever."""
    make(reg, "j1")
    reg.claim("dead-worker")
    reg.update("j1", lease_expires=time.time() - 1)

    assert reg.reap_expired() == ["j1"]
    recovered = reg.claim("fresh-worker")
    assert recovered is not None and recovered.claimed_by == "fresh-worker"


def test_heartbeat_extends_only_your_own_lease(reg):
    make(reg, "j1")
    reg.claim("w1")
    assert reg.heartbeat("j1", "w1") is True
    assert reg.heartbeat("j1", "someone-else") is False


def test_heartbeat_on_a_finished_job_fails(reg):
    make(reg, "j1")
    reg.claim("w1")
    reg.complete("j1")
    assert reg.heartbeat("j1", "w1") is False


def test_complete_clears_the_lease(reg):
    make(reg, "j1")
    reg.claim("w1")
    reg.complete("j1", status=DONE)
    rec = reg.get("j1")
    assert rec.status == DONE and rec.claimed_by is None and rec.lease_expires is None
    assert rec.finished_at is not None


def test_concurrent_claims_never_double_assign(tmp_path):
    """The property a dict-plus-lock gets right by accident and a database
    must get right on purpose."""
    path = tmp_path / "jobs.db"
    seed = JobRegistry(path)
    for i in range(40):
        make(seed, f"j{i:03d}", created=1000.0 + i)

    claimed, errors = [], []
    lock = threading.Lock()

    def worker(wid):
        # Each thread gets its own registry, as separate workers would.
        r = JobRegistry(path)
        try:
            while True:
                rec = r.claim(f"w{wid}")
                if rec is None:
                    return
                with lock:
                    claimed.append(rec.id)
        except Exception as exc:       # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(claimed) == 40, f"claimed {len(claimed)} of 40"
    assert len(set(claimed)) == 40, "a job was claimed by more than one worker"


def test_queue_depth(reg):
    make(reg, "a")
    make(reg, "b")
    reg.claim("w1")
    assert reg.queue_depth() == 1


# --- API keys -------------------------------------------------------------

def test_keys_are_stored_only_as_digests():
    key = new_api_key()
    digest = hash_key(key)
    assert digest != key
    assert len(digest) == 64
    assert key not in digest


def test_hash_is_stable_and_distinct():
    assert hash_key("abc") == hash_key("abc")
    assert hash_key("abc") != hash_key("abd")


def test_new_keys_are_unique_and_prefixed():
    keys = {new_api_key() for _ in range(200)}
    assert len(keys) == 200
    assert all(k.startswith("c3d_") for k in keys)


def test_public_record_hides_scheduling_internals(reg):
    make(reg, "j1")
    reg.claim("worker-1")
    public = reg.get("j1").public()
    assert "claimed_by" not in public and "lease_expires" not in public


# --- key lifecycle --------------------------------------------------------

def test_verify_is_open_until_a_key_exists(reg):
    """An unconfigured single-tenant deployment must still work."""
    from job_registry import KeyStore

    ks = KeyStore(reg)
    assert ks.verify(hash_key("anything")) is True

    ks.issue("first")
    assert ks.verify(hash_key("anything")) is False


def test_the_open_window_can_be_closed(reg):
    """What a hosted deployment runs.

    The open window is a localhost convenience: on a reachable address there
    is a race between the process starting and the operator issuing key one,
    and whoever wins it owns the service. `service.build_app` closes it by
    default and issues the first key against a bootstrap secret instead.
    """
    from job_registry import KeyStore

    ks = KeyStore(reg, open_when_unconfigured=False)
    assert ks.verify(hash_key("i-got-here-first")) is False

    key = ks.issue("admin")
    assert ks.verify(hash_key(key)) is True
    assert ks.verify(hash_key("still-not-me")) is False


def test_issued_key_verifies_then_stops_after_revocation(reg):
    from job_registry import KeyStore

    ks = KeyStore(reg)
    key = ks.issue("ci")
    digest = hash_key(key)

    assert ks.verify(digest) is True
    assert ks.revoke(digest) is True
    assert ks.verify(digest) is False


def test_revoking_twice_reports_no_change(reg):
    from job_registry import KeyStore

    ks = KeyStore(reg)
    digest = hash_key(ks.issue())
    assert ks.revoke(digest) is True
    assert ks.revoke(digest) is False


def test_rotate_issues_a_replacement_and_kills_the_old(reg):
    from job_registry import KeyStore

    ks = KeyStore(reg)
    old = ks.issue("original")
    new = ks.rotate(hash_key(old), "rotated")

    assert new is not None and new != old
    assert ks.verify(hash_key(old)) is False
    assert ks.verify(hash_key(new)) is True


def test_rotating_an_unknown_key_returns_none(reg):
    from job_registry import KeyStore

    ks = KeyStore(reg)
    ks.issue()
    assert ks.rotate(hash_key("never-issued")) is None


def test_listing_exposes_only_a_digest_prefix(reg):
    from job_registry import KeyStore

    ks = KeyStore(reg)
    key = ks.issue("ci")
    entry = ks.list()[0]

    assert key not in str(entry)
    assert hash_key(key) not in str(entry), "full digest is enough to revoke the key"
    assert entry["key_hash_prefix"] == hash_key(key)[:12]
    assert entry["active"] is True


def test_last_used_is_recorded_on_verify(reg):
    from job_registry import KeyStore

    ks = KeyStore(reg)
    key = ks.issue()
    assert ks.list()[0]["last_used"] is None
    ks.verify(hash_key(key))
    assert ks.list()[0]["last_used"] is not None


def test_keys_survive_a_restart(tmp_path):
    from job_registry import KeyStore

    path = tmp_path / "jobs.db"
    key = KeyStore(JobRegistry(path)).issue("persistent")
    assert KeyStore(JobRegistry(path)).verify(hash_key(key)) is True
