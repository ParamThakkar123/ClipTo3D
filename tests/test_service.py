"""Job service API (MPO-244).

Runs against FastAPI's TestClient with the pipeline stubbed, so what is under
test is the service — isolation between callers, quotas, validation, status
reporting and cancellation — not the reconstruction.
"""

import threading
import time

import pytest

pytest.importorskip("fastapi", reason="needs the service extra: uv sync --extra service")

from fastapi.testclient import TestClient  # noqa: E402

import service  # noqa: E402
from job_state import JobState  # noqa: E402

ALICE = {"X-API-Key": "alice-key"}
BOB = {"X-API-Key": "bob-key"}


def stub_runner(root):
    """Stand in for the reconstruction, writing the artifacts it would produce.

    Injected rather than monkeypatched: jobs run on a worker thread and can
    outlive the test that started them, so a patch would already be undone by
    the time the real pipeline was invoked — which is how COLMAP and ffmpeg
    ended up actually running during an API test.
    """
    def run(job_id, video, quality, params):
        from job_paths import JobPaths
        from job_state import JobState as JS

        job = JobPaths(root / job_id).ensure()
        st = JS.load(job.root)
        for stage in ("frames", "depth", "colmap", "fuse", "export"):
            st.start(stage, "fp")
            st.finish(stage, "stubbed")
        (job.export / "cloud.glb").write_bytes(b"glb-bytes")
    return run


def make_app(root, **kw):
    """An app in the localhost auth mode the bulk of these tests assume.

    `auth_mode="open"` is what the fixed API keys below rely on: no key is ever
    issued, so verification has to be open for `alice-key` to mean anything.
    Strict mode — the default, and what a hosted deployment runs — has its own
    tests further down.
    """
    kw.setdefault("auth_mode", "open")
    kw.setdefault("max_workers", 1)
    kw.setdefault("runner", stub_runner(root))
    # Off by default: a fixed-window limiter shared across a test session turns
    # unrelated tests into each other's noisy neighbours.
    kw.setdefault("rate_limit_per_minute", 0)
    return service.build_app(jobs_root=root, **kw)


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "runs"
    app = make_app(root)
    with TestClient(app) as c:
        yield c
    # Drain before the test's tmp_path is torn down. A worker still writing
    # into a directory pytest is deleting fails with PermissionError on
    # Windows, which shows up as an unrelated test flaking.
    app.state.pool.shutdown(wait=True)


def settle(client, tries=200):
    """Wait until nothing is queued or running.

    Jobs now run on a worker that claims from the registry, so a test that
    wants to drive a job's status by hand has to wait for the pool to be done
    with it first — otherwise the worker's `complete()` lands on top of
    whatever the test just wrote.
    """
    reg = client.app.state.registry
    for _ in range(tries):
        if not any(r.status in ("queued", "running") for r in reg.list()):
            return
        time.sleep(0.02)
    raise AssertionError("jobs never settled")


def upload(client, headers=ALICE, name="clip.mp4", data=b"x" * 2048, **params):
    return client.post("/jobs", headers=headers,
                       files={"video": (name, data, "video/mp4")}, params=params)


# --- the happy path -------------------------------------------------------

def test_post_poll_and_fetch_artifacts(client):
    r = upload(client)
    assert r.status_code == 201
    job_id = r.json()["id"]

    status = client.get(f"/jobs/{job_id}", headers=ALICE)
    assert status.status_code == 200
    body = status.json()
    assert body["status"] in ("queued", "running", "done")

    arts = client.get(f"/jobs/{job_id}/artifacts", headers=ALICE)
    assert arts.status_code == 200


def test_status_reports_per_stage_progress(client, tmp_path):
    job_id = upload(client).json()["id"]
    # The worker writes stage state too, so wait for it to be done before
    # writing our own — otherwise its `finished` lands on top of the
    # `running` this test is asserting.
    settle(client)
    # Progress comes from the job's own state file, so write one directly.
    st = JobState(root=tmp_path / "runs" / job_id)
    st.start("colmap", "fp")
    st.emit("colmap", 0.4, "mapping")

    body = client.get(f"/jobs/{job_id}", headers=ALICE).json()
    assert body["stages"]["colmap"]["status"] == "running"
    assert body["progress"]["running"] == "colmap"

    events = client.get(f"/jobs/{job_id}/events", headers=ALICE).json()["events"]
    assert any(e["message"] == "mapping" and e["fraction"] == 0.4 for e in events)


# --- isolation ------------------------------------------------------------

def test_callers_cannot_see_each_others_jobs(client):
    a = upload(client, ALICE).json()["id"]
    b = upload(client, BOB).json()["id"]

    assert [j["id"] for j in client.get("/jobs", headers=ALICE).json()["jobs"]] == [a]
    assert [j["id"] for j in client.get("/jobs", headers=BOB).json()["jobs"]] == [b]


def test_another_callers_job_is_404_not_403(client):
    """403 would confirm the job exists; existence must not leak."""
    a = upload(client, ALICE).json()["id"]
    assert client.get(f"/jobs/{a}", headers=BOB).status_code == 404
    assert client.post(f"/jobs/{a}/cancel", headers=BOB).status_code == 404
    assert client.delete(f"/jobs/{a}", headers=BOB).status_code == 404


def test_api_key_is_required(client):
    assert client.get("/jobs").status_code == 401
    assert client.post("/jobs", files={"video": ("c.mp4", b"x", "video/mp4")}).status_code == 401


# --- validation -----------------------------------------------------------

@pytest.mark.parametrize("name", ["clip.txt", "clip.pdf", "clip"])
def test_unsupported_containers_are_rejected(client, name):
    r = upload(client, name=name)
    assert r.status_code == 400
    assert "unsupported container" in r.json()["detail"]


def test_empty_upload_is_rejected(client):
    assert upload(client, data=b"").status_code == 400


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 1024)
    r = upload(client, data=b"x" * 4096)
    assert r.status_code == 400 and "limit" in r.json()["detail"]


def test_unknown_quality_is_rejected(client):
    r = upload(client, quality="ultra")
    assert r.status_code == 400 and "unknown quality" in r.json()["detail"]


def test_validate_upload_accepts_normal_clips():
    assert service.validate_upload("clip.mp4", 5_000_000) is None
    assert service.validate_upload("clip.MOV", 5_000_000) is None


# --- quotas ---------------------------------------------------------------

def test_active_job_quota(client, monkeypatch):
    """GPU time is the expensive resource; one caller must not fill the queue."""
    monkeypatch.setattr(service, "QUOTA_ACTIVE_JOBS", 1)

    # Keep the first job occupying its slot.
    reg = client.app.state.registry
    first = upload(client).json()["id"]
    settle(client)                    # the worker is done with it; now we own the status
    reg.update(first, status="running")

    r = upload(client)
    assert r.status_code == 429 and "active jobs" in r.json()["detail"]

    reg.update(first, status="done")
    assert upload(client).status_code == 201


def test_total_job_quota(client, monkeypatch):
    monkeypatch.setattr(service, "QUOTA_TOTAL_JOBS", 2)
    upload(client); upload(client)
    assert upload(client).status_code == 429


def test_quota_is_per_caller(client, monkeypatch):
    monkeypatch.setattr(service, "QUOTA_TOTAL_JOBS", 1)
    assert upload(client, ALICE).status_code == 201
    assert upload(client, ALICE).status_code == 429
    assert upload(client, BOB).status_code == 201, "Bob was charged for Alice's usage"


# --- cancellation and deletion -------------------------------------------

def test_cancel_writes_the_sentinel(client, tmp_path):
    job_id = upload(client).json()["id"]
    settle(client)
    client.app.state.registry.update(job_id, status="running")

    r = client.post(f"/jobs/{job_id}/cancel", headers=ALICE)
    assert r.status_code == 200 and r.json()["cancelled"] is True
    assert JobState(root=tmp_path / "runs" / job_id).cancel_requested()


def test_cancelling_a_finished_job_is_a_conflict(client):
    job_id = upload(client).json()["id"]
    settle(client)
    client.app.state.registry.update(job_id, status="done")
    assert client.post(f"/jobs/{job_id}/cancel", headers=ALICE).status_code == 409


def test_delete_removes_the_job_and_its_files(client, tmp_path):
    job_id = upload(client).json()["id"]
    # Let the worker finish before deleting, so this tests deletion rather
    # than the delete-vs-worker race (which has its own test below).
    settle(client)
    client.app.state.registry.update(job_id, status="done")

    assert client.delete(f"/jobs/{job_id}", headers=ALICE).status_code == 204
    assert not (tmp_path / "runs" / job_id).exists()
    assert client.get(f"/jobs/{job_id}", headers=ALICE).status_code == 404


def test_running_job_must_be_cancelled_before_deletion(client):
    job_id = upload(client).json()["id"]
    settle(client)
    client.app.state.registry.update(job_id, status="running")
    assert client.delete(f"/jobs/{job_id}", headers=ALICE).status_code == 409


def test_missing_job_is_404(client):
    assert client.get("/jobs/nope", headers=ALICE).status_code == 404


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["workers"] >= 1


# --- artifacts via storage ------------------------------------------------

def test_artifacts_include_signed_urls_when_storage_provides_them(tmp_path):
    class FakeStore:
        def job(self, job_id):
            class J:
                def signed_url(self, key, expires=3600):
                    return f"https://cdn.example/{job_id}/{key}"

                def upload_tree(self, *a, **k):
                    return []
            return J()

    root = tmp_path / "runs"
    app = make_app(root, storage=FakeStore())
    with TestClient(app) as c:
        job_id = c.post("/jobs", headers=ALICE,
                        files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).json()["id"]
        from job_paths import JobPaths

        job = JobPaths(tmp_path / "runs" / job_id).ensure()
        (job.export / "cloud.glb").write_bytes(b"glb")

        arts = c.get(f"/jobs/{job_id}/artifacts", headers=ALICE).json()["artifacts"]
        assert arts[0]["name"] == "cloud.glb"
        assert arts[0]["url"].endswith(f"{job_id}/cloud.glb")


def test_deleting_a_queued_job_does_not_leave_an_orphan(tmp_path):
    """A worker that starts after deletion must not recreate the directory."""
    import threading

    root = tmp_path / "runs"
    gate = threading.Event()

    def slow_runner(job_id, video, quality, params):
        gate.wait(timeout=5)          # the delete lands while we are in here
        from job_paths import JobPaths
        JobPaths(root / job_id).ensure()

    app = make_app(root, runner=slow_runner)
    with TestClient(app) as c:
        job_id = c.post("/jobs", headers=ALICE,
                        files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).json()["id"]
        # The worker has already claimed it and is parked in slow_runner. Put
        # the row back to `queued` so DELETE is allowed, which is the race this
        # is about: the row goes away while the runner is still working.
        c.app.state.registry.update(job_id, status="queued")
        assert c.delete(f"/jobs/{job_id}", headers=ALICE).status_code == 204
        gate.set()
        c.app.state.pool.shutdown(wait=True)

    assert not (root / job_id).exists(), "worker recreated a deleted job's directory"


# --- durability, presigned upload, hashed keys (MPO-244) ------------------

def test_jobs_survive_a_service_restart(tmp_path):
    """An in-memory registry loses every job when the process dies."""
    root = tmp_path / "runs"
    app1 = make_app(root)
    with TestClient(app1) as c:
        job_id = c.post("/jobs", headers=ALICE,
                        files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).json()["id"]
        app1.state.pool.shutdown(wait=True)

    # A brand-new app over the same directory, as a restarted service would be.
    app2 = make_app(root)
    with TestClient(app2) as c:
        assert c.get(f"/jobs/{job_id}", headers=ALICE).status_code == 200
        assert [j["id"] for j in c.get("/jobs", headers=ALICE).json()["jobs"]] == [job_id]
        app2.state.pool.shutdown(wait=True)


def test_the_api_key_itself_is_never_stored(tmp_path):
    """A leaked jobs database must not hand over working credentials."""
    import sqlite3

    root = tmp_path / "runs"
    app = make_app(root)
    with TestClient(app) as c:
        c.post("/jobs", headers=ALICE, files={"video": ("c.mp4", b"x" * 2048, "video/mp4")})
        app.state.pool.shutdown(wait=True)

    blob = sqlite3.connect(root / "jobs.db").execute(
        "SELECT group_concat(owner_hash) FROM jobs").fetchone()[0]
    assert "alice-key" not in blob
    from job_registry import hash_key
    assert hash_key("alice-key") in blob


def test_health_reports_queue_depth(client):
    body = client.get("/health").json()
    assert "queue_depth" in body and "expired_leases" in body


def test_presigned_upload_requires_a_storage_backend(client):
    r = client.post("/uploads", headers=ALICE, params={"filename": "clip.mp4"})
    assert r.status_code == 503
    assert "CLIPTO3D_STORAGE" in r.json()["detail"]


def test_presigned_upload_returns_a_scoped_url(tmp_path):
    """Large videos should go straight to storage, not through the API."""
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    from storage import S3Storage

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="uploads-test")
        store = S3Storage("uploads-test", "jobs", client=s3)

        root = tmp_path / "runs"
        app = make_app(root, storage=store)
        with TestClient(app) as c:
            r = c.post("/uploads", headers=ALICE, params={"filename": "clip.mp4"})
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["method"] == "PUT"
            assert body["url"].startswith("http")
            assert body["key"].endswith("clip.mp4")

            # Two callers must not be handed keys in the same namespace.
            other = c.post("/uploads", headers=BOB, params={"filename": "clip.mp4"}).json()
            assert other["url"].split("/uploads/")[0] != body["url"].split("/uploads/")[0]
            app.state.pool.shutdown(wait=True)


def test_presigned_upload_validates_the_container(tmp_path):
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    from storage import S3Storage

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="uploads-test")
        root = tmp_path / "runs"
        app = make_app(root, storage=S3Storage("uploads-test", client=s3))
        with TestClient(app) as c:
            r = c.post("/uploads", headers=ALICE, params={"filename": "notes.txt"})
            assert r.status_code == 400
            app.state.pool.shutdown(wait=True)


# --- API key lifecycle (MPO-244) -----------------------------------------

def test_first_key_can_be_issued_without_one(client):
    """Bootstrapping: an unconfigured deployment must be able to mint key one."""
    r = client.post("/keys", params={"label": "bootstrap"})
    assert r.status_code == 201
    assert r.json()["api_key"].startswith("c3d_")


def test_once_a_key_exists_issuing_requires_one(client):
    first = client.post("/keys").json()["api_key"]
    assert client.post("/keys").status_code == 401
    assert client.post("/keys", headers={"X-API-Key": first}).status_code == 201


def test_unknown_key_is_rejected_once_keys_exist(client):
    client.post("/keys")                       # closes the open bootstrap window
    r = client.get("/jobs", headers={"X-API-Key": "not-a-real-key"})
    assert r.status_code == 401


def test_revoked_key_stops_working(client):
    key = client.post("/keys", params={"label": "temp"}).json()["api_key"]
    hdr = {"X-API-Key": key}
    assert client.get("/jobs", headers=hdr).status_code == 200

    prefix = client.get("/keys", headers=hdr).json()["keys"][0]["key_hash_prefix"]
    assert client.delete(f"/keys/{prefix}", headers=hdr).status_code == 200

    # The whole point of revocation: the credential is dead immediately.
    assert client.get("/jobs", headers=hdr).status_code == 401


def test_revocation_keeps_the_jobs(client):
    """Turning off a leaked key must not destroy its history."""
    # Distinct labels: both keys would otherwise be indistinguishable in the
    # listing and the test could revoke the admin key by accident.
    key = client.post("/keys", params={"label": "leaked"}).json()["api_key"]
    admin = client.post("/keys", params={"label": "admin"},
                        headers={"X-API-Key": key}).json()["api_key"]
    hdr = {"X-API-Key": key}

    upload(client, hdr)
    prefix = next(k["key_hash_prefix"] for k in
                  client.get("/keys", headers=hdr).json()["keys"]
                  if k["label"] == "leaked")
    assert client.delete(f"/keys/{prefix}", headers=hdr).status_code == 200
    assert client.get("/jobs", headers=hdr).status_code == 401, "revoked key still works"

    # An admin key can still see the jobs exist.
    assert client.get("/keys", headers={"X-API-Key": admin}).status_code == 200


def test_listing_keys_never_exposes_the_key(client):
    key = client.post("/keys", params={"label": "visible"}).json()["api_key"]
    listed = client.get("/keys", headers={"X-API-Key": key}).json()["keys"]
    blob = str(listed)
    assert key not in blob
    from job_registry import hash_key
    assert hash_key(key) not in blob, "the full digest is enough to revoke someone else's key"
    assert listed[0]["active"] is True


def test_revoking_an_unknown_key_is_404(client):
    key = client.post("/keys").json()["api_key"]
    assert client.delete("/keys/deadbeef1234", headers={"X-API-Key": key}).status_code == 404


# --- serving artifacts and the viewer -------------------------------------
#
# Without these the viewer has nothing to load from a locally-run service:
# artifacts could be listed but not fetched unless S3 was configured.

def wait_for_artifact(client, job_id, name="cloud.glb", tries=100):
    for _ in range(tries):
        arts = client.get(f"/jobs/{job_id}/artifacts", headers=ALICE).json()["artifacts"]
        if any(a["name"] == name for a in arts):
            return arts
        time.sleep(0.05)
    raise AssertionError(f"{name} never appeared")


def test_artifact_bytes_are_downloadable_without_object_storage(client):
    job_id = upload(client).json()["id"]
    arts = wait_for_artifact(client, job_id)
    url = next(a["url"] for a in arts if a["name"] == "cloud.glb")

    r = client.get(url, headers=ALICE)
    assert r.status_code == 200
    assert r.content == b"glb-bytes"
    # A .glb served as text/plain is refused by the browser's GLB path.
    assert r.headers["content-type"] == "model/gltf-binary"


# --- artifact caching -----------------------------------------------------
#
# A job's export directory is written once and never rewritten — a changed
# reconstruction is a new job id — so the bytes behind an artifact URL cannot
# change. These guard the two halves of exploiting that: the response has to
# say so, and a client that took it at its word has to be answered cheaply.
# Starlette's FileResponse does neither on its own, which meant the viewer's
# three-LOD progressive load re-downloaded every LOD on every revisit.

def test_artifacts_are_cacheable_forever(client):
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    r = client.get(f"/jobs/{job_id}/artifacts/cloud.glb", headers=ALICE)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert r.headers.get("etag")


def test_revalidating_an_artifact_costs_no_bytes(client):
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    url = f"/jobs/{job_id}/artifacts/cloud.glb"

    first = client.get(url, headers=ALICE)
    again = client.get(url, headers={**ALICE, "If-None-Match": first.headers["etag"]})

    assert again.status_code == 304
    assert again.content == b""
    # The 304 has to carry the validators forward, or the cache entry it
    # refreshes loses them and the next request is a full fetch again.
    assert again.headers["etag"] == first.headers["etag"]
    assert again.headers["cache-control"] == first.headers["cache-control"]


def test_shared_artifacts_are_cacheable_too(client):
    """The share link is the path that actually gets opened repeatedly.

    It is also the one with no API key, so it must not quietly take a
    different, uncached route through the same file.
    """
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    token = client.post(f"/jobs/{job_id}/share", headers=ALICE).json()["token"]

    first = client.get(f"/shared/{token}/artifacts/cloud.glb")
    assert first.status_code == 200
    assert first.headers["cache-control"] == "public, max-age=31536000, immutable"

    again = client.get(f"/shared/{token}/artifacts/cloud.glb",
                       headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304


@pytest.mark.parametrize("header, expected", [
    ('"abc"', True),
    ('W/"abc"', True),                    # weak validator, same entity
    ('"other", "abc"', True),             # the header is a list, not a value
    ('  "abc"  ', True),
    ("*", True),
    ('"other"', False),
    ("", False),
    ('"ab"', False),
])
def test_if_none_match_understands_the_full_header_grammar(header, expected):
    """A naive `header == etag` passes the first case and fails the rest.

    Browsers send lists and weak validators routinely, so getting this wrong
    means the 304 path silently never fires and nothing looks broken.
    """
    assert service.if_none_match(header, '"abc"') is expected


def test_if_none_match_never_matches_a_missing_etag():
    assert service.if_none_match("*", "") is False


def test_artifact_download_is_scoped_to_the_owner(client):
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    assert client.get(f"/jobs/{job_id}/artifacts/cloud.glb", headers=BOB).status_code == 404
    assert client.get(f"/jobs/{job_id}/artifacts/cloud.glb").status_code == 401


def test_encoded_traversal_over_http_is_refused(client):
    """The one attack string that survives the client's URL normalisation.

    httpx collapses a literal `../..` in a path before sending it, so only the
    percent-encoded form actually reaches the handler. The rest of the
    traversal surface is covered directly against `safe_artifact_path` below,
    which is the honest way to test it.
    """
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    r = client.get(f"/jobs/{job_id}/artifacts/..%2f..%2fjobs.db", headers=ALICE)
    assert r.status_code == 404
    assert b"SQLite" not in r.content


@pytest.mark.parametrize("name", [
    "../../jobs.db",
    "../jobs.db",
    "subdir/../../../jobs.db",
    "..",
    "",
    "/etc/passwd",
    "\\windows\\system32\\config\\sam",
    "C:/Windows/win.ini",
    "missing.glb",
])
def test_safe_artifact_path_rejects_anything_outside_the_job(tmp_path, name):
    export = tmp_path / "job" / "export"
    export.mkdir(parents=True)
    (export / "cloud.glb").write_bytes(b"glb")
    (tmp_path / "jobs.db").write_bytes(b"SQLite format 3\0")
    assert service.safe_artifact_path(export, name) is None, f"{name!r} escaped"


def test_safe_artifact_path_allows_a_real_artifact(tmp_path):
    export = tmp_path / "job" / "export"
    export.mkdir(parents=True)
    (export / "cloud.glb").write_bytes(b"glb")
    assert service.safe_artifact_path(export, "cloud.glb") == (export / "cloud.glb").resolve()


def test_viewer_is_served_from_the_same_origin_as_the_api(client):
    r = client.get("/viewer")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # The page the tests exercise, not a placeholder.
    assert "ClipTo3D viewer" in r.text
    assert "computeSplatOrder" in r.text


# --- resumable upload (MPO-247) -------------------------------------------
#
# A phone loses the network mid-upload or gets backgrounded. What matters is
# that a resume continues from the server's offset and can neither duplicate
# nor skip bytes — a corrupted video that still reconstructs into garbage is
# far worse than a failed upload.

VIDEO = b"".join(bytes([i % 251]) for i in range(20_000))


def start_upload(client, headers=ALICE, name="clip.mp4", total=len(VIDEO)):
    return client.post("/uploads/resumable", headers=headers,
                       params={"filename": name, "total": total})


def send_chunk(client, upload_id, data, offset, headers=ALICE):
    return client.patch(f"/uploads/{upload_id}", headers=headers,
                        params={"offset": offset}, content=data)


def test_chunked_upload_reassembles_the_exact_bytes(client, tmp_path):
    up = start_upload(client).json()
    assert up["offset"] == 0 and up["chunk_size"] > 0

    size = 4096
    for off in range(0, len(VIDEO), size):
        r = send_chunk(client, up["upload_id"], VIDEO[off:off + size], off)
        assert r.status_code == 200, r.text
    assert r.json()["complete"] is True

    job_id = client.post(f"/uploads/{up['upload_id']}/job", headers=ALICE).json()["id"]
    stored = next((tmp_path / "runs" / job_id / "input").glob("*.mp4"))
    assert stored.read_bytes() == VIDEO, "reassembled file differs from the source"


def test_resume_continues_from_the_server_offset(client, tmp_path):
    """The whole point: a dropped connection must not restart from zero."""
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO[:7000], 0)

    # ...connection dies. The client asks where it got to.
    state = client.get(f"/uploads/{up['upload_id']}", headers=ALICE).json()
    assert state["offset"] == 7000
    assert state["total"] == len(VIDEO)

    send_chunk(client, up["upload_id"], VIDEO[7000:], 7000)
    job_id = client.post(f"/uploads/{up['upload_id']}/job", headers=ALICE).json()["id"]
    stored = next((tmp_path / "runs" / job_id / "input").glob("*.mp4"))
    assert stored.read_bytes() == VIDEO


def test_a_replayed_chunk_is_refused_rather_than_appended_twice(client):
    """A retry after an ack that never arrived must not duplicate bytes."""
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO[:5000], 0)

    again = send_chunk(client, up["upload_id"], VIDEO[:5000], 0)
    assert again.status_code == 409
    # And it tells the client where to actually resume from.
    assert again.json()["offset"] == 5000

    assert client.get(f"/uploads/{up['upload_id']}", headers=ALICE).json()["offset"] == 5000


def test_a_chunk_that_would_leave_a_hole_is_refused(client):
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO[:1000], 0)
    ahead = send_chunk(client, up["upload_id"], VIDEO[9000:10000], 9000)
    assert ahead.status_code == 409
    assert ahead.json()["offset"] == 1000


def test_overrunning_the_declared_total_is_refused(client):
    up = start_upload(client, total=1000).json()
    assert send_chunk(client, up["upload_id"], b"x" * 2000, 0).status_code == 400


def test_an_incomplete_upload_cannot_become_a_job(client):
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO[:100], 0)
    r = client.post(f"/uploads/{up['upload_id']}/job", headers=ALICE)
    assert r.status_code == 409
    assert "incomplete" in r.json()["detail"]


def test_upload_validation_happens_before_any_bytes_are_accepted(client):
    """No point taking 400 MB of chunks only to reject the container."""
    assert start_upload(client, name="clip.txt").status_code == 400
    assert start_upload(client, total=0).status_code == 400
    huge = start_upload(client, total=service.MAX_UPLOAD_BYTES + 1)
    assert huge.status_code == 400


def test_uploads_are_scoped_to_the_caller(client):
    up = start_upload(client).json()
    uid = up["upload_id"]
    assert client.get(f"/uploads/{uid}", headers=BOB).status_code == 404
    assert send_chunk(client, uid, b"x" * 10, 0, headers=BOB).status_code == 404
    assert client.post(f"/uploads/{uid}/job", headers=BOB).status_code == 404
    assert client.delete(f"/uploads/{uid}", headers=BOB).status_code == 404


def test_abandoning_an_upload_removes_its_bytes(client, tmp_path):
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO[:3000], 0)
    assert client.delete(f"/uploads/{up['upload_id']}", headers=ALICE).status_code == 204
    assert client.get(f"/uploads/{up['upload_id']}", headers=ALICE).status_code == 404
    assert not list((tmp_path / "runs" / "_uploads").rglob("part"))


def test_starting_a_job_from_an_upload_does_not_leave_the_bytes_behind(client, tmp_path):
    """The clip is moved into the job, not copied — it can be hundreds of MB."""
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO, 0)
    client.post(f"/uploads/{up['upload_id']}/job", headers=ALICE)
    assert not list((tmp_path / "runs" / "_uploads").rglob("part"))


# --- strict auth ----------------------------------------------------------
#
# The open bootstrap window is right for localhost and wrong for anything with
# a public address: between the process becoming reachable and the operator
# issuing key one, whoever asks first owns the deployment. Strict mode is the
# default for that reason, and these are its tests.

BOOT = {"X-Bootstrap-Key": "boot-secret"}


@pytest.fixture
def strict(tmp_path):
    root = tmp_path / "runs"
    app = make_app(root, auth_mode="strict", bootstrap_key="boot-secret")
    with TestClient(app) as c:
        yield c
    app.state.pool.shutdown(wait=True)


def test_an_unknown_key_is_rejected_before_any_key_exists(strict):
    """The land grab, closed. In open mode this request succeeds."""
    assert strict.get("/jobs", headers={"X-API-Key": "i-got-here-first"}).status_code == 401
    assert strict.post("/jobs", headers={"X-API-Key": "x"},
                       files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).status_code == 401


def test_the_first_key_needs_the_bootstrap_secret(strict):
    assert strict.post("/keys").status_code == 401
    assert strict.post("/keys", headers={"X-Bootstrap-Key": "wrong"}).status_code == 401

    r = strict.post("/keys", params={"label": "admin"}, headers=BOOT)
    assert r.status_code == 201
    assert r.json()["api_key"].startswith("c3d_")


def test_the_bootstrap_secret_stops_working_once_a_key_exists(strict):
    """It mints key one, not an unlimited supply — otherwise a leaked
    bootstrap value is a permanent skeleton key."""
    key = strict.post("/keys", headers=BOOT).json()["api_key"]
    assert strict.post("/keys", headers=BOOT).status_code == 401
    assert strict.post("/keys", headers={"X-API-Key": key}).status_code == 201


def test_a_bootstrapped_key_works_normally(strict):
    key = strict.post("/keys", headers=BOOT).json()["api_key"]
    hdr = {"X-API-Key": key}
    assert strict.get("/jobs", headers=hdr).status_code == 200
    assert strict.post("/jobs", headers=hdr,
                       files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).status_code == 201


def test_an_unconfigured_strict_app_generates_a_bootstrap_secret(tmp_path):
    """Usable unconfigured, without leaving the window open: the secret goes
    to the log, which only the operator can read."""
    root = tmp_path / "runs"
    app = make_app(root, auth_mode="strict", bootstrap_key="")
    with TestClient(app) as c:
        generated = app.state.bootstrap_key
        assert generated and generated.startswith("boot_")
        assert c.post("/keys").status_code == 401
        assert c.post("/keys", headers={"X-Bootstrap-Key": generated}).status_code == 201
    app.state.pool.shutdown(wait=True)


def test_health_reports_which_auth_mode_is_live(strict, client):
    assert strict.get("/health").json()["auth"] == "strict"
    assert client.get("/health").json()["auth"] == "open"


def test_an_unknown_auth_mode_is_refused_at_startup(tmp_path):
    """A typo in CLIPTO3D_AUTH must not silently fall back to the open mode."""
    with pytest.raises(ValueError, match="strict"):
        make_app(tmp_path / "runs", auth_mode="stcirt")


# --- upload streaming -----------------------------------------------------

def test_an_oversized_declared_length_is_refused_before_the_body(client, monkeypatch):
    """A 500 MB body should be turned away at the header, not after we have
    already accepted all of it."""
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 1024)
    r = client.post("/jobs", headers={**ALICE, "Content-Length": "999999"},
                    files={"video": ("c.mp4", b"x" * 4096, "video/mp4")})
    assert r.status_code == 400
    assert "limit" in r.json()["detail"]


def test_a_rejected_upload_leaves_no_job_directory(client, tmp_path, monkeypatch):
    """The streaming path creates the job directory before it knows the size,
    so the failure path has to clean up after itself."""
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 1024)
    before = set(p.name for p in (tmp_path / "runs").iterdir())
    assert client.post("/jobs", headers=ALICE,
                       files={"video": ("c.mp4", b"x" * 8192, "video/mp4")}).status_code == 400
    after = set(p.name for p in (tmp_path / "runs").iterdir())
    assert after == before, "a rejected upload left its job directory behind"


def test_the_bytes_written_are_the_bytes_sent(client, tmp_path):
    """Chunked writes must reassemble exactly; a corrupted clip that still
    reconstructs into garbage is worse than a failed upload."""
    payload = bytes(i % 251 for i in range(300_000))
    job_id = client.post("/jobs", headers=ALICE,
                         files={"video": ("c.mp4", payload, "video/mp4")}).json()["id"]
    stored = next((tmp_path / "runs" / job_id / "input").glob("*.mp4"))
    assert stored.read_bytes() == payload


def test_an_unsupported_container_is_rejected_without_writing_anything(client, tmp_path):
    before = set(p.name for p in (tmp_path / "runs").iterdir())
    assert client.post("/jobs", headers=ALICE,
                       files={"video": ("notes.pdf", b"x" * 4096, "application/pdf")}
                       ).status_code == 400
    assert set(p.name for p in (tmp_path / "runs").iterdir()) == before


# --- rate limiting and headers --------------------------------------------

def test_the_rate_limiter_refuses_and_says_when_to_retry(tmp_path):
    root = tmp_path / "runs"
    app = make_app(root, rate_limit_per_minute=3)
    with TestClient(app) as c:
        for _ in range(3):
            assert c.get("/jobs", headers=ALICE).status_code == 200
        r = c.get("/jobs", headers=ALICE)
        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) >= 1
    app.state.pool.shutdown(wait=True)


def test_the_limiter_is_per_caller(tmp_path):
    """One caller exhausting its budget must not lock everyone else out."""
    root = tmp_path / "runs"
    app = make_app(root, rate_limit_per_minute=2)
    with TestClient(app) as c:
        for _ in range(2):
            c.get("/jobs", headers=ALICE)
        assert c.get("/jobs", headers=ALICE).status_code == 429
        assert c.get("/jobs", headers=BOB).status_code == 200
    app.state.pool.shutdown(wait=True)


def test_health_is_never_rate_limited(tmp_path):
    """A load balancer polls it; limiting it takes the service out of
    rotation under exactly the load it should survive."""
    root = tmp_path / "runs"
    app = make_app(root, rate_limit_per_minute=2)
    with TestClient(app) as c:
        for _ in range(10):
            assert c.get("/health").status_code == 200
    app.state.pool.shutdown(wait=True)


def test_responses_carry_the_baseline_security_headers(client):
    r = client.get("/viewer")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert r.headers["X-Frame-Options"] == "SAMEORIGIN"


def test_the_csp_forbids_third_party_scripts(client):
    """The viewer is self-contained by design — no CDN. This is what keeps an
    injected `<script src=…>` from loading at runtime."""
    csp = client.get("/viewer").headers["Content-Security-Policy"]
    assert "script-src 'self' 'unsafe-inline'" in csp
    assert "default-src 'self'" in csp
    # ...while still allowing presigned artifact URLs off-origin.
    assert "connect-src 'self' https:" in csp


def test_the_rate_limiter_can_be_disabled():
    limiter = service.RateLimiter(0)
    for _ in range(1000):
        assert limiter.check("anyone") is None


def test_the_rate_limit_window_slides():
    limiter = service.RateLimiter(2, window=10.0)
    assert limiter.check("a", now=0) is None
    assert limiter.check("a", now=1) is None
    assert limiter.check("a", now=2) is not None
    # Once the first hit ages out of the window there is room again.
    assert limiter.check("a", now=11) is None


# --- retention ------------------------------------------------------------
#
# Quotas cap the job *count* per caller. They do not cap bytes, and a job
# directory is gigabytes — 50 jobs of 4 GB is 200 GB of disk the count-based
# quota is perfectly happy with.

def test_an_abandoned_upload_is_collected(client, tmp_path):
    up = start_upload(client).json()
    send_chunk(client, up["upload_id"], VIDEO[:3000], 0)
    root = tmp_path / "runs"

    # Nothing goes yet: it is still within the retention window.
    assert service.sweep_retention(root, client.app.state.registry)["uploads"] == []
    assert list(root.rglob("part"))

    # A day later, the caller never came back.
    gone = service.sweep_retention(root, client.app.state.registry,
                                   now=time.time() + 25 * 3600)
    assert gone["uploads"] == [up["upload_id"]]
    assert not list(root.rglob("part"))


def test_an_expired_job_is_removed_with_its_bytes(client, tmp_path):
    job_id = upload(client).json()["id"]
    settle(client)
    reg = client.app.state.registry
    root = tmp_path / "runs"
    assert (root / job_id).is_dir()

    gone = service.sweep_retention(root, reg, now=time.time() + 31 * 86400)
    assert gone["jobs"] == [job_id]
    assert not (root / job_id).exists()
    assert reg.get(job_id) is None


def test_retention_never_touches_an_unfinished_job(client, tmp_path):
    """An old job that is still running is not garbage; deleting it out from
    under its worker is how you get a half-written export."""
    job_id = upload(client).json()["id"]
    settle(client)
    reg = client.app.state.registry
    reg.update(job_id, status="running")

    gone = service.sweep_retention(tmp_path / "runs", reg, now=time.time() + 365 * 86400)
    assert gone["jobs"] == []
    assert (tmp_path / "runs" / job_id).exists()


def test_retention_can_be_switched_off(client, tmp_path):
    job_id = upload(client).json()["id"]
    settle(client)
    gone = service.sweep_retention(tmp_path / "runs", client.app.state.registry,
                                   upload_retention_hours=0, job_retention_days=0,
                                   now=time.time() + 365 * 86400)
    assert gone == {"uploads": [], "jobs": []}
    assert (tmp_path / "runs" / job_id).exists()


# --- restart recovery, through the API ------------------------------------

def test_a_job_stranded_by_a_restart_is_picked_up(tmp_path):
    """The property the README claimed and the service did not have: a job
    left `running` by a process that died is recovered, not stranded."""
    root = tmp_path / "runs"
    done = threading.Event()

    app1 = make_app(root, runner=lambda *a: None)
    with TestClient(app1) as c:
        job_id = c.post("/jobs", headers=ALICE,
                        files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).json()["id"]
    app1.state.pool.shutdown(wait=True)

    # Forge the state the old dispatcher left behind: running, no lease.
    app1.state.registry.update(job_id, status="running",
                               claimed_by="dead-worker", lease_expires=None)

    app2 = make_app(root, runner=lambda *a: done.set())
    with TestClient(app2) as c:
        for _ in range(200):
            if c.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "done":
                break
            time.sleep(0.02)
        assert c.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "done"
        assert done.is_set(), "the restarted service never re-ran the stranded job"
    app2.state.pool.shutdown(wait=True)


# --- public (self-service) auth ------------------------------------------
#
# The mode for "anyone can upload a video". Each visitor mints their own key,
# which gives them their own job namespace and their own quota for free — the
# isolation is already keyed on the digest. What has to hold is that a visitor
# cannot turn one key into many and walk past the quota.

@pytest.fixture
def public(tmp_path):
    root = tmp_path / "runs"
    app = make_app(root, auth_mode="public", bootstrap_key="admin-secret")
    with TestClient(app) as c:
        yield c
    app.state.pool.shutdown(wait=True)


def test_anyone_can_mint_their_own_key(public):
    r = public.post("/keys")
    assert r.status_code == 201
    body = r.json()
    assert body["api_key"].startswith("c3d_")
    assert body["admin"] is False
    assert public.get("/jobs", headers={"X-API-Key": body["api_key"]}).status_code == 200


def test_a_self_service_key_cannot_mint_more(public):
    """The containment that makes self-service safe: one visitor stays one
    caller, so the per-caller quota still means something."""
    key = public.post("/keys").json()["api_key"]
    r = public.post("/keys", headers={"X-API-Key": key})
    assert r.status_code == 403
    assert "not permitted" in r.json()["detail"]


def test_signups_are_capped_per_address(public):
    """A quota is only a limit if a fresh identity costs something."""
    codes = [public.post("/keys").status_code for _ in range(service.KEYS_PER_ADDRESS + 3)]
    assert codes.count(201) == service.KEYS_PER_ADDRESS
    assert 429 in codes, codes


def test_a_capped_signup_says_when_to_retry(public):
    for _ in range(service.KEYS_PER_ADDRESS):
        public.post("/keys")
    r = public.post("/keys")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_self_service_callers_are_isolated_from_each_other(public):
    a = public.post("/keys").json()["api_key"]
    b = public.post("/keys").json()["api_key"]
    ha, hb = {"X-API-Key": a}, {"X-API-Key": b}

    job = public.post("/jobs", headers=ha,
                      files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).json()["id"]
    assert [j["id"] for j in public.get("/jobs", headers=hb).json()["jobs"]] == []
    assert public.get(f"/jobs/{job}", headers=hb).status_code == 404


def test_an_unknown_key_is_still_rejected_in_public_mode(public):
    """Self-service means you may *get* a key, not that any string works."""
    assert public.get("/jobs", headers={"X-API-Key": "made-it-up"}).status_code == 401


def test_the_operator_can_still_mint_an_admin_key(public):
    r = public.post("/keys", params={"label": "ops"},
                    headers={"X-Bootstrap-Key": "admin-secret"})
    assert r.status_code == 201 and r.json()["admin"] is True

    # And an admin key issues admin keys — not self-service ones. Getting this
    # wrong silently demotes the credential that administers the deployment.
    second = public.post("/keys", headers={"X-API-Key": r.json()["api_key"]})
    assert second.status_code == 201 and second.json()["admin"] is True


def test_public_mode_is_reported_by_health(public):
    assert public.get("/health").json()["auth"] == "public"


def test_a_strict_admin_key_can_issue_but_a_demoted_one_cannot(strict):
    admin = strict.post("/keys", headers=BOOT).json()["api_key"]
    assert strict.post("/keys", headers={"X-API-Key": admin}).status_code == 201

    demoted = strict.app.state.keys.issue("restricted", admin=False)
    assert strict.post("/keys", headers={"X-API-Key": demoted}).status_code == 403
    # ...but it still works as an ordinary caller.
    assert strict.get("/jobs", headers={"X-API-Key": demoted}).status_code == 200


# --- share links ----------------------------------------------------------

def share_of(client, job_id, headers=ALICE, **params):
    return client.post(f"/jobs/{job_id}/share", headers=headers, params=params)


def test_a_share_link_reads_results_without_a_key(client):
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    token = share_of(client, job_id).json()["token"]

    # No key header at all, which is the entire point.
    arts = client.get(f"/shared/{token}/artifacts")
    assert arts.status_code == 200
    assert any(a["name"] == "cloud.glb" for a in arts.json()["artifacts"])

    blob = client.get(f"/shared/{token}/artifacts/cloud.glb")
    assert blob.status_code == 200
    assert blob.content == b"glb-bytes"
    assert blob.headers["content-type"] == "model/gltf-binary"


def test_the_share_url_is_one_the_viewer_can_actually_open(client):
    """`?job=<base>` fetches `<base>/cloud.glb`, so the base has to be the
    artifacts collection — otherwise the link looks right and loads nothing."""
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    body = share_of(client, job_id).json()

    assert body["viewer_url"] == f"/viewer?job=/shared/{body['token']}/artifacts"
    base = body["viewer_url"].split("?job=")[1]
    assert client.get(f"{base}/cloud.glb").status_code == 200


def test_a_share_link_grants_nothing_but_reading(strict):
    """It is a read credential, not a key: it must not spend GPU time.

    Deliberately in strict mode. The `client` fixture runs open auth, where
    *every* string verifies as a key — so asserting this there would pass or
    fail for reasons that have nothing to do with share tokens.
    """
    key = strict.post("/keys", headers=BOOT).json()["api_key"]
    hdr = {"X-API-Key": key}
    job_id = strict.post("/jobs", headers=hdr,
                         files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}).json()["id"]
    token = strict.post(f"/jobs/{job_id}/share", headers=hdr).json()["token"]

    as_key = {"X-API-Key": token}
    assert strict.get("/jobs", headers=as_key).status_code == 401
    assert strict.post("/jobs", headers=as_key,
                       files={"video": ("c.mp4", b"x" * 2048, "video/mp4")}
                       ).status_code == 401
    assert strict.post(f"/jobs/{job_id}/cancel", headers=as_key).status_code == 401
    # ...while the same token works perfectly well as a share link.
    assert strict.get(f"/shared/{token}").status_code == 200


def test_a_share_link_exposes_status_but_not_the_owner_view(client):
    job_id = upload(client).json()["id"]
    token = share_of(client, job_id).json()["token"]
    body = client.get(f"/shared/{token}").json()

    assert "status" in body and "progress" in body
    # Nothing that belongs to the owner rather than to a viewer.
    assert "error" not in body and "id" not in body


def test_a_revoked_share_stops_working(client):
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    token = share_of(client, job_id).json()["token"]
    assert client.get(f"/shared/{token}/artifacts").status_code == 200

    assert client.delete(f"/jobs/{job_id}/share", headers=ALICE).json()["revoked"] == 1
    assert client.get(f"/shared/{token}/artifacts").status_code == 404


def test_an_expired_share_stops_working(client):
    job_id = upload(client).json()["id"]
    token = share_of(client, job_id, ttl_seconds=-1).json()["token"]
    assert client.get(f"/shared/{token}/artifacts").status_code == 404


def test_deleting_a_job_kills_its_share_links(client):
    """A token outliving its job would point at nothing — or, once an id is
    reused, at someone else's results."""
    job_id = upload(client).json()["id"]
    settle(client)
    token = share_of(client, job_id).json()["token"]
    client.app.state.registry.update(job_id, status="done")

    assert client.delete(f"/jobs/{job_id}", headers=ALICE).status_code == 204
    assert client.get(f"/shared/{token}").status_code == 404


def test_only_the_owner_can_share_a_job(client):
    job_id = upload(client, ALICE).json()["id"]
    assert share_of(client, job_id, headers=BOB).status_code == 404
    assert client.delete(f"/jobs/{job_id}/share", headers=BOB).status_code == 404


def test_an_unknown_token_is_404_not_403(client):
    """Unknown, expired and deleted must be indistinguishable, or a token can
    be probed for which of those it is."""
    assert client.get("/shared/not-a-real-token/artifacts").status_code == 404
    assert client.get("/shared/not-a-real-token").status_code == 404


def test_the_share_token_itself_is_never_stored(client, tmp_path):
    import sqlite3

    job_id = upload(client).json()["id"]
    token = share_of(client, job_id).json()["token"]
    settle(client)

    blob = str(sqlite3.connect(tmp_path / "runs" / "jobs.db").execute(
        "SELECT group_concat(token_hash) FROM job_shares").fetchone()[0])
    assert token not in blob
    from job_registry import hash_key
    assert hash_key(token) in blob


def test_listing_shares_never_exposes_the_token(client):
    job_id = upload(client).json()["id"]
    token = share_of(client, job_id).json()["token"]
    listed = client.get(f"/jobs/{job_id}/share", headers=ALICE).json()["shares"]
    assert len(listed) == 1
    assert token not in str(listed)


def test_share_traversal_is_refused(client):
    """This artifact route is reachable without a key, so its path handling
    matters more, not less."""
    job_id = upload(client).json()["id"]
    wait_for_artifact(client, job_id)
    token = share_of(client, job_id).json()["token"]
    r = client.get(f"/shared/{token}/artifacts/..%2f..%2fjobs.db")
    assert r.status_code == 404
    assert b"SQLite" not in r.content
