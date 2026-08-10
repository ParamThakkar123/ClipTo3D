"""Storage abstraction (MPO-236).

The criterion in the issue is that two jobs can run concurrently without
touching each other's artifacts. That is what most of these tests are about;
the S3 backend is exercised against an injected fake client, so no bucket,
credentials or network are involved.
"""

from pathlib import Path

import pytest

from storage import JobStore, LocalStorage, S3Storage, from_uri


@pytest.fixture
def store(tmp_path):
    return LocalStorage(tmp_path / "store")


@pytest.fixture
def sample(tmp_path):
    f = tmp_path / "cloud.ply"
    f.write_bytes(b"ply\ncontent")
    return f


# --- the isolation property ----------------------------------------------

def test_two_jobs_do_not_collide(store, sample, tmp_path):
    """The whole point: both jobs write the same relative key."""
    a = store.job("job-a")
    b = store.job("job-b")

    other = tmp_path / "other.ply"
    other.write_bytes(b"different")

    a.put(sample, "cloud/fused_cloud.ply")
    b.put(other, "cloud/fused_cloud.ply")

    assert a.open("cloud/fused_cloud.ply").read() == b"ply\ncontent"
    assert b.open("cloud/fused_cloud.ply").read() == b"different"


def test_a_job_only_lists_its_own_artifacts(store, sample):
    store.job("job-a").put(sample, "cloud/a.ply")
    store.job("job-a").put(sample, "export/a.glb")
    store.job("job-b").put(sample, "cloud/b.ply")

    assert store.job("job-a").list() == ["cloud/a.ply", "export/a.glb"]
    assert store.job("job-b").list() == ["cloud/b.ply"]


def test_deleting_one_job_leaves_the_other(store, sample):
    store.job("job-a").put(sample, "cloud/x.ply")
    store.job("job-b").put(sample, "cloud/x.ply")

    store.job("job-a").delete("cloud/x.ply")
    assert not store.job("job-a").exists("cloud/x.ply")
    assert store.job("job-b").exists("cloud/x.ply")


# --- key hygiene ----------------------------------------------------------

@pytest.mark.parametrize("bad", ["../escape", "a/../../b", "/absolute", "..", ""])
def test_traversal_and_absolute_keys_are_rejected(store, sample, bad):
    """On S3 a traversing key silently writes into another tenant's prefix."""
    with pytest.raises(ValueError):
        store.job("job-a").put(sample, bad)


def test_backslashes_are_normalised(store, sample):
    """Windows callers will produce these."""
    store.job("j").put(sample, "cloud\\sub\\x.ply")
    assert store.job("j").exists("cloud/sub/x.ply")


@pytest.mark.parametrize("bad", ["", "..", ".", "has/slash"])
def test_invalid_job_ids_are_rejected(store, bad):
    with pytest.raises(ValueError):
        JobStore(store, bad)


# --- local backend --------------------------------------------------------

def test_round_trip(store, sample, tmp_path):
    job = store.job("j")
    job.put(sample, "cloud/x.ply")
    out = job.get("cloud/x.ply", tmp_path / "back.ply")
    assert out.read_bytes() == sample.read_bytes()
    assert job.size("cloud/x.ply") == len(sample.read_bytes())


def test_missing_key_raises(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.job("j").get("nope.ply", tmp_path / "x")


def test_open_for_write_creates_parents(store):
    with store.job("j").open("deep/nested/x.txt", "wb") as fh:
        fh.write(b"hi")
    assert store.job("j").open("deep/nested/x.txt").read() == b"hi"


def test_local_has_no_signed_url(store, sample):
    store.job("j").put(sample, "x.ply")
    assert store.job("j").signed_url("x.ply") is None


def test_upload_and_download_tree(store, tmp_path):
    src = tmp_path / "job"
    (src / "cloud").mkdir(parents=True)
    (src / "export").mkdir()
    (src / "cloud" / "f.ply").write_bytes(b"a")
    (src / "export" / "c.glb").write_bytes(b"b")

    job = store.job("j")
    assert job.upload_tree(src) == ["cloud/f.ply", "export/c.glb"]

    dst = tmp_path / "down"
    job.download_tree(dst)
    assert (dst / "cloud" / "f.ply").read_bytes() == b"a"
    assert (dst / "export" / "c.glb").read_bytes() == b"b"


# --- URI parsing ----------------------------------------------------------

def test_from_uri_local(tmp_path):
    assert isinstance(from_uri(str(tmp_path)), LocalStorage)
    assert isinstance(from_uri(f"file:///{tmp_path.as_posix().lstrip('/')}"), LocalStorage)


def test_from_uri_s3():
    s = from_uri("s3://my-bucket/jobs/prod")
    assert isinstance(s, S3Storage)
    assert s.bucket == "my-bucket" and s.prefix == "jobs/prod"


def test_from_uri_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unsupported storage scheme"):
        from_uri("gs://bucket/x")


# --- S3 backend against a fake client ------------------------------------

class FakeS3:
    """Just enough of the boto3 S3 client to exercise key handling."""

    def __init__(self):
        self.objects = {}

    def upload_file(self, local, bucket, key):
        self.objects[(bucket, key)] = Path(local).read_bytes()

    def download_file(self, bucket, key, local):
        Path(local).write_bytes(self.objects[(bucket, key)])

    def get_object(self, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix="", **kw):
        keys = [k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in sorted(keys)], "IsTruncated": False}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://{Params['Bucket']}.s3.example/{Params['Key']}?exp={ExpiresIn}"


def test_s3_prefixes_keys_by_job(tmp_path, sample):
    fake = FakeS3()
    s3 = S3Storage("bucket", "jobs", client=fake)
    s3.job("job-a").put(sample, "cloud/x.ply")
    s3.job("job-b").put(sample, "cloud/x.ply")

    assert set(fake.objects) == {
        ("bucket", "jobs/job-a/cloud/x.ply"),
        ("bucket", "jobs/job-b/cloud/x.ply"),
    }


def test_s3_job_listing_is_scoped(sample):
    fake = FakeS3()
    s3 = S3Storage("bucket", "jobs", client=fake)
    s3.job("job-a").put(sample, "cloud/x.ply")
    s3.job("job-b").put(sample, "cloud/y.ply")

    assert s3.job("job-a").list() == ["cloud/x.ply"]


def test_s3_round_trip(tmp_path, sample):
    fake = FakeS3()
    job = S3Storage("bucket", client=fake).job("j")
    job.put(sample, "cloud/x.ply")
    assert job.exists("cloud/x.ply")
    assert job.size("cloud/x.ply") == len(sample.read_bytes())
    assert job.open("cloud/x.ply").read() == sample.read_bytes()
    assert job.get("cloud/x.ply", tmp_path / "back.ply").read_bytes() == sample.read_bytes()


def test_s3_signed_url(sample):
    fake = FakeS3()
    job = S3Storage("bucket", "jobs", client=fake).job("j")
    job.put(sample, "export/cloud.glb")
    url = job.signed_url("export/cloud.glb", expires=60)
    assert "jobs/j/export/cloud.glb" in url and "exp=60" in url


def test_s3_open_is_read_only(sample):
    job = S3Storage("bucket", client=FakeS3()).job("j")
    with pytest.raises(ValueError, match="read-only"):
        job.open("x.ply", "wb")


def test_s3_missing_boto3_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "boto3":
            raise ModuleNotFoundError("No module named 'boto3'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ModuleNotFoundError, match="--extra s3"):
        S3Storage("bucket").client


# --- against a real S3 implementation (moto) ------------------------------
#
# The FakeS3 above shares my assumptions about how S3 behaves. moto is an
# actual S3 implementation, so these catch the cases where those assumptions
# are wrong — pagination, missing-key error types, listing semantics.

moto = pytest.importorskip("moto", reason="needs the dev group")
boto3 = pytest.importorskip("boto3")


@pytest.fixture
def s3_backend(monkeypatch):
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="clipto3d-test")
        yield S3Storage("clipto3d-test", "jobs", client=client)


def test_moto_two_jobs_stay_isolated(s3_backend, sample, tmp_path):
    other = tmp_path / "other.ply"
    other.write_bytes(b"different")

    s3_backend.job("job-a").put(sample, "cloud/fused_cloud.ply")
    s3_backend.job("job-b").put(other, "cloud/fused_cloud.ply")

    assert s3_backend.job("job-a").open("cloud/fused_cloud.ply").read() == b"ply\ncontent"
    assert s3_backend.job("job-b").open("cloud/fused_cloud.ply").read() == b"different"
    assert s3_backend.job("job-a").list() == ["cloud/fused_cloud.ply"]


def test_moto_round_trip_and_size(s3_backend, sample, tmp_path):
    job = s3_backend.job("j")
    job.put(sample, "export/cloud.glb")
    assert job.exists("export/cloud.glb")
    assert job.size("export/cloud.glb") == len(sample.read_bytes())
    assert job.get("export/cloud.glb", tmp_path / "back").read_bytes() == sample.read_bytes()


def test_moto_missing_key_is_absent_not_an_exception(s3_backend):
    """head_object raises ClientError on a real S3; exists() must absorb it."""
    assert s3_backend.job("j").exists("nope.ply") is False


def test_moto_signed_url_is_fetchable_shape(s3_backend, sample):
    job = s3_backend.job("j")
    job.put(sample, "export/cloud.glb")
    url = job.signed_url("export/cloud.glb", expires=120)
    assert url.startswith("http")
    assert "jobs/j/export/cloud.glb" in url
    assert "Expires" in url or "X-Amz-Expires" in url


def test_moto_listing_paginates(s3_backend, tmp_path):
    """list_objects_v2 caps at 1000 keys; the loop must follow the token."""
    f = tmp_path / "x"
    f.write_bytes(b"x")
    job = s3_backend.job("big")
    for i in range(1205):
        job.put(f, f"frames/f{i:05d}.jpg")

    keys = job.list()
    assert len(keys) == 1205, f"pagination dropped keys: got {len(keys)}"


def test_moto_upload_and_download_tree(s3_backend, tmp_path):
    src = tmp_path / "job"
    (src / "export").mkdir(parents=True)
    (src / "export" / "a.glb").write_bytes(b"a")
    (src / "export" / "b.usdz").write_bytes(b"b")

    job = s3_backend.job("j")
    assert sorted(job.upload_tree(src)) == ["export/a.glb", "export/b.usdz"]

    dst = tmp_path / "down"
    job.download_tree(dst)
    assert (dst / "export" / "a.glb").read_bytes() == b"a"
    assert (dst / "export" / "b.usdz").read_bytes() == b"b"


def test_moto_delete_is_scoped(s3_backend, sample):
    s3_backend.job("a").put(sample, "x.ply")
    s3_backend.job("b").put(sample, "x.ply")
    s3_backend.job("a").delete("x.ply")
    assert not s3_backend.job("a").exists("x.ply")
    assert s3_backend.job("b").exists("x.ply")
