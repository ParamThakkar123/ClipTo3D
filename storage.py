"""Storage abstraction for job artifacts (MPO-236).

Every stage used to read and write bare relative directories in the process
CWD — `frames`, `depth_maps`, `gsplat_output` — so two concurrent jobs wrote
to the same place, and nothing could point at a bucket. `JobPaths` (MPO-224)
fixed the first half by making the layout job-scoped; this fixes the second by
making the *backend* pluggable.

    store = from_uri("file:///var/lib/clipto3d")
    store = from_uri("s3://my-bucket/jobs")

    job = store.job("abc123")          # a namespace, not a directory
    job.put(local_ply, "cloud/fused_cloud.ply")
    url = job.signed_url("export/cloud.glb", expires=3600)

Two backends: local filesystem, and anything S3-compatible via boto3 (an
optional extra — the base install does not depend on it).

The reconstruction stages still take real filesystem paths, because COLMAP and
ffmpeg are subprocesses that can only read files. A remote-backed job therefore
works in a local scratch directory and syncs at the boundaries; `JobStore.stage`
is that boundary. Pretending a subprocess can read from a bucket is the trap
this design avoids.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import IO, Iterable, List, Optional
from urllib.parse import urlparse


def _clean_key(key: str) -> str:
    """Normalise a key and refuse anything that escapes its prefix.

    A key like `../../etc/passwd` must not resolve outside the job namespace —
    on the local backend that is a path traversal, and on S3 it silently
    writes to another tenant's prefix.
    """
    p = PurePosixPath(str(key).replace("\\", "/"))
    if p.is_absolute():
        raise ValueError(f"key must be relative, got {key!r}")
    parts = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"key must not traverse upwards: {key!r}")
        parts.append(part)
    if not parts:
        raise ValueError("key must not be empty")
    return "/".join(parts)


class Storage(ABC):
    """Where a job's artifacts live."""

    @abstractmethod
    def put(self, local: Path | str, key: str) -> str: ...

    @abstractmethod
    def get(self, key: str, local: Path | str) -> Path: ...

    @abstractmethod
    def open(self, key: str, mode: str = "rb") -> IO: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> List[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    def signed_url(self, key: str, expires: int = 3600) -> Optional[str]:
        """A URL a browser can fetch, or None when the backend has no notion of one."""
        return None

    def job(self, job_id: str) -> "JobStore":
        return JobStore(self, job_id)


class LocalStorage(Storage):
    """Filesystem backend. The default, and what the CLI uses."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    def _path(self, key: str) -> Path:
        return self.root / _clean_key(key)

    def put(self, local: Path | str, key: str) -> str:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = Path(local)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return str(dst)

    def get(self, key: str, local: Path | str) -> Path:
        src = self._path(key)
        if not src.is_file():
            raise FileNotFoundError(f"{key} not found in {self.root}")
        dst = Path(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst

    def open(self, key: str, mode: str = "rb") -> IO:
        p = self._path(key)
        if "w" in mode or "a" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, mode)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str = "") -> List[str]:
        base = self.root / _clean_key(prefix) if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [base.relative_to(self.root).as_posix()]
        return sorted(
            p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()
        )

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def local_path(self, key: str) -> Path:
        """Real path — only the local backend can offer this."""
        return self._path(key)


class S3Storage(Storage):
    """Any S3-compatible object store, via boto3.

    boto3 is an optional extra and is imported lazily, so the base install
    stays free of it. `client` is injectable, which is what makes this
    testable without a network or a bucket.
    """

    def __init__(self, bucket: str, prefix: str = "", client=None):
        self.bucket = bucket
        self.prefix = _clean_key(prefix) if prefix else ""
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3  # noqa: PLC0415
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "S3 storage needs boto3, which is not in the base install: "
                    "`uv sync --extra s3`."
                ) from exc
            self._client = boto3.client("s3")
        return self._client

    def _key(self, key: str) -> str:
        k = _clean_key(key)
        return f"{self.prefix}/{k}" if self.prefix else k

    def put(self, local: Path | str, key: str) -> str:
        full = self._key(key)
        self.client.upload_file(str(local), self.bucket, full)
        return f"s3://{self.bucket}/{full}"

    def get(self, key: str, local: Path | str) -> Path:
        dst = Path(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(dst))
        return dst

    def open(self, key: str, mode: str = "rb") -> IO:
        if "r" not in mode:
            raise ValueError("S3Storage.open is read-only; use put() to write")
        import io

        body = self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        return io.BytesIO(body)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> List[str]:
        full = self._key(prefix) if prefix else self.prefix
        out: List[str] = []
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kw["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                k = obj["Key"]
                if self.prefix and k.startswith(self.prefix + "/"):
                    k = k[len(self.prefix) + 1:]
                out.append(k)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return sorted(out)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def size(self, key: str) -> int:
        return int(self.client.head_object(Bucket=self.bucket, Key=self._key(key))["ContentLength"])

    def signed_url(self, key: str, expires: int = 3600) -> Optional[str]:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._key(key)},
            ExpiresIn=expires,
        )


class JobStore:
    """One job's namespace within a `Storage`.

    This is what makes concurrent jobs safe: every key is prefixed with the
    job id, so two jobs writing `cloud/fused_cloud.ply` cannot collide.
    """

    def __init__(self, storage: Storage, job_id: str):
        if not job_id or "/" in job_id or job_id in (".", ".."):
            raise ValueError(f"invalid job id: {job_id!r}")
        self.storage = storage
        self.job_id = job_id

    def _key(self, key: str) -> str:
        return f"{self.job_id}/{_clean_key(key)}"

    def put(self, local: Path | str, key: str) -> str:
        return self.storage.put(local, self._key(key))

    def get(self, key: str, local: Path | str) -> Path:
        return self.storage.get(self._key(key), local)

    def open(self, key: str, mode: str = "rb") -> IO:
        return self.storage.open(self._key(key), mode)

    def exists(self, key: str) -> bool:
        return self.storage.exists(self._key(key))

    def size(self, key: str) -> int:
        return self.storage.size(self._key(key))

    def delete(self, key: str) -> None:
        self.storage.delete(self._key(key))

    def list(self, prefix: str = "") -> List[str]:
        full = f"{self.job_id}/{_clean_key(prefix)}" if prefix else self.job_id
        keys = self.storage.list(full)
        return [k[len(self.job_id) + 1:] for k in keys if k.startswith(self.job_id + "/")]

    def signed_url(self, key: str, expires: int = 3600) -> Optional[str]:
        return self.storage.signed_url(self._key(key), expires)

    def upload_tree(self, local_root: Path | str, keys: Optional[Iterable[str]] = None) -> List[str]:
        """Push a finished local job directory into the store."""
        local_root = Path(local_root)
        uploaded = []
        for p in sorted(local_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(local_root).as_posix()
            if keys is not None and rel not in set(keys):
                continue
            self.put(p, rel)
            uploaded.append(rel)
        return uploaded

    def download_tree(self, local_root: Path | str, prefix: str = "") -> List[Path]:
        """Pull a job's artifacts down so subprocesses can read real files."""
        local_root = Path(local_root)
        out = []
        for key in self.list(prefix):
            out.append(self.get(key, local_root / key))
        return out


def from_uri(uri: str) -> Storage:
    """Build a backend from `file://…`, `s3://bucket/prefix`, or a bare path."""
    uri = str(uri)
    # A bare Windows path parses as scheme "c" (from "C:\..."), so it has to be
    # recognised before urlparse gets a chance to mangle it.
    if len(uri) > 1 and uri[1] == ":" and uri[0].isalpha():
        return LocalStorage(uri)

    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = parsed.path or str(uri)
        # file:///C:/x on Windows arrives as /C:/x.
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return LocalStorage(path)
    if parsed.scheme == "s3":
        return S3Storage(parsed.netloc, parsed.path.lstrip("/"))
    raise ValueError(f"unsupported storage scheme {parsed.scheme!r} in {uri!r}")
