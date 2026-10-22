"""Uploaders: components that put a local file somewhere another system can collect it.

An uploader takes a file (a ``FileItemResult``, or a result whose value is a path) and copies it to a
destination -- an S3 bucket, a shared directory -- returning a result that says where it went. This
is the hand-off at the end of a run: ``chai.persistence.jsonl_to_parquet`` (or a ``ParquetStorage``) packs the
run into a Parquet file, an ``S3Uploader`` puts that file in a bucket, and the system that maintains
the central database ingests it from there.

The output is an ``ItemResult`` whose value is ``{"uri", "key", "size", "md5", "source"}``: the
destination URI, the key the file was stored under, its size in bytes, the md5 of its content and
the local path it was uploaded from.
"""

import hashlib
import logging
import os
import shutil
import threading
from datetime import datetime, timezone

from .core import Component, Result
from .result import FileItemResult, ItemResult

logger = logging.getLogger("chai")


def result_file_path(result):
    """The local path *result* stands for: a ``FileItemResult``'s file, or a path given as the value.

    Never reads a ``FileItemResult``'s content. Raises ``ValueError`` when the result names no file.
    """
    if isinstance(result, FileItemResult):
        path = result.file_name
    elif isinstance(result, Result):
        path = result.value
    else:
        path = result
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path:
        raise ValueError(f"Expected a file, or a result whose value is a file path; got {result!r}")
    return path


def file_md5(path, chunk_size=1 << 20):
    """The hex md5 of the file at *path*, read in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Uploader(Component):
    """Copy a local file to a destination and return where it went.

    Subclasses implement ``upload(path, key, info)``, which stores the file at *path* under *key* and
    returns its URI, and may implement ``exists(key)`` so that ``overwrite: false`` can refuse to
    replace a file already there. ``info`` carries the file's ``size``/``md5`` and the run's
    provenance (``workflow_id``, ``component_id``, ``source``) for a destination that can record them
    alongside the file.

    Settings:
        - key: the name to store the file under; ``{name}`` (the file's name), ``{stem}``, ``{ext}``,
          ``{workflow_id}``, ``{date}`` (UTC, YYYY-MM-DD) and ``{timestamp}`` (UTC,
          YYYYMMDDTHHMMSSZ) are filled in (default '{name}')
        - prefix: prepended to the key, e.g. 'incoming/chai/' (default '')
        - overwrite: replace a file already stored under the same key (default true); false raises
          ``FileExistsError`` instead
    """

    def destination_key(self, path):
        """The key *path* is stored under: ``prefix`` plus the ``key`` template filled in."""
        name = os.path.basename(path)
        stem, ext = os.path.splitext(name)
        now = datetime.now(timezone.utc)
        key = str(self.settings.get("key") or "{name}").format(
            name=name,
            stem=stem,
            ext=ext,
            workflow_id=self.workflow.id if self.workflow else "",
            date=now.strftime("%Y-%m-%d"),
            timestamp=now.strftime("%Y%m%dT%H%M%SZ"),
        )
        return f"{self.settings.get('prefix', '')}{key}"

    def exists(self, key):
        """Whether something is already stored under *key*; ``None`` when the destination cannot tell."""
        return None

    def upload(self, path, key, info):
        raise NotImplementedError(f"{self.__class__.__name__} must implement upload()")

    def _process(self, input: Result) -> Result:
        path = result_file_path(input)
        if not os.path.isfile(path):
            raise ValueError(f"{self} cannot upload {path}: no such file")
        key = self.destination_key(path)
        if not self.settings.get("overwrite", True) and self.exists(key):
            raise FileExistsError(f"{self} will not overwrite {key}: it already exists")
        info = {
            "size": os.path.getsize(path),
            "md5": file_md5(path),
            "source": os.path.abspath(path),
            "workflow_id": self.workflow.id if self.workflow else "",
            "component_id": self.id,
        }
        uri = self.upload(path, key, info)
        logger.info(f"{self} uploaded {path} to {uri}")
        value = {
            "uri": uri,
            "key": key,
            "size": info["size"],
            "md5": info["md5"],
            "source": info["source"],
        }
        return ItemResult(value, input=input, processor=self, metadata={"type": "DATA"})


class FileSystemUploader(Uploader):
    """Copy the file into a directory -- a mounted share, or a drop folder another system watches.

    The copy is written under a temporary name and renamed into place, so a watcher never sees a
    partial file. The key may contain '/' to place the file in subdirectories, which are created.

    Settings (as well as ``key``/``prefix``/``overwrite``, see ``Uploader``):
        - directory: the destination directory (required)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not self.settings.get("directory"):
            raise ValueError(f"{self!r} needs the `directory` setting")
        self.directory = os.path.abspath(self.settings["directory"])

    def _target(self, key):
        target = os.path.abspath(os.path.join(self.directory, key))
        if os.path.commonpath([self.directory, target]) != self.directory:
            raise ValueError(f"{self} key {key!r} points outside {self.directory}")
        return target

    def exists(self, key):
        return os.path.exists(self._target(key))

    def upload(self, path, key, info):
        target = self._target(key)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        partial = f"{target}.partial"
        shutil.copyfile(path, partial)
        os.replace(partial, target)
        return target


class S3Uploader(Uploader):
    """Upload the file to an Amazon S3 bucket (or any S3-compatible store, e.g. MinIO).

    Uses ``boto3`` (imported when the first file is uploaded) and its usual credential chain --
    environment variables, ``~/.aws`` profiles, an instance or task role -- so no secrets live in the
    workflow config. Large files are sent as a multipart upload; either way the object only appears in
    the bucket once it is complete, so an ingester listing the prefix never sees a partial file. The
    object is tagged with S3 metadata (``chai-md5``, ``chai-workflow-id``, ``chai-component-id``,
    ``chai-source``) so the far side can check what it received and where it came from. Returns
    ``s3://bucket/key`` as the output's ``uri``.

    Settings (as well as ``key``/``prefix``/``overwrite``, see ``Uploader``):
        - bucket: the bucket to upload to (default: the ``S3_BUCKET`` environment variable)
        - region: AWS region (default: boto3's, e.g. ``AWS_REGION``/``AWS_DEFAULT_REGION``)
        - profile: named AWS profile to take credentials from (default: boto3's)
        - endpoint_url: for an S3-compatible service other than AWS (default: boto3's, which honours
          ``AWS_ENDPOINT_URL``)
        - extra_args: dict passed to boto3 as ``ExtraArgs`` -- e.g. ``{"ServerSideEncryption":
          "aws:kms", "StorageClass": "STANDARD_IA"}``; its ``Metadata`` is merged with chai's
        - content_type: the object's Content-Type (default 'application/vnd.apache.parquet' for
          ``.parquet``, 'application/x-ndjson' for ``.jsonl``, otherwise
          'application/octet-stream')
    """

    CONTENT_TYPES = {
        ".parquet": "application/vnd.apache.parquet",
        ".jsonl": "application/x-ndjson",
        ".json": "application/json",
        ".csv": "text/csv",
    }

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.bucket = self.settings.get("bucket") or os.environ.get("S3_BUCKET", "")
        if not self.bucket:
            raise ValueError(
                f"{self!r} needs the `bucket` setting (or S3_BUCKET in the environment)"
            )
        self._client = None
        self._client_lock = threading.Lock()

    @property
    def client(self):
        """The boto3 S3 client, made on first use; boto3 clients are safe to share across threads."""
        with self._client_lock:
            if self._client is None:
                try:
                    import boto3
                except ImportError as e:  # pragma: no cover - depends on the install
                    raise ImportError(
                        "S3Uploader requires boto3: pip install boto3"
                    ) from e
                session = boto3.session.Session(
                    profile_name=self.settings.get("profile") or None,
                    region_name=self.settings.get("region") or None,
                )
                self._client = session.client(
                    "s3", endpoint_url=self.settings.get("endpoint_url") or None
                )
            return self._client

    def uri(self, key):
        return f"s3://{self.bucket}/{key}"

    def exists(self, key):
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                return False
            raise

    def upload(self, path, key, info):
        extra = dict(self.settings.get("extra_args") or {})
        ext = os.path.splitext(path)[1].lower()
        extra.setdefault(
            "ContentType",
            self.settings.get("content_type")
            or self.CONTENT_TYPES.get(ext, "application/octet-stream"),
        )
        # S3 user metadata must be ASCII; a non-ASCII local path is not worth failing the upload over
        metadata = {
            "chai-md5": info["md5"],
            "chai-workflow-id": info["workflow_id"],
            "chai-component-id": info["component_id"],
            "chai-source": info["source"].encode("ascii", "replace").decode("ascii"),
        }
        metadata.update(extra.get("Metadata") or {})
        extra["Metadata"] = metadata
        self.client.upload_file(path, self.bucket, key, ExtraArgs=extra)
        return self.uri(key)
