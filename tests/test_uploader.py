import hashlib
import json
import os
import shutil
import tempfile
import unittest
import uuid

import pyarrow.parquet as pq

from chai.result import FileItemResult, ItemResult
from chai.storage import jsonl_to_parquet
from chai.workflow import Workflow


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def md5_of(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


RECORDS = [
    {"id": "a", "value": "Ada", "metadata": {"type": "TEXT"}, "score": 1},
    {"id": "b", "value": "Grace", "metadata": {"type": "TEXT"}, "score": 2},
]


class TestFileSystemUploader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.dest = os.path.join(self.dir, "dest")
        self.wf = Workflow({"id": "upload_wf", "type": "workflow.Workflow"})
        self.file = os.path.join(self.dir, "run.parquet")
        with open(self.file, "wb") as fh:
            fh.write(b"PAR1 not really parquet")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def uploader(self, **settings):
        settings.setdefault("directory", self.dest)
        return self.wf._make_step({"type": "uploader.FileSystemUploader", "settings": settings}, self.wf)

    def test_copies_the_file_and_says_where_it_went(self):
        out = self.uploader().process(FileItemResult(self.file))
        target = os.path.join(self.dest, "run.parquet")
        self.assertEqual(out.value["uri"], target)
        self.assertEqual(out.value["key"], "run.parquet")
        self.assertEqual(out.value["size"], os.path.getsize(self.file))
        self.assertEqual(out.value["md5"], md5_of(self.file))
        self.assertEqual(out.value["source"], self.file)
        self.assertEqual(md5_of(target), md5_of(self.file))
        self.assertEqual(os.listdir(self.dest), ["run.parquet"])

    def test_prefix_and_key_template_name_the_destination(self):
        out = self.uploader(prefix="incoming/", key="{workflow_id}/{stem}-x{ext}").process(ItemResult(self.file))
        self.assertEqual(out.value["key"], "incoming/upload_wf/run-x.parquet")
        self.assertTrue(os.path.exists(os.path.join(self.dest, "incoming", "upload_wf", "run-x.parquet")))

    def test_overwrite_false_refuses_to_replace_a_stored_file(self):
        self.uploader().process(ItemResult(self.file))
        self.uploader().process(ItemResult(self.file))  # overwriting is the default
        with self.assertRaises(FileExistsError):
            self.uploader(overwrite=False).process(ItemResult(self.file))

    def test_a_key_cannot_escape_the_directory(self):
        with self.assertRaises(ValueError):
            self.uploader(key="../{name}").process(ItemResult(self.file))

    def test_directory_is_required(self):
        with self.assertRaises(ValueError):
            self.wf._make_step({"type": "uploader.FileSystemUploader"}, self.wf)


class TestUploadWorkflow(unittest.TestCase):
    """The hand-off: .jsonl runs packed into Parquet by jsonl_to_parquet, then uploaded by a workflow."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.dest = os.path.join(self.dir, "dest")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_each_jsonl_file_is_converted_and_uploaded(self):
        files = []
        for n, records in enumerate((RECORDS[:1], RECORDS[1:])):
            jsonl = write_jsonl(os.path.join(self.dir, f"slice-{n}.jsonl"), records)
            files.append(jsonl_to_parquet(jsonl, os.path.join(self.dir, f"slice-{n}.parquet"))["path"])
        tree = {
            "id": "handoff_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "type": "provider.FileListProvider",
                    "input": files,
                    "steps": [
                        {
                            "type": "iterator.Iterator",
                            "steps": [
                                {
                                    "type": "uploader.FileSystemUploader",
                                    "settings": {"directory": self.dest, "prefix": "chai/"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        Workflow(tree).run()
        names = sorted(os.listdir(os.path.join(self.dest, "chai")))
        self.assertEqual(names, ["slice-0.parquet", "slice-1.parquet"])
        values = [pq.read_table(os.path.join(self.dest, "chai", n)).column("value").to_pylist() for n in names]
        self.assertEqual(values, [["Ada"], ["Grace"]])


class TestS3UploaderSettings(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "s3_wf", "type": "workflow.Workflow"})
        self._env = os.environ.pop("S3_BUCKET", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["S3_BUCKET"] = self._env

    def test_bucket_is_required(self):
        with self.assertRaises(ValueError):
            self.wf._make_step({"type": "uploader.S3Uploader"}, self.wf)

    def test_key_and_uri(self):
        up = self.wf._make_step(
            {"type": "uploader.S3Uploader", "settings": {"bucket": "b", "prefix": "in/", "key": "{stem}.pq"}},
            self.wf,
        )
        key = up.destination_key("/tmp/run.parquet")
        self.assertEqual(key, "in/run.pq")
        self.assertEqual(up.uri(key), "s3://b/in/run.pq")


S3_BUCKET = os.environ.get("CHAI_TEST_S3_BUCKET", "")


@unittest.skipUnless(S3_BUCKET, "set CHAI_TEST_S3_BUCKET (and AWS_ENDPOINT_URL for MinIO etc.) to test against S3")
class TestS3Uploader(unittest.TestCase):
    """Uploads to a real bucket; objects go under a fresh prefix that is deleted afterwards."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.prefix = f"chai-test/{uuid.uuid4().hex}/"
        self.wf = Workflow({"id": "s3_wf", "type": "workflow.Workflow"})
        self.file = write_jsonl(os.path.join(self.dir, "run.jsonl"), RECORDS)
        self.up = self.wf._make_step(
            {"type": "uploader.S3Uploader", "settings": {"bucket": S3_BUCKET, "prefix": self.prefix}}, self.wf
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        listed = self.up.client.list_objects_v2(Bucket=S3_BUCKET, Prefix=self.prefix)
        for obj in listed.get("Contents", []):
            self.up.client.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])

    def test_uploads_the_file_with_its_provenance(self):
        out = self.up.process(FileItemResult(self.file))
        key = f"{self.prefix}run.jsonl"
        self.assertEqual(out.value["uri"], f"s3://{S3_BUCKET}/{key}")
        obj = self.up.client.get_object(Bucket=S3_BUCKET, Key=key)
        self.assertEqual(hashlib.md5(obj["Body"].read()).hexdigest(), md5_of(self.file))
        self.assertEqual(obj["ContentType"], "application/x-ndjson")
        self.assertEqual(obj["Metadata"]["chai-md5"], md5_of(self.file))
        self.assertEqual(obj["Metadata"]["chai-workflow-id"], "s3_wf")

    def test_overwrite_false_refuses_to_replace_an_object(self):
        self.up.process(FileItemResult(self.file))
        self.up.settings["overwrite"] = False
        with self.assertRaises(FileExistsError):
            self.up.process(FileItemResult(self.file))


if __name__ == "__main__":
    unittest.main()
