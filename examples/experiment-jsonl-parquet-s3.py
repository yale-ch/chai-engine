"""Pack a run's JSON-Lines output into Parquet and upload it to S3 for ingestion into the central db.

A long run streams its results into a ``.jsonl`` file with ``JsonLinesStorage`` as it goes (see
``experiment-ppl-jsonl.py``). This workflow is the hand-off afterwards:

* ``FileListProvider`` + ``Iterator`` take one or more ``.jsonl`` files (e.g. one per slice of a
  ``SliceIterator`` run);
* ``JsonLinesToParquet`` converts each into a ``.parquet`` file next to it (or into ``--out-dir``);
* ``S3Uploader`` puts each Parquet file in the bucket under ``--prefix``, tagged with its md5 and the
  workflow it came from.

The system that maintains the central database picks the Parquet files up from the bucket and
ingests them; nothing here talks to that database. Credentials come from boto3's usual chain
(``AWS_PROFILE``, ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``, an instance role, ...).

With ``--directory`` the files are copied into a local directory instead (``FileSystemUploader``),
which is a way to try the workflow without a bucket.

Usage (from the repository root):

    python examples/experiment-jsonl-parquet-s3.py results/run.jsonl --bucket my-bucket --prefix incoming/chai/
    python examples/experiment-jsonl-parquet-s3.py results/slice-*.jsonl --bucket my-bucket --region us-east-1
    python examples/experiment-jsonl-parquet-s3.py results/run.jsonl --directory /tmp/dropbox
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

from chai.workflow import Workflow  # noqa: E402


def build_workflow(jsonl_files, uploader, out_dir=None):
    """The workflow config: for each .jsonl file, convert it to Parquet, then upload the Parquet file."""
    converter_settings = {"compression": "zstd"}
    if out_dir:
        converter_settings["directory"] = out_dir
    return {
        "id": "jsonl_parquet_upload_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "id": "jsonl_files",
                "type": "provider.FileListProvider",
                "input": list(jsonl_files),
                "steps": [
                    {
                        "id": "each_jsonl",
                        "type": "iterator.Iterator",
                        "steps": [
                            {
                                "id": "to_parquet",
                                "type": "converter.JsonLinesToParquet",
                                "settings": converter_settings,
                                "next_steps": [uploader],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def s3_uploader(bucket, prefix="", region=None, endpoint_url=None, profile=None):
    """The ``S3Uploader`` step: each Parquet file goes to s3://bucket/prefix/<date>/<name>."""
    settings = {"bucket": bucket, "prefix": prefix, "key": "{date}/{name}", "overwrite": False}
    for name, value in (("region", region), ("endpoint_url", endpoint_url), ("profile", profile)):
        if value:
            settings[name] = value
    return {"id": "to_s3", "type": "uploader.S3Uploader", "settings": settings}


def directory_uploader(directory, prefix=""):
    """The ``FileSystemUploader`` step, for trying the workflow without a bucket."""
    return {
        "id": "to_directory",
        "type": "uploader.FileSystemUploader",
        "settings": {"directory": directory, "prefix": prefix, "key": "{date}/{name}"},
    }


def uploaded(result):
    """Every upload record ({"uri", "key", "size", "md5", "source"}) anywhere in the run's output."""
    value = result.value
    if isinstance(value, dict) and "uri" in value:
        yield value
    elif isinstance(value, list):
        for child in value:
            if hasattr(child, "value"):
                yield from uploaded(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("jsonl", nargs="+", help=".jsonl file(s) to convert and upload")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--bucket", help="S3 bucket to upload to")
    target.add_argument("--directory", help="copy into this directory instead of uploading to S3")
    parser.add_argument("--prefix", default="", help="key prefix, e.g. incoming/chai/")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint (MinIO, LocalStack, ...)")
    parser.add_argument("--profile", help="AWS profile to take credentials from")
    parser.add_argument("--out-dir", help="write the .parquet files here (default: next to each .jsonl)")
    args = parser.parse_args()

    if args.bucket:
        uploader = s3_uploader(args.bucket, args.prefix, args.region, args.endpoint_url, args.profile)
    else:
        uploader = directory_uploader(args.directory, args.prefix)
    result = Workflow(build_workflow(args.jsonl, uploader, args.out_dir)).run()
    for record in uploaded(result):
        print(f"{record['source']} -> {record['uri']} ({record['size']} bytes, md5 {record['md5']})")


if __name__ == "__main__":
    main()
