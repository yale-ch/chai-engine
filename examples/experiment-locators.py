"""Store one row per sentence that points back into the document, without materializing sentences.

A run over a directory of documents: iterate the files, split each into sentences, and store a result
per sentence. Nothing about a sentence is written anywhere except its row -- there is no file, crop or
document per sentence -- and yet every row can be traced to the exact characters it came from:

* the file iterator is marked ``"source": true``, so every row records *that document* as the input
  it was generated from (``input_uri``) plus the md5 of the document's content (``input_hash``),
  rather than whatever step happened to precede it;
* the segmenter runs with ``locate: true``, so each sentence carries the character range it occupies
  in the document, and the storage collects those into ``input_locator``.

A UI holding a stored row therefore has everything it needs: which document to open, whether that
document still hashes the same, and where in it to highlight. The same shape works for images -- a
YOLO region's locator is its bounding box -- and the frames stack when a run segments a segment.

Usage (from the repository root):

    python examples/experiment-locators.py
    python examples/experiment-locators.py --dir path/to/documents --db results/sentences.db
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

from chai.storage import list_results  # noqa: E402
from chai.workflow import Workflow  # noqa: E402

SAMPLES = {
    "hodgkin.txt": (
        "Eliot Hodgkin painted in tempera. He collected fossils and shells. "
        "The 1990 catalogue was introduced by Sir Brinsley Ford."
    ),
    "lovelace.txt": (
        "Ada Lovelace wrote the first algorithm. Babbage called her the enchantress of numbers. "
        "Her notes are longer than the paper they annotate."
    ),
}


def sample_documents(directory):
    """Write the sample documents, so the example runs with nothing else set up."""
    os.makedirs(directory, exist_ok=True)
    for name, text in SAMPLES.items():
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write(text)
    return directory


def build_workflow(directory, database):
    """Read the directory, split each document into located sentences, store one row per sentence."""
    return {
        "id": "locators_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "id": "docs",
                "type": "provider.DirFileProvider",
                "input": directory,
                "steps": [
                    {
                        "id": "each_doc",
                        "type": "iterator.Iterator",
                        # The document is what every row downstream says it was generated from
                        "source": True,
                        "steps": [
                            {
                                "id": "read",
                                "type": "transcriber.TextFileTranscriber",
                                # a transcriber and a segmenter make their own result, so what comes
                                # next runs on that output: next_steps, not steps
                                "next_steps": [
                                    {
                                        "id": "sentences",
                                        "type": "segmenter.TextSegmenter",
                                        # Each sentence keeps its character range instead of a copy
                                        "settings": {"mode": "sentence", "locate": True},
                                        "next_steps": [
                                            {
                                                "id": "each_sentence",
                                                "type": "iterator.Iterator",
                                                "steps": [
                                                    {
                                                        "id": "topics",
                                                        "type": "classifier.KeywordClassifier",
                                                        "settings": {
                                                            "labels": {
                                                                "people": ["ford", "lovelace", "babbage", "hodgkin"],
                                                                "objects": ["fossils", "shells", "catalogue", "paper"],
                                                            }
                                                        },
                                                        "next_steps": [
                                                            {
                                                                "id": "store",
                                                                "type": "storage.SqliteStorage",
                                                                "settings": {"database": database},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=os.path.join(ROOT, "results", "locator-docs"))
    parser.add_argument("--db", default=os.path.join(ROOT, "results", "locators.db"))
    args = parser.parse_args()

    if args.dir == parser.get_default("dir"):
        sample_documents(args.dir)
    if os.path.exists(args.db):
        os.remove(args.db)  # development: the database is rebuilt rather than migrated

    Workflow(build_workflow(args.dir, args.db)).run()

    rows = [r for r in list_results(args.db, processor_id="topics", limit=1000) if r["input_locator"]]
    print(f"{len(rows)} stored rows, not one of which materialized its sentence\n")
    for row in sorted(rows, key=lambda r: (r["input_uri"], r["input_locator"][0]["start"])):
        frame = row["input_locator"][0]
        with open(row["input_uri"]) as fh:  # what a UI would do with the row: go back to the source
            quoted = fh.read()[frame["start"] : frame["end"]]
        labels = ",".join(row["value"]["value"]) or "-"
        print(f"  {os.path.basename(row['input_uri'])} [{row['input_hash'][:8]}] "
              f"{frame['start']:>4}-{frame['end']:<4} {labels:<15} {quoted}")


if __name__ == "__main__":
    sys.exit(main())
