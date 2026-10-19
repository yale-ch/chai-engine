"""Transcribe and translate every page of M.804, each page carrying the one before it as context.

Runs ``workflow-m804.json`` over a directory of page images. Per page:

* ``transcribe_context`` (a ``ContextWindow``) hands the transcriber the last line of the *previous*
  page's transcription along with this page's image, so a word or sentence running across the page
  break can be read correctly and the model stays with the scribe's hand and spelling;
* ``transcribe`` (Gemini) returns the page's text, and ``save_text`` writes it to
  ``results/m804/text/<folio>.txt``;
* ``translate_context`` (a second ``ContextWindow``) hands the translator the whole of the previous
  page's text along with this page's, so names, vocabulary and sentences carry over;
* ``translate`` returns the English, and ``save_translation`` writes it to
  ``results/m804/translation/<folio>.txt`` -- one file per page, named after the page image.

Both AI steps also log one JSON line per result to ``results/m804/run.jsonl`` (``log_transcribe`` and
``log_translate``) as it is produced: the page it came from, the text, and the result's metadata --
which is where each call's ``token_usage`` and ``duration`` live. That is what this script totals at
the end of a run, and what ``--usage-only`` totals afterwards without running anything. The totals
are costed at ``INPUT_PRICE``/``OUTPUT_PRICE`` below -- keep them current with the model's price.

Because each page's context is the page before it, the run is inherently sequential: the iterator
keeps ``workers: 1`` and the input cannot be divided between parallel runs with a ``SliceIterator``.
It writes each page's files as it goes and holds no results in memory (``retain_results: false``),
so a run that is interrupted keeps everything it had finished.

Usage (from the repository root):

    python examples/experiment-m804.py
    python examples/experiment-m804.py --limit 4            # the first 4 pages, to try it out
    python examples/experiment-m804.py --dir inputs/m804/split --model gemini-3.8-flash

    python examples/experiment-m804.py --usage-only           # total an earlier run's tokens and cost
    python examples/experiment-m804.py --usage-only --input-price 0.30 --output-price 2.50  # other rates
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

import ujson as json  # noqa: E402

from chai.result import Result  # noqa: E402
from chai.storage import token_usage_summary  # noqa: E402
from chai.workflow import Workflow  # noqa: E402

BRANCHES = ("steps", "next_steps", "error_steps", "true_steps", "false_steps")

# Component ids from workflow-m804.json
PROVIDER_ID = "m804_pages"
ITERATOR_ID = "each_page"
AI_IDS = ("transcribe", "translate")
OUTPUT_IDS = {"save_text": "text", "save_translation": "translation"}
LOG_IDS = ("log_transcribe", "log_translate")

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")

# Dollars per million tokens for the model the workflow uses, so a run reports what it cost.
# Everything sent (the page image included) is charged at the input price; everything returned, the
# thinking tokens with it, at the output price. Override either with --input-price/--output-price.
# Model prices change: check the provider's pricing page before trusting a total.
INPUT_PRICE = 0.75
OUTPUT_PRICE = 3.75


def find_step(tree, cid):
    """Find the config dict for the step with id *cid* anywhere in a workflow config *tree*."""
    if tree.get("id", None) == cid:
        return tree
    for key in BRANCHES:
        for step in tree.get(key, []):
            found = find_step(step, cid)
            if found is not None:
                return found
    return None


def find_result(result, component):
    """Find the result *component* produced, searching *result* and its nested values."""
    if not isinstance(result, Result):
        return None
    if result.processor is component:
        return result
    # Never touch a FileItemResult's value; reading it would pull the file off disk
    if not getattr(result, "file_name", "") and type(result.value) is list:
        for child in result.value:
            found = find_result(child, component)
            if found is not None:
                return found
    return None


def page_files(directory):
    """The page images in *directory*, in the order the provider would read them."""
    names = [x for x in os.listdir(directory) if x.lower().endswith(IMAGE_TYPES)]
    return sorted(os.path.join(directory, x) for x in names)


def report_usage(log_file, input_price, output_price):
    """Print what the run sent to the model, and what it cost at the configured prices."""
    if not os.path.exists(log_file):
        print(f"No run log at {log_file}, so there is nothing to total")
        return
    summary = token_usage_summary(log_file, input_price=input_price, output_price=output_price)
    if not summary["components"]:
        print(f"{summary['records']} results logged, none of which recorded any token usage")
        return
    print(f"\nToken usage ({log_file}):")
    print(f"  {'step':<20}{'calls':>7}{'in':>12}{'thinking':>10}{'out':>10}{'total':>12}{'cost':>12}")
    for step, counts in list(summary["components"].items()) + [("ALL", summary["total"])]:
        print(
            f"  {step:<20}{counts['calls']:>7,}{counts['input']:>12,}"
            f"{counts['thinking']:>10,}{counts['output']:>10,}{counts['total']:>12,}"
            f"{'$' + format(counts['cost'], '.2f'):>12}"
        )
    if summary["without_usage"]:
        print(f"  ({summary['without_usage']} of {summary['records']} results recorded no usage)")
    print(
        f"  at ${input_price:g} per million tokens in and ${output_price:g} out, "
        f"thinking billed as output"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workflow", default=os.path.join(ROOT, "workflow-m804.json"))
    ap.add_argument("--dir", default=None, help="directory of page images (default: the workflow's own)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages; 0 for all of them")
    ap.add_argument("--text-dir", default=None, help="where the transcriptions go")
    ap.add_argument("--translation-dir", default=None, help="where the translations go")
    ap.add_argument("--model", default=None, help="Gemini model for both steps")
    ap.add_argument("--log", default=None, help="where the per-result JSON-Lines log goes")
    ap.add_argument(
        "--input-price",
        type=float,
        default=INPUT_PRICE,
        help=f"dollars per million input tokens (default {INPUT_PRICE:g})",
    )
    ap.add_argument(
        "--output-price",
        type=float,
        default=OUTPUT_PRICE,
        help=f"dollars per million output tokens (default {OUTPUT_PRICE:g})",
    )
    ap.add_argument(
        "--usage-only",
        action="store_true",
        help="total the tokens in an existing run log and stop, without running anything",
    )
    args = ap.parse_args()

    with open(args.workflow) as fh:
        tree = json.load(fh)

    # Both AI steps log to one file, so it is resolved (and overridden) in one place
    log_file = ""
    for cid in LOG_IDS:
        step = find_step(tree, cid)
        if step is None:
            raise SystemExit(f"No '{cid}' step in {args.workflow}")
        settings = step.setdefault("settings", {})
        if args.log:
            settings["file"] = args.log
        elif not os.path.isabs(settings.get("file", "")):
            settings["file"] = os.path.join(ROOT, settings.get("file", "results.jsonl"))
        log_file = settings["file"]

    if args.usage_only:
        report_usage(log_file, args.input_price, args.output_price)
        return

    provider = find_step(tree, PROVIDER_ID)
    if provider is None:
        raise SystemExit(f"No '{PROVIDER_ID}' step in {args.workflow}")
    if args.dir is not None:
        provider["input"] = args.dir
    directory = provider["input"]
    if not os.path.isabs(directory) and not os.path.exists(directory):
        # Paths in the config are relative to the repository root, not the current directory
        directory = provider["input"] = os.path.join(ROOT, directory)
    if not os.path.isdir(directory):
        raise SystemExit(f"No such directory of page images: {directory}")

    pages = page_files(directory)
    if not pages:
        raise SystemExit(f"No page images in {directory}")
    if args.limit:
        # A directory provider reads the whole directory; an explicit list is how a run takes part
        # of it -- and the pages must stay in folio order for the page-to-page context to mean anything
        pages = pages[: args.limit]
        provider["type"] = "provider.FileListProvider"
        provider["input"] = pages

    for cid in AI_IDS:
        step = find_step(tree, cid)
        if step is None:
            raise SystemExit(f"No '{cid}' step in {args.workflow}")
        if args.model:
            step.setdefault("settings", {})["model"] = args.model

    directories = {}
    for cid, kind in OUTPUT_IDS.items():
        step = find_step(tree, cid)
        if step is None:
            raise SystemExit(f"No '{cid}' step in {args.workflow}")
        settings = step.setdefault("settings", {})
        override = args.text_dir if kind == "text" else args.translation_dir
        if override:
            settings["directory"] = override
        elif not os.path.isabs(settings.get("directory", "")):
            settings["directory"] = os.path.join(ROOT, settings.get("directory", "results"))
        directories[kind] = settings["directory"]

    iterator = find_step(tree, ITERATOR_ID)
    workers = int(iterator.get("settings", {}).get("workers", 1) or 1)
    if workers > 1:
        # Each page's context is the page before it, so the pages must be done in order
        raise SystemExit(f"'{ITERATOR_ID}' must run with workers: 1 to carry context between pages")

    wf = Workflow(tree)
    res = wf.run()

    iter_result = find_result(res, wf.get_component_by_id(ITERATOR_ID))
    processed = iter_result.metadata.get("processed", 0) if iter_result else 0
    errors = iter_result.metadata.get("errors", 0) if iter_result else 0
    print(f"Processed {processed} of {len(pages)} pages ({errors} failed)")
    for kind, path in directories.items():
        written = len([x for x in os.listdir(path) if x.endswith(".txt")]) if os.path.isdir(path) else 0
        print(f"  {written} {kind} files in {path}")
    report_usage(log_file, args.input_price, args.output_price)


if __name__ == "__main__":
    main()
