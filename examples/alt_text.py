"""Write accessibility alt-text for Yale Center for British Art objects.

Two stages -- plain Python, then a chai workflow:

1. **Catalog** (plain Python, outside chai): each catalog URL
   (``https://collections.britishart.yale.edu/catalog/<record id>``) is fetched as JSON
   (``<catalog url>.json``) for the object's collection, object type(s) and IIIF manifest. The
   object type lives in a different field per collection:

       Paintings and Sculpture / Prints and Drawings  ->  object_name_ss
       Rare Books and Manuscripts                     ->  form_genre_ss
       Archives                                       ->  genre_ss

2. **Alt-text** (chai, ``workflow-alt-text.json``): ``IIIFDirFileProvider`` downloads the first
   image of the manifest (``limit: 1``) and ``GeminiDescriber`` writes 4-6 plain sentences that open
   with the object type and otherwise describe only what is visible. The catalogued type goes into
   the prompt's ``{ADDITIONAL_CONTEXT}``.

The workflow is built once per object (each gets its own image directory and object type), and
objects run concurrently on a thread pool. Ported from chai-alt-text's ``run_alt_text.py``.

Run it (from the repository root)::

    python examples/alt_text.py --urls urls.csv                 # a CSV with a `url` column
    python examples/alt_text.py --urls urls.csv --limit 10 --out results/alt_text
    python examples/alt_text.py https://collections.britishart.yale.edu/catalog/tms:1000

Needs the model's credentials in ``.env`` (``GOOGLE_CLOUD_PROJECT`` for Vertex AI, or
``GEMINI_API_KEY``) and network access to the YCBA catalog and IIIF servers.
``tests/test_example_alt_text.py`` runs the same functions offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

from chai.result import FileItemResult, ItemResult, ListResult, Result  # noqa: E402
from chai.workflow import Workflow  # noqa: E402

WORKFLOW_FILE = os.path.join(ROOT, "examples", "workflow-alt-text.json")
MANIFEST_BASE = "https://manifests.collections.yale.edu/ycba/"
HTTP_TIMEOUT = 60
# The catalog sits behind Cloudflare bot management, which returns 403 to python-requests' user
# agent (and to custom ones) but lets curl through.
HTTP_HEADERS = {"Accept": "application/json", "User-Agent": "curl/8.7.1 (chai-alt-text)"}

# Which catalog field holds the object type(s), per collection. The first field that is present
# and non-empty wins; the others are fallbacks for a record of an unexpected shape.
OBJECT_TYPE_FIELDS = {
    "Paintings and Sculpture": ["object_name_ss", "genre_ss", "form_genre_ss"],
    "Prints and Drawings": ["object_name_ss", "genre_ss", "form_genre_ss"],
    "Rare Books and Manuscripts": ["form_genre_ss", "object_name_ss", "genre_ss"],
    "Archives": ["genre_ss", "form_genre_ss", "object_name_ss"],
}
DEFAULT_OBJECT_TYPE_FIELDS = ["object_name_ss", "form_genre_ss", "genre_ss"]

OBJECT_TYPE_KNOWN = (
    "The museum catalogue lists the object type(s) of this work as: {types}.\n"
    "Your first sentence must open by naming this object type. If several object types are listed, "
    'combine them naturally into one opening phrase (for example "engraving" + "map" becomes '
    '"An engraved map of ...").'
)
OBJECT_TYPE_UNKNOWN = (
    "The museum catalogue does not record an object type for this work.\n"
    "Your first sentence must open by naming the kind of object you can see (for example "
    '"A photograph of ...", "A page of handwritten text ...", "A printed document ...").'
)

CATALOG_ID_RE = re.compile(r"/catalog/([A-Za-z_]+:[A-Za-z0-9]+)")
TMS_ID_RE = re.compile(r"tms:(\d+)")


# ----------------------------------------------------------------------------- 1. catalog


def read_catalog_urls(paths: List[str], limit: Optional[int] = None) -> List[str]:
    """The catalog URLs in one or more CSVs (first column, optional `url` header), de-duplicated."""
    urls: List[str] = []
    for path in paths:
        with open(path, newline="") as fh:
            for i, row in enumerate(csv.reader(fh)):
                url = row[0].strip() if row else ""
                if not url or (i == 0 and url.lower() == "url") or url in urls:
                    continue
                urls.append(url)
    return urls[:limit] if limit else urls


def fetch_catalog_record(catalog_url: str, retries: int = 2) -> dict:
    """GET ``<catalog_url>.json`` and return its Solr ``document``."""
    url = catalog_url.rstrip("/")
    if not url.endswith(".json"):
        url += ".json"
    last = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            doc = (data.get("response") or {}).get("document") or data.get("document") or data
            if not isinstance(doc, dict) or not doc:
                raise ValueError("catalog JSON has no document")
            return doc
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"catalog fetch failed: {type(last).__name__}: {last}")


def _first(values):
    if isinstance(values, (list, tuple)):
        return values[0] if values else None
    return values


def clean_object_type(value) -> str:
    """``"Engravings -- Hand-colored."`` -> ``"Engravings (Hand-colored)"``; date qualifiers dropped."""
    s = str(value or "").strip().rstrip(".").strip()
    if not s:
        return ""
    head, *quals = [p.strip().rstrip(".").strip() for p in s.split("--")]
    quals = [q for q in quals if q and not re.match(r"^\d{4}", q)]
    return f"{head} ({', '.join(quals)})" if quals else head


def extract_object_types(doc: dict):
    """``(object_types, field_used)`` from a catalog document, per its collection."""
    collection = _first(doc.get("collection_ss")) or ""
    for field in OBJECT_TYPE_FIELDS.get(collection, DEFAULT_OBJECT_TYPE_FIELDS):
        raw = doc.get(field)
        if not raw:
            continue
        types: List[str] = []
        for value in raw if isinstance(raw, (list, tuple)) else [raw]:
            t = clean_object_type(value)
            if t and t.lower() not in {x.lower() for x in types}:
                types.append(t)
        if types:
            return types, field
    return [], None


def resolve_object(catalog_url: str, fetch: Callable[[str], dict] = fetch_catalog_record) -> dict:
    """Everything the alt-text step needs about one catalog record.

    When the catalog cannot be fetched a ``tms:`` record still gets its manifest
    (``obj/<id>``), with no object type, so a catalog outage degrades rather than fails.
    """
    m = CATALOG_ID_RE.search(catalog_url)
    catalog_id = m.group(1) if m else None
    info = {
        "catalog_url": catalog_url,
        "catalog_id": catalog_id,
        "manifest_url": None,
        "collection": None,
        "title": None,
        "object_types": [],
        "object_type_field": None,
        "catalog_error": "",
    }
    if not catalog_id:
        info["catalog_error"] = "could not parse record id from catalog url"
        return info
    tms = TMS_ID_RE.search(catalog_id)
    fallback = f"{MANIFEST_BASE}obj/{tms.group(1)}" if tms else None
    try:
        doc = fetch(catalog_url)
    except Exception as e:  # noqa: BLE001
        info["catalog_error"] = str(e)
        info["manifest_url"] = fallback
        return info
    info["collection"] = _first(doc.get("collection_ss"))
    info["title"] = _first(doc.get("title_ss")) or _first(doc.get("title_txt"))
    info["object_types"], info["object_type_field"] = extract_object_types(doc)
    manifest = _first(doc.get("manifest_ss"))
    if manifest:
        info["manifest_url"] = manifest if manifest.startswith("http") else MANIFEST_BASE + manifest.lstrip("/")
    else:
        info["manifest_url"] = fallback
    return info


def object_type_context(object_types: List[str]) -> str:
    """The prompt's ``{ADDITIONAL_CONTEXT}``: the catalogued object type(s), or what to do without one."""
    if object_types:
        return OBJECT_TYPE_KNOWN.format(types="; ".join(object_types))
    return OBJECT_TYPE_UNKNOWN


# ----------------------------------------------------------------------------- 2. alt-text (chai)


def alt_text_workflow(directory: str, model: Optional[str] = None, **settings) -> Workflow:
    """``workflow-alt-text.json`` downloading into *directory*; *settings* override the describer's."""
    with open(WORKFLOW_FILE) as fh:
        tree = json.load(fh)
    provider = tree["steps"][0]
    provider["settings"]["directory"] = directory
    describe = provider["steps"][0]["steps"][0]
    if model:
        describe["settings"]["model"] = model
    describe["settings"].update(settings)
    return Workflow(tree)


def describer_results(result) -> List[ItemResult]:
    """Every ``describe`` output in a finished run's result tree."""
    found = []
    if isinstance(result, ItemResult) and getattr(result.processor, "id", None) == "describe":
        found.append(result)
    elif isinstance(result, ListResult):
        for child in result.value:
            if isinstance(child, Result):
                found.extend(describer_results(child))
    return found


def source_file(result) -> Optional[str]:
    """The image file a result was made from (walking its provenance)."""
    while isinstance(result, Result):
        if isinstance(result, FileItemResult):
            return str(result.file_path)
        result = result.input
    return None


def describe_object(
    catalog_url: str,
    downloads: str,
    model: Optional[str] = None,
    fetch: Callable[[str], dict] = fetch_catalog_record,
    **settings,
) -> dict:
    """Look up one catalog record and write the alt-text for its first image: one output row."""
    row = resolve_object(catalog_url, fetch=fetch)
    row.update({"image_file": None, "alt_text": None, "status": "error", "error": row["catalog_error"]})
    if not row["manifest_url"]:
        row["error"] = row["error"] or "no manifest for this record"
        return row
    directory = os.path.join(downloads, re.sub(r"[^A-Za-z0-9._-]+", "_", row["catalog_id"]))
    try:
        wf = alt_text_workflow(directory, model=model, **settings)
        wf.get_component_by_id("describe").substitutions["ADDITIONAL_CONTEXT"] = object_type_context(
            row["object_types"]
        )
        found = describer_results(wf.run(row["manifest_url"]))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    if not found:
        row["error"] = "no alt-text produced"
        return row
    res = found[0]
    usage = res.metadata.get("token_usage") or {}
    row.update(
        {
            "image_file": source_file(res),
            "alt_text": res.value if isinstance(res.value, str) else str(res.value),
            "status": "ok",
            "model": res.metadata.get("model"),
            "tokens": usage.get("total"),
            "cost": res.metadata.get("cost"),
            "duration": round(res.metadata.get("duration") or 0, 2),
        }
    )
    return row


def describe_objects(catalog_urls: List[str], downloads: str, workers: int = 8, **kw) -> List[dict]:
    """``describe_object`` for every URL, *workers* at a time, in input order."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda url: describe_object(url, downloads, **kw), catalog_urls))


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Write YCBA alt-text with chai + Gemini.")
    ap.add_argument("catalog_urls", nargs="*", help="catalog URLs (in addition to --urls)")
    ap.add_argument("--urls", action="append", default=[], metavar="CSV", help="CSV of catalog URLs (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N URLs; 0 for all of them")
    ap.add_argument("--workers", type=int, default=8, help="objects described at once (default 8)")
    ap.add_argument("--model", default=None, help="Gemini model id (default: the workflow's)")
    ap.add_argument("--out", default="results/alt_text", help="output directory (default results/alt_text)")
    args = ap.parse_args(argv)

    urls = list(dict.fromkeys(args.catalog_urls + read_catalog_urls(args.urls)))
    urls = urls[: args.limit] if args.limit else urls
    if not urls:
        ap.error("give catalog URLs or --urls CSV")
    os.makedirs(args.out, exist_ok=True)

    started = time.time()
    rows = describe_objects(urls, os.path.join(args.out, "downloads"), workers=args.workers, model=args.model)
    with open(os.path.join(args.out, "alt_text.json"), "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    for row in rows:
        types = "; ".join(row["object_types"]) or "(no object type)"
        print(f"\n{row['catalog_id']}  {row['collection']} | {types}")
        print(f"  {row['alt_text']}" if row["status"] == "ok" else f"  ERROR: {row['error']}")
    ok = [r for r in rows if r["status"] == "ok"]
    cost = sum(r.get("cost") or 0 for r in ok)
    print(f"\n{len(ok)}/{len(rows)} described in {time.time() - started:.1f}s, ${cost:.4f}")
    print(f"Results: {os.path.join(args.out, 'alt_text.json')}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
    sys.exit(main(sys.argv[1:]))
