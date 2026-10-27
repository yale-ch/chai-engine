"""Providers: components that generate the initial Results of a workflow from raw input.

A Provider is typically the first step of a workflow: it turns a configured or supplied raw value (a
directory path, a list of files, a IIIF manifest URL, a literal value) into a ``Result`` that the rest
of the tree can process.
"""

import csv
import os
import re
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
import ujson as json
from PIL import Image

from .core import Component
from .result import DirectoryListResult, ItemResult, ListResult, Result


class Provider(Component):
    """A Provider generates a Result given a raw input.

    Unlike other components, a Provider can start a run by itself: ``Workflow._run`` calls ``run()``
    (which processes the component's configured ``input``) when the step has its own input, instead of
    feeding it the previous step's output. Subclasses override ``_process`` to turn the raw input into
    a ``Result``; the default ``_process`` returns the input unchanged unless child ``steps`` exist, in
    which case it falls back to the normal pass-down behaviour.
    """

    def run(self) -> Result:
        if self.input:
            return self.process(self.input)
        else:
            raise ValueError("No input value set")

    def _process(self, input):
        if not self.steps:
            return input
        else:
            return super()._process(input)


class StaticProvider(Provider):
    """Provides a fixed, configured value -- handy for driving a workflow from
    its config rather than runtime input.

    Settings:
        - values: list of values to emit as a ListResult (required, unless
                  `value` is set for a single item)
        - value:  a single value to emit in a one-item ListResult
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "values" in self.settings:
            self.values = list(self.settings["values"])
        elif "value" in self.settings:
            self.values = [self.settings["value"]]
        else:
            raise ValueError(f"StaticProvider ({self!r}) needs the `values` (or `value`) setting")

    def run(self):
        # A StaticProvider IS its input; it doesn't need one configured.
        return self.process(self.input)

    def _process(self, input):
        return super()._process(ListResult(self.values, input=input, processor=self))


class IntListProvider(Provider):
    """A provider that returns a fixed list of integers"""

    def _process(self, input):
        return super()._process(ListResult([1, 2, 3]))


class DirFileProvider(Provider):
    """Take a directory name and return one Result entry per file inside it.

    Input is a directory path (string); output is a ``DirectoryListResult`` whose entries wrap each
    file in the directory as a lazily-read ``FileItemResult``. Raises ``ValueError`` when the path does
    not exist. Usually followed by an ``Iterator`` that processes each file.
    """

    def _process(self, input):
        if os.path.exists(input):
            files = os.listdir(input)
            d = DirectoryListResult(
                [os.path.join(input, x) for x in files],
                input=input,
                processor=self,
            )
            return super()._process(d)
        else:
            raise ValueError("input file path does not exist")


class FileListProvider(Provider):
    """Take an explicit list of file paths and return a ``DirectoryListResult``.

    Useful when you want to drive a workflow against a hand-picked subset of
    files rather than every entry in a directory. Accepts either a single
    path (treated as a one-item list) or a list/tuple of paths.
    """

    def _process(self, input):
        if isinstance(input, str):
            paths = [input]
        elif isinstance(input, (list, tuple)):
            paths = list(input)
        else:
            raise ValueError(
                f"FileListProvider expected a list/tuple of paths or a single path, got {type(input)!r}"
            )

        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise ValueError(f"Input files do not exist: {missing}")

        d = DirectoryListResult(paths, input=input, processor=self)
        return super()._process(d)


class CsvFileProvider(Provider):
    """Take a CSV file and return one dict-shaped Result entry per row.

    Input is the path to a CSV file; output is a ``ListResult`` whose entries are ``ItemResult``s
    holding one ``{column: value}`` dict per row. The entries are DATA-typed, so a JSON-shaped
    component such as ``JsonXpathExtractor`` can address a column as ``/name``. Usually followed by an
    ``Iterator`` that processes each row. Raises ``ValueError`` when the path does not exist, or when
    the file has neither a header row nor configured ``columns``.

    The column names end up in the list result's ``columns`` metadata, and each row records its
    zero-based ``row`` (excluding the header) in its own metadata.

    Settings:
        - columns: explicit list of column names; when set the file is read as having no header row
                   (default: take the names from the first row)
        - delimiter: field separator (default ',')
        - quotechar: quoting character (default '"')
        - encoding: file encoding (default 'utf-8')
        - limit: stop after this many rows (default: read all of them)
    """

    def _process(self, input):
        if not os.path.exists(input):
            raise ValueError(f"input file path does not exist: {input}")

        columns = self.settings.get("columns", None)
        limit = int(self.settings.get("limit", 0) or 0)

        rows = ListResult([], input=input, processor=self)
        with open(input, newline="", encoding=self.settings.get("encoding", "utf-8")) as fh:
            reader = csv.DictReader(
                fh,
                fieldnames=list(columns) if columns else None,
                delimiter=self.settings.get("delimiter", ","),
                quotechar=self.settings.get("quotechar", '"'),
                # Ragged rows: pad short ones with '', and collect any surplus fields under a named
                # key rather than csv's default `None`, which no JSON consumer could address.
                restval="",
                restkey="_extra",
            )
            for idx, row in enumerate(reader):
                if limit and idx >= limit:
                    break
                rows.append(
                    ItemResult(
                        dict(row),
                        input=input,
                        processor=self,
                        metadata={"type": "DATA", "row": idx},
                    )
                )
            fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError(f"No columns found in {input}; is the CSV file empty?")
        rows.metadata["columns"] = list(fieldnames)
        return super()._process(rows)


class UrlJsonProvider(Provider):
    """Fetch JSON from a URL and provide it as one DATA result.

    Input is a URL, or a Result whose value is one, or a dict such as a ``CsvFileProvider`` row; the
    ``url`` template is filled from it -- ``{value}`` is a string input, ``{<column>}`` a field of a
    dict -- and the parsed response is returned as a DATA-typed ``ItemResult``, so
    ``JsonXpathExtractor`` can pick values out of it. The URL fetched is its ``uri`` metadata, which
    a stored row records as its input. A response that is not a success, or not JSON, raises, so
    ``retries``/``on_error`` apply.

    Settings:
        - url: URL template, e.g. '{url}.json' (default '{value}' for a string input, '{url}' for a dict)
        - headers: HTTP headers to send, e.g. {"User-Agent": "..."} (default: none)
        - timeout: seconds to wait for the server (default 60)
    """

    def _process(self, input):
        value = input
        while isinstance(value, Result):
            value = value.value
        if isinstance(value, dict):
            fields = {k: "" if v is None else v for k, v in value.items()}
            template = self.settings.get("url") or "{url}"
        elif isinstance(value, str):
            fields = {"value": value.strip()}
            template = self.settings.get("url") or "{value}"
        else:
            raise ValueError(f"{self} needs a URL or a dict to fill its url template, got {value!r}")
        try:
            url = template.format_map(fields)
        except KeyError as e:
            raise ValueError(f"{self}: url template {template!r} names {e}, which the input does not have")
        response = requests.get(
            url, headers=self.settings.get("headers") or None, timeout=float(self.settings.get("timeout", 60))
        )
        response.raise_for_status()
        res = ItemResult(response.json(), input=input, processor=self, metadata={"type": "DATA", "uri": url})
        return super()._process(res)


class IIIFDirFileProvider(DirFileProvider):
    """Downloads the images of a IIIF Presentation manifest and provides them as files.

    Input is a manifest URL -- a string, or a Result whose value is one (or a one-item list of one,
    as a ``JsonXpathExtractor`` returns it), so the provider can follow an earlier step; the manifest
    and its painting-annotation images are fetched (and cached) into a local directory, then a
    ``DirectoryListResult`` of the downloaded JPEG files is returned, in canvas order.

    Settings:
        - directory: where to store the downloaded images (default 'downloads/<component id>')
        - limit: only the first N canvases of the manifest, e.g. 1 for an object's primary view
          (default: all of them)
        - base_url: resolve a relative manifest reference (e.g. 'obj/1000') against this URL
          (default: none; the input must be a full URL)
        - per_manifest: keep each manifest in a subdirectory of its own, named after its URL, so one
          provider can fetch a different manifest per iterator entry (default false)
    """

    def __init__(self, tree, workflow, parent):
        super().__init__(tree, workflow, parent)
        # Now check where to put the images
        if "directory" not in self.settings:
            dy = os.path.join("downloads", self.id)
        else:
            dy = self.settings["directory"]
        if not os.path.exists(dy):
            os.makedirs(dy, exist_ok=True)
        self.images_dir = dy

    def manifest_url(self, input) -> str:
        """The manifest URL *input* names, resolved against ``base_url``."""
        value = input.value if isinstance(input, Result) else input
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{self} needs a manifest URL, got {value!r}")
        base = self.settings.get("base_url")
        return urljoin(base, value.strip()) if base else value.strip()

    def manifest_dir(self, url) -> str:
        """Where the manifest at *url* and its images are kept."""
        if not self.settings.get("per_manifest"):
            return self.images_dir
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", urlparse(url).path.strip("/")) or "manifest"
        directory = os.path.join(self.images_dir, name)
        os.makedirs(directory, exist_ok=True)
        return directory

    def _process(self, input):
        url = self.manifest_url(input)
        directory = self.manifest_dir(url)
        mf = self.get_manifest(url, directory)
        canvases = self.get_images_info(mf)
        limit = int(self.settings.get("limit", 0) or 0)
        if limit:
            canvases = canvases[:limit]
        images = self.download_images(canvases, directory)
        # The images of these canvases, in canvas order -- not whatever else is in the directory
        files = [os.path.join(directory, x) for x in images]
        files = [x for x in files if os.path.isfile(x)]
        d = DirectoryListResult(files, input=input, processor=self)
        return Provider._process(self, d)

    def get_manifest(self, manifest_url, directory=None):
        """Fetch manifest data from the manifest URL"""
        mffn = os.path.join(directory or self.images_dir, "manifest.json")
        if os.path.exists(mffn):
            with open(mffn) as fh:
                return json.load(fh)
        else:
            try:
                response = requests.get(manifest_url)
                response.raise_for_status()
                js = response.json()
                with open(mffn, "w") as fh:
                    json.dump(js, fh)
                return js
            except requests.RequestException as e:
                raise Exception(f"Failed to fetch manifest data: {e}")

    def get_images_info(self, manifest):
        """Extract image information from manifest"""
        canvas_data = []
        try:
            if "items" in manifest:
                for canvas in manifest["items"]:
                    canvas_info = {
                        "id": canvas.get("id", ""),
                        "label": canvas.get("label", {}),
                        "images": [],
                    }
                    if "items" in canvas:
                        for page in canvas["items"]:
                            if "items" in page:
                                for annotation in page["items"]:
                                    if annotation.get("motivation") == "painting":
                                        body = annotation.get("body", {})
                                        if body.get("type") == "Image":
                                            image_info = {
                                                "id": body.get("id"),
                                                "service": body.get("service", []),
                                            }
                                            canvas_info["images"].append(image_info)
                    canvas_data.append(canvas_info)
            return canvas_data
        except Exception as e:
            raise Exception(f"Error extracting image information: {e}")

    def download_images(self, image_urls, directory=None):
        """Download images and store as PIL Image objects"""
        directory = directory or self.images_dir
        images = {}

        for i, cvs in enumerate(image_urls):
            url = cvs["images"][0]["id"]
            filename = f"image_{i:04d}.jpg"
            image_path = os.path.join(directory, filename)
            images[filename] = {"url": url, "canvas": cvs}
            if os.path.exists(image_path):
                continue  # downloaded already; re-saving would re-compress the JPEG every run
            try:
                response = requests.get(url)
                response.raise_for_status()
                # Create PIL Image from response content
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"Warning: Failed to download image {url}: {e}")
                continue
            # Now try to save it
            try:
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(image_path, "JPEG", quality=95)
            except Exception as e:
                print(f"Warning: Failed to save image {url}: {e}")
                continue

        # write images hash to disk
        with open(os.path.join(directory, "_info.json"), "w") as fh:
            json.dump(images, fh)
        return images
