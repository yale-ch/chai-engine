"""Providers: components that generate the initial Results of a workflow from raw input.

A Provider is typically the first step of a workflow: it turns a configured or supplied raw value (a
directory path, a list of files, a IIIF manifest URL, a literal value) into a ``Result`` that the rest
of the tree can process.
"""

import os
from io import BytesIO

import requests
import ujson as json
from PIL import Image

from .core import Component
from .result import DirectoryListResult, ListResult, Result


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


class IIIFDirFileProvider(DirFileProvider):
    """Downloads the images of a IIIF Presentation manifest and provides them as files.

    Input is a manifest URL; the manifest and its painting-annotation images are fetched (and cached)
    into a local directory, then a ``DirectoryListResult`` of the downloaded JPEG files is returned,
    exactly like ``DirFileProvider`` over that directory.

    Settings:
        - directory: where to store the downloaded images (default 'downloads/<component id>')
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

    def _process(self, input):
        # input is a URI to a manifest
        mf = self.get_manifest(input)
        canvases = self.get_images_info(mf)
        self.download_images(canvases)
        files = [os.path.join(self.images_dir, x) for x in os.listdir(self.images_dir) if not x.endswith("json")]
        files = [x for x in files if not os.path.isdir(x)]
        d = DirectoryListResult(files, input=input, processor=self)
        return Provider._process(self, d)

    def get_manifest(self, manifest_url):
        """Fetch manifest data from the manifest URL"""
        mffn = os.path.join(self.images_dir, "manifest.json")
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

    def download_images(self, image_urls):
        """Download images and store as PIL Image objects"""
        images = {}

        for i, cvs in enumerate(image_urls):
            url = cvs["images"][0]["id"]
            filename = f"image_{i:04d}.jpg"
            image_path = os.path.join(self.images_dir, filename)
            images[filename] = {"url": url, "canvas": cvs}
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
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
        with open(os.path.join(self.images_dir, "_info.json"), "w") as fh:
            json.dumps(images, fh)
        return images
