import os
from io import BytesIO

import requests
import ujson as json
from PIL import Image

from .core import Component
from .result import DirectoryListResult, Result


class Provider(Component):
    """A Provider generates a Result given a raw input"""

    def run(self) -> Result:
        if self.input:
            return self.process(self.input)
        else:
            raise ValueError("No input value set")


class DirFileProvider(Provider):
    """Take a director name and return a ListResult of ItemResults for each file"""

    def _process(self, input):
        if os.path.exists(input):
            files = os.listdir(input)
            d = DirectoryListResult([os.path.join(input, x) for x in files], input=input, processor=self)
            return super()._process(d)
        else:
            raise ValueError("input file path does not exist")


class IIIFDirFileProvider(DirFileProvider):
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
        files = os.listdir(self.images_dir)
        d = DirectoryListResult(
            [os.path.join(self.images_dir, x) for x in files if not x.endswith("json")], input=input, processor=self
        )
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
