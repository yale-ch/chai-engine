"""OpenAIComponent turns in-memory IMAGE results into image_url parts, not just files on disk.

The component is the LiteLLM preset (``chai.ai.openai``): the request it sends is read with
``request_kwargs``, and a whole ``process`` run answers through LiteLLM's local mock reply, so
nothing here talks to a server.
"""

import base64
import os
import tempfile
import unittest

from PIL import Image

from chai.ai.llm import sniff_image_mime
from chai.result import FileItemResult, ItemResult, ListResult
from chai.workflow import Workflow

PNG = b"\x89PNG\r\n\x1a\n fake"  # the PNG signature, then anything


def component(**settings):
    wf = Workflow({"id": "wf", "type": "workflow.Workflow"})
    return wf._make_step(
        {
            "id": "llm",
            "type": "ai.openai.OpenAIComponent",
            "settings": {
                "api_host": "localhost:1",
                "model": "fake",
                "prompt": "Describe",
                "expected_output": "text",
                "litellm_params": {"mock_response": "a picture"},
                **settings,
            },
        },
        wf,
    )


def parts_for(comp, input):
    return comp.request_kwargs(input)["messages"][0]["content"]


def data_url(mime, data):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class TestImageInputs(unittest.TestCase):
    def test_an_image_item_result_holding_bytes_is_sent_as_a_data_url(self):
        comp = component()
        inp = ListResult([ItemResult(PNG, metadata={"type": "IMAGE"})])
        out = comp.process(inp)
        self.assertEqual(out.value, "a picture")
        parts = parts_for(comp, inp)
        self.assertEqual([p["type"] for p in parts], ["image_url", "text"])  # attachments, then the prompt
        self.assertEqual(parts[0]["image_url"]["url"], data_url("image/png", PNG))
        self.assertEqual(parts[1]["text"], "Describe")

    def test_an_iterator_wrapped_image_is_unwrapped_the_same_way(self):
        comp = component()
        parts = parts_for(comp, ItemResult(ItemResult(PNG, metadata={"type": "IMAGE"})))
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertEqual(parts[0]["image_url"]["url"], data_url("image/png", PNG))
        self.assertEqual(comp.process(ItemResult(ItemResult(PNG, metadata={"type": "IMAGE"}))).value, "a picture")

    def test_twice_wrapped_file_keeps_its_own_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.jpg")
            Image.new("RGB", (8, 8), "white").save(path, format="JPEG")
            parts = parts_for(component(), ItemResult(ItemResult(FileItemResult(path))))
            with open(path, "rb") as fh:
                self.assertEqual(parts[0]["image_url"]["url"], data_url("image/jpeg", fh.read()))

    def test_a_wrapped_remote_reference_is_passed_as_is(self):
        parts = parts_for(component(), ItemResult(ItemResult("https://example.org/scan.png", metadata={"type": "IMAGE"})))
        self.assertEqual(parts[0]["image_url"]["url"], "https://example.org/scan.png")

    def test_type_detected_from_the_bytes_unless_set(self):
        jpeg = b"\xff\xd8\xff\xe0 fake"
        self.assertEqual(parts_for(component(), ItemResult(jpeg, metadata={"type": "IMAGE"}))[0]["image_url"]["url"], data_url("image/jpeg", jpeg))
        explicit = component(image_mime_type="image/webp")
        self.assertEqual(parts_for(explicit, ItemResult(PNG, metadata={"type": "IMAGE"}))[0]["image_url"]["url"], data_url("image/webp", PNG))
        unknown = b"not an image header"
        self.assertEqual(parts_for(component(), ItemResult(unknown, metadata={"type": "IMAGE"}))[0]["image_url"]["url"], data_url("image/png", unknown))

    def test_signatures(self):
        self.assertEqual(sniff_image_mime(PNG), "image/png")
        self.assertEqual(sniff_image_mime(b"GIF89a..."), "image/gif")
        self.assertEqual(sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertEqual(sniff_image_mime(b"II*\x00rest"), "image/tiff")
        self.assertIsNone(sniff_image_mime(b"plain text"))
        self.assertIsNone(sniff_image_mime(None))


if __name__ == "__main__":
    unittest.main()
