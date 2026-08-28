"""OpenAIComponent turns in-memory IMAGE results into image_url parts, not just files on disk."""

import base64
import unittest
from types import SimpleNamespace

from chai.result import ItemResult, ListResult
from chai.workflow import Workflow


class FakeClient:
    """Records every chat.completions.create call and answers with a fixed reply."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="a picture")
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def component():
    wf = Workflow({"id": "wf", "type": "workflow.Workflow"})
    comp = wf._make_step(
        {
            "id": "llm",
            "type": "ai.openai.OpenAIComponent",
            "settings": {"api_host": "localhost:1", "model": "fake", "prompt": "Describe", "expected_output": "text"},
        },
        wf,
    )
    comp.client = FakeClient()
    return comp


class TestImageInputs(unittest.TestCase):
    def test_an_image_item_result_holding_bytes_is_sent_as_a_data_url(self):
        image = b"\x89PNG fake"
        comp = component()
        out = comp.process(ListResult([ItemResult(image, metadata={"type": "IMAGE"})]))
        self.assertEqual(out.value, "a picture")
        parts = comp.client.calls[0]["messages"][0]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"])
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64," + base64.b64encode(image).decode())

    def test_an_iterator_wrapped_image_is_unwrapped_the_same_way(self):
        image = b"\x89PNG fake"
        comp = component()
        comp.process(ItemResult(ItemResult(image, metadata={"type": "IMAGE"})))
        parts = comp.client.calls[0]["messages"][0]["content"]
        self.assertEqual(parts[1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
