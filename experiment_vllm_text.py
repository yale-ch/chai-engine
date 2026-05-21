"""Text-only smoke test of the vLLM backend.

Drives ``VLLMComponent`` (not the auto-generated ``VLLMTranscriber``) with
plain TEXT inputs so it can be exercised against an Apple Silicon vllm-metal
server, which only supports text models (see ``docs/findings.md``).

Prereqs:
  1. Start the server -- e.g. on Apple Silicon:
       source ~/.venv-vllm-metal/bin/activate
       vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 --max-model-len 4096
  2. ``~/.venv-chai/bin/python experiment_vllm_text.py``
"""

import os
from pathlib import Path

from chai.workflow import Workflow
from chai.result import ItemResult, ListResult  # noqa: E402 -- after Workflow to avoid circular import

IMAGE_PATHS = [
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_00.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_01.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_02.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_03.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_04.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_05.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_06.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_07.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_08.jpeg",
    "/Users/wjm55/data/focus_images/Index 0_Index 01_Index 1_page_09.jpeg",
]

VLLM_API_HOST = os.environ.get("VLLM_API_HOST", "localhost:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")

js = {
    "id": "wf_vllm_text",
    "type": "Workflow",
    "name": "vLLM text smoke test",
    "steps": [
        {
            "id": "img_iter",
            "type": "iterator.Iterator",
            "name": "Filename Iterator",
            "steps": [
                {
                    "id": "vllm_summarize",
                    "type": "ai.VLLMComponent",
                    "name": "vLLM filename summarizer",
                    "settings": {
                        "api_host": VLLM_API_HOST,
                        "model": VLLM_MODEL,
                        "temperature": 0.2,
                        "max_output_tokens": 64,
                        "expected_output": "text",
                        "prompt": (
                            "You are inspecting a JPEG named '{text_input_0}'. "
                            "In ONE short sentence, guess what archival object it likely depicts. "
                            "Do not ask questions. Do not add any preamble."
                        ),
                    },
                }
            ],
        },
        {"id": "iter_debug", "type": "utils.DebugStep"},
    ],
}


def build_text_input(paths):
    items = [
        ItemResult(Path(p).name, metadata={"type": "TEXT"}) for p in paths
    ]
    return ListResult(items)


if __name__ == "__main__":
    wf = Workflow(js)
    res = wf.run(build_text_input(IMAGE_PATHS))
    res.view()
