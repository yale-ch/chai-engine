"""Text-only smoke test of the sglang backend.

Mirror of ``experiment_vllm_text.py`` but targeting an sglang+MLX server.

Prereqs:
  1. Start the server:
       source ~/.venv-sglang/bin/activate
       SGLANG_USE_MLX=1 python -m sglang.launch_server \\
           --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000 --trust-remote-code
  2. ``~/.venv-chai/bin/python experiment_sglang_text.py``
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

SGLANG_API_HOST = os.environ.get("SGLANG_API_HOST", "localhost:30000")
SGLANG_MODEL = os.environ.get("SGLANG_MODEL", "Qwen/Qwen3-0.6B")

js = {
    "id": "wf_sglang_text",
    "type": "Workflow",
    "name": "sglang text smoke test",
    "steps": [
        {
            "id": "img_iter",
            "type": "iterator.Iterator",
            "name": "Filename Iterator",
            "steps": [
                {
                    "id": "sglang_summarize",
                    "type": "ai.SGLangComponent",
                    "name": "sglang filename summarizer",
                    "settings": {
                        "api_host": SGLANG_API_HOST,
                        "model": SGLANG_MODEL,
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
