#!/usr/bin/env python3
"""Smoke-test a self-hosted model backend (vLLM, sglang, MLX-VLM) through a chai workflow.

Transcribes images with the backend's transcriber, or with ``--text`` sends prompts to its plain
component -- the way to test a text-only server, such as vllm-metal or sglang's MLX backend:

    python scripts/smoke_backend.py vllm path/to/images/           # every image in the directory
    python scripts/smoke_backend.py sglang page1.jpg page2.jpg --api-host my-gpu-host:30000
    python scripts/smoke_backend.py mlx-vlm path/to/images/ --limit 3
    python scripts/smoke_backend.py vllm --text "Name three kinds of archival record." --model Qwen/Qwen3-0.6B

The model and host default to the backend's environment variables (``VLLM_MODEL``/``VLLM_API_HOST``,
``SGLANG_MODEL``/``SGLANG_API_HOST``, ``MLX_VLM_MODEL``) and then to the values below. MLX-VLM runs in
process on Apple Silicon and takes no host. Each output is printed with the run's time and usage;
the exit status is 1 if the run or any entry failed.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, str(ROOT))

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# backend -> its component names, environment variables and defaults
BACKENDS = {
    "vllm": {
        "transcriber": "transcriber.VLLMTranscriber",
        "component": "ai.VLLMComponent",
        "env": "VLLM",
        "host": "localhost:8000",
        "image_model": "Qwen/Qwen3-VL-4B-Instruct",
        "text_model": "Qwen/Qwen3-0.6B",
    },
    "sglang": {
        "transcriber": "transcriber.SGLangTranscriber",
        "component": "ai.SGLangComponent",
        "env": "SGLANG",
        "host": "localhost:30000",
        "image_model": "Qwen/Qwen3-VL-4B-Instruct",
        "text_model": "Qwen/Qwen3-0.6B",
    },
    "mlx-vlm": {
        "transcriber": "transcriber.MLXVLMTranscriber",
        "component": "ai.MLXVLMComponent",
        "env": "MLX_VLM",
        "host": None,
        "image_model": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "text_model": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    },
}


def image_files(paths, limit=0):
    """The image files named by *paths* (files, or directories of them), in name order."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_TYPES))
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"No such file or directory: {p}")
    if not files:
        raise SystemExit("No images found")
    return [str(f.resolve()) for f in files[: limit or None]]


def build_workflow(backend, model, api_host, max_output_tokens, images=None):
    """One Iterator running the backend's transcriber over *images* (provided by a FileListProvider),
    or, without them, its plain component on each prompt the run is given."""
    spec = BACKENDS[backend]
    settings = {"model": model, "max_output_tokens": max_output_tokens, "temperature": 0.2,
                "expected_output": "text"}
    if api_host:
        settings["api_host"] = api_host
    if not images:
        step = {"id": "generate", "type": spec["component"], "settings": {**settings, "prompt": "{text_input_0}"}}
    else:
        step = {"id": "transcribe", "type": spec["transcriber"], "settings": {**settings, "max_image_size": 1536}}
    iterator = {"id": "each", "type": "iterator.Iterator", "settings": {"continue_on_error": True},
                "steps": [step]}
    if images:
        iterator = {"id": "images", "type": "provider.FileListProvider", "input": list(images),
                    "steps": [iterator]}
    return {
        "id": f"smoke_{backend.replace('-', '_')}",
        "type": "workflow.Workflow",
        "name": f"{backend} smoke test",
        "steps": [iterator],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backend", choices=sorted(BACKENDS))
    parser.add_argument("images", nargs="*", help="image files or directories of them")
    parser.add_argument("-t", "--text", action="append", default=[],
                        help="send this prompt instead of transcribing images; repeatable")
    parser.add_argument("--model", help="model id (default: $<BACKEND>_MODEL, then a small Qwen model)")
    parser.add_argument("--api-host", help="host:port of the server (default: $<BACKEND>_API_HOST, then localhost)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N images")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="default 2048 for images, 64 for prompts")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    if bool(args.images) == bool(args.text):
        parser.error("give either images or --text prompts")
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    spec = BACKENDS[args.backend]
    text = bool(args.text)
    model = (args.model or os.environ.get(f"{spec['env']}_MODEL")
             or spec["text_model" if text else "image_model"])
    api_host = None
    if spec["host"]:
        api_host = args.api_host or os.environ.get(f"{spec['env']}_API_HOST") or spec["host"]
    max_tokens = args.max_output_tokens or (64 if text else 2048)

    from chai.result import ItemResult, ListResult
    from chai.workflow import Workflow

    if text:
        labels = args.text
        run_input = ListResult([ItemResult(t, metadata={"type": "TEXT"}) for t in args.text])
    else:
        labels = image_files(args.images, args.limit)
        run_input = None
    print(f"{args.backend}: {model}" + (f" at {api_host}" if api_host else "") + f", {len(labels)} inputs")

    wf = Workflow(build_workflow(args.backend, model, api_host, max_tokens, None if text else labels))
    started = time.time()
    res = wf.run(run_input)
    seconds = time.time() - started

    failed = 0
    entries = res.value[0].value  # the iterator's output: one ListResult of step outputs per input
    if not text:
        entries = entries[0].value  # under the provider
    for label, entry in zip(labels, entries):
        out = entry.value[0] if isinstance(entry.value, list) and entry.value else entry
        name = Path(label).name if not text else label[:40]
        if out.metadata.get("type") == "ERROR":
            failed += 1
            print(f"\n[{name}] ERROR {out.metadata.get('error', out.value)}")
        else:
            print(f"\n[{name}]\n{out.value}")

    usage = wf.usage.summary()
    print(f"\n{len(entries) - failed}/{len(labels)} succeeded in {seconds:.1f}s; "
          f"{usage.get('total_tokens', 0)} tokens over {usage.get('calls', 0)} calls")
    return 1 if failed or len(entries) != len(labels) else 0


if __name__ == "__main__":
    sys.exit(main())
