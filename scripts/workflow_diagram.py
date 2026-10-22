#!/usr/bin/env python3
"""Draw a chai workflow definition as a Mermaid flowchart.

Reads a workflow JSON file (the tree handed to ``Workflow``), writes the Mermaid DSL to
``<out>.mmd`` and renders it to ``<out>.png`` (or ``.svg``/``.pdf``) with the Mermaid CLI.

    python scripts/workflow_diagram.py examples/workflow-m804-lines.json
    python scripts/workflow_diagram.py examples/workflow-ppl.json -o docs/ppl --direction LR
    python scripts/workflow_diagram.py examples/workflow-ppl.json --no-render   # DSL only

Rendering uses ``mmdc`` when it is on the PATH, else ``npx -y @mermaid-js/mermaid-cli`` (which
downloads it on first use); everything happens locally.

The diagram reads only the JSON, so it needs nothing of chai installed. Each branch of the tree is
drawn as its own kind of edge:

    steps          parent --> child         the child runs inside the parent, on the parent's input
    next_steps     parent ==> child         the child receives the parent's output
    error_steps    parent -.-> child        run when the parent fails
    true_steps / false_steps                a gate's two branches
    case_steps / default_steps              a SwitchGate's named branches
    file_steps     parent -.-> child        run on a storage's written file once the run has ended
"""

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# branch key -> (mermaid arrow, edge label); a SwitchGate's case_steps and default_steps follow these
BRANCHES = [
    ("steps", "-->", None),
    ("true_steps", "-->", "true"),
    ("false_steps", "-->", "false"),
    ("next_steps", "==>", "next"),
    ("error_steps", "-.->", "on error"),
    ("file_steps", "-.->", "file"),
]

# module prefix of a component type -> (node shape open/close, class)
SHAPES = {
    "workflow": ("([", "])", "workflow"),
    "provider": ("[/", "/]", "provider"),
    "iterator": ("{{", "}}", "iterator"),
    "gate": ("{", "}", "gate"),
    "storage": ("[(", ")]", "storage"),
    "uploader": ("[/", "\\]", "uploader"),
    "reducer": ("[\\", "/]", "reducer"),
    "fanout": ("[\\", "/]", "reducer"),
    "context": ("[[", "]]", "context"),
}
DEFAULT_SHAPE = ("[", "]", "component")

CLASS_DEFS = {
    "workflow": "fill:#37474f,stroke:#263238,color:#fff",
    "provider": "fill:#e3f2fd,stroke:#1565c0,color:#0d47a1",
    "iterator": "fill:#fff3e0,stroke:#ef6c00,color:#e65100",
    "gate": "fill:#fce4ec,stroke:#ad1457,color:#880e4f",
    "storage": "fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20",
    "uploader": "fill:#e0f2f1,stroke:#00695c,color:#004d40",
    "reducer": "fill:#ede7f6,stroke:#4527a0,color:#311b92",
    "context": "fill:#fffde7,stroke:#f9a825,color:#5d4000",
    "component": "fill:#f5f5f5,stroke:#616161,color:#212121",
    "ai": "fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c",
}

# class-name fragments of the generated AI components (GeminiTranscriber, OllamaVectorIndexer, ...)
AI_MARKERS = ("Gemini", "Ollama", "LMStudio", "OpenAI", "Anthropic", "LiteLLM", "VLLM", "Vllm",
              "SGLang", "Sglang", "MLX", "Mlx", "OpenRouter", "Groq", "Yolo")


def escape(text):
    """Make *text* safe inside a quoted Mermaid label."""
    return (str(text).replace("&", "#amp;").replace('"', "#quot;")
            .replace("<", "#lt;").replace(">", "#gt;"))


def wrap(text, width):
    """Wrap *text* into lines no wider than *width*, joined by Mermaid line breaks."""
    return "<br/>".join(escape(line) for line in textwrap.wrap(str(text), width) or [""])


class Diagram:
    def __init__(self, direction="TD", show_names=True, show_settings=(), width=36):
        self.direction = direction
        self.show_names = show_names
        self.show_settings = tuple(show_settings)
        self.width = width
        self.nodes = []
        self.edges = []
        self.count = 0

    def node(self, tree):
        """Add the node for *tree* (and, recursively, its branches); return its Mermaid id."""
        node_id = f"n{self.count}"
        self.count += 1
        ctype = tree.get("type", "component.Component")
        module, _, cls = ctype.rpartition(".")
        module = module or ctype.lower()
        opener, closer, css = SHAPES.get(module, DEFAULT_SHAPE)
        if css == "component" and any(m in cls for m in AI_MARKERS):
            css = "ai"

        lines = []
        if tree.get("id"):
            lines.append(f"<b>{escape(tree['id'])}</b>")
        lines.append(f"<i>{escape(ctype)}</i>")
        if self.show_names and tree.get("name"):
            lines.append(wrap(tree["name"], self.width))
        settings = tree.get("settings") or {}
        for key in self.show_settings:
            if key in settings:
                value = settings[key]
                if not isinstance(value, str):
                    value = json.dumps(value)
                lines.append(wrap(f"{key}: {value}", self.width))
        if tree.get("source") is True:
            lines.append("<small>source</small>")
        self.nodes.append(f'    {node_id}{opener}"{"<br/>".join(lines)}"{closer}:::{css}')

        branches = [(arrow, label, child) for key, arrow, label in BRANCHES
                    for child in tree.get(key) or []]
        branches += [("-->", f"case: {case}", child)
                     for case, children in (tree.get("case_steps") or {}).items() for child in children]
        branches += [("-->", "default", child) for child in tree.get("default_steps") or []]
        for arrow, label, child in branches:
            self.edge(node_id, arrow, label, child)
        return node_id

    def edge(self, src, arrow, label, child):
        """Add the edge from *src* to *child* ahead of the child's own edges, so the layout keeps
        the config's order."""
        slot = len(self.edges)
        self.edges.append(None)
        text = f'|"{escape(label)}"|' if label else ""
        self.edges[slot] = f"    {src} {arrow}{text} {self.node(child)}"

    def render(self, tree):
        self.node(tree)
        # labels are wrapped here, at --width; widen Mermaid's own wrapping so it leaves them be
        out = ['%%{init: {"flowchart": {"wrappingWidth": 1000}}}%%', f"flowchart {self.direction}"]
        out += self.nodes
        out += self.edges
        used = {line.rsplit(":::", 1)[1] for line in self.nodes}
        out += [f"    classDef {name} {style}" for name, style in CLASS_DEFS.items() if name in used]
        return "\n".join(out) + "\n"


def to_mermaid(tree, **options):
    """Return the Mermaid flowchart DSL for the workflow config *tree*."""
    return Diagram(**options).render(tree)


def mermaid_cli():
    if shutil.which("mmdc"):
        return ["mmdc"]
    if shutil.which("npx"):
        return ["npx", "-y", "@mermaid-js/mermaid-cli"]
    raise SystemExit("Rendering needs the Mermaid CLI: `npm install -g @mermaid-js/mermaid-cli` "
                     "(or Node's npx on the PATH); use --no-render for the DSL alone.")


def render(mmd_path, image_path, theme="default", background="white", scale=2):
    """Render the Mermaid file *mmd_path* to *image_path* (format from its suffix)."""
    cmd = mermaid_cli() + ["-i", str(mmd_path), "-o", str(image_path),
                           "-t", theme, "-b", background, "-s", str(scale)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"mermaid-cli failed ({proc.returncode})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow", help="workflow definition (JSON)")
    parser.add_argument("-o", "--output",
                        help="output path without suffix (default: beside the workflow file)")
    parser.add_argument("-f", "--format", choices=["png", "svg", "pdf"], default="png")
    parser.add_argument("-d", "--direction", choices=["TD", "LR", "BT", "RL"], default="TD")
    parser.add_argument("--no-names", action="store_true", help="leave out each component's name")
    parser.add_argument("--settings", default="",
                        help="comma-separated settings to show on each node, e.g. model,file")
    parser.add_argument("--width", type=int, default=36, help="wrap names at this many characters")
    parser.add_argument("--theme", default="default", help="mermaid theme (default, neutral, dark, forest)")
    parser.add_argument("--background", default="white", help="background colour, or transparent")
    parser.add_argument("--scale", type=int, default=2, help="PNG scale factor")
    parser.add_argument("--no-render", action="store_true", help="write the .mmd file only")
    args = parser.parse_args(argv)

    src = Path(args.workflow)
    tree = json.loads(src.read_text())
    dsl = to_mermaid(tree, direction=args.direction, show_names=not args.no_names,
                     show_settings=[s.strip() for s in args.settings.split(",") if s.strip()],
                     width=args.width)

    base = Path(args.output) if args.output else src.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    mmd_path = base.with_name(base.name + ".mmd")
    mmd_path.write_text(dsl)
    print(f"wrote {mmd_path}")

    if not args.no_render:
        image_path = base.with_name(f"{base.name}.{args.format}")
        render(mmd_path, image_path, theme=args.theme, background=args.background, scale=args.scale)
        print(f"wrote {image_path}")


if __name__ == "__main__":
    main()
