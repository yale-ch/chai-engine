from chai.workflow import Workflow

js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {
            "type": "extractor.NameExtractor",
            "settings": {
                "model": "small-models-for-glam/Qwen3.5-0.8B-SFT-name-parser-yaml",
                "prompt": "Parse this person name:\n\n{text_input_0}",
                "expected_output": "yaml",
            },
            "next_steps": [
                {"type": "extractor.JsonXpathExtractor", "settings": {"xpath": "/first_name"}},
            ],
        },
    ],
}

wf = Workflow(js)
from chai.result import ItemResult  # noqa -- avoid circular import timing

inp = ItemResult("Sanderson, Robert D 1976-", metadata={"type": "TEXT"})
res = wf.run(inp)
res.view()
