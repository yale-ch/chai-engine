from chai.result import ItemResult
from chai.workflow import Workflow

js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {
            "id": "wf2_nx",
            "type": "extractor.TransformersExtractor",
            "settings": {
                "model": "small-models-for-glam/Qwen3.5-0.8B-SFT-name-parser-yaml",
                "tie_word_embeddings": False,
                "prompt": "Parse this person name:\n\n{text_input_0}",
                "expected_output": "yaml",
            },
        },
        {"base": "FileSystemStorage"},
        {"id": "wf2_xp", "type": "extractor.JsonXpathExtractor", "settings": {"xpath": "/first_name"}},
        {"base": "FileSystemStorage"},
    ],
}

wf = Workflow(js)
inp = ItemResult("Sanderson, Robert D 1976-", metadata={"type": "TEXT"})
res = wf.run(inp)
# res.view()
