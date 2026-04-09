from chai.workflow import Workflow

js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {
            "type": "provider.DirFileProvider",
            "base": "PII_Transcriber",
        },
        {
            "id": "dfp_debug",
            "type": "utils.DebugStep",  # this gets result from DFP
        },
    ],
}

wf = Workflow(js)
res = wf.run("/Users/rs2668/Development/llm/archives/yul-pipeline/dreier/images_full")
res.view()
