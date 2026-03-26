from chai.core import Workflow

js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {
            "type": "DirFileProvider",
            "steps": [  # Each item gets sent through these
                {
                    "id": "img_iter",
                    "type": "IterateStep",
                    "name": "File Iterator",
                    "steps": [
                        {
                            "id": "pii_classifier",
                            "type": "Classifier",
                            "name": "Has PII Classifier",
                            "register_on": ["img_iter"],
                        },
                        {
                            "type": "LabelTestGate",
                            "name": "Has PII Gate",
                            "settings": {"component": "pii_classifier", "label": ["okay"]},
                            "true_steps": [
                                {
                                    "type": "GeminiTranscriber",
                                    "name": "Cloud Image Transcriber",
                                }
                            ],
                            "false_steps": [
                                {
                                    "type": "LocalTranscriber",
                                    "name": "Local Image Transcriber",
                                }
                            ],
                        },
                    ],
                }  # merged result is complete here
            ],
            "next_steps": [{"id": "iter_debug", "type": "DebugStep"}],  # This gets merged result from the iter
        },
        {
            "id": "dfp_debug",
            "type": "DebugStep",  # this gets result from DFP
        },
    ],
}

wf = Workflow(js)
res = wf.run("/Users/rs2668/Development/llm/archives/yul-pipeline/dreier/images_full")
res.view()
