from chai.workflow import Workflow

js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {
            "type": "provider.DirFileProvider",
            "steps": [  # Each item gets sent through these
                {
                    "id": "img_iter",
                    "type": "iterator.Iterator",
                    "name": "File Iterator",
                    "steps": [
                        {
                            "id": "pii_classifier",
                            "type": "classifier.Classifier",
                            "name": "Has PII Classifier",
                            "register_on": ["img_iter"],
                        },
                        {
                            "type": "gate.LabelTestGate",
                            "name": "Has PII Gate",
                            "settings": {"component": "pii_classifier", "label": ["okay"]},
                            "true_steps": [
                                {
                                    "type": "transcriber.GeminiTranscriber",
                                    "name": "Cloud Image Transcriber",
                                }
                            ],
                            "false_steps": [
                                {
                                    "type": "transcriber.LocalTranscriber",
                                    "name": "Local Image Transcriber",
                                }
                            ],
                        },
                    ],
                }  # merged result is complete here
            ],
            "next_steps": [{"id": "iter_debug", "type": "utils.DebugStep"}],  # This gets merged result from the iter
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
