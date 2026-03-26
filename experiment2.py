from chai.workflow import Workflow



js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {
            "type": "provider.IIIFDirFileProvider",
            "steps": [  # Each item gets sent through these
                {
                    "id": "img_iter",
                    "type": "iterator.Iterator",
                    "name": "File Iterator",
                    "steps": [
                        {
                            "type": "transcriber.GeminiTranscriber",
                            "name": "Cloud Image Transcriber",
                        }
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
res = wf.run("https://collections.library.yale.edu/manifests/16694456")
res.view()
