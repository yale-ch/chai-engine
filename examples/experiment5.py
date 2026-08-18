from chai.workflow import Workflow

js = {
    "id": "wf2",
    "type": "Workflow",
    "name": "Top level workflow",
    "steps": [
        {"type": "provider.IIIFDirFileProvider"},
        {"type": "segmenter.LMStudioSegmenter", "settings": {"max_image_size": 1024}},
    ],
}

wf = Workflow(js)
# res = wf.run("https://collections.library.yale.edu/manifests/16694456")
# res.view()
