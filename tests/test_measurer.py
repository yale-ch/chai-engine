"""Sizing what is in a photograph from the objects of known size beside it -- no models.

``BoxLocator`` maps the boxes a vision model reports (Gemini's ``box_2d``, normalised to 0-1000) onto
the image's pixels as ``bbox`` locators; ``ScaleMeasurer`` reads the rulers, scale bars or charts
among the regions as references and gives every other region a width and height in their unit. A
``StaticProvider`` stands in for the model, so the stored rows are what a real run would write.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chai.result import FileItemResult, ItemResult, ListResult  # noqa: E402
from chai.segmenter import image_size  # noqa: E402
from chai.workflow import Workflow  # noqa: E402

RULERS = {"vertical-ruler": {"height": 92}, "horizontal-ruler": {"width": 61}}


def camera_jpeg(path, stored=(400, 200)):
    """A JPEG stored on its side (EXIF orientation 6), as a camera held upright writes one: shown 200x400."""
    img = Image.new("RGB", stored, "white")
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(path, format="JPEG", exif=exif)
    return path


class Steps(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.wf = Workflow({"id": "measure_wf", "type": "workflow.Workflow", "steps": []})
        self.made = 0

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def step(self, type_, **settings):
        self.made += 1
        return self.wf._make_step({"type": type_, "id": f"step{self.made}", "settings": settings}, self.wf)

    def image(self, name="puppet.jpg", **kw):
        return FileItemResult(camera_jpeg(os.path.join(self.dir, name), **kw))

    def located(self, detections, image=None, **settings):
        """The regions ``BoxLocator`` makes of *detections* reported for *image*."""
        reply = ItemResult(detections, input=image or self.image(), metadata={"type": "DATA"})
        return self.step("segmenter.BoxLocator", **settings).process(reply).value

    def region(self, label, box, size=(1000, 1000), **metadata):
        """A region as a detector leaves it: a label and a ``bbox`` locator."""
        region = ItemResult({"label": label}, metadata=dict(type="DATA", label=label, **metadata))
        region.locator = {"bbox": list(box), "size": list(size)}
        return region

    def measured(self, regions, **settings):
        settings.setdefault("references", RULERS)
        return self.step("measurer.ScaleMeasurer", **settings).process(ListResult(regions))


class TestBoxLocator(Steps):
    def test_the_image_is_measured_upright(self):
        self.assertEqual(image_size(self.image()), (200, 400))
        with open(self.image().file_name, "rb") as fh:
            self.assertEqual(image_size(ItemResult(fh.read(), metadata={"type": "IMAGE"})), (200, 400))

    def test_gemini_boxes_become_pixel_regions(self):
        [region] = self.located([{"label": "shoulder", "box_2d": [100, 250, 300, 750], "note": "left"}])
        self.assertEqual(region.value, {"label": "shoulder", "note": "left", "bbox": [50.0, 40.0, 150.0, 120.0]})
        self.assertEqual(region.locator, {"bbox": [50.0, 40.0, 150.0, 120.0], "size": [200, 400]})
        self.assertEqual(region.metadata["label"], "shoulder")
        self.assertEqual(region.sequence, 0)

    def test_other_box_formats_and_scales(self):
        [xyxy] = self.located([{"class": "rod", "bbox": [10, 20, 30, 60]}], box_format="xyxy", normalized=0)
        self.assertEqual(xyxy.value["bbox"], [10.0, 20.0, 30.0, 60.0])
        self.assertEqual(xyxy.value["label"], "rod")
        [xywh] = self.located([{"label": "rod", "box": [0.1, 0.5, 0.25, 0.25]}], box_format="xywh", normalized=1)
        self.assertEqual(xywh.value["bbox"], [20.0, 200.0, 70.0, 300.0])

    def test_boxes_are_clamped_to_the_image(self):
        [region] = self.located([{"label": "ruler", "box_2d": [-5, 0, 1003, 40]}])
        self.assertEqual(region.value["bbox"], [0.0, 0.0, 8.0, 400.0])

    def test_a_reply_as_text_or_an_object(self):
        text = json.dumps([{"label": "a", "box_2d": [0, 0, 500, 500]}])
        self.assertEqual(len(self.located(text)), 1)
        self.assertEqual(len(self.located(f"```json\n{text}\n```")), 1)  # a reply in a code fence
        with self.assertRaises(ValueError):
            self.located("I could not find any rulers.")
        found = self.located({"objects": [{"label": "a", "box_2d": [0, 0, 500, 500]}], "count": 1})
        self.assertEqual(len(found), 1)
        with self.assertRaises(ValueError):  # two lists: which one?
            self.located({"a": [], "b": []})
        self.assertEqual(len(self.located({"a": [], "b": [{"label": "x", "box_2d": [0, 0, 9, 9]}]}, list_field="b")), 1)

    def test_labels_filter_and_order(self):
        detections = [
            {"label": "low", "box_2d": [800, 0, 900, 100]},
            {"label": "high", "box_2d": [0, 0, 100, 100]},
            {"label": "card", "box_2d": [400, 0, 500, 100]},
        ]
        reply = ItemResult(detections, input=self.image(), metadata={"type": "DATA"})
        out = self.step("segmenter.BoxLocator", labels=["low", "high"], order="top_to_bottom").process(reply)
        self.assertEqual([(r.value["label"], r.sequence) for r in out.value], [("high", 0), ("low", 1)])
        self.assertEqual(out.metadata["dropped"], ["card"])

    def test_a_reply_that_lost_its_labels_is_not_filtered_to_nothing(self):
        # seen from Gemini: the label written under the box's own key, so the label is overwritten
        garbled = '[{"box_2d": "vertical-ruler", "box_2d": [94, 30, 871, 74]}]'
        with self.assertRaises(ValueError):
            self.located(garbled, labels=["vertical-ruler"])
        [region] = self.located(garbled)  # with no labels to keep, an unlabelled box is still a box
        self.assertIsNone(region.value["label"])

    def test_an_unreadable_box_raises(self):
        for bad in ([1, 2, 3], ["a", 0, 1, 1], [500, 500, 500, 600]):
            with self.assertRaises(ValueError):
                self.located([{"label": "x", "box_2d": bad}])
        with self.assertRaises(ValueError):
            self.located([{"label": "x"}])

    def test_no_image_no_regions(self):
        with self.assertRaises(ValueError):
            self.step("segmenter.BoxLocator").process(ItemResult([{"label": "x", "box_2d": [0, 0, 9, 9]}]))

    def test_bad_settings_fail_when_built(self):
        with self.assertRaises(ValueError):
            self.step("segmenter.BoxLocator", box_format="yx")
        with self.assertRaises(ValueError):
            self.step("segmenter.BoxLocator", order="random")


class TestScaleMeasurer(Steps):
    def test_two_rulers_size_each_axis(self):
        regions = [
            self.region("vertical-ruler", [0, 0, 20, 920]),  # 10 px/cm down
            self.region("horizontal-ruler", [0, 980, 1220, 1000]),  # 20 px/cm across
            self.region("body-without-rods", [100, 100, 300, 500]),
        ]
        out = self.measured(regions)
        self.assertEqual(out.metadata["scale"], {"x": 20.0, "y": 10.0})
        vertical, horizontal, body = out.value
        self.assertEqual(
            body.value,
            {"label": "body-without-rods", "bbox": [100.0, 100.0, 300.0, 500.0], "width_px": 200.0, "height_px": 400.0,
             "width": 10.0, "height": 40.0, "unit": "cm"},
        )
        self.assertTrue(body.metadata["measured"])
        self.assertFalse(body.metadata["reference"])
        self.assertEqual(body.metadata["calibrated"], {"width": ["horizontal-ruler"], "height": ["vertical-ruler"]})
        self.assertEqual(body.locator, regions[2].locator)  # so its row keeps the region's bbox in input_segment
        # A reference's own row: its known size, its reading, and what it measures at
        self.assertTrue(vertical.metadata["reference"])
        self.assertEqual(vertical.value["known_size"], {"height": 92, "unit": "cm"})
        self.assertEqual(vertical.value["height"], 92.0)
        self.assertEqual([r["scale"] for r in vertical.metadata["readings"]], [10.0])
        self.assertEqual(horizontal.value["width"], 61.0)

    def test_one_reference_sizes_both_axes(self):
        out = self.measured([self.region("vertical-ruler", [0, 0, 20, 920]), self.region("puppet", [0, 0, 50, 100])])
        puppet = out.value[1]
        self.assertEqual((puppet.value["width"], puppet.value["height"]), (5.0, 10.0))
        self.assertEqual(puppet.metadata["single_axis"], "y")
        self.assertEqual(puppet.metadata["calibrated"], {"width": ["vertical-ruler"], "height": ["vertical-ruler"]})

    def test_a_length_is_the_long_side(self):
        bars = {"scale_bar": {"length": 5}}
        out = self.measured([self.region("scale_bar", [0, 0, 20, 100]), self.region("leaf", [0, 0, 60, 80])], references=bars)
        self.assertEqual(out.metadata["scale"], {"x": None, "y": 20.0})
        self.assertEqual((out.value[1].value["width"], out.value[1].value["height"]), (3.0, 4.0))
        out = self.measured([self.region("scale_bar", [0, 0, 100, 20]), self.region("leaf", [0, 0, 60, 80])], references=bars)
        self.assertEqual(out.metadata["scale"], {"x": 20.0, "y": None})

    def test_references_on_one_axis_are_averaged(self):
        chart = {"color_chart": {"width": 10, "height": 5}, "horizontal-ruler": {"width": 61}}
        regions = [
            self.region("color_chart", [0, 0, 100, 50]),  # 10 px/cm each way
            self.region("horizontal-ruler", [0, 0, 732, 10]),  # 12 px/cm across
            self.region("puppet", [0, 0, 110, 100]),
        ]
        out = self.measured(regions, references=chart)
        self.assertEqual(out.metadata["scale"], {"x": 11.0, "y": 10.0})
        self.assertEqual(out.metadata["calibrated"], {"x": ["color_chart", "horizontal-ruler"], "y": ["color_chart"]})
        self.assertEqual(out.value[2].value["width"], 10.0)

    def test_a_reference_found_twice_uses_the_most_confident(self):
        regions = [
            self.region("horizontal-ruler", [0, 0, 1220, 20], confidence=0.85),  # 20 px/cm
            self.region("horizontal-ruler", [0, 990, 1830, 1000], confidence=0.78),  # another, cut by the frame
            self.region("puppet", [0, 0, 100, 200]),
        ]
        out = self.measured(regions)
        self.assertEqual(out.metadata["scale"], {"x": 20.0, "y": None})
        first, second, puppet = out.value
        self.assertEqual((first.metadata["reference_used"], second.metadata["reference_used"]), (True, False))
        self.assertEqual(second.value["width"], 91.5)  # still sized, by the ruler that was used
        self.assertEqual(puppet.value["width"], 5.0)
        out = self.measured(regions, duplicates="mean")
        self.assertEqual(out.metadata["scale"]["x"], 25.0)

    def test_no_reference_leaves_regions_unmeasured(self):
        out = self.measured([self.region("puppet", [0, 0, 50, 100])])
        [puppet] = out.value
        self.assertFalse(puppet.metadata["measured"])
        self.assertNotIn("width", puppet.value)
        self.assertEqual(puppet.value["width_px"], 50.0)
        self.assertEqual(out.metadata["scale"], {"x": None, "y": None})

    def test_what_to_emit(self):
        regions = [self.region("vertical-ruler", [0, 0, 20, 920]), self.region("puppet", [0, 0, 50, 100]),
                   self.region("shoulder", [0, 0, 10, 10])]
        out = self.measured(regions, measure=["puppet"], references_out=False, unit="centimetres", precision=0)
        [puppet] = out.value
        self.assertEqual((puppet.value["width"], puppet.value["unit"]), (5.0, "centimetres"))

    def test_a_yolo_crop_is_measured_by_its_class(self):
        crop = FileItemResult("crop.png", metadata={"type": "IMAGE", "yolo_class": "puppet", "confidence": 0.9312})
        crop.file_bytes = b"\x89PNG not really"
        crop.locator = {"bbox": [0, 0, 50, 100]}
        [_, puppet] = self.measured([self.region("vertical-ruler", [0, 0, 20, 920]), crop]).value
        self.assertEqual(puppet.value, {"label": "puppet", "bbox": [0.0, 0.0, 50.0, 100.0], "confidence": 0.931,
                                        "width_px": 50.0, "height_px": 100.0, "width": 5.0, "height": 10.0, "unit": "cm"})

    def test_an_uncropped_yolo_region_is_never_read_from_disk(self):
        # YoloSegmenter with crop: false names a crop.png that was never written
        region = FileItemResult(os.path.join(self.dir, "crop.png"), metadata={"type": "IMAGE", "yolo_class": "vertical-ruler"})
        region.locator = {"bbox": [0, 0, 20, 920]}
        [ruler] = self.measured([region]).value
        self.assertEqual(ruler.value["height"], 92.0)

    def test_references_must_be_sizes(self):
        for bad in ({}, {"ruler": 92}, {"ruler": {"depth": 3}}, {"ruler": {"height": -1}},
                    {"ruler": {"length": 5, "width": 5}}, {"ruler": {"height": True}}):
            with self.assertRaises(ValueError):
                self.step("measurer.ScaleMeasurer", references=bad)
        with self.assertRaises(ValueError):
            self.step("measurer.ScaleMeasurer", references=RULERS, duplicates="median")


class TestStoredMeasurements(unittest.TestCase):
    """A photograph, a model's boxes, measured and stored: the CHAI-TEA rows a UI reads."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.images = os.path.join(self.dir, "images")
        os.makedirs(self.images)
        camera_jpeg(os.path.join(self.images, "puppet.jpg"), stored=(1000, 500))  # shown 500x1000
        self.log = os.path.join(self.dir, "run.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_each_region_is_a_row_with_its_box_and_size(self):
        store = {"type": "storage.JsonLinesStorage", "settings": {"file": self.log}}
        detections = [
            {"label": "vertical-ruler", "box_2d": [0, 0, 920, 40]},  # 920 px tall: 10 px/cm
            {"label": "puppet-with-rods", "box_2d": [100, 200, 500, 600]},  # 200 x 400 px
        ]
        wf = Workflow(
            {
                "id": "measured_wf",
                "type": "workflow.Workflow",
                "steps": [
                    {
                        "type": "provider.DirFileProvider",
                        "input": self.images,
                        "steps": [
                            {
                                "type": "iterator.Iterator",
                                "source": True,
                                "steps": [
                                    {
                                        "id": "detect",  # stands in for the model
                                        "type": "provider.StaticProvider",
                                        "settings": {"values": detections},
                                        "next_steps": [
                                            {
                                                "id": "boxes",
                                                "type": "segmenter.BoxLocator",
                                                "next_steps": [
                                                    {
                                                        "id": "measure",
                                                        "type": "measurer.ScaleMeasurer",
                                                        "settings": {"references": {"vertical-ruler": {"height": 92}}},
                                                        "next_steps": [
                                                            {"id": "each_region", "type": "iterator.Iterator", "steps": [store]}
                                                        ],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        wf.run()
        with open(self.log) as fh:
            rows = [json.loads(line) for line in fh]
        self.assertEqual([r["process_id"] for r in rows], ["each_region", "each_region"])  # the iterator's entries
        by_label = {r["value"]["label"]: r for r in rows}
        puppet = by_label["puppet-with-rods"]
        self.assertTrue(puppet["input"].endswith("puppet.jpg"))
        self.assertEqual(json.loads(puppet["input_segment"]), [{"bbox": [100.0, 100.0, 300.0, 500.0], "size": [500, 1000]}])
        self.assertEqual((puppet["value"]["width"], puppet["value"]["height"], puppet["value"]["unit"]), (20.0, 40.0, "cm"))
        self.assertEqual(by_label["vertical-ruler"]["value"]["height"], 92.0)


if __name__ == "__main__":
    unittest.main()
