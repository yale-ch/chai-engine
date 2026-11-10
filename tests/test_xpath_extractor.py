"""Tests for ``JsonXpathExtractor``: what it reads, and that what it cannot read is an error.

It used to print a line and return ``None`` -- a successful result holding nothing -- for an XML
file, an XPath that did not compile and a record with a key it could not render.
"""

import contextlib
import io
import os
import tempfile
import unittest

from chai.extractor import JsonXpathExtractor
from chai.result import FileItemResult, ItemResult
from chai.workflow import Workflow

TEI = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><title>De universo</title></teiHeader></TEI>'


def extractor(xpath, **settings):
    wf = Workflow({"id": "xpath_test_wf", "type": "workflow.Workflow"})
    config = {"type": "extractor.JsonXpathExtractor", "id": "xp", "settings": {"xpath": xpath, **settings}}
    return JsonXpathExtractor(config, wf)


def record(value):
    return ItemResult(value, metadata={"type": "DATA"})


class TestWhatItReads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, content):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return FileItemResult(path)

    def test_a_dict(self):
        out = extractor("/people/name").process(record({"people": [{"name": "Ada"}, {"name": "Grace"}]}))
        self.assertEqual((out.value, out.metadata["type"]), ("Ada", "TEXT"))

    def test_a_json_file_is_parsed(self):
        out = extractor("/shelfmark").process(self.file("record.json", '{"shelfmark": "Ms. L 523", "leaves": 212}'))
        self.assertEqual(out.value, "Ms. L 523")

    def test_json_text_is_parsed(self):
        self.assertEqual(extractor("/leaves").process(record('{"leaves": 212}')).value, 212)

    def test_no_match_is_none_not_an_error(self):
        out = extractor("/medium").process(record({"classification": "Paintings"}))
        self.assertIsNone(out.value)

    def test_booleans_and_numbers_are_quiet_and_usable_in_predicates(self):
        value = {"items": [{"name": "recto", "digitized": True}, {"name": "verso", "digitized": False}], "leaves": 2}
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            out = extractor('/items[digitized="false"]/name').process(record(value))
        self.assertEqual(out.value, "verso")
        self.assertEqual(printed.getvalue(), "")


class TestWhatItRefuses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.xml = os.path.join(self.tmp.name, "record.xml")
        with open(self.xml, "w", encoding="utf-8") as fh:
            fh.write(TEI)

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_xml_file_raises(self):
        with self.assertRaises(ValueError) as cm:
            extractor("/TEI/teiHeader/title").process(FileItemResult(self.xml))
        self.assertIn("is not JSON", str(cm.exception))
        self.assertIn("<TEI", str(cm.exception))

    def test_the_error_policy_applies(self):
        self.assertIsNone(extractor("/TEI/teiHeader/title", on_error="skip").process(FileItemResult(self.xml)))

    def test_plain_text_and_lists_raise(self):
        with self.assertRaises(ValueError):
            extractor("/name").process(record("Dear Sir, I write to you"))
        with self.assertRaises(ValueError) as cm:
            extractor("/name").process(record([{"name": "Ada"}]))
        self.assertIn("got list", str(cm.exception))

    def test_an_xpath_that_does_not_compile_raises(self):
        with self.assertRaises(ValueError) as cm:
            extractor("/people[").process(record({"people": [{"name": "Ada"}]}))
        self.assertIn("'/people['", str(cm.exception))

    def test_a_key_that_cannot_be_a_step_raises_naming_it(self):
        with self.assertRaises(ValueError) as cm:
            extractor("/id").process(record({"id": "7", "Object Name": "ewer"}))
        self.assertIn("'Object Name'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
