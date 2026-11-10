"""Tests for ``DirFileProvider``: which entries of a directory become results.

A digitisation folder holds more than its scans -- ``.DS_Store``, ``Thumbs.db``, subfolders per box
or item -- and a collection is usually a tree. The provider lists files only, and walks the tree
when asked.
"""

import os
import tempfile
import unittest

from chai.provider import DirFileProvider
from chai.workflow import Workflow


def provider(**settings):
    wf = Workflow({"id": "dir_test_wf", "type": "workflow.Workflow"})
    return DirFileProvider({"type": "provider.DirFileProvider", "id": "dir", "settings": settings}, wf)


class TestDirFileProvider(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "box1")
        for path in (
            "page_0001.tif",
            "page_0002.TIF",
            "notes.txt",
            ".DS_Store",
            "Thumbs.db",
            "folder1/page_0003.tif",
            "folder1/item2/page_0004.jpg",
            "folder1/.DS_Store",
            ".thumbnails/page_0001.jpg",
        ):
            full = os.path.join(self.root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as fh:
                fh.write(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def listed(self, **settings):
        out = provider(**settings).process(self.root)
        return [os.path.relpath(path, self.root).replace(os.sep, "/") for path in out.value]

    def test_only_the_directorys_own_files(self):
        # no subfolder, no hidden file, nothing the operating system left behind
        self.assertEqual(self.listed(), ["notes.txt", "page_0001.tif", "page_0002.TIF"])

    def test_every_entry_is_a_readable_typed_file(self):
        out = provider().process(self.root)
        self.assertEqual([item.metadata.get("type") for item in out], ["TEXT", "IMAGE", "IMAGE"])
        self.assertEqual([item.value for item in out], [b"x", b"x", b"x"])

    def test_recursive_walks_the_subfolders_but_not_hidden_ones(self):
        self.assertEqual(
            self.listed(recursive=True),
            ["folder1/item2/page_0004.jpg", "folder1/page_0003.tif", "notes.txt", "page_0001.tif", "page_0002.TIF"],
        )

    def test_pattern_keeps_matching_names_whatever_their_case(self):
        self.assertEqual(self.listed(pattern="*.tif"), ["page_0001.tif", "page_0002.TIF"])
        self.assertEqual(
            self.listed(pattern=["*.tif", "*.jpg"], recursive=True),
            ["folder1/item2/page_0004.jpg", "folder1/page_0003.tif", "page_0001.tif", "page_0002.TIF"],
        )

    def test_hidden_lists_what_is_otherwise_left_out(self):
        self.assertEqual(
            self.listed(hidden=True), [".DS_Store", "Thumbs.db", "notes.txt", "page_0001.tif", "page_0002.TIF"]
        )
        self.assertIn(".thumbnails/page_0001.jpg", self.listed(hidden=True, recursive=True))

    def test_a_missing_path_and_a_file_raise(self):
        with self.assertRaises(ValueError):
            provider().process(os.path.join(self.root, "no-such-folder"))
        with self.assertRaises(ValueError) as cm:
            provider().process(os.path.join(self.root, "notes.txt"))
        self.assertIn("not a directory", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
