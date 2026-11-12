"""Tests for ``importClass``: how a workflow's ``type`` string becomes a class.

chai's own components are named without the ``chai.`` prefix and always win; anything chai does
not have is imported as the dotted path it is, so a project keeps its components in its own
package and names them from the workflow JSON.
"""

import os
import sys
import tempfile
import textwrap
import unittest

from chai.core import Component, importClass
from chai.provider import DirFileProvider
from chai.workflow import Workflow

SHOUT = '''
    from chai.result import ItemResult
    from chai.transcriber import Transcriber
    from chai.utils import text_from_input


    class ShoutTranscriber(Transcriber):
        """Upper-cases its input: a project's own component."""

        def _process(self, input):
            return ItemResult(text_from_input(input).upper(), input=input, processor=self, metadata={"type": "TEXT"})


    class NotAComponent:
        pass
'''


class TestChaiTypes(unittest.TestCase):
    def test_chai_types_resolve_with_and_without_the_prefix(self):
        self.assertIs(importClass("provider.DirFileProvider"), DirFileProvider)
        self.assertIs(importClass("chai.provider.DirFileProvider"), DirFileProvider)
        self.assertIs(importClass("Component"), Component)
        self.assertIsNone(importClass(""))

    def test_a_missing_module_names_both_places_it_looked(self):
        with self.assertRaises(ModuleNotFoundError) as cm:
            importClass("no_such_module_xyz.Thing")
        self.assertIn("chai.no_such_module_xyz", str(cm.exception))
        self.assertIn("there is no module no_such_module_xyz", str(cm.exception))

    def test_a_missing_class_is_an_attribute_error(self):
        with self.assertRaises(AttributeError) as cm:
            importClass("provider.NoSuchProvider")
        self.assertIn("chai.provider has no NoSuchProvider", str(cm.exception))

    def test_a_chai_prefixed_type_is_not_looked_for_elsewhere(self):
        with self.assertRaises(ModuleNotFoundError) as cm:
            importClass("chai.no_such_module_xyz.Thing")
        self.assertNotIn("there is no module no_such_module_xyz", str(cm.exception))


class TestProjectTypes(unittest.TestCase):
    """A package outside chai, on the path the way a project's own code is."""

    MODULES = ("heritage_project", "heritage_project.shout", "heritage_project.broken", "provider")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        package = os.path.join(self.tmp.name, "heritage_project")
        os.makedirs(package)
        self.write(package, "__init__.py", "")
        self.write(package, "shout.py", SHOUT)
        self.write(package, "broken.py", "import a_dependency_that_is_not_installed_xyz\n")
        # a top-level module with the name of one of chai's
        self.write(self.tmp.name, "provider.py", "from chai.provider import Provider\n\n\nclass OnlyHere(Provider):\n    pass\n\n\nclass DirFileProvider(Provider):\n    pass\n")
        sys.path.insert(0, self.tmp.name)

    def tearDown(self):
        sys.path.remove(self.tmp.name)
        for name in self.MODULES:
            sys.modules.pop(name, None)
        self.tmp.cleanup()

    @staticmethod
    def write(directory, name, source):
        with open(os.path.join(directory, name), "w") as fh:
            fh.write(textwrap.dedent(source))

    def test_a_workflow_names_a_projects_component_by_its_dotted_path(self):
        wf = Workflow(
            {
                "type": "workflow.Workflow",
                "id": "project_wf",
                "steps": [{"type": "heritage_project.shout.ShoutTranscriber", "id": "shout"}],
            }
        )
        self.assertEqual(type(wf.steps[0]).__name__, "ShoutTranscriber")
        out = wf.run("ave maria")
        self.assertEqual(out.value[0].value, "AVE MARIA")

    def test_chai_wins_over_a_module_of_the_same_name(self):
        self.assertIs(importClass("provider.DirFileProvider"), DirFileProvider)
        # ... and what chai's module does not have is found in the other one
        self.assertEqual(importClass("provider.OnlyHere").__module__, "provider")

    def test_a_class_that_is_not_a_component_is_refused(self):
        with self.assertRaises(TypeError):
            importClass("heritage_project.shout.NotAComponent")
        with self.assertRaises(TypeError):
            importClass("collections.OrderedDict")

    def test_a_modules_own_failed_import_is_not_reported_as_a_missing_module(self):
        with self.assertRaises(ModuleNotFoundError) as cm:
            importClass("heritage_project.broken.Thing")
        self.assertEqual(cm.exception.name, "a_dependency_that_is_not_installed_xyz")


if __name__ == "__main__":
    unittest.main()
