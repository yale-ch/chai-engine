import tempfile
import unittest

from chai.result import ItemResult
from chai.workflow import Workflow


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "rag_wf", "type": "workflow.Workflow"})
        self.db = tempfile.mktemp(suffix=".db")

    def test_hash_embeddings_deterministic(self):
        from chai.embeddings import embed_texts

        a = embed_texts(["the cat sat"], service="hash")
        b = embed_texts(["the cat sat"], service="hash")
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 512)

    def test_index_then_retrieve(self):
        docs = [
            "Herbarium sheets hold pressed plant specimens.",
            "The barcode identifies each specimen uniquely.",
            "Basketball is played with two teams of five.",
        ]
        indexer = self.wf._make_step(
            {
                "type": "embeddings.VectorIndexer",
                "id": "index",
                "settings": {"database": self.db, "collection": "docs", "documents": docs},
            },
            self.wf,
        )
        retriever = self.wf._make_step(
            {
                "type": "embeddings.VectorRetriever",
                "id": "retrieve",
                "settings": {"database": self.db, "collection": "docs", "top_k": 2},
            },
            self.wf,
        )
        query = ItemResult("pressed plant specimens on sheets")
        passed_through = indexer.process(query)  # indexes docs, passes the query along
        hits = retriever.process(passed_through)
        self.assertEqual(len(hits.value), 2)
        self.assertIn("plant", hits.value[0].value)
        self.assertGreater(hits.value[0].metadata["score"], hits.value[1].metadata["score"])

    def test_indexer_indexes_input_list(self):
        from chai.embeddings import VectorStore

        indexer = self.wf._make_step(
            {
                "type": "embeddings.VectorIndexer",
                "id": "index2",
                "settings": {"database": self.db, "collection": "inputs"},
            },
            self.wf,
        )
        from chai.result import ListResult

        indexer.process(ListResult([ItemResult("alpha"), ItemResult("beta")]))
        self.assertEqual(VectorStore(self.db).count("inputs"), 2)


class TestEvaluators(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "eval_wf", "type": "workflow.Workflow"})

    def test_text_metrics(self):
        ev = self.wf._make_step(
            {
                "type": "evaluator.TextMetricsEvaluator",
                "settings": {"reference": "Herbarium of Yale University"},
            },
            self.wf,
        )
        perfect = ev.process(ItemResult("herbarium of yale university"))
        self.assertTrue(perfect.value["exact"])
        self.assertEqual(perfect.value["cer"], 0)

        typo = ev.process(ItemResult("Herbarium of Yale Universety"))
        self.assertFalse(typo.value["exact"])
        self.assertGreater(typo.value["cer"], 0)
        self.assertAlmostEqual(typo.value["wer"], 0.25)  # 1 of 4 words wrong

    def test_record_metrics(self):
        ev = self.wf._make_step(
            {
                "type": "evaluator.RecordFieldEvaluator",
                "settings": {
                    "expected": {"genus": "Viburnum", "specificEpithet": "lautum", "country": "United States"},
                },
            },
            self.wf,
        )
        out = ev.process(ItemResult({"genus": "Viburnum", "specificEpithet": "latum", "recordedBy": "Moeglein"}))
        m = out.value
        self.assertEqual(m["fields"]["genus"], "correct")
        self.assertEqual(m["fields"]["specificEpithet"], "wrong")
        self.assertEqual(m["fields"]["country"], "missing")
        self.assertEqual(m["fields"]["recordedBy"], "spurious")
        self.assertLess(m["f1"], 1.0)


class TestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "schema_wf", "type": "workflow.Workflow"})

    def test_validate_basics(self):
        from chai.schema import SchemaError, validate

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {"type": {"type": "string", "enum": ["Person", "Location"]}, "text": {"type": "string"}},
            },
        }
        validate([{"type": "Person", "text": "Tom"}], schema)
        with self.assertRaises(SchemaError):
            validate([{"type": "Person"}], schema)  # missing required text
        with self.assertRaises(SchemaError):
            validate([{"type": "Animal", "text": "x"}], schema)  # bad enum

    def test_extractor_schema_retries_invalid_output(self):
        from chai.extractor import Extractor

        class FlakyExtractor(Extractor):
            calls = 0

            def _process(self, input):
                FlakyExtractor.calls += 1
                if FlakyExtractor.calls == 1:
                    return ItemResult({"text": "Tom"}, input=input, processor=self)  # invalid: no type
                return ItemResult({"type": "Person", "text": "Tom"}, input=input, processor=self)

        comp = FlakyExtractor(
            {
                "id": "flaky_extract",
                "settings": {
                    "retries": 2,
                    "retry_delay": 0,
                    "schema": {"type": "object", "required": ["type", "text"]},
                },
            },
            self.wf,
        )
        out = comp.process(ItemResult("Tom went home."))
        self.assertEqual(out.value, {"type": "Person", "text": "Tom"})
        self.assertEqual(FlakyExtractor.calls, 2)


if __name__ == "__main__":
    unittest.main()
