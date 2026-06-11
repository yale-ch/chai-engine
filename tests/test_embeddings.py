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


if __name__ == "__main__":
    unittest.main()
