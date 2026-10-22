import os
import tempfile
import unittest
from unittest import mock

from chai.result import ItemResult
from chai.workflow import Workflow


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "rag_wf", "type": "workflow.Workflow"})
        self.db = tempfile.mktemp(suffix=".db")

    def test_hash_embeddings_deterministic(self):
        from chai.embedder import hash_embed

        a = hash_embed(["the cat sat"])
        b = hash_embed(["the cat sat"])
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
                "type": "embedder.VectorIndexer",
                "id": "index",
                "settings": {"database": self.db, "collection": "docs", "documents": docs},
            },
            self.wf,
        )
        retriever = self.wf._make_step(
            {
                "type": "embedder.VectorRetriever",
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
        from chai.embedder import VectorStore

        indexer = self.wf._make_step(
            {
                "type": "embedder.VectorIndexer",
                "id": "index2",
                "settings": {"database": self.db, "collection": "inputs"},
            },
            self.wf,
        )
        from chai.result import ListResult

        indexer.process(ListResult([ItemResult("alpha"), ItemResult("beta")]))
        self.assertEqual(VectorStore(self.db).count("inputs"), 2)

    def test_backend_composites_embed_through_litellm(self):
        # litellm's mock_response returns the given vector without calling the API
        settings = {
            "database": self.db,
            "collection": "mocked",
            "api_key": "test",
            "litellm_params": {"mock_response": [1.0, 0.0, 0.0]},
        }
        indexer = self.wf._make_step(
            {"type": "embedder.OpenAIVectorIndexer", "id": "oai_index", "settings": {**settings, "documents": ["one"]}},
            self.wf,
        )
        retriever = self.wf._make_step(
            {"type": "embedder.OpenAIVectorRetriever", "id": "oai_retrieve", "settings": settings},
            self.wf,
        )
        self.assertEqual(indexer.model, "openai/text-embedding-3-small")
        hits = retriever.process(indexer.process(ItemResult("query")))
        self.assertEqual([h.value for h in hits.value], ["one"])
        self.assertAlmostEqual(hits.value[0].metadata["score"], 1.0)
        self.assertEqual(hits.metadata["engine"], "openai")
        self.assertEqual(hits.metadata["model"], "openai/text-embedding-3-small")
        self.assertIn("token_usage", hits.metadata)

    def test_backend_embedding_defaults(self):
        ollama = self.wf._make_step({"type": "embedder.OllamaVectorIndexer", "id": "ollama_index"}, self.wf)
        self.assertEqual(ollama.model, "ollama/nomic-embed-text")
        self.assertEqual(ollama.api_base, "http://localhost:11434")
        named = self.wf._make_step(
            {"type": "embedder.LMStudioVectorRetriever", "id": "lms", "settings": {"model": "my-embedder"}}, self.wf
        )
        self.assertEqual(named.model, "lm_studio/my-embedder")

    def test_vector_store_refuses_mismatched_vectors(self):
        from chai.persistence.vectors import VectorStore

        store = VectorStore(self.db)
        with self.assertRaises(ValueError):
            store.add("short", ["one", "two", "three"], [[1.0, 0.0]])
        self.assertEqual(store.count("short"), 0)

    def test_one_vector_for_many_texts_refused(self):
        # litellm's mock_response answers any request with a single vector, as Vertex AI's
        # embedContent does for a batch: the texts must not be paired with it
        indexer = self.wf._make_step(
            {
                "type": "embedder.OpenAIVectorIndexer",
                "id": "oai_short",
                "settings": {"api_key": "test", "litellm_params": {"mock_response": [1.0, 0.0, 0.0]}},
            },
            self.wf,
        )
        self.assertFalse(indexer.embeds_separately())
        with self.assertRaises(ValueError):
            indexer.embed(["one", "two"])

    def test_embed_content_models_get_a_request_per_text(self):
        from chai.persistence.vectors import VectorStore

        vertex = {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "GOOGLE_CLOUD_PROJECT": "test-project", "VERTEXAI_PROJECT": ""}
        settings = {"database": self.db, "collection": "vertex", "litellm_params": {"mock_response": [1.0, 0.0, 0.0]}}
        with mock.patch.dict(os.environ, vertex):
            indexer = self.wf._make_step(
                {
                    "type": "embedder.GeminiVectorIndexer",
                    "id": "vertex_index",
                    "settings": {**settings, "documents": ["one", "two", "three"]},
                },
                self.wf,
            )
            older = self.wf._make_step(
                {"type": "embedder.GeminiVectorIndexer", "id": "vertex_001", "settings": {"model": "gemini-embedding-001"}},
                self.wf,
            )
        self.assertEqual(indexer.model, "vertex_ai/gemini-embedding-2")
        self.assertTrue(indexer.embeds_separately())
        self.assertFalse(older.embeds_separately())
        vectors, call = indexer.embed(["one", "two", "three"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(call["token_usage"]["prompt"], 30)  # the mock reports 10 prompt tokens a request
        indexer.process(ItemResult("query"))
        self.assertEqual(VectorStore(self.db).count("vertex"), 3)

    def test_indexing_usage_reaches_the_ledger(self):
        settings = {
            "database": self.db,
            "collection": "ledger",
            "api_key": "test",
            "litellm_params": {"mock_response": [1.0, 0.0, 0.0]},
        }
        wf = Workflow(
            {
                "id": "ledger_wf",
                "type": "workflow.Workflow",
                "steps": [
                    {
                        "type": "embedder.OpenAIVectorIndexer",
                        "id": "index",
                        "settings": {**settings, "documents": ["one"]},
                        "next_steps": [{"type": "embedder.OpenAIVectorRetriever", "id": "retrieve", "settings": settings}],
                    }
                ],
            }
        )
        wf.run(ItemResult("query"))
        usage = wf.usage_summary()
        self.assertEqual(set(usage["by_component"]), {"index", "retrieve"})
        self.assertEqual(usage["by_component"]["index"]["prompt_tokens"], 10)
        self.assertEqual(usage["calls"], 2)

    def test_service_setting_refused(self):
        with self.assertRaises(ValueError):
            self.wf._make_step(
                {"type": "embedder.VectorIndexer", "id": "old", "settings": {"service": "gemini"}}, self.wf
            )


if __name__ == "__main__":
    unittest.main()
