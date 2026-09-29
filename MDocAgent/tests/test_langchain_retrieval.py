import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MMRAG_ROOT = os.path.join(REPO_ROOT, "MMRAG-DocQA")


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeFAISS:
    def __init__(self, documents):
        self.documents = documents

    @classmethod
    def from_documents(cls, documents, embedding):
        embedding.embed_documents([doc.page_content for doc in documents])
        return cls(documents)

    def similarity_search_with_score(self, query, k=4):
        return [(doc, float(index + 1)) for index, doc in enumerate(self.documents[:k])]


class FakeRunnableLambda:
    def __init__(self, fn):
        self.steps = [fn]
        self.config = None

    def __or__(self, other):
        chain = FakeRunnableLambda(lambda value: value)
        chain.steps = self.steps + other.steps
        return chain

    def with_config(self, config):
        self.config = config
        return self

    def invoke(self, value):
        result = value
        for step in self.steps:
            result = step(result)
        return result


class FakeEmbeddings:
    def __init__(self):
        self.document_inputs = []
        self.query_inputs = []

    def embed_documents(self, texts):
        self.document_inputs.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.query_inputs.append(text)
        return [1.0, 0.0]


def install_fake_langchain_modules():
    sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = FakeDocument
    sys.modules["langchain_core.documents"] = documents_module

    runnables_module = types.ModuleType("langchain_core.runnables")
    runnables_module.RunnableLambda = FakeRunnableLambda
    sys.modules["langchain_core.runnables"] = runnables_module

    sys.modules["langchain_community"] = types.ModuleType("langchain_community")
    vectorstores_module = types.ModuleType("langchain_community.vectorstores")
    vectorstores_module.FAISS = FakeFAISS
    sys.modules["langchain_community.vectorstores"] = vectorstores_module


def load_langchain_support():
    install_fake_langchain_modules()
    module_path = os.path.join(MMRAG_ROOT, "langchain_support.py")
    spec = importlib.util.spec_from_file_location("mmrag_langchain_support", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mmrag_retrieval():
    install_fake_langchain_modules()
    if MMRAG_ROOT not in sys.path:
        sys.path.insert(0, MMRAG_ROOT)

    faiss_module = types.ModuleType("faiss")
    faiss_module.read_index = lambda path: object()
    sys.modules["faiss"] = faiss_module
    sys.modules["numpy"] = types.SimpleNamespace(
        float32="float32",
        array=lambda value, dtype=None: types.SimpleNamespace(reshape=lambda *args: value),
        dot=lambda left, right: 1.0,
        linalg=types.SimpleNamespace(norm=lambda value: 1.0),
    )

    mmrag_config = types.ModuleType("mmrag_config")

    class FakeEmbeddingData:
        embedding = [1.0, 0.0]

    class FakeEmbeddingResponse:
        data = [FakeEmbeddingData()]

    class FakeEmbeddingsClient:
        def create(self, input, model):
            if isinstance(input, list):
                response = types.SimpleNamespace(
                    data=[types.SimpleNamespace(embedding=[1.0, 0.0]) for _ in input]
                )
                return response
            return FakeEmbeddingResponse()

    mmrag_config.get_openai_client = lambda name: types.SimpleNamespace(embeddings=FakeEmbeddingsClient())
    mmrag_config.get_model_name = lambda name: "fake-model"
    sys.modules["mmrag_config"] = mmrag_config

    reranking = types.ModuleType("reranking")
    reranking.LLMReranker = lambda: types.SimpleNamespace(
        rerank_documents=lambda query, documents, documents_batch_size=1, llm_weight=0.7: documents
    )
    sys.modules["reranking"] = reranking

    raptor = types.ModuleType("raptor")
    raptor.RetrievalAugmentation = lambda tree=None: types.SimpleNamespace(
        answer_question=lambda question: ("summary", {})
    )
    sys.modules["raptor"] = raptor

    module_path = os.path.join(MMRAG_ROOT, "retrieval.py")
    spec = importlib.util.spec_from_file_location("mmrag_retrieval_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LangChainRetrievalTests(unittest.TestCase):
    def test_langchain_faiss_backend_preserves_chunk_metadata(self):
        support = load_langchain_support()
        report = {
            "metainfo": {"sha1_name": "doc-a"},
            "content": {
                "chunks": [
                    {"id": 10, "page": 1, "type": "content", "text": "alpha"},
                    {"id": 11, "page": 2, "type": "table", "text": "beta"},
                ],
                "pages": [
                    {"page": 1, "text": "page one"},
                    {"page": 2, "text": "page two"},
                ],
            },
        }
        embeddings = FakeEmbeddings()

        backend = support.LangChainFaissBackend.from_report(report, embeddings)
        results = backend.search("question", top_n=2, return_parent_pages=True)

        self.assertEqual(embeddings.document_inputs, ["alpha", "beta"])
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(results[0]["text"], "page one")
        self.assertEqual(results[0]["metadata"]["doc_id"], "doc-a")
        self.assertEqual(results[0]["metadata"]["chunk_id"], 10)

    def test_lcel_chain_returns_legacy_compatible_retrieval_tuple(self):
        support = load_langchain_support()

        chain = support.build_lcel_retrieval_chain(
            step_back_query=lambda state: {**state, "step_back_query": "expanded"},
            vector_search=lambda state: {**state, "vector_results": [{"text": "candidate"}]},
            raptor_summary=lambda state: {**state, "raptor_summary": "summary"},
            llm_rerank=lambda state: {**state, "reranked_results": [{"text": "reranked"}]},
            assemble_context=lambda state: (state["reranked_results"], state["raptor_summary"]),
            tracing_config=support.TracingConfig(
                enabled=True,
                project="mhier-agent",
                run_name="retrieval",
                tags=["test"],
                metadata={"doc_id": "doc-a"},
            ),
        )

        result = chain.invoke({"query": "q", "document_name": "doc-a"})

        self.assertEqual(result, ([{"text": "reranked"}], "summary"))

    def test_tracing_config_is_noop_when_disabled(self):
        support = load_langchain_support()

        config = support.TracingConfig(enabled=False).to_runnable_config()

        self.assertEqual(config, {})

    def test_tracing_config_sets_langsmith_runnable_metadata(self):
        support = load_langchain_support()
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGSMITH_PROJECT", None)

        config = support.TracingConfig(
            enabled=True,
            project="mhier-agent",
            run_name="retrieval",
            tags=["mhier"],
            metadata={"doc_id": "a"},
        ).to_runnable_config()

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(os.environ["LANGSMITH_PROJECT"], "mhier-agent")
        self.assertEqual(config["run_name"], "retrieval")
        self.assertEqual(config["tags"], ["mhier"])
        self.assertEqual(config["metadata"], {"doc_id": "a"})

    def test_vector_retriever_langchain_backend_does_not_require_raw_faiss_file(self):
        retrieval = load_mmrag_retrieval()
        with tempfile.TemporaryDirectory() as tmpdir:
            documents_dir = os.path.join(tmpdir, "docs")
            vector_dir = os.path.join(tmpdir, "vectors")
            os.makedirs(documents_dir)
            os.makedirs(vector_dir)
            with open(os.path.join(documents_dir, "doc-a.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metainfo": {"sha1_name": "doc-a"},
                        "content": {
                            "chunks": [{"id": 1, "page": 1, "text": "alpha"}],
                            "pages": [{"page": 1, "text": "page alpha"}],
                        },
                    },
                    f,
                )

            retriever = retrieval.VectorRetriever(
                vector_db_dir=Path(vector_dir),
                documents_dir=Path(documents_dir),
                vector_backend="langchain_faiss",
            )

        self.assertEqual(len(retriever.all_dbs), 1)


if __name__ == "__main__":
    unittest.main()
