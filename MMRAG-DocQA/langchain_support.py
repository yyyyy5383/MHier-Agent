import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TracingConfig:
    enabled: bool = False
    project: str = "mhier-agent"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_runnable_config(self):
        if not self.enabled:
            return {}
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", self.project)
        config = {}
        if self.run_name:
            config["run_name"] = self.run_name
        if self.tags:
            config["tags"] = self.tags
        if self.metadata:
            config["metadata"] = self.metadata
        return config


class OpenAICompatibleEmbeddings:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [embedding.embedding for embedding in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


def report_to_documents(report):
    from langchain_core.documents import Document

    doc_id = report.get("metainfo", {}).get("sha1_name") or report.get("metainfo", {}).get("document_name")
    documents = []
    for index, chunk in enumerate(report.get("content", {}).get("chunks", [])):
        chunk_id = chunk.get("id", index)
        metadata = {
            "doc_id": doc_id,
            "page": chunk.get("page"),
            "chunk_id": chunk_id,
            "type": chunk.get("type", "content"),
        }
        documents.append(Document(page_content=chunk.get("text", ""), metadata=metadata))
    return documents


class LangChainFaissBackend:
    def __init__(self, vectorstore, pages_by_number):
        self.vectorstore = vectorstore
        self.pages_by_number = pages_by_number

    @classmethod
    def from_report(cls, report, embeddings):
        from langchain_community.vectorstores import FAISS

        documents = report_to_documents(report)
        vectorstore = FAISS.from_documents(documents, embeddings)
        pages_by_number = {
            page.get("page"): page.get("text", "")
            for page in report.get("content", {}).get("pages", [])
        }
        return cls(vectorstore, pages_by_number)

    def search(self, query, top_n=3, return_parent_pages=False):
        results = self.vectorstore.similarity_search_with_score(query, k=top_n)
        converted = []
        seen_pages = set()
        for document, score in results:
            metadata = dict(document.metadata)
            page = metadata.get("page")
            text = document.page_content
            if return_parent_pages:
                if page in seen_pages:
                    continue
                seen_pages.add(page)
                text = self.pages_by_number.get(page, text)
            converted.append({
                "distance": round(float(score), 4),
                "page": page,
                "text": text,
                "metadata": metadata,
            })
        return converted


def build_lcel_retrieval_chain(
    *,
    step_back_query: Callable[[dict], dict],
    vector_search: Callable[[dict], dict],
    raptor_summary: Callable[[dict], dict],
    llm_rerank: Callable[[dict], dict],
    assemble_context: Callable[[dict], Any],
    tracing_config: TracingConfig | None = None,
):
    from langchain_core.runnables import RunnableLambda

    chain = (
        RunnableLambda(step_back_query)
        | RunnableLambda(vector_search)
        | RunnableLambda(raptor_summary)
        | RunnableLambda(llm_rerank)
        | RunnableLambda(assemble_context)
    )
    runnable_config = (tracing_config or TracingConfig()).to_runnable_config()
    if runnable_config:
        chain = chain.with_config(runnable_config)
    return chain
