import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
import importlib.util
import sys

from tqdm import tqdm

from retrieval.base_retrieval import BaseRetrieval
from mydatasets.base_dataset import BaseDataset


class MHierRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.repo_root = Path(__file__).resolve().parents[2]
        self.mmrag_root = self.repo_root / "MMRAG-DocQA"

        self.vector_db_dir = self._resolve_path(getattr(config, "vector_db_dir", None))
        self.documents_dir = self._resolve_documents_dir(self._resolve_path(getattr(config, "documents_dir", None)))
        self.raptor_tree_dir = self._resolve_path(getattr(config, "raptor_tree_dir", None))
        self.images_dir = self._resolve_path(getattr(config, "images_dir", None))

        if self.vector_db_dir is None:
            self.vector_db_dir = self.repo_root / "data" / "vector_db"
        if self.documents_dir is None:
            self.documents_dir = self.repo_root / "data" / "processed_json" / "03_chunked_reports"
        if self.raptor_tree_dir is None:
            self.raptor_tree_dir = self.repo_root / "data" / "raptor_tree"

        self.hybrid_retriever = self._load_hybrid_retriever()
        self.image_index = self._build_image_index(self.images_dir)

    def _resolve_path(self, value):
        if value is None:
            return None
        raw = Path(str(value))
        if raw.is_absolute():
            return raw
        path = (self.repo_root / raw).resolve()
        if path.exists():
            return path
        alt = (self.repo_root / "MDocAgent" / raw).resolve()
        if alt.exists():
            return alt
        return path

    def _resolve_documents_dir(self, path: Path):
        if path is None:
            return None
        if path.exists():
            chunked = path / "03_chunked_reports"
            if chunked.exists():
                return chunked
            return path
        fallback = self.repo_root / "data" / "processed_json" / "03_chunked_reports"
        if fallback.exists():
            return fallback
        return path

    def _load_mmrag_module(self, filename: str):
        if str(self.mmrag_root) not in sys.path:
            sys.path.append(str(self.mmrag_root))
        module_path = self.mmrag_root / filename
        spec = importlib.util.spec_from_file_location("mmrag_retrieval", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_hybrid_retriever(self):
        module = self._load_mmrag_module("retrieval.py")
        hybrid_class = getattr(module, "HybridRetriever")
        tracing_class = getattr(module, "TracingConfig")
        tracing_config = getattr(self.config, "tracing", None)
        tracing = tracing_class(
            enabled=bool(getattr(tracing_config, "enabled", False)),
            project=str(getattr(tracing_config, "project", "mhier-agent")),
            run_name=str(getattr(tracing_config, "run_name", "mhier-retrieval")),
            tags=list(getattr(tracing_config, "tags", [])),
            metadata=dict(getattr(tracing_config, "metadata", {})),
        )
        return hybrid_class(
            self.vector_db_dir,
            self.documents_dir,
            vector_backend=str(getattr(self.config, "vector_backend", "raw_faiss")),
            chain_backend=str(getattr(self.config, "chain_backend", "legacy")),
            tracing_config=tracing,
        )

    @contextmanager
    def _chdir(self, path: Path):
        if path is None:
            yield
            return
        path.mkdir(parents=True, exist_ok=True)
        prev = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev)

    def _extract_doc_name(self, sample: dict) -> str:
        doc_value = sample.get(self.config.doc_key, "")
        return Path(str(doc_value)).stem

    def _build_image_index(self, images_dir: Path) -> Dict[str, Dict[int, str]]:
        index: Dict[str, Dict[int, str]] = {}
        if images_dir is None or not images_dir.exists():
            return index
        for image_path in images_dir.glob("*.png"):
            stem = image_path.stem
            doc_part = None
            page_num = None

            if "_page_" in stem:
                doc_part, _, page_part = stem.rpartition("_page_")
                if page_part.isdigit():
                    page_num = int(page_part)
            if doc_part is None or page_num is None:
                doc_part, sep, page_part = stem.rpartition("_")
                if sep and page_part.isdigit():
                    page_num = int(page_part)
            if doc_part is None or page_num is None:
                match = re.match(r"^(?P<doc>.+)page(?P<page>\d+)$", stem)
                if match:
                    doc_part = match.group("doc").rstrip("_-")
                    page_num = int(match.group("page"))

            if doc_part is None or page_num is None:
                continue
            index.setdefault(doc_part, {})[page_num] = str(image_path)
        return index

    def _get_image_path(self, doc_name: str, page_num: int):
        if self.images_dir is None:
            return None
        doc_map = self.image_index.get(doc_name, {})
        if page_num in doc_map:
            return doc_map[page_num]
        if page_num > 0 and (page_num - 1) in doc_map:
            return doc_map[page_num - 1]
        matches = list(self.images_dir.glob(f"{doc_name}_page_{page_num}*.png"))
        if not matches:
            matches = list(self.images_dir.glob(f"{doc_name}_{page_num}*.png"))
        if matches:
            return str(matches[0])
        return None

    def find_top_k(self, dataset: BaseDataset):
        top_k = self.config.top_k
        llm_reranking_sample_size = getattr(self.config, "llm_reranking_sample_size", 28)
        documents_batch_size = getattr(self.config, "documents_batch_size", 1)
        llm_weight = getattr(self.config, "llm_weight", 0.7)
        return_parent_pages = getattr(self.config, "return_parent_pages", True)
        llm_rerank = getattr(self.config, "llm_rerank", True)
        if not llm_rerank:
            llm_weight = 0.0

        samples = dataset.load_data(use_retreival=True)
        for sample in tqdm(samples):
            query = sample[self.config.text_question_key]
            doc_name = self._extract_doc_name(sample)
            try:
                with self._chdir(self.raptor_tree_dir):
                    retrieval_output = self.hybrid_retriever.retrieve_by_document_name(
                        document_name=doc_name,
                        query=query,
                        llm_reranking_sample_size=llm_reranking_sample_size,
                        documents_batch_size=documents_batch_size,
                        top_n=top_k,
                        llm_weight=llm_weight,
                        return_parent_pages=return_parent_pages,
                        is_picture=False,
                    )
            except Exception as e:
                print(f"[MHierRetrieval] Retrieval failed for {doc_name}: {e}")
                sample[self.config.r_text_key] = []
                sample[self.config.r_image_key] = []
                sample["texts"] = []
                sample["images"] = []
                continue

            reranked_results = None
            raptor_summary = None
            if isinstance(retrieval_output, tuple):
                if len(retrieval_output) >= 1:
                    reranked_results = retrieval_output[0]
                if len(retrieval_output) >= 2:
                    raptor_summary = retrieval_output[1]
            else:
                reranked_results = retrieval_output

            texts = []
            if raptor_summary:
                if isinstance(raptor_summary, list):
                    texts.append("\n".join(str(item) for item in raptor_summary))
                else:
                    texts.append(str(raptor_summary))
            if reranked_results:
                texts.extend(
                    item.get("text", "")
                    for item in reranked_results
                    if item.get("text")
                )

            images = []
            if reranked_results:
                for item in reranked_results:
                    page_num = item.get("page")
                    if page_num is None:
                        continue
                    try:
                        page_num = int(page_num)
                    except (TypeError, ValueError):
                        continue
                    image_path = self._get_image_path(doc_name, page_num)
                    if image_path:
                        images.append(image_path)

            sample[self.config.r_text_key] = texts
            sample[self.config.r_image_key] = images
            sample["texts"] = texts
            sample["images"] = images

        path = dataset.dump_data(samples, use_retreival=True)
        print(f"Save retrieval results at {path}.")
