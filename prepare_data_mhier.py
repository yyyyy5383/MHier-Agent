import argparse
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import pymupdf
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
MMRAG_DIR = ROOT_DIR / "MMRAG-DocQA"

# Ensure MMRAG-DocQA modules are importable
if str(MMRAG_DIR) not in os.sys.path:
    os.sys.path.append(str(MMRAG_DIR))

from pdf_parsing import PDFParser
from parsed_reports_merging import PageTextPreparation
from text_splitter import TextSplitter
from ingestion import VectorDBIngestor


@contextmanager
def chdir(path: Path):
    prev = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def parse_pdfs(pdf_dir: Path, parsed_dir: Path, parallel: bool, chunk_size: int, max_workers: int):
    logging.basicConfig(level=logging.INFO)
    pdf_parser = PDFParser(output_dir=parsed_dir)
    if parallel:
        input_doc_paths = list(pdf_dir.glob("*.pdf"))
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size,
        )
    else:
        pdf_parser.parse_and_export(doc_dir=pdf_dir)


def extract_page_images(pdf_dir: Path, images_dir: Path, dpi: int, overwrite: bool):
    images_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in tqdm(list(pdf_dir.glob("*.pdf")), desc="Extracting page images"):
        with pymupdf.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                page_num = page_index + 1
                out_path = images_dir / f"{pdf_path.stem}_{page_num}.png"
                if out_path.exists() and not overwrite:
                    continue
                pix = page.get_pixmap(dpi=dpi)
                pix.save(out_path)


def prepare_reports(parsed_dir: Path, merged_dir: Path, chunked_dir: Path, raptor_tree_dir: Path):
    ptp = PageTextPreparation(use_serialized_tables=False)
    ptp.process_reports(reports_dir=parsed_dir, output_dir=merged_dir)

    splitter = TextSplitter()
    with chdir(raptor_tree_dir):
        splitter.split_all_reports(merged_dir, chunked_dir)


def build_vector_db(chunked_dir: Path, vector_db_dir: Path):
    vdb = VectorDBIngestor()
    vdb.process_reports(chunked_dir, vector_db_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare MHier data with MMRAG pipeline.")
    parser.add_argument("--pdf-dir", default="data/raw_pdfs", help="Directory containing raw PDFs.")
    parser.add_argument("--processed-json-dir", default="data/processed_json", help="Directory for parsed/merged/chunked JSON.")
    parser.add_argument("--vector-db-dir", default="data/vector_db", help="Directory for Faiss vector DBs.")
    parser.add_argument("--raptor-tree-dir", default="data/raptor_tree", help="Directory for RAPTOR trees.")
    parser.add_argument("--images-dir", default="data/page_images", help="Directory for page image outputs.")
    parser.add_argument("--image-dpi", type=int, default=144, help="DPI for page image rendering.")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite existing page images.")
    parser.add_argument("--parallel-parse", action="store_true", help="Enable parallel PDF parsing.")
    parser.add_argument("--chunk-size", type=int, default=2, help="Chunk size for parallel parsing.")
    parser.add_argument("--max-workers", type=int, default=10, help="Max workers for parallel parsing.")
    args = parser.parse_args()

    pdf_dir = resolve_path(args.pdf_dir)
    processed_json_dir = resolve_path(args.processed_json_dir)
    vector_db_dir = resolve_path(args.vector_db_dir)
    raptor_tree_dir = resolve_path(args.raptor_tree_dir)
    images_dir = resolve_path(args.images_dir)

    parsed_dir = processed_json_dir / "01_parsed_reports"
    merged_dir = processed_json_dir / "02_merged_reports"
    chunked_dir = processed_json_dir / "03_chunked_reports"

    for path in [parsed_dir, merged_dir, chunked_dir, vector_db_dir, raptor_tree_dir, images_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("Step 1: Parsing PDFs with Docling...")
    parse_pdfs(pdf_dir, parsed_dir, args.parallel_parse, args.chunk_size, args.max_workers)

    print("Step 2: Extracting page images...")
    extract_page_images(pdf_dir, images_dir, args.image_dpi, args.overwrite_images)

    print("Step 3: Merging reports and building RAPTOR trees...")
    prepare_reports(parsed_dir, merged_dir, chunked_dir, raptor_tree_dir)

    print("Step 4: Building vector DBs (Faiss)...")
    build_vector_db(chunked_dir, vector_db_dir)

    print("Done.")
    print(f"Parsed JSON: {parsed_dir}")
    print(f"Merged JSON: {merged_dir}")
    print(f"Chunked JSON: {chunked_dir}")
    print(f"Vector DB: {vector_db_dir}")
    print(f"RAPTOR tree: {raptor_tree_dir}")
    print(f"Page images: {images_dir}")


if __name__ == "__main__":
    main()
