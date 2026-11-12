"""
End-to-end ingestion and RAG pipeline test using sample_report.pdf.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.embeddings import build_index
from backend.pdf_processor import extract_documents_from_pdf
from backend.query_engine import query_index
from config.settings import get_google_api_key
from utils.helpers import validate_pdf


def main() -> int:
    pdf_path = BASE_DIR / "sample_report.pdf"
    print(f"==================================================", flush=True)
    print(f"Testing Complete Ingestion Pipeline on: {pdf_path.name}", flush=True)
    print(f"==================================================", flush=True)

    # 1. Check API Key
    api_key = get_google_api_key()
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY is not set.", flush=True)
        return 1
    print(f"✅ Google API Key loaded (length: {len(api_key)})", flush=True)

    # 2. Read and Validate PDF
    if not pdf_path.exists():
        print(f"❌ Error: File not found at {pdf_path}", flush=True)
        return 1

    file_bytes = pdf_path.read_bytes()
    is_valid, err = validate_pdf(file_bytes, pdf_path.name)
    if not is_valid:
        print(f"❌ Validation failed: {err}", flush=True)
        return 1
    print(f"✅ File validation passed: {len(file_bytes):,} bytes, valid %PDF- header", flush=True)

    # 3. Extract Documents per Page
    print("\n--- Step 1: Extracting All Pages (PyMuPDF) ---", flush=True)
    start_t = time.time()
    documents = extract_documents_from_pdf(file_bytes, source_name=pdf_path.name)
    extract_duration = time.time() - start_t
    print(f"✅ Successfully extracted {len(documents)} page documents in {extract_duration:.2f}s", flush=True)
    for doc in documents[:5]:
        page_num = doc.metadata.get("page_number")
        total_p = doc.metadata.get("total_pages")
        print(f"   • Page {page_num}/{total_p}: {len(doc.text)} chars | excerpt: {doc.text[:65]!r}...", flush=True)
    if len(documents) > 5:
        print(f"   • ... and {len(documents) - 5} more pages successfully extracted.", flush=True)

    # 4. Chunk & Build Vector Index (Gemini Embeddings)
    print("\n--- Step 2: Generating Embeddings & Storing in Vector DB ---", flush=True)
    start_t = time.time()
    index = build_index(documents)
    embed_duration = time.time() - start_t
    num_nodes = len(index.docstore.docs)
    print(f"✅ Vector index built successfully in {embed_duration:.2f}s", flush=True)
    print(f"   • Total vector chunks indexed: {num_nodes}", flush=True)

    # 5. Query and Verify Question Answering
    test_questions = [
        "What is Retrieval-Augmented Generation (RAG) and what are its key components according to the report?",
        "What are the key considerations and metrics for model evaluation discussed in the document?",
    ]

    print("\n--- Step 3: Verifying RAG Question Answering ---", flush=True)
    for q_idx, question in enumerate(test_questions, start=1):
        print(f"\n==================================================", flush=True)
        print(f"[Test Query {q_idx}]: {question}", flush=True)
        print(f"==================================================", flush=True)
        start_t = time.time()
        result = query_index(index, question)
        query_duration = time.time() - start_t

        print(f"⏱️  Answer generated in {query_duration:.2f}s", flush=True)
        print(f"\n🤖 AI Answer:\n{result.answer.strip()}\n", flush=True)
        print(f"📚 Cited Sources ({len(result.sources)} chunks retrieved):", flush=True)
        for src in result.sources:
            score = src.get("relevance_score")
            score_str = f" (relevance: {score:.4f})" if score else ""
            print(f"   • Page {src['page_number']}{score_str} — {src['source']}", flush=True)
            print(f"     Excerpt: “{src['text_snippet'][:130]}...”", flush=True)

        if not result.answer or not result.sources:
            print("❌ Query returned empty answer or no sources!", flush=True)
            return 1

    print("\n==================================================", flush=True)
    print("🎉 Ingestion pipeline and Q&A test completed successfully!", flush=True)
    print("==================================================", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
