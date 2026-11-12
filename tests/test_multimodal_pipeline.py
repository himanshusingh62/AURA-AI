"""
Comprehensive Multimodal Ingestion & Cross-Modal Testing Suite.
Tests:
  1. PDF validation and ingestion.
  2. PNG and JPG image validation and metadata extraction.
  3. Image Question Answering via Gemini vision (OCR, diagram comprehension).
  4. PDF Question Answering via LlamaIndex RAG.
  5. Combined Cross-Modal Q&A (PDF + Image joint reasoning).
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
from backend.image_processor import process_uploaded_image
from backend.multimodal_engine import query_multimodal
from backend.pdf_processor import extract_documents_from_pdf
from config.settings import get_google_api_key
from utils.helpers import validate_image, validate_pdf
import pytest


@pytest.mark.live
def test_live_multimodal_suite():
    """Live end-to-end multimodal ingestion, vision, and cross-modal reasoning test."""
    api_key = get_google_api_key()
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not configured for live testing.")
    import socket
    try:
        socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=3)
    except OSError:
        pytest.skip("External Google GenAI network is unreachable in this environment.")
    assert run_tests() == 0


def run_tests() -> int:
    print("==================================================", flush=True)
    print("STARTING COMPLETE MULTIMODAL TEST SUITE", flush=True)
    print("==================================================", flush=True)

    # Check API Key
    api_key = get_google_api_key()
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY is not set.", flush=True)
        return 1
    print(f"✅ Google API Key loaded (length: {len(api_key)})", flush=True)

    pdf_path = BASE_DIR / "sample_report.pdf"
    if not pdf_path.exists():
        pdf_path = Path(__file__).parent / "sample_report.pdf"

    png_path = BASE_DIR / "sample_architecture.png"
    if not png_path.exists():
        png_path = Path(__file__).parent / "sample_architecture.png"

    jpg_path = BASE_DIR / "sample_chart.jpg"
    if not jpg_path.exists():
        jpg_path = Path(__file__).parent / "sample_chart.jpg"

    # ──────────────────────────────────────────────────────────
    # TEST 1: PDF Validation & Ingestion
    # ──────────────────────────────────────────────────────────
    print("\n--- TEST 1: PDF Document Ingestion ---", flush=True)
    pdf_bytes = pdf_path.read_bytes()
    is_valid, err = validate_pdf(pdf_bytes, pdf_path.name)
    assert is_valid, f"PDF validation failed: {err}"
    print(f"✅ PDF validation passed: {pdf_path.name} ({len(pdf_bytes):,} bytes)", flush=True)

    documents = extract_documents_from_pdf(pdf_bytes, pdf_path.name)
    assert len(documents) > 0, "No documents extracted from PDF!"
    print(f"✅ PyMuPDF extracted {len(documents)} pages", flush=True)

    print("Building vector index with Gemini embeddings...", flush=True)
    t0 = time.time()
    pdf_index = build_index(documents)
    print(f"✅ Vector index created in {time.time()-t0:.2f}s ({len(pdf_index.docstore.docs)} chunks)", flush=True)

    # ──────────────────────────────────────────────────────────
    # TEST 2: Image Processing (PNG & JPG)
    # ──────────────────────────────────────────────────────────
    print("\n--- TEST 2: Image Validation & Processing (PNG & JPG) ---", flush=True)

    # PNG Test
    png_bytes = png_path.read_bytes()
    is_v, err = validate_image(png_bytes, png_path.name)
    assert is_v, f"PNG validation failed: {err}"
    proc_png = process_uploaded_image(png_bytes, png_path.name)
    assert proc_png.width > 0 and proc_png.height > 0
    assert len(proc_png.thumbnail_base64) > 0
    print(f"✅ PNG Processed: {proc_png.filename} ({proc_png.width}x{proc_png.height}, {proc_png.format}, {proc_png.size_mb} MB)", flush=True)

    # JPG Test
    jpg_bytes = jpg_path.read_bytes()
    is_v, err = validate_image(jpg_bytes, jpg_path.name)
    assert is_v, f"JPG validation failed: {err}"
    proc_jpg = process_uploaded_image(jpg_bytes, jpg_path.name)
    assert proc_jpg.width > 0 and proc_jpg.height > 0
    assert len(proc_jpg.thumbnail_base64) > 0
    print(f"✅ JPG Processed: {proc_jpg.filename} ({proc_jpg.width}x{proc_jpg.height}, {proc_jpg.format}, {proc_jpg.size_mb} MB)", flush=True)

    # ──────────────────────────────────────────────────────────
    # TEST 3: Image-Only Question Answering (Vision & OCR)
    # ──────────────────────────────────────────────────────────
    print("\n--- TEST 3: Image Vision & Diagram Understanding ---", flush=True)
    img_question = "What are the 5 architectural steps shown in this workflow diagram?"
    print(f"Query on {proc_png.filename}: '{img_question}'", flush=True)

    t0 = time.time()
    img_result = query_multimodal(
        question=img_question,
        pdf_index=None,
        images=[proc_png],
    )
    print(f"⏱️  Answer generated in {img_result.latency_seconds:.2f}s", flush=True)
    print(f"🤖 Answer:\n{img_result.answer}\n", flush=True)
    assert len(img_result.answer) > 20, "Image answer too short!"
    assert len(img_result.image_sources) == 1, "Expected 1 image source cited!"
    assert img_result.image_sources[0]["filename"] == proc_png.filename
    print("✅ Image Question Answering verified with image source attribution!", flush=True)

    # ──────────────────────────────────────────────────────────
    # TEST 4: Chart Understanding & OCR on JPG
    # ──────────────────────────────────────────────────────────
    print("\n--- TEST 4: Chart Data Extraction on JPG ---", flush=True)
    chart_question = "Which model has the highest accuracy score in this benchmark chart and what is the percentage?"
    print(f"Query on {proc_jpg.filename}: '{chart_question}'", flush=True)

    chart_result = query_multimodal(
        question=chart_question,
        pdf_index=None,
        images=[proc_jpg],
    )
    print(f"⏱️  Answer generated in {chart_result.latency_seconds:.2f}s", flush=True)
    print(f"🤖 Answer:\n{chart_result.answer}\n", flush=True)
    assert "97" in chart_result.answer or "Gemini" in chart_result.answer, "Expected Gemini 97% to be detected in chart!"
    print("✅ Chart data extraction and visual analysis verified!", flush=True)

    # ──────────────────────────────────────────────────────────
    # TEST 5: Combined PDF + Image Joint Reasoning
    # ──────────────────────────────────────────────────────────
    print("\n--- TEST 5: Combined PDF + Image Multimodal Reasoning ---", flush=True)
    combined_question = (
        "Compare the 'Vector Embed (Gemini 3072-dim)' and 'In-Memory Index' shown in the architecture diagram "
        "with how Retrieval-Augmented Generation is explained in the PDF document."
    )
    print(f"Cross-modal Query: '{combined_question}'", flush=True)

    t0 = time.time()
    comb_result = query_multimodal(
        question=combined_question,
        pdf_index=pdf_index,
        images=[proc_png],
        top_k=4,
    )
    print(f"⏱️  Combined answer generated in {comb_result.latency_seconds:.2f}s", flush=True)
    print(f"🤖 Answer:\n{comb_result.answer}\n", flush=True)
    print(f"📚 Cited PDF Sources ({len(comb_result.pdf_sources)}):", flush=True)
    for src in comb_result.pdf_sources:
        print(f"   • Page {src['page_number']} (Score: {src.get('relevance_score')}) — {src['source']}", flush=True)

    print(f"🖼️ Cited Image Sources ({len(comb_result.image_sources)}):", flush=True)
    for isrc in comb_result.image_sources:
        print(f"   • Image: {isrc['filename']} ({isrc['dimensions']})", flush=True)

    assert len(comb_result.pdf_sources) > 0, "Expected PDF sources cited in combined query!"
    assert len(comb_result.image_sources) > 0, "Expected Image sources cited in combined query!"
    assert comb_result.query_mode == "combined", "Query mode should be 'combined'!"
    print("✅ Combined Cross-Modal Q&A verified with dual citations!", flush=True)

    print("\n==================================================", flush=True)
    print("🎉 ALL MULTIMODAL TESTS PASSED SUCCESSFULLY!", flush=True)
    print("==================================================", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
