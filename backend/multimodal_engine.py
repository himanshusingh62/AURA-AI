"""
Multimodal Query Engine for AURA.AI using the modern official google-genai SDK.
Seamlessly routes queries across PDF-only, Image-only, and Combined Cross-Modal synthesis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from google.genai import errors, types
from llama_index.core import VectorStoreIndex

from backend.gemini_client import get_genai_client, translate_genai_error
from backend.query_engine import query_index
from backend.rate_limit import parse_quota_error
from config.settings import GEMINI_CHAT_MODEL, TOP_K_RESULTS
from utils.usage_tracker import record_api_request

if TYPE_CHECKING:
    from backend.image_processor import ProcessedImage


@dataclass
class MultimodalQueryResult:
    """Unified response containing grounded answer and citations for both PDFs and images."""

    answer: str
    query_mode: str  # 'pdf_only' | 'image_only' | 'combined' | 'none'
    pdf_sources: list[dict] = field(default_factory=list)
    # [{"page_number": int, "source": str, "text_snippet": str, "relevance_score": float}]
    image_sources: list[dict] = field(default_factory=list)
    # [{"filename": str, "thumbnail_base64": str, "format": str, "dimensions": str}]
    latency_seconds: float = 0.0


def _generate_multimodal_content(
    contents: list,
    model_name: str | None = None,
    system_instruction: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
    initial_backoff: float = 1.5,
) -> str:
    """Generate multimodal response using official google-genai SDK with retry on rate limits."""
    from utils.usage_tracker import reserve_api_request
    target_model = model_name or GEMINI_CHAT_MODEL
    reservation_id = reserve_api_request("multimodal", target_model)

    client = get_genai_client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
    )

    backoff = initial_backoff
    last_error_msg: str | None = None

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config,
            )
            record_api_request("multimodal", "success", latency=time.time() - t0, model_name=target_model, reservation_id=reservation_id)
            return (
                response.text.strip()
                if hasattr(response, "text") and response.text
                else "Unable to generate an answer from the provided visual content."
            )
        except errors.ClientError as exc:
            latency = time.time() - t0
            quota_info = parse_quota_error(exc, model_name=target_model)
            last_error_msg = quota_info.friendly_message

            if quota_info.is_429:
                record_api_request(
                    "multimodal",
                    "429_quota_exceeded",
                    latency=latency,
                    is_429=True,
                    error_message=quota_info.friendly_message,
                    model_name=target_model,
                    reservation_id=reservation_id,
                )
                reservation_id = None
                if quota_info.is_daily_limit:
                    raise RuntimeError(quota_info.friendly_message) from None

                delay = quota_info.retry_delay_seconds
                if delay is not None and delay <= 6.0 and attempt < max_retries:
                    time.sleep(delay)
                    continue
                elif delay is None and attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                else:
                    raise RuntimeError(quota_info.friendly_message) from None
            else:
                record_api_request("multimodal", "error", latency=latency, error_message=str(exc), model_name=target_model, reservation_id=reservation_id)
                reservation_id = None
                raise RuntimeError(translate_genai_error(exc, model_name=target_model)) from None
        except Exception as exc:
            latency = time.time() - t0
            quota_info = parse_quota_error(exc, model_name=target_model)
            if quota_info.is_429:
                record_api_request(
                    "multimodal",
                    "429_quota_exceeded",
                    latency=latency,
                    is_429=True,
                    error_message=quota_info.friendly_message,
                    model_name=target_model,
                    reservation_id=reservation_id,
                )
                reservation_id = None
                if quota_info.is_daily_limit or attempt >= max_retries:
                    raise RuntimeError(quota_info.friendly_message) from None
                time.sleep(backoff)
                backoff *= 2.0
                continue

            record_api_request("multimodal", "error", latency=latency, error_message=str(exc), model_name=target_model, reservation_id=reservation_id)
            reservation_id = None
            raise RuntimeError(translate_genai_error(exc, model_name=target_model)) from None

    raise RuntimeError(last_error_msg or f"Multimodal generation failed after {max_retries} attempts.")


def query_multimodal(
    question: str,
    pdf_index: VectorStoreIndex | None = None,
    images: list[ProcessedImage] | None = None,
    top_k: int = TOP_K_RESULTS,
    api_key: str | None = None,
    doc_filter: str | None = None,
    model_name: str | None = None,
) -> MultimodalQueryResult:
    """Execute a multimodal query routing automatically between PDF-only, Image-only, or Combined."""
    start_time = time.time()
    images = images or []

    # Case 1: Neither PDF nor Images loaded
    if pdf_index is None and not images:
        return MultimodalQueryResult(
            answer="Please upload at least one PDF document or image file to begin research.",
            query_mode="none",
            latency_seconds=round(time.time() - start_time, 2),
        )

    # Case 2: Only PDF is active (no images)
    if pdf_index is not None and not images:
        res = query_index(
            pdf_index,
            question,
            top_k=top_k,
            api_key=api_key,
            doc_filter=doc_filter,
            model_name=model_name,
        )
        return MultimodalQueryResult(
            answer=res.answer,
            query_mode="pdf_only",
            pdf_sources=res.sources,
            image_sources=[],
            latency_seconds=round(time.time() - start_time, 2),
        )

    # Build image source citations
    image_sources = [
        {
            "filename": img.filename,
            "thumbnail_base64": img.thumbnail_base64,
            "format": img.format,
            "dimensions": f"{img.width}x{img.height}",
        }
        for img in images
    ]

    # Case 3: Only Images active (no PDF)
    if pdf_index is None and images:
        system_prompt = (
            "You are an expert Visual AI Research Assistant. "
            "Analyze the provided image(s) with high precision. "
            "Extract readable text, interpret diagrams, identify chart trends, "
            "and describe objects or structures accurately. "
            "Cite visual features explicitly and do not hallucinate details."
        )

        contents = [img.pil_image for img in images]
        contents.append(f"User Question: {question}")

        answer_text = _generate_multimodal_content(
            contents=contents,
            model_name=model_name,
            system_instruction=system_prompt,
            api_key=api_key,
        )

        return MultimodalQueryResult(
            answer=answer_text,
            query_mode="image_only",
            pdf_sources=[],
            image_sources=image_sources,
            latency_seconds=round(time.time() - start_time, 2),
        )

    # Case 4: Combined PDF + Image(s) Multimodal Reasoning
    retrieval_k = max(top_k * 5, 50) if doc_filter else top_k
    retriever = pdf_index.as_retriever(similarity_top_k=retrieval_k)
    source_nodes = retriever.retrieve(question)

    pdf_sources: list[dict] = []
    seen_keys: set[tuple[str, int]] = set()
    context_blocks: list[str] = []

    for node_with_score in source_nodes:
        node = getattr(node_with_score, "node", node_with_score)
        score = getattr(node_with_score, "score", None)
        meta = getattr(node, "metadata", {}) or {}
        page = meta.get("page_number", 0)
        source_doc = meta.get("source", "uploaded.pdf")

        if doc_filter and source_doc != doc_filter:
            continue

        dedup_key = (source_doc, page)
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            pdf_sources.append(
                {
                    "page_number": page,
                    "source": source_doc,
                    "text_snippet": node.text[:300] + ("…" if len(node.text) > 300 else ""),
                    "relevance_score": round(score, 4) if score is not None else None,
                }
            )

        context_blocks.append(f"--- [From Document: {source_doc}, Page: {page}] ---\n{node.text}")
        if len(context_blocks) >= top_k:
            break

    combined_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an advanced Multimodal Research Assistant. "
        "You have access to verified textual evidence from uploaded PDF document(s) "
        "AND one or more uploaded visual images (diagrams, charts, screenshots, or documents).\n\n"
        "INSTRUCTIONS:\n"
        "1. Synthesize insights from BOTH the visual image(s) and the PDF textual context.\n"
        "2. When referencing facts from a document, cite the specific document name and page number(s) (e.g. [Document: report.pdf, Page X]).\n"
        "3. When referencing visual information, diagrams, or charts, explicitly refer to the image by its filename.\n"
        "4. If comparing the image and the document, clearly explain alignments, discrepancies, or complementary details.\n"
        "5. Be direct, rigorous, and never fabricate ungrounded claims."
    )

    multimodal_prompt = (
        f"PDF DOCUMENT CONTEXT:\n{combined_context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Please provide a complete, well-structured answer integrating evidence from the image(s) and the PDF context."
    )

    contents = [img.pil_image for img in images]
    contents.append(multimodal_prompt)

    answer_text = _generate_multimodal_content(
        contents=contents,
        model_name=model_name,
        system_instruction=system_prompt,
        api_key=api_key,
    )

    return MultimodalQueryResult(
        answer=answer_text,
        query_mode="combined",
        pdf_sources=sorted(pdf_sources, key=lambda s: (s["source"], s["page_number"])),
        image_sources=image_sources,
        latency_seconds=round(time.time() - start_time, 2),
    )
