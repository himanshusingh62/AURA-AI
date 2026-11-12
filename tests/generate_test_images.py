"""
Generate sample testing images (PNG diagram & JPG chart) for multimodal test suite.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_sample_architecture_png(output_path: Path) -> None:
    """Create a clean architecture workflow diagram as PNG."""
    width, height = 700, 360
    img = Image.new("RGB", (width, height), color=(15, 23, 42))  # Dark blue background
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((20, 20), "Multimodal RAG Architecture Workflow", fill=(248, 250, 252))

    # Box 1: PDF Document
    draw.rectangle([40, 80, 200, 160], outline=(99, 102, 241), width=2, fill=(30, 41, 59))
    draw.text((55, 105), "1. PDF Ingestion", fill=(226, 232, 240))
    draw.text((55, 125), "(PyMuPDF Pages)", fill=(148, 163, 184))

    # Arrow 1 -> 2
    draw.line([200, 120, 260, 120], fill=(99, 102, 241), width=2)

    # Box 2: Embeddings
    draw.rectangle([260, 80, 420, 160], outline=(16, 185, 129), width=2, fill=(30, 41, 59))
    draw.text((275, 105), "2. Vector Embed", fill=(226, 232, 240))
    draw.text((275, 125), "(Gemini 3072-dim)", fill=(148, 163, 184))

    # Arrow 2 -> 3
    draw.line([420, 120, 480, 120], fill=(16, 185, 129), width=2)

    # Box 3: Vector Store
    draw.rectangle([480, 80, 640, 160], outline=(245, 158, 11), width=2, fill=(30, 41, 59))
    draw.text((495, 105), "3. In-Memory Index", fill=(226, 232, 240))
    draw.text((495, 125), "(LlamaIndex Top-K)", fill=(148, 163, 184))

    # Box 4: Multimodal Vision
    draw.rectangle([40, 200, 300, 300], outline=(236, 72, 153), width=2, fill=(30, 41, 59))
    draw.text((55, 225), "4. Image Understanding (Vision)", fill=(244, 114, 182))
    draw.text((55, 250), "• OCR Text Extraction\n• Diagram & Chart Analysis", fill=(203, 213, 225))

    # Box 5: Unified Gemini Synthesis
    draw.rectangle([380, 200, 640, 300], outline=(139, 92, 246), width=2, fill=(30, 41, 59))
    draw.text((395, 225), "5. Gemini 3.5 Flash Synthesis", fill=(196, 181, 253))
    draw.text((395, 250), "• Grounded Answers\n• Source Citations (Page + Image)", fill=(203, 213, 225))

    # Connect to synthesis
    draw.line([300, 250, 380, 250], fill=(236, 72, 153), width=2)
    draw.line([560, 160, 560, 200], fill=(139, 92, 246), width=2)

    img.save(output_path, format="PNG")
    print(f"Created PNG architecture diagram: {output_path}")


def create_sample_chart_jpg(output_path: Path) -> None:
    """Create a sample benchmark bar chart as JPG."""
    width, height = 600, 320
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((20, 20), "Model Benchmark Accuracy Scores (%)", fill=(15, 23, 42))

    # Coordinate axes
    draw.line([60, 60, 60, 260], fill=(100, 116, 139), width=2)
    draw.line([60, 260, 540, 260], fill=(100, 116, 139), width=2)

    # Bars
    bars = [
        ("Baseline SVM", 68, (148, 163, 184)),
        ("Random Forest", 82, (59, 130, 246)),
        ("ResNet-50", 91, (16, 185, 129)),
        ("Gemini Multimodal", 97, (99, 102, 241)),
    ]

    for idx, (label, score, color) in enumerate(bars):
        x = 90 + idx * 110
        bar_h = score * 1.8
        y = 260 - bar_h
        draw.rectangle([x, y, x + 70, 260], fill=color)
        draw.text((x + 15, y - 20), f"{score}%", fill=(15, 23, 42))
        draw.text((x, 270), label, fill=(51, 65, 85))

    img.save(output_path, format="JPEG", quality=90)
    print(f"Created JPG benchmark chart: {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "tests"
    create_sample_architecture_png(out_dir / "sample_architecture.png")
    create_sample_chart_jpg(out_dir / "sample_chart.jpg")
