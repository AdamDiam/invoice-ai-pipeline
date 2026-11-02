"""Read PDFs: text layer when there is one, page images when the document is a scan."""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

MIN_TEXT_CHARS = 40  # below this, a page is treated as a scan


@dataclass
class PdfContent:
    path: Path
    sha256: str
    n_pages: int
    text: str                      # all pages, with "--- page N ---" markers
    is_scanned: bool               # no usable text layer -> needs vision
    page_images: list[str] = field(default_factory=list)  # base64 PNGs (only for scans)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_page_png(path: Path, page_index: int = 0, scale: float = 2.0) -> bytes:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        buf = io.BytesIO()
        doc[page_index].render(scale=scale).to_pil().save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def read_pdf(path: Path, max_pages: int = 5) -> PdfContent:
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "").strip() for p in reader.pages[:max_pages]]
    text = "\n".join(f"--- page {i} ---\n{t}" for i, t in enumerate(pages, 1))
    scanned = sum(len(t) for t in pages) < MIN_TEXT_CHARS * max(1, len(pages))
    images = []
    if scanned:
        images = [base64.b64encode(render_page_png(path, i)).decode() for i in range(min(len(reader.pages), max_pages))]
    return PdfContent(path, file_sha256(path), len(reader.pages), text if not scanned else "", scanned, images)
