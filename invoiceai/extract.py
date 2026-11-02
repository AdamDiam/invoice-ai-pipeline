"""LLM extraction: PDF content -> validated `Invoice` + per-field confidence.

Text-layer PDFs are sent as text (cheap, and lets us verify every value
against the source afterwards). Scans are sent as page images (vision).
If the JSON does not validate against the schema, the validation error is
sent back and the model corrects itself.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from invoiceai.config import OUR_COMPANY
from invoiceai.llm import LLMClient
from invoiceai.pdf import PdfContent
from invoiceai.schema import HEADER_FIELDS, Invoice

SYSTEM_PROMPT = f"""You extract structured data from supplier invoices for an accounts-payable team.
The invoices are addressed to our company, {OUR_COMPANY['name']} (VAT {OUR_COMPANY['vat_id']}): it is the CUSTOMER.
The issuer of the invoice is the SUPPLIER.

Rules:
- Copy values exactly as printed. Never compute, correct or invent values: if the printed total looks wrong,
  still return the printed total (a separate validation step checks the arithmetic).
- Numbers: return plain JSON numbers with a dot as decimal separator (the invoice may print 1.234,56 or 1,234.56).
- Dates: return ISO format YYYY-MM-DD. Interpret dd/mm/yyyy for European invoices; "March 30, 2026" style is month-first.
- VAT ids: return as printed, keep the country prefix if printed (e.g. DE123456789); a Greek ΑΦΜ printed without
  prefix should be returned as its 9 digits.
- IBAN: without spaces. currency: ISO 4217 code (€ -> EUR, £ -> GBP, $ -> USD).
- vat_rate per line in percent (24 for 24%, 0 for reverse charge / no VAT).
- Missing optional values: null. Greek text stays in Greek.
- Also return "confidence": an object mapping each of these fields to a number 0..1 reflecting how sure you are
  the value is printed and read correctly: {", ".join(HEADER_FIELDS)}.
  For an optional field that is genuinely not printed (you returned null), report 1.0 if you are sure it is absent.

Return ONE JSON object with the invoice fields and "confidence". JSON schema of the invoice fields:
{json.dumps(Invoice.model_json_schema(), separators=(",", ":"))}"""


@dataclass
class Extraction:
    invoice: Invoice | None
    confidence: dict = field(default_factory=dict)
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    mode: str = "text"  # text | vision

    @property
    def ok(self) -> bool:
        return self.invoice is not None


def build_user_message(pdf: PdfContent) -> dict:
    if not pdf.is_scanned:
        return {"role": "user", "content": f"Invoice text (extracted from PDF):\n\n{pdf.text}"}
    parts: list[dict] = [{"type": "text", "text": "This invoice is a scanned image. Read it carefully."}]
    for b64 in pdf.page_images:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
    return {"role": "user", "content": parts}


def extract_invoice(llm: LLMClient, pdf: PdfContent, max_retries: int = 2) -> Extraction:
    t0 = time.perf_counter()
    ex = Extraction(invoice=None, mode="vision" if pdf.is_scanned else "text")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, build_user_message(pdf)]
    for _ in range(max_retries + 1):
        ex.attempts += 1
        try:
            resp = llm.complete_json(messages)
        except Exception as exc:  # network / auth / rate limit
            ex.errors.append(f"LLM call failed: {exc}")
            break
        ex.prompt_tokens += resp.prompt_tokens
        ex.completion_tokens += resp.completion_tokens
        data = dict(resp.data)
        conf = data.pop("confidence", {}) or {}
        try:
            ex.invoice = Invoice.model_validate(data)
            ex.confidence = {k: float(v) for k, v in conf.items() if isinstance(v, (int, float))}
            break
        except ValidationError as err:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:10])
            ex.errors.append(msg)
            messages += [
                {"role": "assistant", "content": resp.raw},
                {"role": "user", "content": f"The JSON does not match the schema: {msg}. Return the corrected full JSON object."},
            ]
    ex.latency_ms = (time.perf_counter() - t0) * 1000
    return ex
