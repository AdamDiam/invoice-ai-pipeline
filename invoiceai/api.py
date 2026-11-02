"""REST API: lets an ERP, an email robot or any other system use the pipeline without the UI.

    POST /invoices                  upload a PDF -> extracted, validated, decided (auto_approved / needs_review / duplicate)
    GET  /invoices?status=...       list processed documents (e.g. the review queue)
    GET  /invoices/{id}             full result: invoice data, issues, per-field confidence, cost
    POST /invoices/{id}/approve     human decision, optionally with corrected data (logged field by field)
    POST /invoices/{id}/reject
    GET  /suppliers                 supplier master: volume, amount range, known bank accounts
    GET  /health

Run:   uvicorn invoiceai.api:app --reload      then open http://127.0.0.1:8000/docs
Auth:  if INVOICEAI_API_KEY is set in .env, every request needs the header  X-API-Key: <key>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from invoiceai.config import DATA_DIR, Settings
from invoiceai.llm import LLMClient
from invoiceai.pipeline import process_file
from invoiceai.schema import Invoice
from invoiceai.store import Store
from invoiceai.validate import check_rules

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Status = Literal["auto_approved", "needs_review", "duplicate", "approved", "rejected", "failed"]


# --------------------------------------------------------------------------- response models
class IssueOut(BaseModel):
    code: str = Field(examples=["IBAN_CHANGED"])
    severity: Literal["error", "warning"]
    field: str | None = None
    message: str


class DocumentSummary(BaseModel):
    doc_id: int
    file_name: str
    status: Status
    supplier_name: str | None = None
    invoice_number: str | None = None
    issue_date: str | None = None
    currency: str | None = None
    total: float | None = None
    confidence: float | None = None
    issue_codes: list[str] = []


class DocumentDetail(BaseModel):
    doc_id: int
    file_name: str
    status: Status
    already_processed: bool = Field(False, description="True when the same file (same SHA-256) was uploaded before")
    confidence: float | None = None
    field_confidence: dict[str, float] = {}
    issues: list[IssueOut] = []
    invoice: Invoice | None = None
    read_as: str | None = Field(None, description="text (PDF text layer) or vision (scanned page)")
    model: str | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    processed_at: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None
    error: str | None = None


class SupplierOut(BaseModel):
    supplier_name: str
    supplier_vat: str | None = None
    supplier_country: str | None = None
    currency: str
    invoices: int
    total_spend: float
    min_total: float
    max_total: float
    first_invoice: str
    last_invoice: str
    ibans_seen: list[str] = []
    flagged: int


class ApproveIn(BaseModel):
    note: str = ""
    invoice: Invoice | None = Field(None, description="Corrected invoice data. Omit to approve the extracted data as is.")


class RejectIn(BaseModel):
    note: str = ""


# --------------------------------------------------------------------------- helpers
def _detail(store: Store, doc_id: int, already_processed: bool = False) -> DocumentDetail:
    d = store.document(doc_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {doc_id} not found")
    return DocumentDetail(
        doc_id=d["doc_id"], file_name=d["file_name"], status=d["status"], already_processed=already_processed,
        confidence=d["confidence"], field_confidence=json.loads(d["field_confidence"] or "{}"),
        issues=[IssueOut(code=i["code"], severity=i["severity"], field=i["field"] or None, message=i["message"])
                for i in store.issues(doc_id)],
        invoice=Invoice.model_validate_json(d["final_json"]) if d["final_json"] else None,
        read_as=d["mode"], model=d["model"], cost_usd=d["cost_usd"],
        latency_s=round(d["latency_ms"] / 1000, 2) if d["latency_ms"] is not None else None,
        processed_at=d["processed_at"], reviewed_at=d["reviewed_at"], review_note=d["review_note"], error=d["error"],
    )


def _safe_name(name: str | None) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", Path(name or "invoice.pdf").name).strip("._") or "invoice"
    return stem if stem.lower().endswith(".pdf") else stem + ".pdf"


# --------------------------------------------------------------------------- app factory
def create_app(store: Store | None = None, llm: LLMClient | None = None, settings: Settings | None = None,
               inbox: Path | None = None, api_key: str | None = None) -> FastAPI:
    """Build the API. Tests pass their own store / fake LLM; in production everything comes from .env."""
    settings = settings or Settings.from_env()
    store = store or Store(settings.db_path)
    inbox = Path(inbox or DATA_DIR / "inbox")
    api_key = api_key if api_key is not None else os.getenv("INVOICEAI_API_KEY", "")
    state: dict = {"llm": llm}
    write_lock = threading.Lock()  # one SQLite connection is shared by the worker threads; serialise writes

    def get_llm() -> LLMClient:
        if state["llm"] is None:  # created on first upload, so read-only endpoints work without an OpenAI key
            from invoiceai.llm import OpenAIClient
            state["llm"] = OpenAIClient(settings.model)
        return state["llm"]

    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def check_key(key: str | None = Security(header)) -> None:
        if api_key and not (key and secrets.compare_digest(key, api_key)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key header")

    app = FastAPI(
        title="Invoice AI API",
        version="1.0.0",
        description="Accounts-payable automation: LLM extraction that is verified before it is trusted.",
        dependencies=[Depends(check_key)],
    )

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "model": settings.model, "documents": len(store.documents())}

    @app.post("/invoices", response_model=DocumentDetail, status_code=status.HTTP_201_CREATED, tags=["invoices"],
              summary="Upload and process one invoice PDF")
    def upload_invoice(file: UploadFile = File(..., description="Invoice PDF (max 10 MB)")) -> DocumentDetail:
        # a plain `def` endpoint runs in a worker thread, so the ~2 s LLM call does not block other requests
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File is larger than 10 MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are accepted")
        sha = hashlib.sha256(data).hexdigest()
        with write_lock:
            existing = store.document_by_sha(sha)
            if existing is not None:  # idempotent: re-sending the same file returns the earlier result
                return _detail(store, existing["doc_id"], already_processed=True)
            inbox.mkdir(parents=True, exist_ok=True)
            path = inbox / f"{sha[:12]}_{_safe_name(file.filename)}"
            path.write_bytes(data)
            result = process_file(path, get_llm(), store, settings)
        return _detail(store, result.doc_id)

    @app.get("/invoices", response_model=list[DocumentSummary], tags=["invoices"],
             summary="List processed documents (filter by status, e.g. the review queue)")
    def list_invoices(status_: Status | None = Query(None, alias="status"),
                      limit: int = Query(100, ge=1, le=1000)) -> list[DocumentSummary]:
        rows = [r for r in store.documents() if status_ is None or r["status"] == status_]
        rows = sorted(rows, key=lambda r: r["doc_id"], reverse=True)[:limit]
        return [DocumentSummary(doc_id=r["doc_id"], file_name=r["file_name"], status=r["status"],
                                supplier_name=r["supplier_name"], invoice_number=r["invoice_number"],
                                issue_date=r["issue_date"], currency=r["currency"], total=r["total"],
                                confidence=r["confidence"],
                                issue_codes=r["issue_codes"].split(", ") if r["issue_codes"] else [])
                for r in rows]

    @app.get("/invoices/{doc_id}", response_model=DocumentDetail, tags=["invoices"])
    def get_invoice(doc_id: int) -> DocumentDetail:
        return _detail(store, doc_id)

    @app.post("/invoices/{doc_id}/approve", response_model=DocumentDetail, tags=["review"],
              summary="Approve, optionally with corrected data (corrections are logged)")
    def approve_invoice(doc_id: int, body: ApproveIn | None = None) -> DocumentDetail:
        body = body or ApproveIn()
        doc = store.document(doc_id)
        if doc is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {doc_id} not found")
        inv = body.invoice or (Invoice.model_validate_json(doc["final_json"]) if doc["final_json"] else None)
        if inv is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Extraction failed: send the corrected invoice data to approve")
        with write_lock:
            remaining = check_rules(inv) + store.history_issues(inv, exclude_doc=doc_id)
            store.review(doc_id, "approved", inv if body.invoice else None, body.note, remaining)
        return _detail(store, doc_id)

    @app.post("/invoices/{doc_id}/reject", response_model=DocumentDetail, tags=["review"])
    def reject_invoice(doc_id: int, body: RejectIn | None = None) -> DocumentDetail:
        if store.document(doc_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {doc_id} not found")
        with write_lock:
            store.review(doc_id, "rejected", note=(body or RejectIn()).note)
        return _detail(store, doc_id)

    @app.get("/suppliers", response_model=list[SupplierOut], tags=["suppliers"],
             summary="Supplier master built from history: amounts and known bank accounts")
    def suppliers() -> list[SupplierOut]:
        return [SupplierOut(**{**dict(r), "ibans_seen": r["ibans_seen"].split(",") if r["ibans_seen"] else [],
                               "flagged": r["flagged"] or 0})
                for r in store.suppliers()]

    return app


def __getattr__(name: str):
    """`uvicorn invoiceai.api:app` builds the app from .env on first access (so importing this module has no side effects)."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
