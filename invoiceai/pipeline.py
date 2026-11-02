"""End-to-end processing: PDF -> extraction -> validation -> decision -> storage.

Idempotent: a file whose SHA-256 is already in the database is skipped, so
the same inbox folder can be processed again safely.

Usage:
    python -m invoiceai.pipeline samples/invoices          # process a folder
    python -m invoiceai.pipeline path/to/invoice.pdf
    python -m invoiceai.pipeline samples/invoices --reset  # start from an empty database
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from invoiceai.config import Settings
from invoiceai.extract import extract_invoice
from invoiceai.llm import LLMClient
from invoiceai.pdf import file_sha256, read_pdf
from invoiceai.schema import HEADER_FIELDS, get_field
from invoiceai.store import Store
from invoiceai.validate import Issue, check_rules, decide, field_confidence, grounding, grounding_issues


@dataclass
class Result:
    file_name: str
    status: str
    doc_id: int | None = None
    issues: list[Issue] = field(default_factory=list)
    confidence: float | None = None
    cost_usd: float = 0.0
    skipped: bool = False


def process_file(path: Path, llm: LLMClient, store: Store, settings: Settings) -> Result:
    path = Path(path)
    sha = file_sha256(path)
    if store.has_file(sha):
        return Result(path.name, "skipped", skipped=True)

    pdf = read_pdf(path)
    ex = extract_invoice(llm, pdf, max_retries=settings.max_retries)
    cost = settings.cost_usd(ex.prompt_tokens, ex.completion_tokens)
    common = dict(file_name=path.name, file_path=str(path.resolve()), sha256=sha, mode=ex.mode, model=llm.model, attempts=ex.attempts,
                  prompt_tokens=ex.prompt_tokens, completion_tokens=ex.completion_tokens, cost_usd=cost,
                  latency_ms=ex.latency_ms)

    if not ex.ok:
        issue = Issue("EXTRACTION_FAILED", "error", "", "; ".join(ex.errors) or "No valid extraction")
        doc_id = store.add_document(status="failed", confidence=None, field_confidence=None, invoice=None,
                                    issues=[issue], error=issue.message, **common)
        return Result(path.name, "failed", doc_id, [issue], None, cost)

    inv = ex.invoice
    issues = check_rules(inv)
    if pdf.is_scanned:
        issues.append(Issue("SCANNED", "warning", "", "Scanned document (no text layer): read with vision, values could not be verified against text"))
        grounded: dict = {}
    else:
        grounded = grounding(inv, pdf.text)
        issues += grounding_issues(grounded)
    issues += store.history_issues(inv)
    values = {f: get_field(inv.model_dump(), f) for f in HEADER_FIELDS}
    conf = field_confidence(ex.confidence, grounded, pdf.is_scanned, values)
    status, doc_conf = decide(issues, conf, pdf.is_scanned, settings.auto_approve_confidence)
    doc_id = store.add_document(status=status, confidence=doc_conf, field_confidence=conf, invoice=inv,
                                issues=issues, **common)
    return Result(path.name, status, doc_id, issues, doc_conf, cost)


def iter_pdfs(target: Path) -> list[Path]:
    target = Path(target)
    return sorted(target.glob("*.pdf")) if target.is_dir() else [target]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract and validate invoices.")
    p.add_argument("target", type=Path, help="a PDF file or a folder of PDFs")
    p.add_argument("--reset", action="store_true", help="empty the database first")
    args = p.parse_args()

    from invoiceai.llm import OpenAIClient

    settings = Settings.from_env()
    store = Store(settings.db_path)
    if args.reset:
        store.reset()
    llm = OpenAIClient(settings.model)
    files = iter_pdfs(args.target)
    if not files:
        sys.exit(f"No PDFs found at {args.target}")
    counts: dict[str, int] = {}
    total_cost = 0.0
    for f in files:
        r = process_file(f, llm, store, settings)
        counts[r.status] = counts.get(r.status, 0) + 1
        total_cost += r.cost_usd
        codes = ", ".join(sorted({i.code for i in r.issues}))
        print(f"  {r.status:<14} {f.name:<40} {codes}")
    print(f"\n{len(files)} files: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" | cost ${total_cost:.4f}")


if __name__ == "__main__":
    main()
