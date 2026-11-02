"""Business-rule validation, grounding checks and confidence scoring.

An LLM can misread a number or invent a value, and suppliers make mistakes too.
Every extraction therefore goes through three independent checks before it
can be auto-approved:

1. Business rules: line math, subtotal / VAT / total arithmetic, VAT-id
   format and Greek AFM checksum, IBAN mod-97, date logic, addressed to us.
2. Grounding: each extracted value must literally appear in the PDF text
   (in any locale format). A value that is not in the source is treated as a
   possible hallucination.
3. Confidence: the model's self-reported confidence, capped by grounding.

Duplicate and outlier checks need history and live in `store.py`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from invoiceai.config import OUR_COMPANY
from invoiceai.identifiers import iban_is_valid, vat_id_problem
from invoiceai.normalize import amount_variants, date_variants, norm_id, norm_text
from invoiceai.schema import HEADER_FIELDS, Invoice, get_field

TOL = 0.02  # rounding tolerance in currency units


@dataclass
class Issue:
    code: str
    severity: str  # "error" | "warning"
    field: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# 1. business rules
# ---------------------------------------------------------------------------
def check_rules(inv: Invoice, today: date | None = None) -> list[Issue]:
    issues: list[Issue] = []
    today = today or date.today()

    for i, li in enumerate(inv.line_items):
        if not _close(li.quantity * li.unit_price, li.net_amount, max(TOL, abs(li.net_amount) * 0.001)):
            issues.append(Issue("MATH_LINE", "error", f"line_items[{i}]",
                                f"Line {i + 1}: {li.quantity:g} × {li.unit_price:.2f} ≠ {li.net_amount:.2f}"))
    lines_sum = round(sum(li.net_amount for li in inv.line_items), 2)
    if inv.line_items and not _close(lines_sum, inv.subtotal):
        issues.append(Issue("MATH_SUBTOTAL", "error", "subtotal", f"Sum of lines {lines_sum:.2f} ≠ subtotal {inv.subtotal:.2f}"))
    expected_vat = round(sum(li.net_amount * li.vat_rate / 100 for li in inv.line_items), 2)
    if inv.line_items and not _close(expected_vat, inv.vat_amount, max(TOL, 0.01 * len(inv.line_items))):
        issues.append(Issue("MATH_VAT", "error", "vat_amount", f"VAT per line rates {expected_vat:.2f} ≠ printed VAT {inv.vat_amount:.2f}"))
    if not _close(inv.subtotal + inv.vat_amount, inv.total):
        issues.append(Issue("MATH_TOTAL", "error", "total",
                            f"Subtotal {inv.subtotal:.2f} + VAT {inv.vat_amount:.2f} = {inv.subtotal + inv.vat_amount:.2f}, "
                            f"but the invoice says {inv.total:.2f}"))

    for role in ("supplier", "customer"):
        party = getattr(inv, role)
        problem = vat_id_problem(party.vat_id, party.country)
        if problem:
            issues.append(Issue("VAT_ID_INVALID", "error", f"{role}.vat_id", f"{role.title()} VAT id {party.vat_id}: {problem}"))
    if not inv.supplier.vat_id:
        issues.append(Issue("MISSING_FIELD", "warning", "supplier.vat_id", "Supplier VAT / tax id is missing"))

    if inv.iban and not iban_is_valid(inv.iban):
        issues.append(Issue("IBAN_INVALID", "error", "iban", f"IBAN {inv.iban} fails the mod-97 checksum: possible typo or fraud"))

    if inv.due_date and inv.due_date < inv.issue_date:
        issues.append(Issue("DATE_ORDER", "error", "due_date", "Due date is before the issue date"))
    if inv.issue_date > today:
        issues.append(Issue("DATE_FUTURE", "warning", "issue_date", f"Issue date {inv.issue_date} is in the future"))

    ours = norm_id(OUR_COMPANY["vat_id"])
    cust = norm_id(inv.customer.vat_id)
    if cust and cust not in (ours, ours[2:]):
        issues.append(Issue("WRONG_CUSTOMER", "error", "customer.vat_id", f"Invoice is addressed to VAT {inv.customer.vat_id}, not to us"))
    if inv.document_type != "invoice":
        issues.append(Issue("NOT_AN_INVOICE", "warning", "document_type", f"Document type is '{inv.document_type}'"))
    return issues


# ---------------------------------------------------------------------------
# 2. grounding against the source text
# ---------------------------------------------------------------------------
def is_grounded(value, source_norm: str, source_compact: str) -> bool | None:
    """True if the value appears in the source text; None if not checkable."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return any(v in source_norm or v.replace(" ", "") in source_compact for v in amount_variants(float(value)))
    if isinstance(value, date):
        return any(norm_text(v) in source_norm for v in date_variants(value))
    s = str(value)
    if len(s) == 3 and s.isupper():  # currency code: may be printed as a symbol
        return None
    if norm_text(s) in source_norm:
        return True
    ident = norm_id(s)
    if len(ident) >= 5:  # ids may be printed with spaces / dashes / without country prefix
        return ident in source_compact or ident[2:] in source_compact
    return False


def grounding(inv: Invoice, source_text: str) -> dict[str, bool | None]:
    src = norm_text(source_text)
    compact = norm_id(source_text)
    values = {f: get_field(inv.model_dump(), f) for f in HEADER_FIELDS}
    for f in ("issue_date", "due_date"):
        values[f] = getattr(inv, f)
    out = {f: is_grounded(v, src, compact) for f, v in values.items()}
    for i, li in enumerate(inv.line_items):
        out[f"line_items[{i}].net_amount"] = is_grounded(li.net_amount, src, compact)
    return out


# ---------------------------------------------------------------------------
# 3. confidence and decision
# ---------------------------------------------------------------------------
def field_confidence(llm_conf: dict, grounded: dict[str, bool | None], scanned: bool,
                     values: dict | None = None) -> dict[str, float]:
    """Per-field confidence. Optional fields that are simply absent (e.g. no IBAN on a US invoice)
    are not scored: the model's uncertainty about a value that doesn't exist must not block a document."""
    conf = {}
    for f in HEADER_FIELDS:
        if values is not None and values.get(f) in (None, ""):
            continue
        c = float(llm_conf.get(f, 0.9))
        g = grounded.get(f)
        if g is False:
            c = min(c, 0.3)  # value not found in the document: possible hallucination
        if scanned:
            c = min(c, 0.8)  # nothing to verify against
        conf[f] = round(c, 3)
    return conf


def grounding_issues(grounded: dict[str, bool | None]) -> list[Issue]:
    return [
        Issue("NOT_GROUNDED", "warning", f, f"Extracted {f} was not found in the document text")
        for f, g in grounded.items()
        if g is False
    ]


def decide(issues: list[Issue], confidence: dict[str, float], scanned: bool, threshold: float) -> tuple[str, float]:
    """Return (status, document_confidence). Only clean, confident, text-verified documents are auto-approved."""
    doc_conf = round(min(confidence.values()) if confidence else 0.0, 3)
    if any(i.code == "DUPLICATE" for i in issues):
        return "duplicate", doc_conf
    if issues or scanned or doc_conf < threshold:
        return "needs_review", doc_conf
    return "auto_approved", doc_conf
