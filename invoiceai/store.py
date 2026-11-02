"""SQLite storage, plus the checks that need history: duplicates and amount outliers."""
from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import date, datetime
from pathlib import Path

from invoiceai.identifiers import iban_is_valid
from invoiceai.normalize import norm_id, norm_text
from invoiceai.schema import Invoice
from invoiceai.validate import Issue

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id            INTEGER PRIMARY KEY,
    file_name         TEXT NOT NULL,
    file_path         TEXT,
    sha256            TEXT NOT NULL UNIQUE,
    processed_at      TEXT NOT NULL,
    status            TEXT NOT NULL,          -- auto_approved | needs_review | duplicate | approved | rejected | failed
    mode              TEXT,                   -- text | vision
    model             TEXT,
    attempts          INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    latency_ms        REAL,
    confidence        REAL,
    field_confidence  TEXT,                   -- JSON
    extracted_json    TEXT,                   -- what the model returned (validated)
    final_json        TEXT,                   -- after human review (else = extracted)
    error             TEXT,
    reviewed_at       TEXT,
    review_note       TEXT
);
CREATE TABLE IF NOT EXISTS issues (
    doc_id   INTEGER NOT NULL REFERENCES documents (doc_id),
    code     TEXT NOT NULL,
    severity TEXT NOT NULL,
    field    TEXT,
    message  TEXT
);
CREATE TABLE IF NOT EXISTS invoices (
    doc_id          INTEGER PRIMARY KEY REFERENCES documents (doc_id),
    supplier_name   TEXT, supplier_vat TEXT, supplier_country TEXT,
    invoice_number  TEXT, issue_date TEXT, due_date TEXT, currency TEXT,
    subtotal REAL, vat_amount REAL, total REAL, iban TEXT, language TEXT
);
CREATE TABLE IF NOT EXISTS line_items (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    line_no INTEGER, description TEXT, quantity REAL, unit_price REAL, vat_rate REAL, net_amount REAL
);
CREATE TABLE IF NOT EXISTS corrections (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    field TEXT, old_value TEXT, new_value TEXT, corrected_at TEXT
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    # ------------------------------------------------------------------ reads
    def has_file(self, sha256: str) -> bool:
        return self.con.execute("SELECT 1 FROM documents WHERE sha256 = ?", (sha256,)).fetchone() is not None

    def documents(self) -> list[sqlite3.Row]:
        return self.con.execute(
            """SELECT d.*, i.supplier_name, i.invoice_number, i.issue_date, i.currency, i.total,
                      (SELECT GROUP_CONCAT(code, ', ') FROM issues WHERE issues.doc_id = d.doc_id) AS issue_codes
               FROM documents d LEFT JOIN invoices i USING (doc_id) ORDER BY d.doc_id"""
        ).fetchall()

    def document(self, doc_id: int) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()

    def document_by_sha(self, sha256: str) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()

    def issues(self, doc_id: int) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM issues WHERE doc_id = ? ORDER BY severity, code", (doc_id,)).fetchall()

    # ------------------------------------------------------------------ history checks
    def history_issues(self, inv: Invoice, exclude_doc: int | None = None) -> list[Issue]:
        """Duplicate, near-duplicate and amount-outlier checks against previously stored invoices."""
        issues: list[Issue] = []
        rows = self.con.execute(
            """SELECT i.*, d.file_name, d.status FROM invoices i JOIN documents d USING (doc_id)
               WHERE d.status NOT IN ('rejected', 'failed') AND (? IS NULL OR i.doc_id <> ?)""",
            (exclude_doc, exclude_doc),
        ).fetchall()
        sup_vat, sup_name = norm_id(inv.supplier.vat_id), norm_text(inv.supplier.name)

        def same_supplier(r) -> bool:
            if sup_vat and r["supplier_vat"]:
                a, b = sup_vat, norm_id(r["supplier_vat"])
                return a == b or a[2:] == b or a == b[2:]
            return norm_text(r["supplier_name"]) == sup_name

        history = [r for r in rows if same_supplier(r)]
        num = norm_id(inv.invoice_number)
        for r in history:
            if norm_id(r["invoice_number"]) == num:
                issues.append(Issue("DUPLICATE", "error", "invoice_number",
                                    f"Invoice {inv.invoice_number} from this supplier was already received ({r['file_name']})"))
                return issues
        for r in history:
            days = abs((date.fromisoformat(r["issue_date"]) - inv.issue_date).days)
            if abs(r["total"] - inv.total) < 0.01 and days <= 14:
                issues.append(Issue("POSSIBLE_DUPLICATE", "warning", "total",
                                    f"Same supplier and same amount {inv.total:,.2f} {inv.currency} {days} day(s) apart "
                                    f"({r['file_name']}, no. {r['invoice_number']})"))
                break
        # Bank-account change: the classic invoice-fraud pattern (a valid IBAN that isn't the supplier's).
        # Only trusted history counts, i.e. invoices that were auto-approved or approved by a reviewer.
        # An IBAN that fails its checksum is already reported as IBAN_INVALID, so it is not double-counted here.
        trusted_ibans = {norm_id(r["iban"]) for r in history if r["iban"] and r["status"] in ("auto_approved", "approved")}
        if inv.iban and iban_is_valid(inv.iban) and trusted_ibans and norm_id(inv.iban) not in trusted_ibans:
            known = ", ".join(sorted(trusted_ibans))
            issues.append(Issue("IBAN_CHANGED", "error", "iban",
                                f"Bank account {inv.iban} differs from this supplier's known IBAN ({known}). "
                                "Verify by phone before paying: a changed IBAN is the most common invoice-fraud pattern."))
        totals = [r["total"] for r in history if r["currency"] == inv.currency]
        if len(totals) >= 3:
            med = statistics.median(totals)
            if med > 0 and inv.total > 4 * med:
                issues.append(Issue("AMOUNT_OUTLIER", "warning", "total",
                                    f"Total {inv.total:,.2f} is {inv.total / med:.1f}× this supplier's median ({med:,.2f})"))
        return issues

    def suppliers(self) -> list[sqlite3.Row]:
        """Supplier master data built from history: volume, typical amounts and known bank accounts."""
        return self.con.execute(
            """SELECT i.supplier_name, i.supplier_vat, i.supplier_country, i.currency,
                      COUNT(*) AS invoices, ROUND(SUM(i.total), 2) AS total_spend,
                      ROUND(MIN(i.total), 2) AS min_total, ROUND(MAX(i.total), 2) AS max_total,
                      MIN(i.issue_date) AS first_invoice, MAX(i.issue_date) AS last_invoice,
                      GROUP_CONCAT(DISTINCT i.iban) AS ibans_seen,
                      SUM(d.status IN ('needs_review', 'duplicate')) AS flagged
               FROM invoices i JOIN documents d USING (doc_id)
               WHERE d.status NOT IN ('rejected', 'failed')
               GROUP BY COALESCE(i.supplier_vat, i.supplier_name), i.currency
               ORDER BY total_spend DESC"""
        ).fetchall()

    # ------------------------------------------------------------------ writes
    def add_document(self, *, file_name: str, file_path: str | None = None, sha256: str, status: str, mode: str | None, model: str | None,
                     attempts: int, prompt_tokens: int, completion_tokens: int, cost_usd: float, latency_ms: float,
                     confidence: float | None, field_confidence: dict | None, invoice: Invoice | None,
                     issues: list[Issue], error: str | None = None) -> int:
        inv_json = invoice.model_dump_json() if invoice else None
        cur = self.con.execute(
            """INSERT INTO documents (file_name, file_path, sha256, processed_at, status, mode, model, attempts, prompt_tokens,
               completion_tokens, cost_usd, latency_ms, confidence, field_confidence, extracted_json, final_json, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (file_name, file_path, sha256, datetime.now().isoformat(timespec="seconds"), status, mode, model, attempts,
             prompt_tokens, completion_tokens, cost_usd, latency_ms, confidence,
             json.dumps(field_confidence) if field_confidence else None, inv_json, inv_json, error),
        )
        doc_id = int(cur.lastrowid)
        self._write_issues(doc_id, issues)
        if invoice:
            self._write_invoice(doc_id, invoice)
        self.con.commit()
        return doc_id

    def _write_issues(self, doc_id: int, issues: list[Issue]) -> None:
        self.con.execute("DELETE FROM issues WHERE doc_id = ?", (doc_id,))
        self.con.executemany("INSERT INTO issues VALUES (?,?,?,?,?)",
                             [(doc_id, i.code, i.severity, i.field, i.message) for i in issues])

    def _write_invoice(self, doc_id: int, inv: Invoice) -> None:
        self.con.execute("DELETE FROM invoices WHERE doc_id = ?", (doc_id,))
        self.con.execute("DELETE FROM line_items WHERE doc_id = ?", (doc_id,))
        self.con.execute(
            "INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, inv.supplier.name, inv.supplier.vat_id, inv.supplier.country, inv.invoice_number,
             inv.issue_date.isoformat(), inv.due_date.isoformat() if inv.due_date else None, inv.currency,
             inv.subtotal, inv.vat_amount, inv.total, inv.iban, inv.language),
        )
        self.con.executemany(
            "INSERT INTO line_items VALUES (?,?,?,?,?,?,?)",
            [(doc_id, n, li.description, li.quantity, li.unit_price, li.vat_rate, li.net_amount)
             for n, li in enumerate(inv.line_items, 1)],
        )

    def review(self, doc_id: int, decision: str, invoice: Invoice | None = None, note: str = "",
               remaining_issues: list[Issue] | None = None) -> None:
        """Record a human decision ('approved' or 'rejected'), with optional corrected data."""
        row = self.document(doc_id)
        if row is None:
            raise KeyError(doc_id)
        now = datetime.now().isoformat(timespec="seconds")
        if invoice is not None:
            old = json.loads(row["final_json"] or "{}")
            new = json.loads(invoice.model_dump_json())
            for path, a, b in _diff(old, new):
                self.con.execute("INSERT INTO corrections VALUES (?,?,?,?,?)", (doc_id, path, str(a), str(b), now))
            self.con.execute("UPDATE documents SET final_json = ? WHERE doc_id = ?", (invoice.model_dump_json(), doc_id))
            self._write_invoice(doc_id, invoice)
        if remaining_issues is not None:
            self._write_issues(doc_id, remaining_issues)
        self.con.execute("UPDATE documents SET status = ?, reviewed_at = ?, review_note = ? WHERE doc_id = ?",
                         (decision, now, note, doc_id))
        self.con.commit()

    def reset(self) -> None:
        for t in ("corrections", "line_items", "invoices", "issues", "documents"):
            self.con.execute(f"DELETE FROM {t}")
        self.con.commit()


def _diff(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            yield from _diff(old.get(k), new.get(k), f"{prefix}.{k}" if prefix else k)
    elif isinstance(old, list) and isinstance(new, list):
        for i in range(max(len(old), len(new))):
            yield from _diff(old[i] if i < len(old) else None, new[i] if i < len(new) else None, f"{prefix}[{i}]")
    elif old != new:
        yield prefix, old, new
