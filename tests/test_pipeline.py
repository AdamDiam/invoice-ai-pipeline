"""End-to-end on the 64 labelled samples with an oracle LLM (no API key)."""
import json
from pathlib import Path

import pytest

from fakes import OracleLLM
from invoiceai.config import GROUND_TRUTH_PATH, SAMPLES_DIR, Settings
from invoiceai.eval.run_eval import evaluate
from invoiceai.extract import extract_invoice
from invoiceai.llm import LLMResponse
from invoiceai.pdf import read_pdf
from invoiceai.pipeline import process_file
from invoiceai.store import Store

SETTINGS = Settings.from_env()
TRUTH = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def oracle_eval():
    llm = OracleLLM()
    return evaluate(llm, SETTINGS, verbose=False, on_file=llm.set_file)


def test_every_planted_problem_is_caught(oracle_eval):
    _, s = oracle_eval
    for code, frac in s["issue_recall"].items():
        got, total = map(int, frac.split("/"))
        assert got == total, (code, frac)


def test_no_false_alarms_and_no_unsafe_approvals(oracle_eval):
    _, s = oracle_eval
    assert s["false_alarm_docs"].startswith("0/")
    assert s["unsafe_approvals"] == 0
    assert s["field_accuracy_pct"] == 100.0 and s["line_item_f1_pct"] == 100.0


def test_misread_value_is_never_auto_approved(tmp_path):
    clean = next(f for f, v in TRUTH.items() if not v["expected_issues"])

    def misread(d):  # the model reads 1 digit wrong in the total
        d["total"] = round(d["total"] + 9, 2)
        return d

    llm = OracleLLM(corrupt={clean: misread})
    llm.set_file(clean)
    r = process_file(SAMPLES_DIR / clean, llm, Store(tmp_path / "t.db"), SETTINGS)
    assert r.status == "needs_review"
    assert {"MATH_TOTAL", "NOT_GROUNDED"} <= {i.code for i in r.issues}


def test_processing_is_idempotent(tmp_path):
    f = sorted(SAMPLES_DIR.glob("*.pdf"))[0]
    llm = OracleLLM(); llm.set_file(f.name)
    store = Store(tmp_path / "t.db")
    assert not process_file(f, llm, store, SETTINGS).skipped
    assert process_file(f, llm, store, SETTINGS).skipped
    assert llm.calls == 1


def test_scans_use_vision_and_need_review(tmp_path):
    scanned = next(f for f, v in TRUTH.items() if "SCANNED" in v["expected_issues"])
    pdf = read_pdf(SAMPLES_DIR / scanned)
    assert pdf.is_scanned and pdf.page_images and pdf.text == ""
    llm = OracleLLM(); llm.set_file(scanned)
    r = process_file(SAMPLES_DIR / scanned, llm, Store(tmp_path / "t.db"), SETTINGS)
    assert r.status == "needs_review"


class SchemaFixLLM:
    """First answer misses required fields; second answer is valid."""
    model = "fix"

    def __init__(self, good):
        self.replies = [{"invoice_number": "X"}, good]
        self.calls = []

    def complete_json(self, messages):
        self.calls.append(messages)
        d = self.replies.pop(0)
        return LLMResponse(d, json.dumps(d), 10, 10)


def test_schema_errors_are_fed_back_and_fixed():
    f = sorted(SAMPLES_DIR.glob("*.pdf"))[0]
    llm = SchemaFixLLM(TRUTH[f.name]["invoice"])
    ex = extract_invoice(llm, read_pdf(f))
    assert ex.ok and ex.attempts == 2
    assert "does not match the schema" in llm.calls[1][-1]["content"]


def test_review_records_corrections(tmp_path):
    f = sorted(SAMPLES_DIR.glob("*.pdf"))[0]
    llm = OracleLLM(); llm.set_file(f.name)
    store = Store(tmp_path / "t.db")
    r = process_file(f, llm, store, SETTINGS)
    from invoiceai.schema import Invoice
    inv = Invoice.model_validate_json(store.document(r.doc_id)["final_json"])
    inv.invoice_number = "CORRECTED-1"
    store.review(r.doc_id, "approved", inv, "fixed number")
    assert store.document(r.doc_id)["status"] == "approved"
    rows = store.con.execute("SELECT field, new_value FROM corrections").fetchall()
    assert [tuple(x) for x in rows] == [("invoice_number", "CORRECTED-1")]


def test_changed_bank_account_is_flagged_as_fraud_risk(tmp_path):
    from invoiceai.schema import Invoice

    store = Store(tmp_path / "t.db")
    llm = OracleLLM()
    # three clean invoices from the same supplier build trusted history
    sup = next(v["supplier_key"] for v in TRUTH.values() if "IBAN_CHANGED" in v["expected_issues"])
    clean = [f for f, v in TRUTH.items() if v["supplier_key"] == sup and not v["expected_issues"]][:3]
    for f in clean:
        llm.set_file(f)
        assert process_file(SAMPLES_DIR / f, llm, store, SETTINGS).status == "auto_approved"
    inv = Invoice.model_validate(TRUTH[clean[0]]["invoice"])
    inv.invoice_number = "NEW-1"
    inv.iban = "GB82WEST12345698765432"  # valid checksum, but not this supplier's account
    codes = {i.code for i in store.history_issues(inv)}
    assert "IBAN_CHANGED" in codes
    inv.iban = TRUTH[clean[0]]["invoice"]["iban"]
    assert "IBAN_CHANGED" not in {i.code for i in store.history_issues(inv)}


def test_supplier_master_view(oracle_eval, tmp_path):
    store = Store(tmp_path / "t.db")
    llm = OracleLLM()
    for f in sorted(SAMPLES_DIR.glob("*.pdf"))[:20]:
        llm.set_file(f.name)
        process_file(f, llm, store, SETTINGS)
    rows = store.suppliers()
    assert rows and all(r["invoices"] >= 1 and r["min_total"] <= r["max_total"] for r in rows)
