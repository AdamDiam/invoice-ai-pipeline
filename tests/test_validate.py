import copy
import json
from datetime import date

from invoiceai.config import GROUND_TRUTH_PATH
from invoiceai.schema import Invoice
from invoiceai.validate import check_rules, decide, field_confidence, grounding, grounding_issues

TRUTH = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
TODAY = date(2026, 11, 2)


def clean_invoice():
    for v in TRUTH.values():
        if not v["expected_issues"] and v["invoice"]["vat_amount"] > 0 and len(v["invoice"]["line_items"]) > 1:
            return copy.deepcopy(v["invoice"])
    raise AssertionError


def codes(inv_dict):
    return {i.code for i in check_rules(Invoice.model_validate(inv_dict), today=TODAY)}


def test_clean_invoice_has_no_issues():
    assert codes(clean_invoice()) == set()


def test_arithmetic_errors_are_caught():
    d = clean_invoice(); d["total"] += 5
    assert "MATH_TOTAL" in codes(d)
    d = clean_invoice(); d["line_items"][0]["net_amount"] += 3
    assert {"MATH_LINE", "MATH_SUBTOTAL"} <= codes(d)
    d = clean_invoice(); d["vat_amount"] += 2; d["total"] += 2
    assert "MATH_VAT" in codes(d)


def test_identifier_and_date_rules():
    d = clean_invoice(); d["iban"] = d["iban"][:-1] + str((int(d["iban"][-1]) + 1) % 10)
    assert "IBAN_INVALID" in codes(d)
    d = clean_invoice(); d["due_date"] = "2020-01-01"
    assert "DATE_ORDER" in codes(d)
    d = clean_invoice(); d["customer"]["vat_id"] = "EL094019245"
    assert "WRONG_CUSTOMER" in codes(d)


def test_grounding_flags_values_not_in_source():
    d = clean_invoice()
    inv = Invoice.model_validate(d)
    # build a source text from the true values in a different locale format
    from invoiceai.normalize import fmt_amount
    src = " ".join([d["invoice_number"], d["supplier"]["name"], d["supplier"]["vat_id"][2:], d["customer"]["name"],
                    "801234560", d["iban"], fmt_amount(d["subtotal"], "eu"), fmt_amount(d["vat_amount"], "eu"),
                    fmt_amount(d["total"], "eu"), date.fromisoformat(d["issue_date"]).strftime("%d/%m/%Y"),
                    date.fromisoformat(d["due_date"]).strftime("%d/%m/%Y") if d["due_date"] else ""]
                   + [fmt_amount(li["net_amount"], "eu") for li in d["line_items"]])
    g = grounding(inv, src)
    assert all(v is not False for v in g.values()), g
    hallucinated = Invoice.model_validate({**d, "total": d["total"] + 0.9})
    assert grounding(hallucinated, src)["total"] is False


def test_decision_policy():
    conf = {"total": 0.95, "iban": 0.99}
    assert decide([], conf, scanned=False, threshold=0.85) == ("auto_approved", 0.95)
    assert decide([], conf, scanned=True, threshold=0.85)[0] == "needs_review"
    assert decide([], {"total": 0.5}, scanned=False, threshold=0.85)[0] == "needs_review"
    g = {"total": False, "iban": True}
    c = field_confidence({"total": 0.99}, g, scanned=False)
    assert c["total"] <= 0.3
    issues = grounding_issues(g)
    assert [i.code for i in issues] == ["NOT_GROUNDED"]


def test_absent_optional_fields_do_not_lower_confidence():
    llm_conf = {"iban": 0.1, "total": 0.95, "due_date": 0.2}
    values = {"iban": None, "total": 120.0, "due_date": None}
    c = field_confidence(llm_conf, {"total": True}, scanned=False, values=values)
    assert "iban" not in c and "due_date" not in c
    assert decide([], c, scanned=False, threshold=0.85)[0] == "auto_approved"
    # but a low confidence on a value that IS present still blocks
    values["iban"] = "GB82WEST12345698765432"
    c = field_confidence(llm_conf, {"total": True, "iban": True}, scanned=False, values=values)
    assert decide([], c, scanned=False, threshold=0.85)[0] == "needs_review"
