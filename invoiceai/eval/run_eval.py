"""Evaluate the pipeline on the labelled sample set.

Metrics
  * Field accuracy        : per header field, extracted vs ground truth (normalised compare)
  * Line-item F1          : lines matched on amount + description similarity
  * Document accuracy     : every header field and every line correct
  * Issue detection       : recall per planted problem type + false alarms on clean documents
  * Straight-through rate : share of documents auto-approved without a human
  * Unsafe approvals      : auto-approved documents that were wrong or had a planted problem (target: 0)
  * Cost / latency        : USD and seconds per document

Usage:
    python -m invoiceai.eval.run_eval              # uses OPENAI_MODEL from .env
    python -m invoiceai.eval.run_eval --limit 10
"""
from __future__ import annotations

import argparse
import difflib
import json
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from invoiceai.config import EVAL_RESULTS_DIR, GROUND_TRUTH_PATH, SAMPLES_DIR, Settings
from invoiceai.llm import LLMClient
from invoiceai.normalize import norm_id, norm_text
from invoiceai.pipeline import process_file
from invoiceai.schema import HEADER_FIELDS, get_field
from invoiceai.store import Store

PLANTED = ["MATH_TOTAL", "VAT_ID_INVALID", "IBAN_INVALID", "IBAN_CHANGED", "DUPLICATE", "POSSIBLE_DUPLICATE", "AMOUNT_OUTLIER", "SCANNED"]


def _vat_eq(a, b) -> bool:
    x, y = norm_id(a), norm_id(b)
    return x == y or (len(x) > 2 and x[2:] == y) or (len(y) > 2 and y[2:] == x)


def field_equal(name: str, pred, truth) -> bool:
    if truth is None or truth == "":
        return pred is None or pred == ""
    if pred is None:
        return False
    if name in ("subtotal", "vat_amount", "total"):
        return abs(float(pred) - float(truth)) <= 0.011
    if name.endswith("vat_id"):
        return _vat_eq(pred, truth)
    if name in ("invoice_number", "iban"):
        return norm_id(pred) == norm_id(truth)
    if name.endswith("name"):
        return difflib.SequenceMatcher(None, norm_text(pred), norm_text(truth)).ratio() >= 0.92
    return str(pred) == str(truth)


def line_scores(pred_lines: list, true_lines: list) -> tuple[int, int, int]:
    """Return (matched, n_pred, n_true)."""
    unused = list(range(len(pred_lines)))
    matched = 0
    for t in true_lines:
        for j in unused:
            p = pred_lines[j]
            same_amount = abs(float(p["net_amount"]) - float(t["net_amount"])) <= 0.011
            same_desc = difflib.SequenceMatcher(None, norm_text(p["description"]), norm_text(t["description"])).ratio() >= 0.8
            if same_amount and same_desc:
                matched += 1
                unused.remove(j)
                break
    return matched, len(pred_lines), len(true_lines)


def evaluate(llm: LLMClient, settings: Settings, samples: Path = SAMPLES_DIR, truth_path: Path = GROUND_TRUTH_PATH,
             limit: int | None = None, verbose: bool = True, on_file=None) -> tuple[list[dict], dict]:
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    files = sorted(samples.glob("*.pdf"))[: limit or None]
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.db")
        rows = []
        for f in files:
            if on_file:
                on_file(f.name)
            r = process_file(f, llm, store, settings)
            doc = store.document(r.doc_id)
            gt = truth[f.name]
            pred = json.loads(doc["extracted_json"]) if doc["extracted_json"] else {}
            fields = {name: field_equal(name, get_field(pred, name), get_field(gt["invoice"], name)) for name in HEADER_FIELDS}
            m, n_pred, n_true = line_scores(pred.get("line_items", []), gt["invoice"]["line_items"])
            detected = sorted({i.code for i in r.issues})
            expected = gt["expected_issues"]
            row = {
                "file": f.name, "status": r.status, "mode": doc["mode"], "confidence": r.confidence,
                "fields": fields, "lines_matched": m, "lines_pred": n_pred, "lines_true": n_true,
                "doc_correct": all(fields.values()) and m == n_true == n_pred,
                "expected_issues": expected, "detected_issues": detected,
                "cost_usd": doc["cost_usd"] or 0.0, "latency_s": round((doc["latency_ms"] or 0) / 1000, 2),
                "tokens": (doc["prompt_tokens"] or 0) + (doc["completion_tokens"] or 0),
                "low_confidence_fields": {k: v for k, v in json.loads(doc["field_confidence"] or "{}").items()
                                          if v < settings.auto_approve_confidence},
            }
            row["unsafe_approval"] = r.status == "auto_approved" and (bool(expected) or not row["doc_correct"])
            rows.append(row)
            if verbose:
                wrong = [k for k, ok in fields.items() if not ok]
                flag = "UNSAFE" if row["unsafe_approval"] else ""
                print(f"  {r.status:<14} {f.name:<34} fields {sum(fields.values())}/{len(fields)} "
                      f"lines {m}/{n_true}  {','.join(detected) or '-':<40} {('wrong: ' + ','.join(wrong)) if wrong else ''} {flag}")
    return rows, summarize(rows)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    field_acc = {f: round(100 * sum(r["fields"][f] for r in rows) / n, 1) for f in HEADER_FIELDS}
    all_fields = [ok for r in rows for ok in r["fields"].values()]
    m = sum(r["lines_matched"] for r in rows)
    prec = m / max(1, sum(r["lines_pred"] for r in rows))
    rec = m / max(1, sum(r["lines_true"] for r in rows))
    recall_by_issue = {}
    for code in PLANTED:
        exp = [r for r in rows if code in r["expected_issues"]]
        if exp:
            recall_by_issue[code] = f"{sum(code in r['detected_issues'] for r in exp)}/{len(exp)}"
    clean = [r for r in rows if not r["expected_issues"]]
    false_alarm_docs = [r for r in clean if r["detected_issues"]]
    statuses = Counter(r["status"] for r in rows)
    return {
        "documents": n,
        "field_accuracy_pct": round(100 * sum(all_fields) / max(1, len(all_fields)), 1),
        "field_accuracy_by_field": field_acc,
        "line_item_f1_pct": round(100 * (2 * prec * rec / (prec + rec)) if prec + rec else 0, 1),
        "document_accuracy_pct": round(100 * sum(r["doc_correct"] for r in rows) / n, 1),
        "issue_recall": recall_by_issue,
        "false_alarm_docs": f"{len(false_alarm_docs)}/{len(clean)}",
        "false_alarm_codes": dict(Counter(c for r in false_alarm_docs for c in r["detected_issues"])),
        "straight_through_pct": round(100 * statuses.get("auto_approved", 0) / n, 1),
        "clean_docs_auto_approved": f"{sum(r['status'] == 'auto_approved' for r in clean)}/{len(clean)}",
        "unsafe_approvals": sum(r["unsafe_approval"] for r in rows),
        "statuses": dict(statuses),
        "cost_per_doc_usd": round(statistics.mean(r["cost_usd"] for r in rows), 5) if rows else 0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "median_latency_s": statistics.median(r["latency_s"] for r in rows) if rows else 0,
        "vision_docs": sum(r["mode"] == "vision" for r in rows),
    }


def write_report(rows: list[dict], s: dict, model: str, out_dir: Path = EVAL_RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M}_{model}.md"
    L = [
        f"# Invoice extraction evaluation: {model}", "",
        f"- Documents: **{s['documents']}** ({s['vision_docs']} scanned, read with vision)",
        f"- Field accuracy: **{s['field_accuracy_pct']}%** · line-item F1: **{s['line_item_f1_pct']}%** · "
        f"documents fully correct: **{s['document_accuracy_pct']}%**",
        f"- Straight-through processing: **{s['straight_through_pct']}%** "
        f"(clean documents auto-approved: {s['clean_docs_auto_approved']})",
        f"- **Unsafe approvals: {s['unsafe_approvals']}** (auto-approved but wrong or problematic)",
        f"- False alarms on clean documents: {s['false_alarm_docs']} {s['false_alarm_codes'] or ''}",
        f"- Cost: ${s['cost_per_doc_usd']:.4f} per document (total ${s['total_cost_usd']:.4f}) · median latency {s['median_latency_s']} s",
        "", "## Planted problems detected", "", "| Problem | Detected |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in s["issue_recall"].items()],
        "", "## Field accuracy", "", "| Field | Accuracy |", "|---|---|",
        *[f"| {k} | {v}% |" for k, v in s["field_accuracy_by_field"].items()],
        "", "## Documents", "",
        "| File | Status | Wrong fields | Lines | Expected | Detected | Low-confidence fields |", "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wrong = ", ".join(k for k, ok in r["fields"].items() if not ok)
        low = ", ".join(f"{k} {v:.2f}" for k, v in r.get("low_confidence_fields", {}).items())
        L.append(f"| {r['file']} | {r['status']}{' ⚠️' if r['unsafe_approval'] else ''} | {wrong} | "
                 f"{r['lines_matched']}/{r['lines_true']} | {', '.join(r['expected_issues'])} | "
                 f"{', '.join(r['detected_issues'])} | {low} |")
    path.write_text("\n".join(L), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({"summary": s, "rows": rows}, indent=2), encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate extraction + validation on the labelled samples.")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    from invoiceai.llm import OpenAIClient

    settings = Settings.from_env()
    print(f"Evaluating with {settings.model} ...\n")
    rows, s = evaluate(OpenAIClient(settings.model), settings, limit=args.limit)
    path = write_report(rows, s, settings.model)
    print(f"\nField accuracy {s['field_accuracy_pct']}% | line F1 {s['line_item_f1_pct']}% | "
          f"STP {s['straight_through_pct']}% | unsafe approvals {s['unsafe_approvals']} | "
          f"${s['cost_per_doc_usd']:.4f}/doc")
    print(f"Issue recall: {s['issue_recall']} | false alarms {s['false_alarm_docs']}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
