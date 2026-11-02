"""Invoice AI: accounts-payable automation UI.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from invoiceai.config import DATA_DIR, SAMPLES_DIR, Settings  # noqa: E402
from invoiceai.pdf import render_page_png  # noqa: E402
from invoiceai.pipeline import iter_pdfs, process_file  # noqa: E402
from invoiceai.schema import Invoice  # noqa: E402
from invoiceai.store import Store  # noqa: E402
from invoiceai.validate import check_rules  # noqa: E402

st.set_page_config(page_title="Invoice AI: AP Automation", page_icon="🧾", layout="wide")

STATUS_LABEL = {
    "auto_approved": "✅ Auto-approved",
    "approved": "✅ Approved (reviewed)",
    "needs_review": "🟡 Needs review",
    "duplicate": "⛔ Duplicate",
    "rejected": "❌ Rejected",
    "failed": "⚠️ Failed",
}
INBOX_DIR = DATA_DIR / "inbox"


@st.cache_resource
def get_store(path: str) -> Store:
    return Store(Path(path))


@st.cache_resource
def get_llm(model: str):
    from invoiceai.llm import OpenAIClient

    return OpenAIClient(model)


@st.cache_data(show_spinner=False)
def page_image(path: str) -> bytes | None:
    try:
        return render_page_png(Path(path), 0, scale=1.6)
    except Exception:
        return None


def process(paths: list[Path], store: Store, settings: Settings) -> None:
    llm = get_llm(settings.model)
    bar = st.progress(0.0, text="Processing…")
    done = {}
    for i, p in enumerate(paths, 1):
        r = process_file(p, llm, store, settings)
        done[r.status] = done.get(r.status, 0) + 1
        bar.progress(i / len(paths), text=f"{i}/{len(paths)} · {p.name} → {STATUS_LABEL.get(r.status, r.status)}")
    bar.empty()
    st.toast("Done: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in done.items()))


def docs_frame(store: Store) -> pd.DataFrame:
    rows = [dict(r) for r in store.documents()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["status_label"] = df["status"].map(STATUS_LABEL).fillna(df["status"])
    return df


# --------------------------------------------------------------------------- sidebar
settings = Settings.from_env()
store = get_store(str(settings.db_path))

with st.sidebar:
    st.markdown("## 🧾 Invoice AI")
    st.caption("Reads supplier invoices (PDF, incl. scans), extracts the data with an LLM, validates it against "
               "business rules and the source text, and only auto-approves what it can verify.")
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        st.warning("Set OPENAI_API_KEY in .env to process documents.")
    n_samples = len(list(SAMPLES_DIR.glob("*.pdf")))
    if st.button(f"▶️ Process the {n_samples} sample invoices", disabled=not has_key or n_samples == 0):
        process(iter_pdfs(SAMPLES_DIR), store, settings)
    uploads = st.file_uploader("…or upload your own PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Process uploaded files", disabled=not has_key):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for u in uploads:
            p = INBOX_DIR / Path(u.name).name
            p.write_bytes(u.getvalue())
            paths.append(p)
        process(paths, store, settings)
    st.divider()
    if st.button("🗑️ Reset database"):
        store.reset()
        st.rerun()
    st.caption(f"Model: `{settings.model}` · auto-approve ≥ {settings.auto_approve_confidence:.0%} confidence")

# --------------------------------------------------------------------------- header
st.title("Invoice AI: accounts-payable automation")
df = docs_frame(store)
if df.empty:
    st.info("No documents yet. Use **Process the sample invoices** in the sidebar (about 1–2 minutes for 64 PDFs) or upload your own.")
    st.stop()

n = len(df)
auto = int((df["status"] == "auto_approved").sum())
review = int((df["status"] == "needs_review").sum())
dups = df[df["status"] == "duplicate"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Documents", f"{n}")
c2.metric("Straight-through", f"{auto / n:.0%}", help="Auto-approved without a human")
c3.metric("Waiting for review", f"{review}")
c4.metric("Duplicates blocked", f"{len(dups)}", help=f"Value: {dups['total'].fillna(0).sum():,.2f}")
c5.metric("AI cost", f"${df['cost_usd'].fillna(0).sum():.3f}", help=f"${df['cost_usd'].fillna(0).mean():.4f} per document")

tab_inbox, tab_review, tab_dash, tab_sup, tab_export = st.tabs(
    ["📥 Inbox", "🔍 Review", "📊 Dashboard", "🏢 Suppliers", "⬇️ Export"]
)

# --------------------------------------------------------------------------- inbox
with tab_inbox:
    statuses = st.multiselect("Status", list(STATUS_LABEL.values()), default=list(STATUS_LABEL.values()))
    view = df[df["status_label"].isin(statuses)][
        ["doc_id", "status_label", "file_name", "supplier_name", "invoice_number", "issue_date", "total", "currency",
         "confidence", "issue_codes", "mode", "cost_usd"]
    ]
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "doc_id": "ID", "status_label": "Status", "file_name": "File", "supplier_name": "Supplier",
            "invoice_number": "Invoice no.", "issue_date": "Date",
            "total": st.column_config.NumberColumn("Total", format="%.2f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "issue_codes": "Issues", "mode": "Read as",
            "cost_usd": st.column_config.NumberColumn("Cost $", format="%.4f"),
        },
    )

# --------------------------------------------------------------------------- review
with tab_review:
    queue = df[df["status"].isin(["needs_review", "duplicate", "failed"])]
    if queue.empty:
        st.success("Nothing to review 🎉")
    else:
        options = {f"#{r.doc_id} · {STATUS_LABEL.get(r.status)} · {r.file_name}": int(r.doc_id) for r in queue.itertuples()}
        doc_id = options[st.selectbox(f"{len(queue)} document(s) in the queue", list(options))]
        doc = store.document(doc_id)
        left, right = st.columns([1, 1])
        with left:
            img = page_image(doc["file_path"]) if doc["file_path"] else None
            if img:
                st.image(img, caption=doc["file_name"])
            else:
                st.info("Preview not available.")
        with right:
            for i in store.issues(doc_id):
                (st.error if i["severity"] == "error" else st.warning)(f"**{i['code']}**: {i['message']}")
            if not doc["final_json"]:
                st.error(doc["error"] or "Extraction failed.")
                if st.button("❌ Reject", key=f"rej_{doc_id}"):
                    store.review(doc_id, "rejected", note="extraction failed")
                    st.rerun()
            else:
                inv = json.loads(doc["final_json"])
                conf = json.loads(doc["field_confidence"] or "{}")

                def label(text: str, field: str) -> str:
                    c = conf.get(field)
                    return f"{text} · {c:.0%}{' ⚠️' if c is not None and c < 0.6 else ''}" if c is not None else text

                with st.form(f"review_{doc_id}"):
                    a, b = st.columns(2)
                    sup_name = a.text_input(label("Supplier", "supplier.name"), inv["supplier"]["name"])
                    sup_vat = b.text_input(label("Supplier VAT id", "supplier.vat_id"), inv["supplier"]["vat_id"] or "")
                    number = a.text_input(label("Invoice number", "invoice_number"), inv["invoice_number"])
                    currency = b.text_input(label("Currency", "currency"), inv["currency"])
                    issue = a.date_input(label("Issue date", "issue_date"), date.fromisoformat(inv["issue_date"]))
                    due = b.date_input(label("Due date", "due_date"),
                                       date.fromisoformat(inv["due_date"]) if inv["due_date"] else None)
                    iban = st.text_input(label("IBAN", "iban"), inv.get("iban") or "")
                    lines = st.data_editor(pd.DataFrame(inv["line_items"]), num_rows="dynamic", key=f"lines_{doc_id}")
                    x, y, z = st.columns(3)
                    subtotal = x.number_input(label("Subtotal", "subtotal"), value=float(inv["subtotal"]), format="%.2f")
                    vat = y.number_input(label("VAT", "vat_amount"), value=float(inv["vat_amount"]), format="%.2f")
                    total = z.number_input(label("Total", "total"), value=float(inv["total"]), format="%.2f")
                    note = st.text_input("Review note (optional)")
                    approve = st.form_submit_button("✅ Approve")
                    reject = st.form_submit_button("❌ Reject")

                if approve or reject:
                    inv.update(invoice_number=number, currency=currency, issue_date=issue.isoformat(),
                               due_date=due.isoformat() if due else None, iban=iban or None,
                               subtotal=subtotal, vat_amount=vat, total=total,
                               line_items=lines.dropna(how="all").to_dict("records"))
                    inv["supplier"].update(name=sup_name, vat_id=sup_vat or None)
                    try:
                        corrected = Invoice.model_validate(inv)
                    except Exception as exc:
                        st.error(f"Invalid data: {exc}")
                        st.stop()
                    remaining = check_rules(corrected) + store.history_issues(corrected, exclude_doc=doc_id)
                    store.review(doc_id, "approved" if approve else "rejected", corrected, note, remaining)
                    st.rerun()

# --------------------------------------------------------------------------- dashboard
with tab_dash:
    ok = df[df["status"].isin(["auto_approved", "approved"])]
    if ok.empty:
        st.info("No approved invoices yet.")
    else:
        cur = st.radio("Currency", sorted(ok["currency"].dropna().unique()), horizontal=True)
        sub = ok[ok["currency"] == cur].copy()
        sub["month"] = sub["issue_date"].str[:7]
        a, b = st.columns(2)
        by_sup = sub.groupby("supplier_name", as_index=False)["total"].sum().sort_values("total", ascending=True)
        a.plotly_chart(px.bar(by_sup, x="total", y="supplier_name", orientation="h", title=f"Spend by supplier ({cur})",
                              labels={"total": cur, "supplier_name": ""}))
        by_month = sub.groupby("month", as_index=False)["total"].sum()
        b.plotly_chart(px.bar(by_month, x="month", y="total", title=f"Spend by month ({cur})", labels={"total": cur, "month": ""}))
        issues = pd.read_sql("SELECT code, COUNT(*) AS n FROM issues GROUP BY code ORDER BY n DESC", store.con)
        if not issues.empty:
            a.plotly_chart(px.bar(issues, x="code", y="n", title="Issues caught by type", labels={"n": "documents", "code": ""}))
        status_counts = df["status_label"].value_counts().reset_index()
        status_counts.columns = ["status", "n"]
        b.plotly_chart(px.pie(status_counts, names="status", values="n", hole=0.5, title="Processing outcome"))
        corr = pd.read_sql("SELECT field, COUNT(*) AS n FROM corrections GROUP BY field ORDER BY n DESC", store.con)
        if not corr.empty:
            st.markdown("**Fields corrected by reviewers** (where the AI is weakest)")
            st.dataframe(corr, hide_index=True)

# --------------------------------------------------------------------------- suppliers
with tab_sup:
    st.caption("Supplier master data learned from the processed invoices. New invoices are compared with it: "
               "**a different IBAN than the supplier's known account is blocked for review** (the most common "
               "invoice-fraud pattern), and amounts far above the supplier's usual range are flagged.")
    sup = pd.DataFrame([dict(r) for r in store.suppliers()])
    if sup.empty:
        st.info("No suppliers yet.")
    else:
        sup["ibans_seen"] = sup["ibans_seen"].fillna("").str.replace(",", "\n")
        st.dataframe(
            sup,
            hide_index=True,
            column_config={
                "supplier_name": "Supplier", "supplier_vat": "VAT id", "supplier_country": "Country",
                "currency": "Cur.", "invoices": "Invoices",
                "total_spend": st.column_config.NumberColumn("Total spend", format="%.2f"),
                "min_total": st.column_config.NumberColumn("Min", format="%.2f"),
                "max_total": st.column_config.NumberColumn("Max", format="%.2f"),
                "first_invoice": "First", "last_invoice": "Last", "ibans_seen": "IBANs seen",
                "flagged": "Flagged",
            },
        )

# --------------------------------------------------------------------------- export
with tab_export:
    approved = pd.read_sql(
        """SELECT i.*, d.status, d.file_name FROM invoices i JOIN documents d USING (doc_id)
           WHERE d.status IN ('auto_approved', 'approved') ORDER BY i.issue_date""", store.con)
    lines = pd.read_sql(
        """SELECT l.* FROM line_items l JOIN documents d USING (doc_id)
           WHERE d.status IN ('auto_approved', 'approved') ORDER BY l.doc_id, l.line_no""", store.con)
    st.markdown(f"**{len(approved)} approved invoices**, {len(lines)} line items")
    st.dataframe(approved, hide_index=True)
    st.download_button("Download invoices (CSV)", approved.to_csv(index=False).encode("utf-8"), "invoices.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        approved.to_excel(xl, sheet_name="invoices", index=False)
        lines.to_excel(xl, sheet_name="line_items", index=False)
    st.download_button("Download invoices + line items (Excel)", buf.getvalue(), "invoices.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
