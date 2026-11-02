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

st.set_page_config(page_title="Invoice AI", page_icon="🧾", layout="wide", initial_sidebar_state="collapsed")

ACCENT, TEAL, RED, MUTED = "#F5A524", "#2DD4BF", "#F87171", "#94A3B8"
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [ACCENT, TEAL, "#818CF8", "#F472B6", "#A3E635", "#38BDF8"]

st.markdown(
    f"""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 4.2rem; max-width: 1400px; }}
  .hero {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 24px; border-radius:14px;
          margin-bottom:14px;
          background: linear-gradient(120deg, #111A2E 0%, #1A2440 60%, #2A2213 100%); border:1px solid #243049; }}
  .brand {{ font-size:1.6rem; font-weight:800; letter-spacing:.06em; color:#F8FAFC; }}
  .brand span {{ color:{ACCENT}; }}
  .tagline {{ color:{MUTED}; font-size:.92rem; margin-top:2px; }}
  .chip {{ font-family: ui-monospace, Menlo, monospace; font-size:.78rem; color:{TEAL}; border:1px solid #1F4D4A;
          padding:4px 10px; border-radius:999px; background:#0F2A2A; }}
  .kpis {{ display:grid; grid-template-columns: repeat(6, 1fr); gap:12px; margin:10px 0 8px; }}
  .kpi {{ background:#111A2E; border:1px solid #243049; border-radius:12px; padding:14px 16px; }}
  .kpi .label {{ color:{MUTED}; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }}
  .kpi .value {{ color:#F8FAFC; font-size:1.65rem; font-weight:700; font-family: ui-monospace, Menlo, monospace; }}
  .kpi .sub {{ color:{MUTED}; font-size:.75rem; }}
  .kpi.good .value {{ color:{TEAL}; }} .kpi.warn .value {{ color:{ACCENT}; }} .kpi.bad .value {{ color:{RED}; }}
  .flow {{ display:flex; align-items:stretch; gap:8px; margin:6px 0 18px; }}
  .step {{ flex:1; background:#0E1526; border:1px solid #243049; border-radius:10px; padding:10px 12px; text-align:center; }}
  .step b {{ display:block; font-size:1.25rem; color:#F8FAFC; font-family: ui-monospace, Menlo, monospace; }}
  .step small {{ color:{MUTED}; }}
  .arrow {{ align-self:center; color:#475569; font-size:1.2rem; }}
  .step.ok {{ border-color:#1F4D4A; }} .step.rev {{ border-color:#5A4316; }} .step.blk {{ border-color:#5B2424; }}
  .step.ok b {{ color:{TEAL}; }} .step.rev b {{ color:{ACCENT}; }} .step.blk b {{ color:{RED}; }}
  @media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} .flow {{ flex-wrap:wrap; }} }}
</style>
""",
    unsafe_allow_html=True,
)

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


def style(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(color="#E2E8F0"), title_font=dict(size=15))
    return fig


def docs_frame(store: Store) -> pd.DataFrame:
    rows = [dict(r) for r in store.documents()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["status_label"] = df["status"].map(STATUS_LABEL).fillna(df["status"])
    return df


# --------------------------------------------------------------------------- header + controls
settings = Settings.from_env()
store = get_store(str(settings.db_path))
has_key = bool(os.getenv("OPENAI_API_KEY"))

st.markdown(
    f"""<div class="hero"><div><div class="brand">INVOICE<span>·</span>AI</div>
    <div class="tagline">Accounts-payable automation: LLM extraction that is verified before it is trusted</div></div>
    <div class="chip">{settings.model} · auto-approve ≥ {settings.auto_approve_confidence:.0%}</div></div>""",
    unsafe_allow_html=True,
)

n_samples = len(list(SAMPLES_DIR.glob("*.pdf")))
c1, c2, c3, _ = st.columns([3, 1.6, 1, 7.4], gap="small", vertical_alignment="center")
if c1.button(f"▶ Process {n_samples} sample invoices", type="primary", disabled=not has_key or n_samples == 0):
    process(iter_pdfs(SAMPLES_DIR), store, settings)
with c2.popover("⬆ Upload PDFs", disabled=not has_key):
    uploads = st.file_uploader("Supplier invoices", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Process uploaded files"):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for u in uploads:
            p = INBOX_DIR / Path(u.name).name
            p.write_bytes(u.getvalue())
            paths.append(p)
        process(paths, store, settings)
if c3.button("Reset"):
    store.reset()
    st.rerun()
if not has_key:
    st.warning("Set OPENAI_API_KEY in .env to process documents.")

df = docs_frame(store)
if df.empty:
    st.info("The inbox is empty. Click **Process sample invoices** (about 2 minutes for 64 PDFs) or upload your own.")
    st.stop()

n = len(df)
counts = df["status"].value_counts().to_dict()
auto = counts.get("auto_approved", 0)
approved = counts.get("approved", 0)
review = counts.get("needs_review", 0) + counts.get("failed", 0)
blocked = counts.get("duplicate", 0) + counts.get("rejected", 0)
fraud = int(df["issue_codes"].fillna("").str.contains("IBAN_CHANGED").sum())
dups = df[df["status"] == "duplicate"]


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


st.markdown(
    '<div class="kpis">'
    + kpi("Documents", f"{n}", f"{int((df['mode'] == 'vision').sum())} scanned")
    + kpi("Straight-through", f"{auto / n:.0%}", "no human needed", "good")
    + kpi("Needs review", f"{review}", "with reasons attached", "warn")
    + kpi("Duplicates blocked", f"{len(dups)}", f"{dups['total'].fillna(0).sum():,.2f} not paid twice", "bad")
    + kpi("Fraud alerts", f"{fraud}", "changed bank account", "bad" if fraud else "")
    + kpi("AI cost", f"${df['cost_usd'].fillna(0).sum():.2f}", f"${df['cost_usd'].fillna(0).mean():.4f} / invoice")
    + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="flow">
    <div class="step"><b>{n}</b><small>received</small></div><div class="arrow">→</div>
    <div class="step"><b>{n - counts.get('failed', 0)}</b><small>extracted</small></div><div class="arrow">→</div>
    <div class="step"><b>{n - counts.get('failed', 0)}</b><small>rules · grounding · history</small></div><div class="arrow">→</div>
    <div class="step ok"><b>{auto + approved}</b><small>approved</small></div>
    <div class="step rev"><b>{review}</b><small>in review</small></div>
    <div class="step blk"><b>{blocked}</b><small>blocked</small></div></div>""",
    unsafe_allow_html=True,
)

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
        a.plotly_chart(style(px.bar(by_sup, x="total", y="supplier_name", orientation="h", title=f"Spend by supplier ({cur})",
                              labels={"total": cur, "supplier_name": ""})))
        by_month = sub.groupby("month", as_index=False)["total"].sum()
        b.plotly_chart(style(px.bar(by_month, x="month", y="total", title=f"Spend by month ({cur})", labels={"total": cur, "month": ""})))
        issues = pd.read_sql("SELECT code, COUNT(*) AS n FROM issues GROUP BY code ORDER BY n DESC", store.con)
        if not issues.empty:
            a.plotly_chart(style(px.bar(issues, x="code", y="n", title="Issues caught by type", labels={"n": "documents", "code": ""},
                                        color_discrete_sequence=[RED])))
        status_counts = df["status_label"].value_counts().reset_index()
        status_counts.columns = ["status", "n"]
        b.plotly_chart(style(px.pie(status_counts, names="status", values="n", hole=0.55, title="Processing outcome")))
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
