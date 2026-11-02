"""REST API tests (FastAPI TestClient + oracle LLM, no API key or network needed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from fakes import OracleLLM  # noqa: E402
from invoiceai.api import create_app  # noqa: E402
from invoiceai.config import SAMPLES_DIR, Settings  # noqa: E402
from invoiceai.store import Store  # noqa: E402

SETTINGS = Settings.from_env()
CLEAN = "001_hispania_20260110.pdf"
MATH = "011_athpaper_20260312.pdf"


@pytest.fixture
def ctx(tmp_path):
    llm = OracleLLM()
    store = Store(tmp_path / "api.db")
    app = create_app(store=store, llm=llm, settings=SETTINGS, inbox=tmp_path / "inbox", api_key="")
    return TestClient(app), llm, store


def upload(client, llm, name):
    llm.set_file(name)
    with open(SAMPLES_DIR / name, "rb") as fh:
        return client.post("/invoices", files={"file": (name, fh, "application/pdf")})


def test_health(ctx):
    client, _, _ = ctx
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_clean_invoice_is_auto_approved(ctx):
    client, llm, _ = ctx
    r = upload(client, llm, CLEAN)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "auto_approved"
    assert body["issues"] == []
    assert body["invoice"]["supplier"]["name"] == "Hispania Print S.L."
    assert body["read_as"] == "text" and body["cost_usd"] > 0


def test_same_file_twice_is_not_processed_again(ctx):
    client, llm, _ = ctx
    first = upload(client, llm, CLEAN).json()
    calls = llm.calls
    again = upload(client, llm, CLEAN).json()
    assert again["doc_id"] == first["doc_id"] and again["already_processed"] is True
    assert llm.calls == calls  # no second LLM call, no second charge


def test_problem_invoice_goes_to_review_with_reason(ctx):
    client, llm, _ = ctx
    body = upload(client, llm, MATH).json()
    assert body["status"] == "needs_review"
    assert "MATH_TOTAL" in {i["code"] for i in body["issues"]}
    queue = client.get("/invoices", params={"status": "needs_review"}).json()
    assert [d["doc_id"] for d in queue] == [body["doc_id"]]
    assert queue[0]["issue_codes"] == ["MATH_TOTAL"]


def test_approve_with_correction_is_logged(ctx):
    client, llm, store = ctx
    body = upload(client, llm, MATH).json()
    inv = body["invoice"]
    inv["total"] = round(inv["subtotal"] + inv["vat_amount"], 2)  # reviewer fixes the total
    r = client.post(f"/invoices/{body['doc_id']}/approve", json={"note": "confirmed with supplier", "invoice": inv})
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "approved" and out["review_note"] == "confirmed with supplier"
    assert "MATH_TOTAL" not in {i["code"] for i in out["issues"]}
    fields = [row["field"] for row in store.con.execute("SELECT field FROM corrections")]
    assert fields == ["total"]


def test_reject(ctx):
    client, llm, _ = ctx
    doc_id = upload(client, llm, MATH).json()["doc_id"]
    r = client.post(f"/invoices/{doc_id}/reject", json={"note": "not our order"})
    assert r.json()["status"] == "rejected"


def test_rejects_non_pdf_and_unknown_ids(ctx):
    client, _, _ = ctx
    r = client.post("/invoices", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert client.get("/invoices/999").status_code == 404
    assert client.post("/invoices/999/approve").status_code == 404
    assert client.get("/invoices", params={"status": "nonsense"}).status_code == 422


def test_suppliers(ctx):
    client, llm, _ = ctx
    upload(client, llm, CLEAN)
    sup = client.get("/suppliers").json()
    assert sup[0]["supplier_name"] == "Hispania Print S.L." and sup[0]["invoices"] == 1
    assert len(sup[0]["ibans_seen"]) == 1


def test_api_key_is_enforced_when_configured(tmp_path):
    app = create_app(store=Store(tmp_path / "k.db"), llm=OracleLLM(), settings=SETTINGS, api_key="s3cret")
    client = TestClient(app)
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health", headers={"X-API-Key": "s3cret"}).status_code == 200
