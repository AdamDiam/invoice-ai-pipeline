"""Generate a realistic, labelled set of supplier invoices (PDF + ground truth).

The set mimics an accounts-payable inbox of a Greek company:
  * 12 suppliers from GR, DE, NL, FR, IE, GB, IT, ES and the US
  * 4 visual layouts, English and Greek, local number/date formats
    (1,234.56 vs 1.234,56; 12 March 2026 vs 12/03/2026 vs 12.03.2026)
  * EU reverse-charge invoices (0% VAT), Greek 24% / 13% VAT, US invoices without VAT
  * deliberately planted problems the pipeline must catch:
      duplicates (re-sent invoices), a near-duplicate, arithmetic errors,
      invalid VAT numbers / AFM checksums, invalid IBANs, an amount outlier,
      a changed bank account (invoice fraud) and scanned (image-only) documents
Ground truth for every document is written to samples/ground_truth.json.

Usage:
    python -m invoiceai.synth.generate                 # 60 base invoices + planted cases
    python -m invoiceai.synth.generate --n 30 --seed 7 --out /tmp/invoices
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from invoiceai.config import GROUND_TRUTH_PATH, OUR_COMPANY, SAMPLES_DIR
from invoiceai.identifiers import afm_check_digit, make_iban
from invoiceai.normalize import fmt_amount, fmt_date

OUR_ADDRESS = "Leof. Kifisias 120, 151 25 Marousi, Athens, Greece"


@dataclass
class Supplier:
    key: str
    name: str
    country: str
    vat_id: str
    address: str
    currency: str
    lang: str          # en | el
    layout: str        # classic | modern | greek | compact
    num_style: str     # en | eu | fr
    date_style: str
    number_fmt: str    # python format with {n} and {y}
    catalog: list = field(default_factory=list)  # (description, low, high, vat_rate, qty_kind)
    weight: float = 1.0
    vat_regime: str = "domestic"  # domestic | reverse_charge | export | none


def _afm(rng: random.Random) -> str:
    first8 = f"{rng.randint(10_000_000, 99_999_999)}"
    return first8 + afm_check_digit(first8)


def build_suppliers(rng: random.Random) -> list[Supplier]:
    S = Supplier
    return [
        S("athpaper", "ΑΘΗΝΑΪΚΗ ΧΑΡΤΟΠΟΙΪΑ Α.Ε.", "GR", "EL" + _afm(rng), "Πειραιώς 210, 177 78 Ταύρος, Αθήνα",
          "EUR", "el", "greek", "eu", "dmy_slash", "ΤΠΥ Α-{n}",
          [("Χαρτί φωτοτυπικό Α4 (κιβώτιο 5 δεσμίδων)", 21, 32, 24, "int"),
           ("Φάκελοι αλληλογραφίας Α4 (500 τεμ.)", 18, 30, 24, "int"),
           ("Μελάνι εκτυπωτή μαύρο", 35, 75, 24, "int"),
           ("Ντοσιέ αρχειοθέτησης", 2.5, 6, 24, "int")], 1.4),
        S("konst", "Κωνσταντίνου Τεχνική Ο.Ε.", "GR", "EL" + _afm(rng), "Λ. Μεσογείων 355, 152 31 Χαλάνδρι",
          "EUR", "el", "greek", "eu", "dmy_slash", "ΤΠΥ Β-{n}",
          [("Συντήρηση κλιματιστικών μονάδων", 60, 140, 24, "int"),
           ("Εργασία ηλεκτρολόγου (ώρες)", 28, 40, 24, "hours"),
           ("Ανταλλακτικά ηλεκτρολογικού υλικού", 15, 220, 24, "int")], 1.0),
        S("aegean", "Aegean Fresh Foods Μ.Ι.Κ.Ε.", "GR", "EL" + _afm(rng), "Ermou 54, 105 63 Athens, Greece",
          "EUR", "en", "classic", "eu", "dmy_slash", "AFF-{y}-{n:04d}",
          [("Seasonal fruit (crate)", 14, 26, 13, "int"),
           ("Extra virgin olive oil 5L", 32, 48, 13, "int"),
           ("Espresso coffee beans 1kg", 18, 29, 24, "int"),
           ("Bottled water (pack of 24)", 4.5, 7, 13, "int")], 1.6),
        S("delta", "Δέλτα Λογιστική Ι.Κ.Ε.", "GR", "EL" + _afm(rng), "Αγίου Κωνσταντίνου 8, 151 24 Μαρούσι",
          "EUR", "el", "greek", "eu", "dmy_slash", "ΤΠΥ {n}",
          [("Λογιστική υποστήριξη μηνός", 350, 650, 24, "one"),
           ("Σύνταξη μισθοδοσίας", 80, 160, 24, "one"),
           ("Υποβολή περιοδικής δήλωσης ΦΠΑ", 60, 120, 24, "one")], 0.9),
        S("bergmann", "Bergmann Industrietechnik GmbH", "DE", "DE" + str(rng.randint(100_000_000, 999_999_999)),
          "Industriestraße 14, 70565 Stuttgart, Germany", "EUR", "en", "modern", "eu", "dmy_dot", "RE-{y}-{n:05d}",
          [("Hydraulic pump service", 300, 900, 0, "one"),
           ("Stainless steel fittings set", 40, 120, 0, "int"),
           ("Maintenance labour (hours)", 55, 85, 0, "hours")], 0.8, "reverse_charge"),
        S("vandijk", "Van Dijk Logistics B.V.", "NL", f"NL{rng.randint(100_000_000, 999_999_999)}B01",
          "Waalhaven Zuidzijde 21, 3089 JH Rotterdam, Netherlands", "EUR", "en", "classic", "eu", "dmy_slash",
          "VDL{y}{n:05d}",
          [("Pallet transport Rotterdam-Athens", 850, 1600, 0, "int"),
           ("Warehouse storage (pallet/month)", 12, 20, 0, "int"),
           ("Customs clearance fee", 60, 120, 0, "one")], 1.0, "reverse_charge"),
        S("lumiere", "Lumière Conseil SAS", "FR", f"FR{rng.randint(10, 99)}{rng.randint(100_000_000, 999_999_999)}",
          "18 rue de la Paix, 75002 Paris, France", "EUR", "en", "modern", "fr", "dmy_slash", "F{y}-{n:04d}",
          [("Strategy workshop (day)", 900, 1500, 0, "one"),
           ("Consulting hours", 120, 180, 0, "hours")], 0.6, "reverse_charge"),
        S("northwind", "Northwind Office Supplies Ltd", "IE", f"IE{rng.randint(1_000_000, 9_999_999)}W",
          "Unit 4, Park West Business Park, Dublin 12, Ireland", "EUR", "en", "classic", "en", "long_dmy",
          "NW-{n:06d}",
          [("Ergonomic office chair", 180, 420, 0, "int"),
           ("Height-adjustable desk", 350, 690, 0, "int"),
           ("Whiteboard markers (pack of 12)", 8, 15, 0, "int"),
           ("LED desk lamp", 25, 60, 0, "int")], 0.9, "reverse_charge"),
        S("brightline", "Brightline Software Ltd", "GB", f"GB{rng.randint(100_000_000, 999_999_999)}",
          "1 Canada Square, London E14 5AB, United Kingdom", "GBP", "en", "modern", "en", "long_dmy",
          "BL-{y}{n:04d}",
          [("Analytics Pro licence (seat/month)", 25, 60, 0, "int"),
           ("Premium support plan", 200, 500, 0, "one"),
           ("Onboarding and training", 400, 900, 0, "one")], 1.0, "export"),
        S("cloudnest", "CloudNest Hosting Inc.", "US", f"{rng.randint(10, 99)}-{rng.randint(1_000_000, 9_999_999)}",
          "500 Howard St, San Francisco, CA 94105, USA", "USD", "en", "compact", "en", "long_mdy", "CN{n:07d}",
          [("Cloud compute instances", 300, 1200, 0, "one"),
           ("Object storage (TB-month)", 20, 25, 0, "int"),
           ("Data egress (TB)", 80, 95, 0, "int")], 1.0, "none"),
        S("marconi", "Studio Marconi S.r.l.", "IT", f"IT{rng.randint(10**10, 10**11 - 1)}",
          "Via Tortona 31, 20144 Milano, Italy", "EUR", "en", "classic", "eu", "dmy_slash", "{n}/{y}",
          [("Brand identity design", 1500, 3500, 0, "one"),
           ("Website UI mockups", 800, 2000, 0, "one")], 0.4, "reverse_charge"),
        S("hispania", "Hispania Print S.L.", "ES", f"ESB{rng.randint(10_000_000, 99_999_999)}",
          "Calle de Alcalá 95, 28009 Madrid, Spain", "EUR", "en", "modern", "eu", "dmy_slash", "HP-{y}-{n:04d}",
          [("Brochure printing A5 (1,000 units)", 250, 450, 0, "int"),
           ("Business cards (500 units)", 35, 60, 0, "int"),
           ("Roll-up banner", 70, 120, 0, "int")], 0.6, "reverse_charge"),
    ]


IBAN_BBAN = {"GR": lambda r: f"0{r.randint(10, 99)}0{r.randint(1000, 9999)}" + f"{r.randint(10**15, 10**16 - 1)}",
             "DE": lambda r: f"{r.randint(10**17, 10**18 - 1)}",
             "NL": lambda r: "ABNA" + f"{r.randint(10**9, 10**10 - 1)}",
             "FR": lambda r: f"{r.randint(10**22, 10**23 - 1)}",
             "IE": lambda r: "AIBK" + f"{r.randint(10**13, 10**14 - 1)}",
             "GB": lambda r: "BARC" + f"{r.randint(10**13, 10**14 - 1)}",
             "IT": lambda r: "X" + f"{r.randint(10**21, 10**22 - 1)}",
             "ES": lambda r: f"{r.randint(10**19, 10**20 - 1)}"}


# ---------------------------------------------------------------------------
# invoice content
# ---------------------------------------------------------------------------
def make_invoice(sup: Supplier, rng: random.Random, issue: date, seq: int, iban: str | None) -> dict:
    n_lines = rng.choice([1, 1, 2, 2, 3, 3, 4, 5]) if len(sup.catalog) > 1 else 1
    items = rng.sample(sup.catalog, k=min(n_lines, len(sup.catalog)))
    lines = []
    for desc, lo, hi, rate, kind in items:
        qty = {"int": rng.randint(1, 12), "hours": rng.choice([2, 3.5, 4, 6, 7.5, 8, 12]), "one": 1}[kind]
        price = round(rng.uniform(lo, hi), 2)
        lines.append({"description": desc, "quantity": float(qty), "unit_price": price,
                      "vat_rate": float(rate), "net_amount": round(qty * price, 2)})
    subtotal = round(sum(li["net_amount"] for li in lines), 2)
    vat = round(sum(li["net_amount"] * li["vat_rate"] / 100 for li in lines), 2)
    due = issue + timedelta(days=rng.choice([15, 30, 30, 30, 45])) if rng.random() > 0.15 else None
    number = sup.number_fmt.format(n=seq, y=issue.year)
    return {
        "document_type": "invoice",
        "language": sup.lang,
        "invoice_number": number,
        "issue_date": issue.isoformat(),
        "due_date": due.isoformat() if due else None,
        "currency": sup.currency,
        "supplier": {"name": sup.name, "vat_id": sup.vat_id, "country": sup.country, "address": sup.address},
        "customer": {"name": OUR_COMPANY["name"], "vat_id": OUR_COMPANY["vat_id"], "country": "GR", "address": OUR_ADDRESS},
        "line_items": lines,
        "subtotal": subtotal,
        "vat_amount": vat,
        "total": round(subtotal + vat, 2),
        "iban": iban,
    }


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]

LABELS = {
    "en": dict(title="INVOICE", number="Invoice No.", date="Invoice date", due="Due date", due_none="Due on receipt",
               bill_to="Bill to", vat="VAT No.", desc="Description", qty="Qty", price="Unit price", rate="VAT %",
               amount="Amount", subtotal="Subtotal", vat_total="VAT", total="Total due", bank="Bank details",
               reverse="Reverse charge: VAT to be accounted for by the recipient (Art. 196, Directive 2006/112/EC).",
               export="Supply of services outside the scope of UK VAT.", none="No sales tax applies to this invoice."),
    "el": dict(title="ΤΙΜΟΛΟΓΙΟ ΠΑΡΟΧΗΣ ΥΠΗΡΕΣΙΩΝ / ΠΩΛΗΣΗΣ", number="Αριθμός", date="Ημερομηνία έκδοσης",
               due="Ημερομηνία λήξης", due_none="Εξόφληση με την παραλαβή", bill_to="Στοιχεία πελάτη", vat="ΑΦΜ",
               desc="Περιγραφή", qty="Ποσ.", price="Τιμή μον.", rate="ΦΠΑ %", amount="Καθαρή αξία",
               subtotal="Σύνολο καθαρής αξίας", vat_total="ΦΠΑ", total="Πληρωτέο ποσό", bank="Τράπεζα",
               reverse="", export="", none=""),
}


def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists():
            pdfmetrics.registerFont(TTFont("Inv", regular))
            pdfmetrics.registerFont(TTFont("Inv-Bold", bold if Path(bold).exists() else regular))
            return
    raise RuntimeError("No Unicode TTF font found (DejaVuSans or Arial) for rendering Greek text.")


def _vat_display(sup: Supplier, inv: dict) -> str:
    """Greek suppliers print the bare AFM; others print the full VAT id."""
    v = inv["supplier"]["vat_id"]
    return v[2:] if sup.lang == "el" and v.startswith("EL") else v


def render_pdf(inv: dict, sup: Supplier, path: Path, stamp: str | None = None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.pdfgen import canvas

    L = dict(LABELS[sup.lang])
    if sup.country == "US":
        L.update(vat="EIN", rate="Tax %", vat_total="Sales tax")
    W, H = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"{inv['invoice_number']} {sup.name}")
    num = lambda x: fmt_amount(x, sup.num_style)  # noqa: E731
    dt = lambda s: fmt_date(date.fromisoformat(s), sup.date_style)  # noqa: E731
    cur = {"EUR": "€", "GBP": "£", "USD": "$"}[inv["currency"]]
    accent = {"classic": "#1F3A5F", "modern": "#0E7C66", "greek": "#3B3B98", "compact": "#222222"}[sup.layout]

    def text(x, y, s, size=9, bold=False, color=black, align="left"):
        c.setFillColor(color)
        c.setFont("Inv-Bold" if bold else "Inv", size)
        {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[align](x, y, s)

    # ---- header -----------------------------------------------------------------
    if sup.layout == "modern":
        c.setFillColor(HexColor(accent))
        c.rect(0, H - 95, W, 95, stroke=0, fill=1)
        text(40, H - 50, sup.name, 16, True, white)
        text(40, H - 68, sup.address, 8.5, color=white)
        text(W - 40, H - 50, L["title"], 20, True, white, "right")
        y = H - 125
    elif sup.layout == "greek":
        text(W - 40, H - 36, L["title"], 11, True, HexColor(accent), "right")
        text(40, H - 62, sup.name, 15, True, HexColor(accent))
        text(40, H - 77, sup.address, 8.5)
        text(40, H - 89, f"{L['vat']}: {_vat_display(sup, inv)}", 8.5)
        y = H - 122
    else:
        text(40, H - 55, sup.name, 15, True, HexColor(accent))
        text(40, H - 70, sup.address, 8.5)
        text(40, H - 82, f"{L['vat']}: {_vat_display(sup, inv)}", 8.5)
        text(W - 40, H - 55, L["title"], 20, True, HexColor(accent), "right")
        y = H - 115
        if sup.layout == "compact":
            text(W - 40, H - 75, f"Amount due: {cur}{num(inv['total'])}", 11, True, align="right")

    # ---- meta block --------------------------------------------------------------
    meta = [(L["number"], inv["invoice_number"]), (L["date"], dt(inv["issue_date"])),
            (L["due"], dt(inv["due_date"]) if inv["due_date"] else L["due_none"])]
    if sup.layout == "modern":
        meta.append((L["vat"], _vat_display(sup, inv)))
    for i, (k, v) in enumerate(meta):
        text(W - 220, y - i * 14, f"{k}:", 9, True)
        text(W - 40, y - i * 14, v, 9, align="right")

    text(40, y, L["bill_to"], 9, True, HexColor(accent))
    cust = inv["customer"]
    text(40, y - 14, cust["name"], 9.5, True)
    text(40, y - 27, cust["address"], 8.5)
    cust_vat = cust["vat_id"][2:] if sup.lang == "el" else cust["vat_id"]
    text(40, y - 40, f"{'VAT No.' if sup.country == 'US' else L['vat']}: {cust_vat}", 8.5)

    # ---- line items --------------------------------------------------------------
    y -= 80
    cols = [40, 300, 350, 430, 480, W - 40]
    c.setFillColor(HexColor(accent if sup.layout != "compact" else "#EEEEEE"))
    c.rect(36, y - 4, W - 72, 18, stroke=0, fill=1)
    hdr_color = white if sup.layout != "compact" else black
    for x, label, align in zip(cols, [L["desc"], L["qty"], L["price"], L["rate"], L["amount"]],
                               ["left", "right", "right", "right", "right"]):
        xx = cols[cols.index(x) + 1] - 4 if align == "right" else x
        text(xx if label != L["amount"] else W - 44, y + 1, label, 8.5, True, hdr_color, align)
    y -= 20
    for li in inv["line_items"]:
        q = li["quantity"]
        q_str = (str(int(q)) if float(q).is_integer() else num(q).rstrip("0").rstrip(",."))
        text(cols[0] + 2, y, li["description"], 8.5)
        text(cols[2] - 4, y, q_str, 8.5, align="right")
        text(cols[3] - 4, y, num(li["unit_price"]), 8.5, align="right")
        text(cols[4] - 4, y, f"{li['vat_rate']:g}%", 8.5, align="right")
        text(W - 44, y, num(li["net_amount"]), 8.5, align="right")
        if sup.layout == "classic":
            c.setStrokeColor(HexColor("#CCCCCC"))
            c.line(36, y - 5, W - 36, y - 5)
        y -= 17

    # ---- totals ------------------------------------------------------------------
    y -= 10
    totals = [(L["subtotal"], inv["subtotal"]), (L["vat_total"], inv["vat_amount"]), (L["total"], inv["total"])]
    for i, (k, v) in enumerate(totals):
        last = i == len(totals) - 1
        text(W - 220, y - i * 15, k, 9.5 if last else 9, last)
        text(W - 44, y - i * 15, f"{cur} {num(v)}" if sup.lang == "en" else f"{num(v)} {cur}", 9.5 if last else 9, last, align="right")
    y -= 70

    note = {"reverse_charge": L["reverse"], "export": L["export"], "none": L["none"]}.get(sup.vat_regime, "")
    if note:
        text(40, y, note, 7.5)
        y -= 16
    if inv.get("iban"):
        text(40, y, f"{L['bank']}: IBAN {_spaced(inv['iban'])}", 8.5)
    if stamp:
        c.saveState()
        c.setFillColor(HexColor("#C0392B"))
        c.setFont("Inv-Bold", 28)
        c.translate(W / 2, H / 2)
        c.rotate(25)
        c.drawCentredString(0, 0, stamp)
        c.restoreState()
    c.showPage()
    c.save()


def _spaced(iban: str) -> str:
    return " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))


def make_scanned(pdf_path: Path, rng: random.Random) -> None:
    """Re-save a PDF as an image-only 'scan' (no text layer): slight rotation + noise."""
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter

    doc = pdfium.PdfDocument(str(pdf_path))
    img = doc[0].render(scale=150 / 72).to_pil().convert("L")
    doc.close()
    img = img.rotate(rng.uniform(-0.8, 0.8), expand=False, fillcolor=255, resample=Image.BICUBIC)
    noise = Image.effect_noise(img.size, 18).point(lambda v: 255 if v > 120 else 230)
    img = Image.blend(img, noise.convert("L"), 0.06).filter(ImageFilter.GaussianBlur(0.4))
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=150)
    pdf_path.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# dataset with planted problems
# ---------------------------------------------------------------------------
def generate(n: int = 60, seed: int = 42, out_dir: Path = SAMPLES_DIR, gt_path: Path = GROUND_TRUTH_PATH) -> dict:
    rng = random.Random(seed)
    _register_fonts()
    suppliers = build_suppliers(rng)
    ibans = {s.key: (make_iban(s.country, IBAN_BBAN[s.country](rng)) if s.country in IBAN_BBAN else None) for s in suppliers}
    seqs = {s.key: rng.randint(100, 900) for s in suppliers}

    start = date(2026, 1, 5)
    docs = []  # (issue_date, supplier, invoice, expected_issues, extras)
    for _ in range(n):
        sup = rng.choices(suppliers, weights=[s.weight for s in suppliers])[0]
        issue = start + timedelta(days=rng.randint(0, 290))
        seqs[sup.key] += rng.randint(1, 9)
        docs.append([issue, sup, make_invoice(sup, rng, issue, seqs[sup.key], ibans[sup.key]), set(), {}])
    docs.sort(key=lambda d: d[0])

    def pick(cond=lambda d: True):
        free = [d for d in docs if not d[3] and not d[4]]
        choices = [d for d in free if cond(d)] or free  # small sets: relax the condition
        return rng.choice(choices)

    # arithmetic errors: printed total does not equal subtotal + VAT (supplier mistake)
    for delta in (10.0, -1.0, 90.0):
        d = pick(lambda d: d[1].vat_regime == "domestic")
        d[2]["total"] = round(d[2]["total"] + delta, 2)
        d[3].add("MATH_TOTAL")
    # invalid Greek AFM (wrong check digit) / invalid VAT format
    for _ in range(2):
        d = pick(lambda d: d[1].country == "GR")
        v = d[2]["supplier"]["vat_id"]
        d[2]["supplier"]["vat_id"] = v[:-1] + str((int(v[-1]) + 3) % 10)
        d[3].add("VAT_ID_INVALID")
    # invalid IBAN (typo in one digit)
    for _ in range(2):
        d = pick(lambda d: d[2]["iban"] is not None)
        ib = d[2]["iban"]
        pos = rng.randint(6, len(ib) - 2)
        d[2]["iban"] = ib[:pos] + str((int(ib[pos]) + 1) % 10 if ib[pos].isdigit() else 7) + ib[pos + 1 :]
        d[3].add("IBAN_INVALID")
    # amount outlier: unusually large invoice from a frequent supplier, late in the year
    counts: dict[str, int] = {}
    for d in docs:
        counts[d[1].key] = counts.get(d[1].key, 0) + 1
    top = max(counts, key=counts.get)
    d = pick(lambda d: d[1].key == top and d[0] > date(2026, 8, 1))
    for li in d[2]["line_items"]:
        li["quantity"] = float(li["quantity"] * 12 if li["quantity"] >= 1 else 12)
        li["net_amount"] = round(li["quantity"] * li["unit_price"], 2)
    d[2]["subtotal"] = round(sum(li["net_amount"] for li in d[2]["line_items"]), 2)
    d[2]["vat_amount"] = round(sum(li["net_amount"] * li["vat_rate"] / 100 for li in d[2]["line_items"]), 2)
    d[2]["total"] = round(d[2]["subtotal"] + d[2]["vat_amount"], 2)
    d[3].add("AMOUNT_OUTLIER")
    # scanned documents: no text layer, must be read with vision and reviewed
    for _ in range(3):
        d = pick()
        d[4]["scanned"] = True
        d[3].add("SCANNED")

    # duplicates: the same invoice re-sent later (exact copy with a "COPY" stamp)
    extra = []
    for _ in range(3):
        d = pick(lambda d: d[0] < date(2026, 9, 1))
        dup = [d[0] + timedelta(days=rng.randint(10, 40)), d[1], copy.deepcopy(d[2]), {"DUPLICATE"}, {"stamp": "COPY"}]
        extra.append(dup)
    # near-duplicate: same supplier and amount a few days later, different number
    d = pick(lambda d: d[0] < date(2026, 9, 1))
    near = copy.deepcopy(d[2])
    near["invoice_number"] = d[1].number_fmt.format(n=seqs[d[1].key] + 50, y=d[0].year)
    new_issue = d[0] + timedelta(days=3)
    near["issue_date"] = new_issue.isoformat()
    if near["due_date"]:
        near["due_date"] = (date.fromisoformat(near["due_date"]) + timedelta(days=3)).isoformat()
    extra.append([new_issue, d[1], near, {"POSSIBLE_DUPLICATE"}, {}])
    # invoice fraud: a familiar supplier suddenly asks to be paid into a different (but valid) account
    by_sup: dict[str, list] = {}
    for d in docs:
        by_sup.setdefault(d[1].key, []).append(d)
    candidates = [k for k, v in by_sup.items() if v[0][2]["iban"] and len([x for x in v if not x[3] and not x[4]]) >= 5]
    fraud_sup = sorted(candidates, key=lambda k: -len(by_sup[k]))[min(1, len(candidates) - 1)] if candidates else None
    if fraud_sup:
        earlier_clean = 0
        for d in by_sup[fraud_sup]:
            if not d[3] and not d[4] and earlier_clean >= 3:
                sup = d[1]
                d[2]["iban"] = make_iban(sup.country, IBAN_BBAN[sup.country](rng))
                d[3].add("IBAN_CHANGED")
                break
            if not d[3] and not d[4]:
                earlier_clean += 1

    docs = sorted(docs + extra, key=lambda d: d[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pdf"):
        old.unlink()
    truth = {}
    for i, (issue, sup, inv, issues, extras) in enumerate(docs, start=1):
        fname = f"{i:03d}_{sup.key}_{issue:%Y%m%d}.pdf"
        path = out_dir / fname
        render_pdf(inv, sup, path, stamp=extras.get("stamp"))
        if extras.get("scanned"):
            make_scanned(path, rng)
        truth[fname] = {"invoice": inv, "expected_issues": sorted(issues), "supplier_key": sup.key}
    gt_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return truth


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic supplier invoices with ground truth.")
    p.add_argument("--n", type=int, default=60, help="number of base invoices (planted cases are added on top)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=SAMPLES_DIR)
    args = p.parse_args()
    gt = GROUND_TRUTH_PATH if args.out == SAMPLES_DIR else args.out / "ground_truth.json"
    truth = generate(args.n, args.seed, args.out, gt)
    n_issues = sum(1 for t in truth.values() if t["expected_issues"])
    print(f"Wrote {len(truth)} invoices to {args.out} ({n_issues} with planted issues). Ground truth: {gt}")


if __name__ == "__main__":
    main()
