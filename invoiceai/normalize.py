"""Locale-aware formatting and parsing helpers.

Invoices print the same number as 1,234.56 (EN), 1.234,56 (GR/DE) or
1 234,56 (FR). These helpers are used both to render the synthetic invoices
and to check that an extracted value really appears in the source text.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]


def fmt_amount(x: float, style: str = "en") -> str:
    """Format a number in a locale style: en 1,234.56 | eu 1.234,56 | fr 1 234,56 | plain 1234.56."""
    s = f"{abs(x):,.2f}"
    if style == "eu":
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    elif style == "fr":
        s = s.replace(",", " ").replace(".", ",")
    elif style == "plain":
        s = s.replace(",", "")
    return ("-" if x < 0 else "") + s


def amount_variants(x: float) -> set[str]:
    """All the ways an amount could be printed (2 decimals), used for grounding checks."""
    out = {fmt_amount(x, s) for s in ("en", "eu", "fr", "plain")}
    if float(x).is_integer():
        out |= {f"{int(x):,}", f"{int(x):,}".replace(",", "."), str(int(x))}
    return out


def parse_amount(s: str) -> float | None:
    """Parse '1.234,56', '1,234.56', '€ 1 234,56', '-12,5' ... into a float."""
    if s is None:
        return None
    t = re.sub(r"[^\d,.\-]", "", str(s))
    if not re.search(r"\d", t):
        return None
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
    elif "," in t:
        dec = "," if re.search(r",\d{1,2}$", t) else None
    elif "." in t:
        dec = "." if (t.count(".") == 1 and not re.search(r"\.\d{3}$", t)) else None
    else:
        dec = None
    if dec == ",":
        t = t.replace(".", "").replace(",", ".")
    elif dec == ".":
        t = t.replace(",", "")
    else:
        t = t.replace(",", "").replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def fmt_date(d: date, style: str) -> str:
    if style == "dmy_slash":
        return d.strftime("%d/%m/%Y")
    if style == "dmy_dot":
        return d.strftime("%d.%m.%Y")
    if style == "long_dmy":
        return f"{d.day} {EN_MONTHS[d.month - 1]} {d.year}"
    if style == "long_mdy":
        return f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "iso":
        return d.isoformat()
    raise ValueError(style)


def date_variants(d: date) -> set[str]:
    out = {fmt_date(d, s) for s in ("dmy_slash", "dmy_dot", "long_dmy", "long_mdy", "iso")}
    out |= {f"{d.day}/{d.month}/{d.year}", f"{d.month}/{d.day}/{d.year}", d.strftime("%m/%d/%Y"),
            f"{EN_MONTHS[d.month - 1][:3]} {d.day}, {d.year}", f"{d.day} {EN_MONTHS[d.month - 1][:3]} {d.year}"}
    return out


def norm_text(s: str) -> str:
    """Casefold, strip accents (Greek tonos), collapse whitespace: for robust matching."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().casefold()


def norm_id(s: str | None) -> str:
    """Normalise identifiers (VAT ids, IBANs, invoice numbers): uppercase letters/digits only (any script)."""
    return re.sub(r"[\W_]", "", s or "").upper()
