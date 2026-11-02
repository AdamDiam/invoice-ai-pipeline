"""Checksum and format validation for VAT ids (incl. Greek AFM) and IBANs."""
from __future__ import annotations

import re

from invoiceai.normalize import norm_id

# EU VAT number formats (after the 2-letter prefix), plus UK and US EIN.
VAT_FORMATS = {
    "EL": r"\d{9}",
    "DE": r"\d{9}",
    "NL": r"\d{9}B\d{2}",
    "FR": r"[0-9A-Z]{2}\d{9}",
    "IE": r"\d{7}[A-Z]{1,2}|\d[A-Z]\d{5}[A-Z]",
    "IT": r"\d{11}",
    "ES": r"[0-9A-Z]\d{7}[0-9A-Z]",
    "GB": r"\d{9}",
}


def afm_is_valid(afm: str) -> bool:
    """Greek tax number (ΑΦΜ) checksum: 9 digits, weighted sum of the first 8 by 2^8..2^1, mod 11 mod 10."""
    afm = re.sub(r"\D", "", afm or "")
    if len(afm) != 9 or afm == "000000000":
        return False
    total = sum(int(d) * 2 ** (8 - i) for i, d in enumerate(afm[:8]))
    return (total % 11) % 10 == int(afm[8])


def afm_check_digit(first8: str) -> str:
    total = sum(int(d) * 2 ** (8 - i) for i, d in enumerate(first8))
    return str((total % 11) % 10)


def vat_id_problem(vat_id: str | None, country: str | None = None) -> str | None:
    """Return a human-readable problem with a VAT id, or None if it looks valid."""
    if not vat_id:
        return None
    raw = norm_id(vat_id)
    if raw.startswith("GR"):  # Greek VAT ids use the EL prefix in the EU
        raw = "EL" + raw[2:]
    prefix, body = raw[:2], raw[2:]
    if prefix not in VAT_FORMATS:
        if country == "GR" and re.fullmatch(r"\d{9}", raw):  # bare AFM printed without prefix
            prefix, body = "EL", raw
        elif country == "US":
            return None if re.fullmatch(r"\d{9}", raw) else "US EIN should have 9 digits"
        else:
            return None  # unknown scheme: do not guess
    if not re.fullmatch(VAT_FORMATS[prefix], body):
        return f"{prefix} VAT number has an invalid format"
    if prefix == "EL" and not afm_is_valid(body):
        return "Greek AFM checksum is invalid"
    return None


def iban_is_valid(iban: str | None) -> bool:
    s = norm_id(iban)
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def make_iban(country: str, bban: str) -> str:
    """Build an IBAN with correct check digits (for the synthetic data)."""
    digits = "".join(str(int(c, 36)) for c in (bban + country + "00"))
    check = 98 - int(digits) % 97
    return f"{country}{check:02d}{bban}"
