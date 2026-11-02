import pytest

from invoiceai.identifiers import afm_check_digit, afm_is_valid, iban_is_valid, make_iban, vat_id_problem
from invoiceai.normalize import amount_variants, norm_id, parse_amount


@pytest.mark.parametrize("afm, ok", [("094019245", True), ("801234560", True), ("801234561", False), ("12345", False), ("000000000", False)])
def test_greek_afm_checksum(afm, ok):
    assert afm_is_valid(afm) is ok


def test_afm_check_digit_roundtrip():
    for first8 in ("12345678", "80123456", "99999999"):
        assert afm_is_valid(first8 + afm_check_digit(first8))


@pytest.mark.parametrize("iban, ok", [("GB82WEST12345698765432", True), ("GB82 WEST 1234 5698 7654 32", True),
                                      ("GB82WEST12345698765433", False), ("XX00", False), (None, False)])
def test_iban(iban, ok):
    assert iban_is_valid(iban) is ok


def test_make_iban_is_valid():
    assert iban_is_valid(make_iban("GR", "01101250000000012300695"))


@pytest.mark.parametrize("vat, country, problem", [
    ("EL801234560", "GR", False), ("801234560", "GR", False), ("GR801234560", "GR", False),
    ("EL801234561", "GR", True), ("DE123456789", "DE", False), ("DE12345", "DE", True),
    ("NL123456789B01", "NL", False), ("79-2458591", "US", False), ("XY-anything", None, False)])
def test_vat_ids(vat, country, problem):
    assert (vat_id_problem(vat, country) is not None) is problem


@pytest.mark.parametrize("s, v", [("1.234,56", 1234.56), ("1,234.56", 1234.56), ("€ 1 234,56", 1234.56),
                                  ("12,5", 12.5), ("1.234", 1234.0), ("-45.10", -45.1), ("abc", None)])
def test_parse_amount(s, v):
    assert parse_amount(s) == v


def test_amount_variants_cover_locales():
    assert {"1,234.50", "1.234,50", "1 234,50", "1234.50"} <= amount_variants(1234.5)


def test_norm_id_keeps_greek_letters():
    assert norm_id("ΤΠΥ Α-1452") == "ΤΠΥΑ1452"
