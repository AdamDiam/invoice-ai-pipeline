"""The target data model for an extracted invoice (Pydantic v2).

The LLM must return JSON that validates against `Invoice`. Validation errors
are fed back to the model for a corrected attempt.
"""
from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Party(BaseModel):
    name: str = Field(description="Legal name as printed")
    vat_id: Optional[str] = Field(None, description="VAT / tax id incl. country prefix if printed, e.g. EL123456789, DE123456789")
    country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code, e.g. GR, DE, US")
    address: Optional[str] = Field(None, description="Postal address on one line")

    @field_validator("country")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else v


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float = Field(description="Net unit price")
    vat_rate: float = Field(description="VAT rate in percent, e.g. 24 for 24%")
    net_amount: float = Field(description="Line net amount (before VAT) as printed")


class Invoice(BaseModel):
    document_type: Literal["invoice", "credit_note", "other"] = "invoice"
    language: str = Field(description="ISO 639-1 code of the document language, e.g. en, el")
    invoice_number: str
    issue_date: date
    due_date: Optional[date] = None
    currency: str = Field(description="ISO 4217 code, e.g. EUR, GBP, USD")
    supplier: Party
    customer: Party
    line_items: List[LineItem]
    subtotal: float = Field(description="Total net amount as printed")
    vat_amount: float = Field(description="Total VAT as printed")
    total: float = Field(description="Total amount payable as printed")
    iban: Optional[str] = Field(None, description="Supplier IBAN without spaces, if printed")

    @field_validator("currency")
    @classmethod
    def _currency(cls, v: str) -> str:
        return v.strip().upper()


# Fields the LLM reports a confidence for, and that the evaluation scores.
HEADER_FIELDS = [
    "invoice_number",
    "issue_date",
    "due_date",
    "currency",
    "supplier.name",
    "supplier.vat_id",
    "customer.name",
    "customer.vat_id",
    "subtotal",
    "vat_amount",
    "total",
    "iban",
]


def get_field(inv: dict, path: str):
    cur = inv
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur
