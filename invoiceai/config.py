"""Settings, paths and model pricing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = PROJECT_ROOT / "samples" / "invoices"
GROUND_TRUTH_PATH = PROJECT_ROOT / "samples" / "ground_truth.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

# The company receiving the invoices (accounts-payable side). Used to tell
# supplier and customer apart and to check that invoices are addressed to us.
OUR_COMPANY = {"name": "Helios Analytics S.A.", "vat_id": "EL801234560", "country": "GR"}


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model: str
    price_input_per_1m: float   # USD per 1M input tokens
    price_output_per_1m: float  # USD per 1M output tokens
    auto_approve_confidence: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.getenv("INVOICEAI_DB_PATH", str(DATA_DIR / "invoices.db"))),
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            # gpt-5.4-mini list prices (OpenAI docs, Nov 2026). Override in .env for other models.
            price_input_per_1m=float(os.getenv("PRICE_INPUT_PER_1M", "0.75")),
            price_output_per_1m=float(os.getenv("PRICE_OUTPUT_PER_1M", "4.50")),
            auto_approve_confidence=float(os.getenv("AUTO_APPROVE_CONFIDENCE", "0.85")),
            max_retries=int(os.getenv("INVOICEAI_MAX_RETRIES", "2")),
        )

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens / 1e6 * self.price_input_per_1m + completion_tokens / 1e6 * self.price_output_per_1m
