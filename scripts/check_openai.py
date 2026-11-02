"""Verify the OpenAI key and model. Usage: python scripts/check_openai.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from invoiceai.config import Settings  # noqa: E402

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
from openai import OpenAI  # noqa: E402

model = Settings.from_env().model
try:
    r = OpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": 'Reply with {"ok": true}'}],
                                         response_format={"type": "json_object"})
    print(f"OK  {model}: {r.choices[0].message.content}")
except Exception as exc:  # noqa: BLE001
    sys.exit(f"FAILED with {model}: {exc}")
