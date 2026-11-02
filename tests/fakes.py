"""Test doubles: an 'oracle' LLM that returns the ground truth for the current file."""
import json

from invoiceai.config import GROUND_TRUTH_PATH
from invoiceai.llm import LLMResponse
from invoiceai.schema import HEADER_FIELDS


class OracleLLM:
    model = "oracle"

    def __init__(self, truth_path=GROUND_TRUTH_PATH, corrupt=None):
        self.truth = json.loads(open(truth_path, encoding="utf-8").read())
        self.current = None
        self.corrupt = corrupt or {}  # file -> function(dict) -> dict
        self.calls = 0

    def set_file(self, name):
        self.current = name

    def complete_json(self, messages):
        self.calls += 1
        data = json.loads(json.dumps(self.truth[self.current]["invoice"]))
        if self.current in self.corrupt:
            data = self.corrupt[self.current](data)
        data["confidence"] = {f: 0.97 for f in HEADER_FIELDS}
        return LLMResponse(data=data, raw=json.dumps(data), prompt_tokens=1500, completion_tokens=400)
