import json

from invoiceai.synth.generate import generate


def test_generator_is_deterministic_and_labelled(tmp_path):
    a = generate(n=25, seed=3, out_dir=tmp_path / "a", gt_path=tmp_path / "a.json")
    b = generate(n=25, seed=3, out_dir=tmp_path / "b", gt_path=tmp_path / "b.json")
    strip = lambda t: {k: v["invoice"] for k, v in t.items()}  # noqa: E731
    assert strip(a) == strip(b)
    assert len(list((tmp_path / "a").glob("*.pdf"))) == len(a)
    planted = {c for v in a.values() for c in v["expected_issues"]}
    assert {"DUPLICATE", "MATH_TOTAL", "SCANNED"} <= planted
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")).keys() == a.keys()
