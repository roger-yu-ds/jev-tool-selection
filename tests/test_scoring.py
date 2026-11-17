"""Outcomes, run names and the paired statistics."""

import pytest
from scipy.stats import binomtest

import toolpick as tp

LAB = {"gap": False, "acceptable": ["a.right"], "partial": ["a.first_step"], "ask_ok": False}
GAP = {"gap": True, "acceptable": [tp.HAND], "partial": ["a.first_step"], "ask_ok": False}


@pytest.mark.parametrize(
    ("pick", "lab", "kind", "expected"),
    [
        ("a.right", LAB, "tool_call", "right"),
        ("a.first_step", LAB, "tool_call", "defensible"),
        (tp.HAND, LAB, "tool_call", "handed_off"),
        ("b.other", LAB, "tool_call", "wrong"),
        (None, LAB, "text", "asked"),
        (None, LAB | {"ask_ok": True}, "text", "right"),
        (tp.ASK, LAB | {"ask_ok": True}, "tool_call", "right"),
        (tp.HAND, GAP, "tool_call", "right"),
        ("b.other", GAP, "tool_call", "bluffed"),
        ("a.first_step", GAP, "tool_call", "defensible"),
    ],
)
def test_outcome(pick, lab, kind, expected):
    assert tp.outcome(pick, lab, kind) == expected


def test_a_batch_is_right_only_if_no_call_is_wrong():
    assert tp.outcome("a.right", LAB, calls=["a.right", "a.first_step"]) == "right"
    assert tp.outcome("a.right", LAB, calls=["a.right", "b.other"]) == "wrong"
    assert tp.outcome("a.first_step", LAB, calls=["a.first_step", "a.first_step"]) == "defensible"


def test_every_outcome_has_a_verdict():
    assert set(tp.VERDICT) == set(tp.OUTCOMES)
    assert set(tp.VERDICT.values()) == set(tp.VERDICTS)


def test_mcnemar_matches_the_exact_binomial_test():
    a = [True] * 37 + [False] * 25 + [True] * 500
    b = [False] * 37 + [True] * 25 + [True] * 500
    m = tp.mcnemar(a, b)
    assert (m["only_a"], m["only_b"], m["n_pairs"]) == (37, 25, 562)
    assert m["p"] == pytest.approx(binomtest(25, 62, 0.5).pvalue)
    assert tp.mcnemar(b, a)["p"] == pytest.approx(m["p"])
    assert tp.mcnemar(a, a)["p"] == 1.0


def test_wilson_interval_contains_the_estimate_and_stays_in_range():
    lo, hi = tp.wilson(572, 600)
    assert 0 < lo < 572 / 600 < hi < 1
    assert tp.wilson(0, 10)[0] == 0.0 and tp.wilson(10, 10)[1] == 1.0
    assert tp.wilson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("jev", {"mode": "jev", "model": "jev", "prompt": "base", "sample": "first"}),
        ("native_sonnet_policy", {"mode": "native", "model": "sonnet", "prompt": "policy", "sample": "first"}),
        ("router_deepseek_gepa", {"mode": "router", "model": "deepseek", "prompt": "gepa", "sample": "first"}),
        ("fresh/native_haiku", {"mode": "native", "model": "haiku", "prompt": "final", "sample": "fresh"}),
        ("fresh/jev_ask", {"mode": "jev", "model": "jev", "prompt": "final_ask", "sample": "fresh"}),
    ],
)
def test_parse_run_name(name, expected):
    assert tp.parse_run_name(name) == expected
