"""The headline numbers, recomputed from the committed runs. If a change to the answer key or the
scoring moves a published number, this fails and says which."""

import pytest

import toolpick as tp

FRESH = {r.id for r in tp.requests("fresh")}
CORRECT = {"fresh/jev": 572, "fresh/native_sonnet": 560, "fresh/native_haiku": 493, "fresh/native_deepseek": 283,
           "fresh/router_deepseek": 545}  # fmt: skip


@pytest.mark.parametrize("name", sorted(CORRECT))
def test_final_runs_cover_the_fresh_600_once_each(name):
    assert set(tp.load_run(name)) == FRESH


@pytest.mark.parametrize(("name", "right"), sorted(CORRECT.items()))
def test_correct_counts(name, right):
    assert sum(r["right"] for r in tp.scored_rows(name)) == right


def test_three_harmful_answers_in_2400():
    names = ["fresh/jev", "fresh/native_sonnet", "fresh/native_haiku", "fresh/native_deepseek"]
    assert sum(r["wrong_call"] for n in names for r in tp.scored_rows(n)) == 3


def test_jev_against_sonnet_is_a_tie():
    p = tp.paired("fresh/jev", "fresh/native_sonnet")
    assert (p["only_a"], p["only_b"]) == (37, 25)
    assert p["p"] == pytest.approx(0.162, abs=0.001)
    lo, hi = tp.paired_by_intent("fresh/jev", "fresh/native_sonnet")
    assert lo < 0 < hi


def test_the_audit_moved_the_development_scores():
    audited = tp.summarise(tp.scored_rows("jev_policy"))["correct"]
    first_pass = tp.summarise(tp.scored_rows("jev_policy", "first_pass"))["correct"]
    assert audited == pytest.approx(0.855) and first_pass < audited


def test_the_read_only_rule_only_lowers_harm():
    for name in ("fresh/jev", "fresh/native_sonnet", "fresh/native_deepseek"):
        with_rule = sum(r["wrong_call"] for r in tp.scored_rows(name))
        without = sum(r["wrong_call"] for r in tp.scored_rows(name, "audited_strict"))
        assert without >= with_rule


def test_the_judge_labelled_every_message_an_agent_asked_about():
    asked = {r["id"] for n in ("fresh/native_sonnet", "fresh/native_haiku", "fresh/native_deepseek")
             for r in tp.scored_rows(n) if r["outcome"] == "asked" and r["gap"]}  # fmt: skip
    assert asked == set(tp.judgements()) and len(asked) == 180
    assert sum(j["label"] == "unclear" for j in tp.judgements().values()) == 29


@pytest.mark.parametrize(("name", "right"), [("fresh/jev", 572), ("fresh/native_sonnet", 568), ("fresh/native_haiku", 516),
                                             ("fresh/native_deepseek", 312)])  # fmt: skip
def test_correct_counts_when_asking_about_an_unclear_message_counts(name, right):
    assert sum(tp.right_if_asking_when_unclear(r) for r in tp.scored_rows(name)) == right
