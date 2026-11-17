"""The data the study rests on: two disjoint samples, 91 tools, and an answer key that only names real tools."""

import toolpick as tp
from toolpick.data import ASK, CUSTOMER_WORDS


def test_samples_are_600_each_and_disjoint():
    first, fresh = tp.requests("first"), tp.requests("fresh")
    assert len(first) == len(fresh) == 600
    assert len({r.id for r in first} | {r.id for r in fresh}) == 1200
    assert not {r.text for r in first} & {r.text for r in fresh}


def test_91_tools_with_unique_names():
    names = [t["name"] for t in tp.tools()]
    assert len(tp.catalog()) == 90
    assert len(names) == len(set(names)) == 91
    assert tp.HAND in names


def test_ask_option_adds_one_tool():
    assert [t["name"] for t in tp.tools(ask=True)][-1] == ASK
    assert len(tp.tools(ask=True)) == 92


def test_customer_words_change_only_six_descriptions():
    plain = {t["name"]: t["description"] for t in tp.tools()}
    worded = {t["name"]: t["description"] for t in tp.tools(customer_words=True)}
    changed = {k for k in plain if plain[k] != worded[k]}
    assert changed == set(CUSTOMER_WORDS) and len(changed) == 6


def test_answer_key_names_only_real_tools():
    names = {t["name"] for t in tp.tools()}
    for version in ("audited", "audited_strict", "first_pass"):
        for intent, lab in tp.labels(version).items():
            assert lab["acceptable"], intent
            assert set(lab["acceptable"]) | set(lab["partial"]) <= names, intent
            assert not set(lab["acceptable"]) & set(lab["partial"]), intent


def test_every_message_has_a_label_and_gap_intents_hand_off():
    lab = tp.labels()
    assert {r.intent for s in ("first", "fresh") for r in tp.requests(s)} == set(lab)
    gaps = [k for k, v in lab.items() if v["gap"]]
    assert len(gaps) == 14
    assert all(lab[k]["acceptable"] == [tp.HAND] for k in gaps)
