"""The judge: a stable cached prefix, nothing in the prompt that could bias it, and one call end to end
against a fake server (no network, no key)."""

import json

import anthropic
import httpx2
import pytest

import toolpick as tp
from toolpick.judge import EXAMPLES, FALLBACK_BETA, cost_usd, judge, judge_request, judge_system, parse_judgement
from toolpick.prompts import POLICY


def test_the_system_prompt_is_identical_on_every_call():
    a, b = judge_request("where is my order"), judge_request("register")
    assert a["system"] == b["system"]
    assert a["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert a["messages"] != b["messages"]


def test_the_judge_sees_the_policy_and_all_91_tools():
    system = judge_system()
    assert POLICY.strip() in system
    assert all(f"- {t['name']}: " in system for t in tp.tools())
    assert len(system) > 4 * 512  # comfortably above Opus 5.5's 512-token minimum for a cached prefix


def test_the_judge_is_not_told_a_model_an_intent_or_a_reply():
    kwargs = judge_request("using pro account")
    text = (kwargs["system"][0]["text"] + kwargs["messages"][0]["content"]).lower()
    assert not any(word in text for word in ("sonnet", "haiku", "deepseek", "claude", "switch_account", "assistant's reply"))


def test_the_worked_examples_are_not_messages_being_judged():
    fresh = {r.text for r in tp.requests("fresh")}
    assert not fresh & {message for message, _, _ in EXAMPLES}


def test_parse_judgement_rejects_unknown_labels():
    assert parse_judgement('{"reason": "could mean several things", "label": "unclear"}')["label"] == "unclear"
    with pytest.raises(ValueError):
        parse_judgement('{"reason": "", "label": "fine"}')


def test_one_call_end_to_end_against_a_fake_server():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"], seen["beta"] = json.loads(request.content), request.headers.get("anthropic-beta", "")
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "end_turn",
            "stop_sequence": None, "content": [{"type": "text", "text": '{"reason": "no request stated", "label": "unclear"}'}],
            "usage": {"input_tokens": 60, "cache_read_input_tokens": 10_000, "cache_creation_input_tokens": 0, "output_tokens": 100},
        })  # fmt: skip

    client = anthropic.Anthropic(api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    row = judge(client, "using pro account")
    assert (row["label"], row["cache_read_tokens"], row["model"]) == ("unclear", 10_000, "claude-opus-5-5")
    assert row["cost_usd"] == pytest.approx((60 * 4 + 10_000 * 0.2 + 100 * 20) / 1e6)
    body = seen["body"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["format"]["type"] == "json_schema" and body["fallbacks"] == "default"
    assert FALLBACK_BETA in seen["beta"] and "betas" not in body
    assert "temperature" not in body and "thinking" not in body
    assert cost_usd(type("U", (), {"input_tokens": 0, "cache_read_input_tokens": None, "cache_creation_input_tokens": 1_000_000,
                                   "output_tokens": 0})) == pytest.approx(5.0)  # fmt: skip
