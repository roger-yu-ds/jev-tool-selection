"""An LLM judge for one question: is this customer message clear enough to act on?

The answer key marks a question back as a miss on every request no tool can handle. Some of those
messages are too vague to act on ("using pro account"), and there a question is the sensible reply.
The judge (Claude Opus 5.5) reads one customer message, with the same policy and tool list the agents
had, and labels it:

    unclear   no reading stands out, so a question back is justified
    no_tool   the customer wants something none of the tools can do: hand off
    tool      the customer wants something a tool can do

The judge sees the message only: not the agent's reply, not which model wrote it, not Bitext's
intent. A first version showed it the reply as well (runs/judge/calibration_replies.jsonl). On 15
messages judged twice, once beside each of two agents' replies, the clarity label changed with the
reply on 2, so the reply was swaying a label that should depend on the message alone. Judging the
message alone removes that, and with it any chance of a Claude judge favouring a Claude reply.

Caching: the system prompt (instructions, policy, worked examples, 91 tool descriptions; about 10,000
tokens) is the same bytes on every call and carries the cache marker. Only the short user message
varies, and it comes after the marker.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from .data import tools
from .prompts import POLICY

JUDGE_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LABELS = ["unclear", "no_tool", "tool"]
PRICE = {"input": 4.00, "cache_read": 0.20, "cache_write": 5.00, "output": 20.00}  # US$ per million tokens, 2026-11-17

SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "label": {"type": "string", "enum": LABELS}},
    "required": ["reason", "label"],
    "additionalProperties": False,
}

# Worked examples that set the standard, all from the development sample (data/requests.jsonl), so
# none is a message being judged. They are a judgement call: edit them to move the line.
EXAMPLES = [
    ("how to use the premium profile", "unclear", "It doesn't say what the customer wants to do with the profile."),
    ("add something to order", "unclear",
     "It could mean adding to a cart, which a tool can do, or to a placed order, which none can. Neither reading stands out."),
    ("new Platinum acount", "no_tool", "Short, but the natural reading is opening an account, and no tool does that."),
    ("update standard account", "no_tool", "An account change. A subscription change is possible but less natural."),
    ("can I talk to a person?", "no_tool", "The customer wants a human."),
    ("assistance cancelling purchase ORD-451497", "no_tool", "Cancelling a placed order, which no tool can do."),
    ("can I sign up to the company newsletter?", "no_tool", "A newsletter sign-up, which no tool can do."),
    ("help seeing order ORD-402373 status", "tool", "An order-status tool does exactly this."),
]

INSTRUCTIONS = """You are reviewing messages sent to the customer-support assistant of an online store. The store \
runs on Shopify, takes payments through Stripe and uses Intercom as its helpdesk; its staff also use GitHub and Notion.

The assistant gets one message from a customer, with no earlier conversation and no knowledge of who the \
customer is. It can call any of the tools listed below, including hand_off_to_human, or reply with a question. \
It works under this routing policy:

{policy}
Your job is to say how clear the customer's message is. The messages are short and full of typos; read through \
the typos to the meaning. Give one label:

- "unclear": the message doesn't say what the customer wants, or it has several readings and none stands out. \
A question back would be the sound next step.
- "no_tool": the customer wants something none of the tools can do (rule 3 of the policy). The assistant should \
hand off.
- "tool": the customer wants something one of the tools can do.

If one reading is clearly the most natural, judge by that reading, even when other readings are possible. \
Answer "unclear" only when no reading stands out. A short message is not unclear for being short.

Give the reason in one sentence, then the label.

Examples:

{examples}

The tools the assistant has (name: description):

{tools}"""


def judge_system() -> str:
    """The judge's system prompt: instructions, policy, examples and the tool list. Stable bytes."""
    lines = [f"- {t['name']}: {' '.join(t['description'].split())}" for t in tools(customer_words=True)]
    examples = [f'Message: "{m}"\n{json.dumps({"reason": why, "label": label})}' for m, label, why in EXAMPLES]
    return INSTRUCTIONS.format(policy=POLICY, examples="\n\n".join(examples), tools="\n".join(lines))


def judge_request(message: str) -> dict:
    """Arguments for `client.beta.messages.create`. Everything before the user message is cached."""
    return {
        "model": JUDGE_MODEL,
        "max_tokens": 4000,  # room for the model's thinking; the answer itself is under 100 tokens
        "system": [{"type": "text", "text": judge_system(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"Customer's message:\n{message}"}],
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",  # if Opus 5.5 declines a message, another model answers; `model` in the row says which
    }


def parse_judgement(text: str) -> dict:
    """The judge's JSON answer, checked against the allowed labels."""
    out = json.loads(text)
    if out.get("label") not in LABELS:
        raise ValueError(f"unexpected label in {text[:200]!r}")
    return {"label": out["label"], "reason": out.get("reason", "")}


def cost_usd(usage) -> float:
    """Billed cost of one call from its usage (the Anthropic API reports tokens, not dollars)."""
    return (usage.input_tokens * PRICE["input"] + (usage.cache_read_input_tokens or 0) * PRICE["cache_read"]
            + (usage.cache_creation_input_tokens or 0) * PRICE["cache_write"] + usage.output_tokens * PRICE["output"]) / 1e6  # fmt: skip


def judge(client, message: str) -> dict:
    """One call to the judge: the label, the usage and the full response. `client` is `anthropic.Anthropic()`."""
    t0, ts = time.perf_counter(), datetime.now(tz=UTC).isoformat()
    resp = client.beta.messages.create(**judge_request(message))
    if resp.stop_reason == "refusal":
        raise ValueError(f"the judge declined: {resp.stop_details}")
    text = next(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    return {**parse_judgement(text), "model": resp.model, "input_tokens": u.input_tokens,
            "cache_read_tokens": u.cache_read_input_tokens or 0, "cache_write_tokens": u.cache_creation_input_tokens or 0,
            "output_tokens": u.output_tokens, "cost_usd": cost_usd(u), "latency_s": round(time.perf_counter() - t0, 3),
            "ts": ts, "response": resp.to_dict()}  # fmt: skip
