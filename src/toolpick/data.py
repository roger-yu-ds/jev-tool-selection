"""The data: customer messages, 90 real MCP tools, and the labels that say which tool fits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"

HAND = "hand_off_to_human"
FAQ = "shopify_storefront.search_shop_policies_and_faqs"
ORDER_LOOKUPS = ["shopify_customer_account.get_order_status", "shopify_customer_account.get_most_recent_order_status",
                 "shopify_ucp.get_order"]  # fmt: skip

# The one tool that isn't real: added so a selector can say "no tool can do this".
HANDOFF_TOOL = {
    "name": HAND,
    "description": "None of the other tools can do what the customer is asking. Hand the conversation "
    "to a human support agent, with a short reason.",
    "inputSchema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
}

# Jev's equivalent of a native agent replying with a question (`tools(ask=True)`). In production the
# agent reads it and asks the customer; it is scored exactly like an LLM's text reply.
ASK = "need_more_information"
ASK_TOOL = {
    "name": ASK,
    "description": "The customer wants something one of the tools can do, but hasn't given the details it "
    "needs (for example which order, which item, or what went wrong). Ask the customer for them first.",
    "inputSchema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
}

# One line in customers' words, appended to six real descriptions (`tools(customer_words=True)`).
# Real descriptions are written for developers ("This tool will fetch a list of Invoices from
# Stripe"); customers say "download my bill". Written while looking at the first sample's misses,
# so the fresh sample (data/requests_fresh.jsonl) is the clean test of them.
CUSTOMER_WORDS = {
    "shopify_ucp.search_catalog": "Customers say: 'I want to buy an article', 'help me order some items', 'I need to shop for a product'.",
    "shopify_ucp.create_cart": "Customers say: 'I want to buy this', 'add it to my cart'.",
    "stripe.list_invoices": "Customers say: 'where is my invoice', 'download my bill #123', 'check my bills'.",
    "shopify_customer_account.get_order_status": "Customers say: 'status of order ORD-123', 'check bill/invoice #123', 'any news on my refund for order ...'.",
    "shopify_customer_account.get_most_recent_order_status": "Customers say: 'where is my order', 'any news on my refund', 'I'm waiting for my reimbursement'.",
    "shopify_customer_account.request_return": "Customers say: 'I want my money back', 'how do I get a refund', 'I want a compensation'.",
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    intent: str
    text: str


SAMPLES = {"first": "requests.jsonl", "fresh": "requests_fresh.jsonl"}


@cache
def requests(sample: str = "first") -> list[Request]:
    """600 Bitext messages. ``first`` is the sample the study was developed on; ``fresh`` is a
    disjoint, varied 600 drawn afterwards (scripts/make_fresh_sample.py) for the final run."""
    path = DATA / SAMPLES[sample]
    return [Request(**json.loads(x)) for x in path.read_text().splitlines() if x.strip()]


@cache
def all_requests_by_id() -> dict[str, Request]:
    return {r.id: r for s in SAMPLES if (DATA / SAMPLES[s]).exists() for r in requests(s)}


@cache
def catalog() -> list[dict]:
    """90 tools from 7 official MCP servers, verbatim (see data/REPORT.md for provenance)."""
    return json.loads((DATA / "catalog.json").read_text())


def tools(customer_words: bool = False, ask: bool = False, params: bool = False) -> list[dict]:
    """The 91 tools a selector chooses from: ``{name, description, inputSchema}`` (92 with ``ask``).
    ``params`` appends a short "Inputs:" summary of the input schema to each description: for Jev,
    which reads only names and descriptions (the LLMs get the full schema anyway)."""
    out = []
    for t in catalog():
        desc = t["description"]
        if customer_words and t["qualified_name"] in CUSTOMER_WORDS:
            desc = desc.rstrip() + "\n" + CUSTOMER_WORDS[t["qualified_name"]]
        out.append({"name": t["qualified_name"], "description": desc, "inputSchema": t["input_schema"] or {"type": "object"}})
    out += [HANDOFF_TOOL] + ([ASK_TOOL] if ask else [])
    if params:
        out = [{**t, "description": t["description"].rstrip() + "\n" + inputs_text(t["inputSchema"])} for t in out]
    return out


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def inputs_text(schema: dict, desc_chars: int = 90) -> str:
    """One line per tool: each top-level input with its type, whether it is required, a clipped
    description, and its fields or allowed values. UCP's ``meta`` is left out: the harness fills it in."""
    props = {k: v for k, v in (schema.get("properties") or {}).items() if k != "meta"}
    if not props:
        return "Inputs: none."
    required = set(schema.get("required") or [])
    parts = []
    for name, p in props.items():
        kind = p.get("type", "object")
        items = p.get("items") or {}
        if kind == "array":
            kind = f"list of {items.get('type', 'object')}"
        bits = [kind, "required" if name in required else "optional"]
        line = f"{name} ({', '.join(bits)})"
        if p.get("description"):
            line += ": " + _clip(p["description"], desc_chars).rstrip(".")
        enum = p.get("enum") or items.get("enum")
        fields = (p.get("properties") or items.get("properties") or {}).keys()
        if enum:
            line += " [one of " + ", ".join(map(str, enum[:10])) + "]"
        elif fields:
            line += " [fields: " + ", ".join(list(fields)[:8]) + "]"
        parts.append(line)
    return "Inputs: " + "; ".join(parts) + "."


# -- labels -------------------------------------------------------------------------------------
# data/intent_map.json is the research pass: per intent, the tools that fit (acceptable), the ones
# that are a sensible first step without finishing the job (partial), and notes. Intents with no
# acceptable tool are "gap" intents: the only right answer is hand_off_to_human.
#
# Two versions. `first_pass` is the research map plus four early overrides. `audited` (the default)
# is the result of reviewing every failure by hand: each change below has its reason, and every
# reported number uses it.

FIRST_PASS_OVERRIDES: dict[str, tuple[list[str], list[str], str]] = {
    "place_order": (["shopify_ucp.search_catalog", "shopify_ucp.lookup_catalog", "shopify_ucp.get_product"], [],
                    "The task asks for the FIRST call: finding the product comes before creating a cart."),
    "delivery_period": (["shopify_ucp.get_order"], [], "get_order is accepted for track_order, so it is accepted here too."),
    "contact_customer_service": ([], [HAND], "Handing to a person is a defensible reading of 'contact customer service'."),
    "track_refund": ([], ["shopify_customer_account.get_most_recent_order_status"],
                     "get_order_status is listed as partial; its no-argument twin does the same job."),
}  # fmt: skip

# (intent, acceptable to add, partial to add, gap?, reason). gap=None leaves it unchanged.
AUDIT: list[tuple[str, list[str], list[str], bool | None, str]] = [
    ("track_refund", [*ORDER_LOOKUPS, "stripe.list_payment_intents"], ["shopify_customer_account.get_store_credit_balances"], False,
     "Looking up the order (or the payment) shows whether the refund went through; the routing policy itself says to."),
    ("check_invoice", [*ORDER_LOOKUPS, "stripe.list_customers"], [], None,
     "For a shopper the bill is usually the order receipt; Stripe needs the customer before it can list their invoices."),
    ("get_invoice", [*ORDER_LOOKUPS, "stripe.list_customers"], [], None,
     "Same as check_invoice."),
    ("get_refund", [*ORDER_LOOKUPS, "stripe.list_payment_intents"], [], None,
     "Both refund tools need an order or payment ID first, so the lookup is the right first call."),
    ("contact_customer_service", [HAND], [], None, "'I want to call customer support': a person is customer service."),
    ("payment_issue", [HAND], [], None, "Someone reporting a payment problem should reach a person."),
    ("cancel_order", [], ORDER_LOOKUPS, None, "Checking the order first is sensible, though nothing can cancel it."),
    ("change_order", [], ORDER_LOOKUPS, None, "Same as cancel_order."),
    ("change_shipping_address", [], ORDER_LOOKUPS, None, "Same as cancel_order."),
    ("review", [], ["intercom.add_internal_note"], None, "Recording the feedback as a note is a reasonable first step."),
]
# Reading the customer's own records or the store's public information is at worst a wasted step, so
# in the audited key these are benign on every intent where they aren't correct. Harmful means a write
# that wasn't asked for, or reaching into the wrong system: GitHub, Notion, the helpdesk's internal
# conversations and contacts, Stripe's developer docs or the merchant's own balance and catalogue.
READ_ONLY_BENIGN = [
    FAQ, "shopify_ucp.get_checkout", "shopify_ucp.get_cart", "shopify_ucp.get_order", "shopify_ucp.search_catalog",
    "shopify_ucp.lookup_catalog", "shopify_ucp.get_product", "shopify_ucp.search_locations", "shopify_ucp.lookup_locations",
    *ORDER_LOOKUPS[:2], "shopify_customer_account.get_store_credit_balances",
    "stripe.list_customers", "stripe.list_invoices", "stripe.list_payment_intents", "stripe.list_subscriptions", "stripe.list_disputes",
    "intercom.search_articles", "intercom.list_articles", "intercom.get_article",
]  # fmt: skip
AUDIT_READ_ONLY = ("Reading the customer's own records or the store's public information (including the FAQ) is at worst "
                   "a wasted step: benign, never harmful.")  # fmt: skip
# A clarifying question is right when every acceptable tool needs something the message doesn't give.
ASK_OK = {
    "place_order": "Searching the catalogue needs to know what the customer wants to buy.",
    "get_refund": "Starting a return needs the order and the items.",
    "payment_issue": "Investigating needs to know what went wrong.",
}


@cache
def _intent_map() -> dict[str, dict]:
    return json.loads((DATA / "intent_map.json").read_text())


def labels(version: str = "audited") -> dict[str, dict]:
    """Per intent: ``gap``, ``acceptable``, ``partial``, ``ask_ok``, the research ``note`` and the
    ``changes`` made to it (with reasons). ``audited_strict`` is the audited key without the
    read-only rule (READ_ONLY_BENIGN), to measure what that rule does to the harmful counts."""
    if version not in ("audited", "audited_strict", "first_pass"):
        raise ValueError(version)
    out = {}
    for intent, v in _intent_map().items():
        acc, part = list(v["acceptable"]), list(v.get("partial") or [])
        gap = not acc
        if gap:
            acc = [HAND]
        changes = []
        if intent in FIRST_PASS_OVERRIDES:
            add_acc, add_part, why = FIRST_PASS_OVERRIDES[intent]
            acc += [t for t in add_acc if t not in acc]
            part = [t for t in part if t not in add_acc] + [t for t in add_part if t not in part]
            changes.append(why)
        ask_ok = False
        if version != "first_pass":
            for a_intent, add_acc, add_part, a_gap, why in AUDIT:
                if a_intent != intent:
                    continue
                if a_gap is False:
                    gap, acc = False, [t for t in acc if t != HAND]
                acc += [t for t in add_acc if t not in acc]
                part += [t for t in add_part if t not in part]
                changes.append(why)
            if version == "audited":
                part += [t for t in READ_ONLY_BENIGN if t not in acc and t not in part]
            ask_ok = intent in ASK_OK
            if ask_ok:
                changes.append(ASK_OK[intent])
        part = [t for t in part if t not in acc]
        out[intent] = {"gap": gap, "acceptable": acc, "partial": part, "ask_ok": ask_ok, "note": v.get("notes", ""),
                       "changes": changes}  # fmt: skip
    return out


def tool_tokens_by_server() -> dict[str, int]:
    """Characters of tool JSON per server: where the prompt budget goes."""
    out: dict[str, int] = {}
    for t in catalog():
        s = json.dumps({"name": t["qualified_name"], "description": t["description"], "inputSchema": t["input_schema"]},
                       separators=(",", ":"))  # fmt: skip
        out[t["server"]] = out.get(t["server"], 0) + len(s)
    return out
