"""Are the arguments right? A free, rule-based check of the values an agent passed to a tool.

There is no answer key for argument values: most messages carry none ("where is my order"), and
free text such as an FAQ query has many right wordings. What can be checked without judgement:

- **valid**: the arguments match the tool's input schema (required fields, types, enums). UCP's
  `meta` (the agent's profile URL) is supplied by the harness, not the model, so it is filled in
  before checking and never scored.
- **grounded**: every identifier (order number, order or product ID, payment intent, customer,
  email) comes from the message. An order number the customer never gave is made up.
- **carried**: when the message has an order number and the tool has an order field, the number
  is in it.
- **amount**: Stripe refund amounts are in cents, so "$181" must be 18100.
- **policy_type**: when the FAQ search is narrowed to policy types, they fit the request
  (a soft check, reported but not part of `ok`).

`ok` is valid and grounded and not failing carried or amount. Free text (queries, hand-off
reasons, notes) is not scored beyond being non-empty.
"""

from __future__ import annotations

import re
from functools import cache

from jsonschema import validators

from .data import FAQ, HAND, tools

HARNESS_META = {"ucp-agent": {"profile": "https://agent.example.com/profile.json"}}
ID_KEYS = {"order_number", "order_id", "id", "ids", "payment_intent", "customer", "email", "line_item_id", "product_id",
           "variant_id", "invoice"}  # fmt: skip
ORDER_FIELDS = {"shopify_customer_account.get_order_status": ["order_number"], "shopify_ucp.get_order": ["id"],
                "shopify_customer_account.request_return": ["order_number", "order_id"]}  # fmt: skip
FREE_TEXT = {"query", "reason", "context", "customer_note", "body"}
ORDER_RE = re.compile(r"(?:ORD-|#)\s?(\d{3,})", re.IGNORECASE)
AMOUNT_RE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
POLICY_FOR = {
    "check_refund_policy": {"refund"}, "get_refund": {"refund"}, "track_refund": {"refund"},
    "check_cancellation_fee": {"refund", "terms_of_sale", "terms_of_service"},
    "delivery_options": {"shipping"}, "delivery_period": {"shipping"},
    "check_payment_methods": {"terms_of_sale", "terms_of_service"},
    "contact_customer_service": {"contact"}, "contact_human_agent": {"contact"},
}  # fmt: skip


@cache
def _schemas() -> dict[str, dict]:
    return {t["name"]: t["inputSchema"] for t in tools(customer_words=True)}


def _walk(value, key: str = ""):
    """(key, leaf value) pairs, lists flattened under their parent key."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v, key)
    else:
        yield key, value


def _grounded(value, text: str) -> bool:
    v = str(value).strip()
    if not v:
        return False
    if v.lower() in text.lower():
        return True
    digits = re.findall(r"\d{3,}", v)
    return bool(digits) and any(d in text for d in digits)


def check(tool: str, args: dict | None, text: str, intent: str) -> dict:
    """Rule-based checks of one tool call's arguments against the message it answers."""
    schema = _schemas().get(tool) or {"type": "object"}
    args = dict(args or {})
    if "_unparsed" in args:
        return {"valid": False, "grounded": True, "carried": None, "amount": None, "policy_type": None, "ok": False,
                "problems": ["arguments are not valid JSON"]}  # fmt: skip
    if "meta" in (schema.get("properties") or {}):
        args["meta"] = HARNESS_META
    problems = []
    validator = validators.validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
    problems += [f"schema: {'/'.join(map(str, e.path)) or '(top)'}: {e.message[:120]}" for e in errors]
    for k, v in _walk({k: v for k, v in args.items() if k != "meta"}):
        if k in FREE_TEXT and isinstance(v, str) and not v.strip():
            problems.append(f"empty {k}")
    valid = not problems

    made_up = [f"{k}={v!r}" for k, v in _walk({k: v for k, v in args.items() if k != "meta"})
               if k in ID_KEYS and tool != HAND and not _grounded(v, text)]  # fmt: skip
    problems += [f"not in the message: {m}" for m in made_up]

    carried = None
    order_ids = ORDER_RE.findall(text)
    if order_ids and tool in ORDER_FIELDS:
        given = " ".join(str(args.get(f, "")) for f in ORDER_FIELDS[tool])
        carried = any(o in given for o in order_ids)
        if not carried:
            problems.append(f"order number {order_ids[0]} not passed")

    amount = None
    if tool == "stripe.create_refund" and "amount" in args:
        dollars = AMOUNT_RE.findall(text)
        amount = bool(dollars) and any(round(float(d) * 100) == args["amount"] for d in dollars)
        if not amount:
            problems.append(f"amount {args['amount']} (cents) does not match {dollars or 'nothing in the message'}")

    policy_type = None
    if tool == FAQ and args.get("policy_types") and intent in POLICY_FOR:
        policy_type = bool(set(args["policy_types"]) & POLICY_FOR[intent])

    ok = valid and not made_up and carried is not False and amount is not False
    return {"valid": valid, "grounded": not made_up, "carried": carried, "amount": amount, "policy_type": policy_type,
            "ok": ok, "problems": problems}  # fmt: skip
