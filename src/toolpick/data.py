"""The data: 600 customer messages, 90 real MCP tools, and the labels that say which tool fits."""

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

# The one tool that isn't real: added so a selector can say "no tool can do this".
HANDOFF_TOOL = {
    "name": HAND,
    "description": "None of the other tools can do what the customer is asking. Hand the conversation "
    "to a human support agent, with a short reason.",
    "inputSchema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
}


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    intent: str
    text: str


@cache
def requests() -> list[Request]:
    """The 600 messages, a fixed sample of Bitext's customer-support dataset (27 intents)."""
    return [Request(**json.loads(x)) for x in (DATA / "requests.jsonl").read_text().splitlines() if x.strip()]


@cache
def catalog() -> list[dict]:
    """90 tools from 7 official MCP servers, verbatim (see data/REPORT.md for provenance)."""
    return json.loads((DATA / "catalog.json").read_text())


def tools() -> list[dict]:
    """The 91 tools a selector chooses from: ``{name, description, inputSchema}``."""
    out = [{"name": t["qualified_name"], "description": t["description"],
            "inputSchema": t["input_schema"] or {"type": "object"}} for t in catalog()]  # fmt: skip
    return out + [HANDOFF_TOOL]


# -- labels -------------------------------------------------------------------------------------
# data/intent_map.json is the research pass: per intent, the tools that fit (acceptable), the ones
# that are a defensible first step without finishing the job (partial), and notes. The overrides
# below are the author's first-pass corrections, each with its reason. The 15 intents with no
# acceptable tool are "gap" intents: the only right answer is hand_off_to_human.

OVERRIDES: dict[str, tuple[list[str], list[str], str]] = {
    "place_order": (["shopify_ucp.search_catalog", "shopify_ucp.lookup_catalog", "shopify_ucp.get_product"], [],
                    "The task asks for the FIRST call: finding the product comes before creating a cart."),
    "delivery_period": (["shopify_ucp.get_order"], [],
                        "get_order is accepted for track_order, so it is accepted here too."),
    "contact_customer_service": ([], [HAND], "Handing to a person is a defensible reading of 'contact customer service'."),
    "track_refund": ([], ["shopify_customer_account.get_most_recent_order_status"],
                     "get_order_status is listed as partial; its no-argument twin does the same job."),
}

# The revised labels (v2) also accept the readings the LLMs made consistently and defensibly.
V2_EXTRA: dict[str, tuple[list[str], str]] = {
    "payment_issue": ([HAND], "Handing 'I have a payment problem' to a person is a fair first step."),
    "get_refund": ([FAQ], "'How do I get a refund?' can be answered from the refund policy."),
}
V2_GAP_FAQ = "On intents no tool covers, a how-to question can be answered from the store's FAQ."


@cache
def _intent_map() -> dict[str, dict]:
    return json.loads((DATA / "intent_map.json").read_text())


def labels(version: str = "v1") -> dict[str, dict]:
    """Per intent: ``gap``, ``acceptable``, ``partial``, the research ``note`` and any ``override``."""
    out = {}
    for intent, v in _intent_map().items():
        acc, part = list(v["acceptable"]), list(v.get("partial") or [])
        gap = not acc
        if gap:
            acc = [HAND]
        note = None
        if intent in OVERRIDES:
            add_acc, add_part, note = OVERRIDES[intent]
            acc += [t for t in add_acc if t not in acc]
            part = [t for t in part if t not in add_acc] + [t for t in add_part if t not in part]
        row = {"gap": gap, "acceptable": acc, "partial": part, "note": v.get("notes", ""), "override": note, "v2": None}
        if version == "v2":
            if gap and FAQ not in acc:
                acc.append(FAQ)
                row["partial"] = [t for t in part if t != FAQ]
                row["v2"] = V2_GAP_FAQ
            if intent in V2_EXTRA:
                add, why = V2_EXTRA[intent]
                acc += [t for t in add if t not in acc]
                row["partial"] = [t for t in row["partial"] if t not in add]
                row["v2"] = why
        elif version != "v1":
            raise ValueError(version)
        out[intent] = row
    return out


def tool_tokens_by_server() -> dict[str, int]:
    """Characters of tool JSON per server: where the prompt budget goes."""
    out: dict[str, int] = {}
    for t in catalog():
        s = json.dumps({"name": t["qualified_name"], "description": t["description"], "inputSchema": t["input_schema"]},
                       separators=(",", ":"))  # fmt: skip
        out[t["server"]] = out.get(t["server"], 0) + len(s)
    return out
