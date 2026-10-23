"""Why does Jev hand off too often, and can it be fixed? Variants of the Jev question, on a split.

    uv run python scripts/jev_experiments.py --split dev            # every variant on the 201 dev messages
    uv run python scripts/jev_experiments.py --split held --only X  # one variant on the 399 held-out messages

Same development/held-out split as the GEPA experiment (scripts/optimise_prompt.py). Each variant
changes one thing from the `policy` setup (the policy instructions, the 91 tools in catalogue order,
hand_off_to_human last):

    baseline      the policy setup, re-run (checks run-to-run noise)
    handoff_first hand_off_to_human moved to the top of the option list
    shuffled      options in a random order (seed 0)
    handoff_desc  a narrower hand-off description: what it is for, not "none of the others"
    no_handoff_line  the policy instructions without the "pick hand_off_to_human if..." sentence
    no_distractors   GitHub and Notion tools removed (a customer assistant wouldn't load them)
    descriptions  the 91 real descriptions, plus one customer-language line on the six tools Jev
                  misses (catalogue search, invoices, order status, returns): what customers say
    combined      handoff_desc + no_handoff_line + descriptions

Rows go to runs/experiments/jev_<variant>_<split>.jsonl with the full response. Jev costs about
US$0.06 per 200 messages.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parents[1] / ".env")  # when this folder sits inside the course repo, its .env works too
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from toolpick import HAND, RUNS, requests, tools
from toolpick.jev import Jev
from toolpick.prompts import JEV_INSTRUCTIONS, POLICY

OUT = RUNS / "experiments"

HANDOFF_NARROW = ("Transfer the conversation to a human support agent. Use this for requests that need a person "
                  "and that no other tool supports: complaints, account changes (sign-up, password, profile, "
                  "deleting or switching accounts), cancelling or editing a placed order, newsletters and reviews.")  # fmt: skip

# One line in customers' words, appended to the real description. Only for tools Jev misses.
CUSTOMER_WORDS = {
    "shopify_ucp.search_catalog": "Customers say: 'I want to buy an article', 'help me order some items', 'I need to shop for a product'.",
    "shopify_ucp.create_cart": "Customers say: 'I want to buy this', 'add it to my cart'.",
    "stripe.list_invoices": "Customers say: 'where is my invoice', 'download my bill #123', 'check my bills'.",
    "shopify_customer_account.get_order_status": "Customers say: 'status of order ORD-123', 'check bill/invoice #123', 'any news on my refund for order ...'.",
    "shopify_customer_account.get_most_recent_order_status": "Customers say: 'where is my order', 'any news on my refund', 'I'm waiting for my reimbursement'.",
    "shopify_customer_account.request_return": "Customers say: 'I want my money back', 'how do I get a refund', 'I want a compensation'.",
}


def variant_setup(name: str) -> tuple[list[dict], str]:
    tl = [dict(t) for t in tools()]
    instructions = POLICY + JEV_INSTRUCTIONS
    if name == "handoff_first":
        tl = [t for t in tl if t["name"] == HAND] + [t for t in tl if t["name"] != HAND]
    elif name == "shuffled":
        random.Random(0).shuffle(tl)
    elif name == "handoff_desc":
        for t in tl:
            if t["name"] == HAND:
                t["description"] = HANDOFF_NARROW
    elif name == "no_handoff_line":
        instructions = POLICY + JEV_INSTRUCTIONS.replace(" Pick hand_off_to_human if no tool can do what the customer asks.", "")
    elif name == "no_distractors":
        tl = [t for t in tl if not t["name"].startswith(("github.", "notion."))]
    elif name == "descriptions":
        for t in tl:
            if t["name"] in CUSTOMER_WORDS:
                t["description"] = t["description"].rstrip() + "\n" + CUSTOMER_WORDS[t["name"]]
    elif name == "combined":  # narrower hand-off, no hand-off line, customer words
        for t in tl:
            if t["name"] == HAND:
                t["description"] = HANDOFF_NARROW
            if t["name"] in CUSTOMER_WORDS:
                t["description"] = t["description"].rstrip() + "\n" + CUSTOMER_WORDS[t["name"]]
        instructions = POLICY + JEV_INSTRUCTIONS.replace(" Pick hand_off_to_human if no tool can do what the customer asks.", "")
    elif name != "baseline":
        raise ValueError(name)
    return tl, instructions


VARIANTS = ["baseline", "handoff_first", "shuffled", "handoff_desc", "no_handoff_line", "no_distractors", "descriptions", "combined"]


def split_ids(split: str) -> set[str]:
    g = json.loads((ROOT / "data" / "gepa_deepseek.json").read_text())
    return set(g["dev_ids"] if split == "dev" else g["held_ids"])


def run(variant: str, split: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"jev_{variant}_{split}.jsonl"
    done = {json.loads(x)["id"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    ids = split_ids(split)
    reqs = [r for r in requests() if r.id in ids and r.id not in done]
    tl, instructions = variant_setup(variant)
    criteria = {t["name"]: t["description"] for t in tl}
    jev = Jev()
    lock = threading.Lock()

    def one(r):
        ts = datetime.now(tz=UTC).isoformat()
        body = jev.choose(r.text, instructions, criteria)
        a = body["answers"]["tool"]
        row = {"id": r.id, "pick": a["choice"], "kind": "tool_call", "confidence": a["confidence"],
               "probabilities": a["probabilities"], "latency_s": round(body["latency_s"], 4),
               "input_tokens": body["usage"]["input_tokens"], "cost_usd": body["cost_usd"], "ts": ts,
               "model": body.get("model"), "variant": variant}  # fmt: skip
        with lock, out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, reqs))
    print(f"{variant}/{split}: {len(reqs)} calls", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "held"], default="dev")
    ap.add_argument("--only", nargs="*", choices=VARIANTS)
    args = ap.parse_args()
    for v in args.only or VARIANTS:
        run(v, args.split)


if __name__ == "__main__":
    main()
