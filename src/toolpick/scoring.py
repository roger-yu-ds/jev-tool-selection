"""Score a run against the labels; paired statistics between runs.

A run is one selector under one condition, stored as ``runs/<name>.jsonl`` with one row per
request. Names: ``jev``, ``router_<model>`` (the LLM told to pick a tool by name) and
``native_<model>`` (the LLM with the tools in the API's ``tools`` parameter), each with a
``_policy`` variant where the routing policy was added to the prompt. ``fresh/<name>`` runs are the
final configuration (policy prompt, customer-worded tool descriptions) on the fresh 600 messages.

Outcomes per request, from the pick and the intent's labels:

    right       an acceptable tool (on a gap intent: hand_off_to_human)
    defensible  a partial tool: a sensible first step that doesn't finish the job
    handed_off  hand_off_to_human on an intent a real tool could handle (a miss)
    asked       no tool call: the model replied in text, or (router) wrote prose instead of a name
                (right instead, when the intent's ``ask_ok`` says a question is needed first)
    wrong       the wrong tool on a covered intent
    bluffed     a tool that can't do it, on a gap intent

For reporting, the six outcomes fold into three verdicts:

    correct     right
    benign      incorrect, but nothing bad happens: handed_off, asked, defensible (the customer
                waits for a person, answers a question, or gets a sensible first step)
    harmful     incorrect, and the assistant acted wrongly: wrong, bluffed (a tool that can't do
                what was asked, so it looks as if something happened and nothing did)

Only ``right`` counts as correct. The harmful rate is what a customer would notice.

A native model may call several tools at once. The batch is ``right`` if every call is acceptable
or defensible and at least one is acceptable; a batch with any wrong call is a wrong call (the
worst outcome in it); otherwise the first call's outcome stands.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from functools import cache

from .arguments import check
from .data import ASK, HAND, RUNS, Request, all_requests_by_id, labels, tools

OUTCOMES = ["right", "defensible", "handed_off", "asked", "wrong", "bluffed"]
WRONG_CALL = {"wrong", "bluffed"}
VERDICTS = ["correct", "benign", "harmful"]
VERDICT = {"right": "correct", "defensible": "benign", "handed_off": "benign", "asked": "benign",
           "wrong": "harmful", "bluffed": "harmful"}  # fmt: skip
MODEL_NAMES = {"jev": "Jev", "deepseek": "DeepSeek V4.1 Flash", "haiku": "Claude Haiku 5.5", "sonnet": "Claude Sonnet 5.5"}


SEVERITY = {"bluffed": 0, "wrong": 1, "handed_off": 2, "asked": 3, "defensible": 4, "right": 5}


def outcome(pick: str | None, lab: dict, kind: str = "tool_call", calls: Sequence[str] | None = None) -> str:
    """The outcome of one pick (or of a parallel batch of ``calls``) against an intent's labels."""
    if calls and len(calls) > 1:
        outs = [outcome(c, lab) for c in calls]
        if any(o in WRONG_CALL for o in outs):
            return min(outs, key=SEVERITY.get)
        if "right" in outs and all(o in ("right", "defensible") for o in outs):
            return "right"
        return outs[0]
    if kind == "text" or pick is None or pick == ASK:
        return "right" if lab.get("ask_ok") else "asked"
    if pick in lab["acceptable"]:
        return "right"
    if pick in lab["partial"]:
        return "defensible"
    if lab["gap"]:
        return "bluffed"
    return "handed_off" if pick == HAND else "wrong"


def parse_run_name(name: str) -> dict[str, str]:
    """``native_sonnet_policy`` -> mode native, model sonnet, prompt policy (or gepa, or base);
    ``fresh/native_sonnet`` -> the same with sample fresh and prompt final."""
    sample, _, name = name.rpartition("/")
    if name == "jev" or name.startswith("jev_"):  # jev, jev_policy, jev_ask, jev_params, jev_ask_params
        variant = name.removeprefix("jev").strip("_")
        if sample == "fresh":
            return {"mode": "jev", "model": "jev", "prompt": "final" + (f"_{variant}" if variant else ""), "sample": "fresh"}
        return {"mode": "jev", "model": "jev", "prompt": variant or "base", "sample": "first"}
    if sample == "fresh":
        out = parse_run_name(name)
        return {**out, "prompt": "final", "sample": "fresh"}
    prompt = next((p for p in ("policy", "gepa") if name.endswith("_" + p)), "base")
    base = name.removesuffix("_" + prompt) if prompt != "base" else name
    if base == "jev":
        mode, model = "jev", "jev"
    else:
        mode, _, model = base.partition("_")
    return {"mode": mode, "model": model, "prompt": prompt, "sample": "first"}


@cache
def load_run(name: str) -> dict[str, dict]:
    """Rows of ``runs/<name>.jsonl`` keyed by request id (without the full API response)."""
    p = RUNS / f"{name}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r.pop("response", None)
            out[r["id"]] = r
    return out


def available_runs(sample: str = "first") -> list[str]:
    """Run names for one sample (``first``: runs/*.jsonl; ``fresh``: runs/fresh/*.jsonl)."""
    if sample == "fresh":
        return sorted(f"fresh/{p.stem}" for p in (RUNS / "fresh").glob("*.jsonl"))
    return sorted(p.stem for p in RUNS.glob("*.jsonl"))


# Jev only picks a tool; a second call fills in its arguments (scripts/run_jev_fill.py). For the final
# run, that call's cost, latency and tokens are added to Jev's, and its arguments are Jev's arguments.
FILLS = {"fresh/jev": "fill/jev_fresh", "fresh/jev_ask": "fill/jev_ask_fresh", "fresh/jev_params": "fill/jev_params_fresh",
         "fresh/jev_ask_params": "fill/jev_ask_params_fresh"}  # fmt: skip


def scored_rows(name: str, version: str = "audited") -> list[dict]:
    """One dict per request the run answered: request fields + pick + outcome + usage. ``cost_usd``,
    ``latency_s`` and the token counts include the argument-filling call where there is one
    (``pick_*`` and ``fill_*`` give the two parts)."""
    lab, by_id = labels(version), all_requests_by_id()
    fills = load_run(FILLS[name]) if name in FILLS else {}
    rows = []
    for rid, r in load_run(name).items():
        req = by_id[rid]
        calls = [c["name"] for c in r.get("tool_calls") or []]
        kind = r.get("kind", "tool_call")
        o = outcome(r["pick"], lab[req.intent], kind, calls)
        f = fills.get(rid) or {}
        src = f if fills else r  # only native runs and filled Jev rows have arguments
        args = (src.get("tool_calls") or [{}])[0].get("arguments")
        chk = check(r["pick"], args, req.text, req.intent) if src.get("tool_calls") else None
        pick_cost, fill_cost = r.get("cost_usd", 0.0), f.get("cost_usd", 0.0)
        rows.append({"id": rid, "intent": req.intent, "gap": lab[req.intent]["gap"], "text": req.text, "run": name,
                     **parse_run_name(name), "pick": r["pick"], "kind": kind, "outcome": o,
                     "right": o == "right", "wrong_call": o in WRONG_CALL, "verdict": VERDICT[o], "confidence": r.get("confidence"),
                     "latency_s": r["latency_s"] + f.get("latency_s", 0.0),
                     "input_tokens": (r.get("input_tokens") or 0) + f.get("input_tokens", 0),
                     "cached_tokens": r.get("cached_tokens", 0) + f.get("cached_tokens", 0),
                     "output_tokens": r.get("output_tokens", 0) + f.get("output_tokens", 0),
                     "cost_usd": pick_cost + fill_cost, "pick_cost_usd": pick_cost, "fill_cost_usd": fill_cost,
                     "pick_latency_s": r["latency_s"], "fill_latency_s": f.get("latency_s", 0.0),
                     "reply": r.get("raw") or r.get("text") or "", "arguments": args, "n_calls": max(1, len(calls)),
                     "args_ok": chk["ok"] if chk else None, "args_problems": "; ".join(chk["problems"]) if chk else ""})  # fmt: skip
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def summarise(rows: Sequence[dict]) -> dict:
    """Headline numbers for one run (or any subset of its rows)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    right = sum(r["right"] for r in rows)
    mix = Counter(r["outcome"] for r in rows)
    return {
        "n": n,
        "right": right / n,
        "right_ci": wilson(right, n),
        "wrong_call": sum(r["wrong_call"] for r in rows) / n,
        "correct": right / n,
        "benign": sum(VERDICT[r["outcome"]] == "benign" for r in rows) / n,
        "harmful": sum(r["wrong_call"] for r in rows) / n,
        "handoff": sum(r["pick"] == HAND for r in rows) / n,
        "asked": mix["asked"] / n,
        "mix": {o: mix[o] / n for o in OUTCOMES},
        "latency_median_s": statistics.median(r["latency_s"] for r in rows),
        "latency_p95_s": sorted(r["latency_s"] for r in rows)[int(0.95 * (n - 1))],
        "input_tokens_median": statistics.median(r["input_tokens"] or 0 for r in rows),
        "output_tokens_median": statistics.median(r["output_tokens"] or 0 for r in rows),
        "cost_usd": sum(r["cost_usd"] for r in rows),
        "cost_per_1000": 1000 * sum(r["cost_usd"] for r in rows) / n,
    }


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired right/wrong outcomes: how often only a or only b was right."""
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "n_pairs": len(a), "p": p}


def paired(name_a: str, name_b: str, version: str = "audited", seed: int = 0, boots: int = 4000) -> dict:
    """Compare two runs on the requests both answered: McNemar and a bootstrap CI on the difference."""
    a = {r["id"]: r for r in scored_rows(name_a, version)}
    b = {r["id"]: r for r in scored_rows(name_b, version)}
    ids = [i for i in a if i in b]
    if not ids:
        return {}
    ra, rb = [a[i]["right"] for i in ids], [b[i]["right"] for i in ids]
    d = [int(x) - int(y) for x, y in zip(ra, rb, strict=True)]
    rng = random.Random(seed)
    bs = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(boots))
    return {**mcnemar(ra, rb), "diff": statistics.fmean(d), "ci": (bs[int(0.025 * boots)], bs[int(0.975 * boots) - 1]),
            "right_a": statistics.fmean(ra), "right_b": statistics.fmean(rb)}  # fmt: skip


def per_intent(name: str, version: str = "audited") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in scored_rows(name, version):
        d = out.setdefault(r["intent"], {"n": 0, "right": 0, "gap": r["gap"], "picks": Counter()})
        d["n"] += 1
        d["right"] += r["right"]
        d["picks"][r["pick"] if r["kind"] != "text" else "(asked)"] += 1
    for d in out.values():
        d["right_rate"] = d["right"] / d["n"]
    return out


def grid(version: str = "audited") -> dict[str, dict]:
    """Every available run summarised, keyed by run name."""
    return {name: summarise(scored_rows(name, version)) for name in available_runs()}


def gate_curve(name: str = "jev", version: str = "audited", thresholds: Sequence[float] = (0, .3, .4, .5, .6, .7, .8, .9, .95)) -> list[dict]:
    """Jev's confidence as a dial: below the threshold, hand off instead of calling a tool."""
    lab = labels(version)
    rows = scored_rows(name, version)
    out = []
    for t in thresholds:
        acted = [r for r in rows if r["pick"] != HAND and (r["confidence"] or 0) >= t]
        wrong = sum(r["wrong_call"] for r in acted)
        gated_right = sum(
            outcome(HAND if (r["pick"] != HAND and (r["confidence"] or 0) < t) else r["pick"], lab[r["intent"]]) == "right"
            for r in rows)
        out.append({"t": t, "acted": len(acted) / len(rows), "wrong_of_acted": wrong / max(1, len(acted)),
                    "right": gated_right / len(rows)})  # fmt: skip
    return out


def requests_by_id() -> dict[str, Request]:
    return all_requests_by_id()


# The "ask first" stipulation (a sensitivity check, applied to every selector alike): on a request
# where a question back is right (refunds, placing an order, payment problems) and the message gives
# no order number, invoice number or amount, the only correct answers are asking, handing off where
# the key accepts it, or a tool that needs nothing from the customer. Acting straight away (a return,
# a refund, a cart) needs details the customer hasn't given, so it drops to benign.
DETAILS = re.compile(r"ORD-\d+|#\s?\d{3,}|INV-[\d-]+|\$\s?\d+|\d{5,}", re.IGNORECASE)


@cache
def _required_inputs() -> dict[str, set[str]]:
    return {t["name"]: set(t["inputSchema"].get("required") or []) - {"meta"} for t in tools(customer_words=True, ask=True)}


def stipulated_right(row: dict) -> bool:
    """``row["right"]`` under the ask-first stipulation (rows from ``scored_rows``, audited key)."""
    lab = labels()[row["intent"]]
    if not lab["ask_ok"] or DETAILS.search(row["text"]):
        return row["right"]
    if row["kind"] == "text" or row["pick"] in (None, ASK):
        return True
    return row["pick"] in lab["acceptable"] and (row["pick"] == HAND or not _required_inputs().get(row["pick"]))
