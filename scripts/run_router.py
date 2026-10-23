"""The router condition: Jev, or an LLM told to answer with one tool's name.

    uv run python scripts/run_router.py --arm jev [--policy] [--limit 5]
    uv run python scripts/run_router.py --arm deepseek|haiku|sonnet [--policy]

Jev answers one `choice` question whose options are the 91 tool names and descriptions. The LLMs
get the full MCP definitions (name, description, inputSchema) as JSON in a cached system prompt
and reply with a name. Rows go to runs/jev[_policy].jsonl or runs/router_<arm>[_policy].jsonl,
with the full API response, the start timestamp, usage, billed cost and latency. Re-running skips
ids already done.

Needs TYPESAFE_API_KEY (Jev) and OPENROUTER_API_KEY (the LLMs) in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parents[1] / ".env")  # when this folder sits inside the course repo, its .env works too
sys.path.insert(0, str(ROOT / "src"))

from toolpick import HAND, RUNS, requests, tools
from toolpick.jev import Jev
from toolpick.prompts import jev_instructions, router_system

MODELS = {"deepseek": "deepseek/deepseek-v4.1-flash", "haiku": "anthropic/claude-haiku-5.5",
          "sonnet": "anthropic/claude-sonnet-5.5"}  # fmt: skip
BUDGET_USD = {"deepseek": 3.0, "haiku": 2.0, "sonnet": 8.0}
CONCURRENCY = 8


def parse_pick(text: str, names: set[str]) -> str | None:
    words = (text or "").strip().strip("`'\" .").split()
    if not words:
        return None
    w = words[0].strip("`'\" .,")
    if w in names:
        return w
    bare = [n for n in names if n.split(".")[-1] == w]
    return bare[0] if len(bare) == 1 else None


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["id"] for x in path.read_text().splitlines() if x.strip()}


def run_jev(reqs, tool_list, out: Path, policy: bool) -> None:
    jev = Jev()
    criteria = {t["name"]: t["description"] for t in tool_list}
    instructions = jev_instructions(policy)
    lock = threading.Lock()

    def one(r):
        ts = datetime.now(tz=UTC).isoformat()
        body = jev.choose(r.text, instructions, criteria)
        a = body["answers"]["tool"]
        top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:8]
        row = {"id": r.id, "pick": a["choice"], "kind": "tool_call", "confidence": a["confidence"],
               "top": [[k, round(v, 5)] for k, v in top], "latency_s": round(body["latency_s"], 4),
               "input_tokens": body["usage"]["input_tokens"], "output_tokens": body["usage"].get("output_tokens", 0),
               "cost_usd": body["cost_usd"], "ts": ts, "model": body.get("model"),
               "response": {k: v for k, v in body.items() if k not in ("latency_s", "cost_usd")}}  # fmt: skip
        with lock, out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, reqs))


def run_llm(arm: str, reqs, tool_list, out: Path, policy: bool, system_text: str | None = None) -> None:
    import openai

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                           max_retries=6, timeout=120)  # fmt: skip
    model = MODELS[arm]
    names = {t["name"] for t in tool_list}
    system_block = {"type": "text", "text": system_text or router_system(json.dumps(tool_list, separators=(",", ":")), policy),
                    "cache_control": {"type": "ephemeral"}}  # fmt: skip
    extra: dict = {"usage": {"include": True}, "reasoning": {"effort": "low" if arm == "sonnet" else "none"}}
    if arm == "deepseek":  # one provider, so its prefix cache stays warm
        extra["provider"] = {"order": ["deepinfra/fp8", "relace"], "allow_fallbacks": True}
    spent = [0.0]
    lock = threading.Lock()

    def one(r):
        with lock:
            if spent[0] >= BUDGET_USD[arm]:
                return
        for attempt in range(4):
            try:
                t0, ts = time.perf_counter(), datetime.now(tz=UTC).isoformat()
                resp = client.chat.completions.create(
                    model=model, max_tokens=1500 if arm == "sonnet" else 40, temperature=0,
                    messages=[{"role": "system", "content": [system_block]}, {"role": "user", "content": r.text}],
                    extra_body=extra,
                )  # fmt: skip
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{r.id}: attempt {attempt + 1} failed: {type(exc).__name__}", flush=True)
                time.sleep(2 * (attempt + 1))
        else:
            return
        lat = time.perf_counter() - t0
        u = resp.usage
        d = getattr(u, "prompt_tokens_details", None)
        cached = int(getattr(d, "cached_tokens", 0) or 0)
        cost = getattr(u, "cost", None) or (getattr(u, "model_extra", None) or {}).get("cost") or 0.0
        text = resp.choices[0].message.content or ""
        row = {"id": r.id, "pick": parse_pick(text, names), "kind": "tool_call", "raw": text, "latency_s": round(lat, 3),
               "input_tokens": u.prompt_tokens, "cached_tokens": cached, "output_tokens": u.completion_tokens,
               "cost_usd": float(cost), "provider": getattr(resp, "provider", None) or
               (getattr(resp, "model_extra", None) or {}).get("provider"), "ts": ts,
               "response": resp.model_dump(mode="json")}  # fmt: skip
        with lock:
            spent[0] += float(cost)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    if reqs:  # the first call writes the cache; the rest read it
        one(reqs[0])
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        list(pool.map(one, reqs[1:]))
    print(f"{arm}: spent ${spent[0]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["jev", *MODELS], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", choices=["first", "fresh"], default="first",
                    help="fresh = the final setup (policy prompt + customer-worded tools) on data/requests_fresh.jsonl, into runs/fresh/")
    ap.add_argument("--policy", action="store_true", help="add the routing policy to the prompt")
    args = ap.parse_args()
    fresh = args.sample == "fresh"
    if fresh:
        args.policy = True
    tool_list = tools(customer_words=fresh)
    reqs = requests(args.sample)[: args.limit] if args.limit else requests(args.sample)
    base = "jev" if args.arm == "jev" else f"router_{args.arm}"
    name = f"fresh/{base}" if fresh else base + ("_policy" if args.policy else "")
    out = RUNS / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in reqs if r.id not in done_ids(out)]
    print(f"{name}: {len(todo)} to run", flush=True)
    t0 = time.perf_counter()
    if args.arm == "jev":
        run_jev(todo, tool_list, out, args.policy)
    else:
        run_llm(args.arm, todo, tool_list, out, args.policy)
    print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)
    assert HAND in {t["name"] for t in tool_list}


if __name__ == "__main__":
    main()
