"""Was each agent's question justified? Claude Opus 5.5 labels the messages (see src/toolpick/judge.py).

    uv run python scripts/judge_asks.py --dry-run     # what would be sent, and the cost estimate (free)
    uv run python scripts/judge_asks.py --limit 3     # three messages: checks the prompt cache is hit
    uv run python scripts/judge_asks.py --sample 20   # a random 20, to read before trusting the rest
    uv run python scripts/judge_asks.py               # all 180 (about US$1.50)

In the final runs the agents replied in text on 251 requests no tool can handle (Sonnet 15, Haiku 62,
DeepSeek 174). Those replies are to 180 different messages. The judge labels each message once, so
every agent that asked about a message gets the same verdict: the question was justified if the
message is unclear. Rows go to runs/judge/messages_opus.jsonl as they arrive, with the full API
response; a re-run skips what is done, and a changed prompt re-judges everything. The first call
goes alone so it writes the prompt cache and the rest read it. The summary prints how many calls
read the cache: if it is low, stop and look. Needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

import toolpick as tp
from toolpick.judge import JUDGE_MODEL, PRICE, judge, judge_request, judge_system

ROOT = Path(__file__).resolve().parents[1]
AGENTS = {"sonnet": "fresh/native_sonnet", "haiku": "fresh/native_haiku", "deepseek": "fresh/native_deepseek"}
OUT = tp.RUNS / "judge" / "messages_opus.jsonl"
BUDGET_USD = 3.00
CONCURRENCY = 4


def asked() -> dict[str, list[dict]]:
    """Per agent, the scored rows where it replied in text on a request no tool can handle."""
    return {agent: [r for r in tp.scored_rows(run) if r["outcome"] == "asked" and r["gap"]] for agent, run in AGENTS.items()}


def items() -> list[dict]:
    """The different messages behind those replies, in a fixed order."""
    texts = {r["id"]: r["text"] for rows in asked().values() for r in rows}
    return [{"id": rid, "message": texts[rid]} for rid in sorted(texts)]


def prompt_id() -> str:
    """A short hash of the judge's system prompt, stored with every row."""
    return hashlib.sha256(judge_system().encode()).hexdigest()[:12]


def done_rows() -> dict[str, dict]:
    """Judgements on disk made with the current prompt, by message id. Rows from an older prompt are
    ignored, so a changed prompt re-judges everything instead of mixing two judges in one table."""
    if not OUT.exists():
        return {}
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows if r.get("prompt_id") == prompt_id()}


def dry_run(todo: list[dict]) -> None:
    system = judge_system()
    tokens = len(system) / 2.85  # measured on the first version of this prompt: 27,187 characters were 9,528 tokens
    n = len(todo)
    per_call = 60 * PRICE["input"] + 300 * PRICE["output"]  # a short message in; a little thinking and the answer out
    cached = (tokens * PRICE["cache_write"] + max(0, n - 1) * tokens * PRICE["cache_read"] + n * per_call) / 1e6
    uncached = n * (tokens * PRICE["input"] + per_call) / 1e6
    print(f"{n} messages to judge with {JUDGE_MODEL}")
    print(f"system prompt: {len(system):,} characters, roughly {tokens:,.0f} tokens (cached after the first call)")
    print(f"estimated cost: ${cached:.2f} with the cache, ${uncached:.2f} without")
    if todo:
        print("\n--- first user message ---\n" + judge_request(todo[0]["message"])["messages"][0]["content"])


def summary() -> None:
    rows = done_rows()
    if not rows:
        return
    reads = sum(r["cache_read_tokens"] > 0 for r in rows.values())
    print(f"\n{len(rows)} messages judged, ${sum(r['cost_usd'] for r in rows.values()):.3f}, "
          f"cache read on {reads} of {len(rows)} calls, labels {dict(Counter(r['label'] for r in rows.values()))}")  # fmt: skip
    if len(rows) > 1 and reads < 0.8 * (len(rows) - 1):
        print("WARNING: few cache reads. Check that the system prompt is identical across calls.")
    other = Counter(r["model"] for r in rows.values() if not r["model"].startswith(JUDGE_MODEL))
    if other:
        print(f"WARNING: some messages were judged by a fallback model: {dict(other)}")
    print(f"\n{'agent':10s} {'asked':>6s} {'judged':>7s} {'unclear':>8s} {'no_tool':>8s} {'tool':>5s} {'correct now':>12s} {'if asks on unclear count':>25s}")
    for agent, mine in asked().items():
        scored = tp.scored_rows(AGENTS[agent])
        c = Counter(rows[r["id"]]["label"] for r in mine if r["id"] in rows)
        justified = {r["id"] for r in mine if rows.get(r["id"], {}).get("label") == "unclear"}
        rescored = sum(s["right"] or s["id"] in justified for s in scored) / len(scored)
        print(f"{agent:10s} {len(mine):6d} {sum(c.values()):7d} {c['unclear']:8d} {c['no_tool']:8d} {c['tool']:5d} "
              f"{sum(s['right'] for s in scored) / len(scored):12.1%} {rescored:25.1%}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N messages still to do")
    ap.add_argument("--sample", type=int, default=None, help="judge a random N of the messages still to do (seed 0)")
    ap.add_argument("--workers", type=int, default=CONCURRENCY, help="calls in flight at once, after the first has written the cache")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and a cost estimate; make no calls")
    args = ap.parse_args()
    done = done_rows()
    todo = [i for i in items() if i["id"] not in done]
    if args.sample:
        todo = random.Random(0).sample(todo, min(args.sample, len(todo)))
    todo = todo[: args.limit]
    if args.dry_run:
        return dry_run(todo)
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parents[1] / ".env")  # when this folder sits inside the course repo, its .env works too
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example)")
    import anthropic

    client = anthropic.Anthropic(max_retries=4, timeout=120)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spent, lock = [0.0], threading.Lock()

    def one(item: dict) -> None:
        if spent[0] >= BUDGET_USD:
            return
        try:
            row = {"id": item["id"], "message": item["message"], "prompt_id": prompt_id(), **judge(client, item["message"])}
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError, ValueError) as exc:
            # still failing after the SDK's own retries, or an unusable answer: left undone, the next run picks it up.
            # A bad request or a bad key is not caught: it would fail the same way on every message, so it stops the run.
            print(f"{item['id']}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return
        with lock, OUT.open("a") as f:
            spent[0] += row["cost_usd"]
            f.write(json.dumps(row) + "\n")

    print(f"{len(todo)} to judge ({len(done)} already done)", flush=True)
    for item in todo[:1]:
        one(item)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo[1:]))
    summary()


if __name__ == "__main__":
    main()
