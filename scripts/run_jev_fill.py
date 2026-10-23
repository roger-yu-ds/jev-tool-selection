"""The second step Jev needs: fill in the arguments of the tool Jev picked.

    uv run python scripts/run_jev_fill.py [--limit 5]
    uv run python scripts/run_jev_fill.py --run fresh/jev_ask     # Jev with need_more_information

Jev only picks a tool; a working agent still has to fill in the tool's arguments, which the native
agents do in the same call. This makes that call for every row of runs/fresh/jev.jsonl, so Jev's
cost and latency can include it. The filler is DeepSeek V4.1 Flash (the cheapest arm), pinned to the
same provider as the native runs, with the same system prompt plus one line, and only the one
chosen tool, which it is forced to call: Jev has made the decision, this call only fills it in.
Hand-offs get a call too (it writes the reason), as the native agents write one. Rows go to
runs/fill/jev_fresh.jsonl; `toolpick.scored_rows("fresh/jev")` adds them to Jev's cost, latency and
tokens.

`--ask` is a variant that may reply with a question instead (runs/fill/jev_fresh_ask.jsonl). It
asked on 208 of 600 messages, 136 of them where Jev had rightly handed off, so it overrides Jev's
decision rather than filling it in; kept as a record, not used in the totals.
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_router import CONCURRENCY, done_ids

from toolpick import RUNS, requests, tools
from toolpick.live import native_request, parse_native
from toolpick.prompts import native_system
from toolpick.scoring import load_run

FILL_ARM = "deepseek"
FILL_SYSTEM = native_system(True) + "\n\nThe tool for this message has already been chosen: call it."
FILL_SYSTEM_ASK = native_system(True) + ("\n\nThe tool for this message has already been chosen: call it, or, if the "
                                         "customer hasn't given what it needs, ask them for it.")
OUT = RUNS / "fill" / "jev_fresh.jsonl"
OUT_ASK = RUNS / "fill" / "jev_fresh_ask.jsonl"
BUDGET_USD = 0.50


def fill_request(text: str, tool: dict, forced: bool = True) -> tuple[dict, dict[str, str]]:
    kwargs, back = native_request(FILL_ARM, text, [tool], FILL_SYSTEM if forced else FILL_SYSTEM_ASK)
    if forced:
        kwargs["tool_choice"] = {"type": "function", "function": {"name": tool["name"].replace(".", "__")}}
    return kwargs, back


def main() -> None:
    import openai

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ask", action="store_true", help="variant: may ask a question instead of calling the tool")
    ap.add_argument("--run", choices=["fresh/jev", "fresh/jev_ask", "fresh/jev_params", "fresh/jev_ask_params"], default="fresh/jev",
                    help="fresh/jev_ask: Jev with the need_more_information option; the filling call writes the question")
    args = ap.parse_args()
    out = OUT_ASK if args.ask else OUT
    if args.run != "fresh/jev":
        out = RUNS / "fill" / f"{args.run.split('/')[1]}_fresh.jsonl"
    by_name = {t["name"]: t for t in tools(customer_words=True, ask=True)}
    jev = load_run(args.run)
    reqs = [r for r in requests("fresh") if r.id in jev]
    reqs = reqs[: args.limit] if args.limit else reqs
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in reqs if r.id not in done_ids(out)]
    print(f"fill: {len(todo)} to run", flush=True)
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                           max_retries=6, timeout=180)  # fmt: skip
    spent, lock = [0.0], threading.Lock()

    def one(r):
        with lock:
            if spent[0] >= BUDGET_USD:
                return
        tool = by_name[jev[r.id]["pick"]]
        for attempt in range(4):
            try:
                t0, ts = time.perf_counter(), datetime.now(tz=UTC).isoformat()
                kwargs, back = fill_request(r.text, tool, forced=not args.ask)
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{r.id}: attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                time.sleep(2 * (attempt + 1))
        else:
            return
        row = {"id": r.id, "jev_pick": tool["name"], **parse_native(resp, back),
               "latency_s": round(time.perf_counter() - t0, 3), "ts": ts}  # fmt: skip
        with lock:
            spent[0] += row["cost_usd"]
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        list(pool.map(one, todo))
    print(f"fill: spent ${spent[0]:.4f} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
