"""The native condition: the LLM gets the 91 tools in the API's `tools` parameter and decides itself.

    uv run python scripts/run_native.py --arm deepseek|haiku|sonnet [--policy] [--limit 5]

This is how an agent really picks tools: a plain assistant prompt, the tool definitions in the
provider's own format, `tool_choice: auto`, and the model may reply in text (usually a clarifying
question) instead of calling anything. We score the first tool call; a text reply is its own
outcome, "asked". MCP names are `server.tool`; function names can't contain dots, so the API sees
`server__tool` and we map it back. A cache breakpoint on the system text covers the tools too
(the cached prefix is tools, then system, then messages). DeepSeek is pinned to DeepInfra, which
serves cache hits with tools. Rows go to runs/native_<arm>[_policy].jsonl with the full API
response, start timestamp and the tool-call arguments.
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

from run_router import BUDGET_USD, CONCURRENCY, MODELS, done_ids

from toolpick import RUNS, requests, tools
from toolpick.prompts import native_system


def native_tools(tool_list: list[dict]) -> tuple[list[dict], dict[str, str]]:
    out, back = [], {}
    for t in tool_list:
        name = t["name"].replace(".", "__")
        back[name] = t["name"]
        out.append({"type": "function", "function": {"name": name, "description": t["description"],
                                                      "parameters": t["inputSchema"]}})  # fmt: skip
    return out, back


def run_native(arm: str, reqs, tool_list, out: Path, policy: bool, system_text: str | None = None) -> None:
    import openai

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                           max_retries=6, timeout=180)  # fmt: skip
    model = MODELS[arm]
    fn_tools, back = native_tools(tool_list)
    system_block = {"type": "text", "text": system_text or native_system(policy), "cache_control": {"type": "ephemeral"}}
    extra: dict = {"usage": {"include": True}, "reasoning": {"effort": "low" if arm == "sonnet" else "none"}}
    if arm == "deepseek":
        extra["provider"] = {"order": ["deepinfra/fp8", "deepinfra/fp8/priority"], "allow_fallbacks": True}
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
                    model=model, max_tokens=400, temperature=0,
                    messages=[{"role": "system", "content": [system_block]}, {"role": "user", "content": r.text}],
                    tools=fn_tools, tool_choice="auto", extra_body=extra,
                )  # fmt: skip
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{r.id}: attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                time.sleep(2 * (attempt + 1))
        else:
            return
        lat = time.perf_counter() - t0
        ch, u = resp.choices[0], resp.usage
        d = getattr(u, "prompt_tokens_details", None)
        cached = int(getattr(d, "cached_tokens", 0) or 0)
        cost = getattr(u, "cost", None) or (getattr(u, "model_extra", None) or {}).get("cost") or 0.0
        calls = []
        for tc in ch.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": tc.function.arguments}
            calls.append({"name": back.get(tc.function.name, tc.function.name), "arguments": args})
        row = {"id": r.id, "pick": calls[0]["name"] if calls else None, "kind": "tool_call" if calls else "text",
               "tool_calls": calls, "text": ch.message.content or "", "finish_reason": ch.finish_reason,
               "latency_s": round(lat, 3), "input_tokens": u.prompt_tokens, "cached_tokens": cached,
               "output_tokens": u.completion_tokens, "cost_usd": float(cost),
               "provider": getattr(resp, "provider", None) or (getattr(resp, "model_extra", None) or {}).get("provider"),
               "ts": ts, "response": resp.model_dump(mode="json")}  # fmt: skip
        with lock:
            spent[0] += float(cost)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    if reqs:  # the first call writes the cache; the rest read it
        one(reqs[0])
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        list(pool.map(one, reqs[1:]))
    print(f"{arm}: spent ${spent[0]:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(MODELS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", choices=["first", "fresh"], default="first",
                    help="fresh = the final setup (policy prompt + customer-worded tools) on data/requests_fresh.jsonl, into runs/fresh/")
    ap.add_argument("--policy", action="store_true")
    args = ap.parse_args()
    fresh = args.sample == "fresh"
    if fresh:
        args.policy = True
    reqs = requests(args.sample)[: args.limit] if args.limit else requests(args.sample)
    name = f"fresh/native_{args.arm}" if fresh else f"native_{args.arm}{'_policy' if args.policy else ''}"
    out = RUNS / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in reqs if r.id not in done_ids(out)]
    print(f"{name}: {len(todo)} to run", flush=True)
    t0 = time.perf_counter()
    run_native(args.arm, todo, tools(customer_words=fresh), out, args.policy)
    print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
