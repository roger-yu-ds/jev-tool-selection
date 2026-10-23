"""One live call, exactly as the runs make it. Used by the scripts and by notebooks/rerun_failures.ipynb.

    from toolpick.live import ask
    row = ask("sonnet", "i need help to get refunds of my money")   # the final setup by default

`ask` returns the same row shape as runs/*.jsonl (pick, kind, tool_calls, text, usage, cost, latency,
full response), so `toolpick.outcome` scores it the same way. Needs OPENROUTER_API_KEY (LLMs) or
TYPESAFE_API_KEY (Jev) in the environment.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from functools import cache

from .data import tools
from .prompts import jev_instructions, native_system

MODELS = {"deepseek": "deepseek/deepseek-v4.1-flash", "haiku": "anthropic/claude-haiku-5.5",
          "sonnet": "anthropic/claude-sonnet-5.5"}  # fmt: skip


def native_tools(tool_list: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """MCP tools as function-calling tools. Names can't contain dots: `server.tool` -> `server__tool`."""
    out, back = [], {}
    for t in tool_list:
        name = t["name"].replace(".", "__")
        back[name] = t["name"]
        out.append({"type": "function", "function": {"name": name, "description": t["description"],
                                                      "parameters": t["inputSchema"]}})  # fmt: skip
    return out, back


def native_extra(arm: str) -> dict:
    """OpenRouter options: billed usage, no hidden reasoning (Sonnet 5.5 can't turn it off, so low),
    and DeepSeek pinned to the provider whose prompt cache works with tools."""
    extra: dict = {"usage": {"include": True}, "reasoning": {"effort": "low" if arm == "sonnet" else "none"}}
    if arm == "deepseek":
        extra["provider"] = {"order": ["deepinfra/fp8", "deepinfra/fp8/priority"], "allow_fallbacks": True}
    return extra


def native_request(arm: str, text: str, tool_list: list[dict], system_text: str) -> tuple[dict, dict[str, str]]:
    fn_tools, back = native_tools(tool_list)
    system_block = {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
    kwargs = {"model": MODELS[arm], "max_tokens": 400, "temperature": 0,
              "messages": [{"role": "system", "content": [system_block]}, {"role": "user", "content": text}],
              "tools": fn_tools, "tool_choice": "auto", "extra_body": native_extra(arm)}  # fmt: skip
    return kwargs, back


def parse_native(resp, back: dict[str, str]) -> dict:
    ch, u = resp.choices[0], resp.usage
    d = getattr(u, "prompt_tokens_details", None)
    cost = getattr(u, "cost", None) or (getattr(u, "model_extra", None) or {}).get("cost") or 0.0
    calls = []
    for tc in ch.message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": tc.function.arguments}
        calls.append({"name": back.get(tc.function.name, tc.function.name), "arguments": args})
    return {"pick": calls[0]["name"] if calls else None, "kind": "tool_call" if calls else "text", "tool_calls": calls,
            "text": ch.message.content or "", "finish_reason": ch.finish_reason, "input_tokens": u.prompt_tokens,
            "cached_tokens": int(getattr(d, "cached_tokens", 0) or 0), "output_tokens": u.completion_tokens,
            "cost_usd": float(cost),
            "provider": getattr(resp, "provider", None) or (getattr(resp, "model_extra", None) or {}).get("provider"),
            "response": resp.model_dump(mode="json")}  # fmt: skip


@cache
def _openrouter():
    import openai

    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         max_retries=6, timeout=180)  # fmt: skip


@cache
def _jev():
    from .jev import Jev

    return Jev()


def ask(arm: str, text: str, *, policy: bool = True, customer_words: bool = True) -> dict:
    """One decision for one message. ``arm`` is jev, deepseek, haiku or sonnet; LLMs run as native
    tool-calling agents. Defaults are the final setup (policy prompt, customer-worded tools)."""
    tool_list = tools(customer_words=customer_words)
    ts = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    if arm == "jev":
        body = _jev().choose(text, jev_instructions(policy), {t["name"]: t["description"] for t in tool_list})
        a = body["answers"]["tool"]
        top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:8]
        return {"pick": a["choice"], "kind": "tool_call", "confidence": a["confidence"], "top": [[k, round(v, 5)] for k, v in top],
                "latency_s": round(body["latency_s"], 4), "input_tokens": body["usage"]["input_tokens"],
                "output_tokens": body["usage"].get("output_tokens", 0), "cost_usd": body["cost_usd"], "ts": ts,
                "model": body.get("model"), "response": {k: v for k, v in body.items() if k not in ("latency_s", "cost_usd")}}  # fmt: skip
    kwargs, back = native_request(arm, text, tool_list, native_system(policy))
    resp = _openrouter().chat.completions.create(**kwargs)
    return {**parse_native(resp, back), "latency_s": round(time.perf_counter() - t0, 3), "ts": ts}
