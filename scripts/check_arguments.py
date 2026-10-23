"""Argument check on the final run: are the values passed to the tools right? (free: no API calls)

    uv run python scripts/check_arguments.py [--examples 3]

Scores the first tool call of every native-agent row, and Jev's argument-filling call
(runs/fill/jev_fresh.jsonl), with the rules in toolpick.arguments. The headline is "right tool, right
arguments": of the calls whose tool was correct (hand-offs left out, their reason is free text), how
many also passed every argument check. Writes data/arguments.json for the deck and README.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp
from toolpick.arguments import check
from toolpick.scoring import FILLS, load_run

RUNS = {"jev": "fresh/jev", "deepseek": "fresh/native_deepseek", "haiku": "fresh/native_haiku", "sonnet": "fresh/native_sonnet"}
OUT = ROOT / "data" / "arguments.json"


def rate(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{sum(xs)}/{len(xs)} ({sum(xs) / len(xs):.0%})" if xs else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    by_id = tp.data.all_requests_by_id()
    report = {}
    for model, run in RUNS.items():
        rows = {r["id"]: r for r in tp.scored_rows(run)}
        src = load_run(FILLS[run]) if run in FILLS else load_run(run)
        checks = []
        for rid, raw in src.items():
            if not raw.get("tool_calls") or rid not in rows:
                continue
            row, req = rows[rid], by_id[rid]
            call = raw["tool_calls"][0]
            c = check(call["name"], call["arguments"], req.text, req.intent)
            checks.append({**c, "id": rid, "tool": call["name"], "right": row["right"], "text": req.text,
                           "arguments": call["arguments"]})  # fmt: skip
        scope = [c for c in checks if c["right"] and c["tool"] != tp.HAND]
        n_right_tool = sum(rows[c["id"]]["right"] for c in checks)
        total = len(rows)
        both = sum(c["ok"] for c in scope)
        handoffs_right = sum(1 for r in rows.values() if r["right"] and r["pick"] == tp.HAND)
        asked_right = sum(1 for r in rows.values() if r["right"] and r["kind"] == "text")
        summary = {
            "calls_scored": len(scope),
            "args_ok": both,
            "args_ok_rate": both / len(scope) if scope else None,
            "valid": rate([c["valid"] for c in scope]),
            "grounded": rate([c["grounded"] for c in scope]),
            "carried": rate([c["carried"] for c in scope]),
            "amount": rate([c["amount"] for c in scope]),
            "policy_type": rate([c["policy_type"] for c in scope]),
            # end to end: right tool and right arguments, or a right hand-off / question (no values to get wrong)
            "right_end_to_end": (both + handoffs_right + asked_right) / total,
            "right_tool": sum(r["right"] for r in rows.values()) / total,
            "problems": Counter(p.split(":")[0] if p.startswith("schema") else p.split("=")[0].replace("'", "")
                                for c in scope for p in c["problems"]).most_common(8),
            "by_tool": {t: rate([c["ok"] for c in scope if c["tool"] == t]) for t in sorted({c["tool"] for c in scope})},
        }
        report[model] = summary
        print(f"\n== {tp.MODEL_NAMES[model]} ({run}) ==")
        print(f"tool calls with arguments: {len(checks)} (on a correct tool: {n_right_tool}); scored (correct, not hand-off): {len(scope)}")
        for k in ("valid", "grounded", "carried", "amount", "policy_type"):
            print(f"  {k:12s} {summary[k]}")
        print(f"  ALL OK       {rate([c['ok'] for c in scope])}")
        print(f"  right tool {summary['right_tool']:.1%} -> right tool and arguments {summary['right_end_to_end']:.1%}")
        print("  by tool:", summary["by_tool"])
        print("  problems:", summary["problems"])
        for c in [c for c in scope if not c["ok"]][: args.examples]:
            print(f"    {c['id']} {c['text'][:60]!r} -> {c['tool'].split('.')[-1]} {json.dumps(c['arguments'])[:140]}")
            print(f"       {c['problems'][:2]}")
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
