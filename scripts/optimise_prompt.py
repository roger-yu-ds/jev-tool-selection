"""Prompt optimisation with DSPy GEPA: can an optimiser find the routing policy from data alone?

    uv run python scripts/optimise_prompt.py --arm deepseek            # ~US$1, ~15 minutes
    uv run python scripts/optimise_prompt.py --arm deepseek --evaluate # run the found prompt on the held-out 400

The hand-written policy was written by someone who had seen the labels. The honest version of
"tell the model the policy" is to let an optimiser learn the preamble from a development split
and score it on messages it never saw:

- split: 200 development messages (stratified by intent, seed 0) for GEPA; the other 400 are
  held out and only ever scored once, with the final prompt.
- program: the router prompt as a DSPy signature whose instruction is the preamble GEPA may
  rewrite. The tool JSON is an input field, so it stays fixed and cached, and never appears in
  the reflection LM's rewrite.
- metric: 1 if the pick is an acceptable tool for the intent (original labels), else 0, with
  feedback naming the acceptable tools so the reflection LM (Claude Sonnet 5.5) can reason.
- the found preamble is then run through the same runner as every other condition
  (`run_router.run_llm` with a custom system text) on the held-out 400 -> runs/router_<arm>_gepa.jsonl.

Results land in data/gepa_<arm>.json (the prompt, the dev score per candidate, the held-out score).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parents[1] / ".env")  # when this folder sits inside the course repo, its .env works too
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from toolpick import HAND, RUNS, labels, outcome, requests, tools
from toolpick.prompts import ROUTER_SYSTEM, ROUTER_TOOLS_HEADER

MODEL_IDS = {"deepseek": "openrouter/deepseek/deepseek-v4.1-flash", "haiku": "openrouter/anthropic/claude-haiku-5.5"}
REFLECTION_MODEL = "openrouter/anthropic/claude-sonnet-5.5"


def split(seed: int = 0, dev_per_intent_frac: float = 1 / 3) -> tuple[list, list]:
    by_intent = defaultdict(list)
    for r in requests():
        by_intent[r.intent].append(r)
    rng = random.Random(seed)
    dev, held = [], []
    for intent, rs in sorted(by_intent.items()):
        rs = rs[:]
        rng.shuffle(rs)
        k = max(1, round(len(rs) * dev_per_intent_frac))
        dev += rs[:k]
        held += rs[k:]
    return dev, held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(MODEL_IDS), default="deepseek")
    ap.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    ap.add_argument("--evaluate", action="store_true", help="run the saved prompt on the held-out split")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out_path = ROOT / "data" / f"gepa_{args.arm}.json"
    dev, held = split()
    lab = labels("first_pass")  # what GEPA was trained and reported against

    if args.evaluate:
        from run_router import run_llm

        found = json.loads(out_path.read_text())["instruction"]
        tools_json = json.dumps(tools(), separators=(",", ":"))
        out = RUNS / f"router_{args.arm}_gepa.jsonl"
        run_llm(args.arm, held, tools(), out, policy=False, system_text=found + "\n\n" + ROUTER_TOOLS_HEADER + tools_json)
        return

    import dspy

    lm = dspy.LM(MODEL_IDS[args.arm], api_key=os.environ["OPENROUTER_API_KEY"], temperature=0.0, max_tokens=40,
                 cache=False, extra_body={"reasoning": {"effort": "none"},
                                          "provider": {"order": ["deepinfra/fp8"], "allow_fallbacks": True}})  # fmt: skip
    reflection = dspy.LM(REFLECTION_MODEL, api_key=os.environ["OPENROUTER_API_KEY"], temperature=1.0, max_tokens=8000)
    dspy.configure(lm=lm)

    class Route(dspy.Signature):
        __doc__ = ROUTER_SYSTEM.strip()
        tools_json: str = dspy.InputField(desc="MCP tools/list JSON: every tool's name, description and inputSchema")
        message: str = dspy.InputField(desc="the customer's message")
        tool_name: str = dspy.OutputField(desc="the exact name of the one tool to call first")

    tools_json = json.dumps(tools(), separators=(",", ":"))
    names = {t["name"] for t in tools()}

    def example(r):
        return dspy.Example(tools_json=tools_json, message=r.text, intent=r.intent).with_inputs("tools_json", "message")

    def metric(ex, pred, trace=None, pred_name=None, pred_trace=None):
        pick = (pred.tool_name or "").strip().strip("`'\" .").split()[0] if (pred.tool_name or "").strip() else None
        if pick and pick not in names:
            bare = [n for n in names if n.split(".")[-1] == pick]
            pick = bare[0] if len(bare) == 1 else None
        lb = lab[ex.intent]
        o = outcome(pick, lb)
        score = 1.0 if o == "right" else 0.0
        if o == "right":
            fb = f"Correct: {pick} is acceptable for a '{ex.intent}' message."
        elif lb["gap"]:
            fb = (f"Wrong: no tool can handle a '{ex.intent}' message, so the right answer is {HAND}; "
                  f"the model picked {pick}. Searching the FAQ or looking up an order does not do what the customer asked.")
        else:
            fb = (f"Wrong: the model picked {pick or 'nothing parseable'}; acceptable tools for '{ex.intent}' are "
                  f"{', '.join(lb['acceptable'])}. " + ("Handing off when a tool could do the job is a miss." if pick == HAND else ""))
        return dspy.Prediction(score=score, feedback=fb)

    program = dspy.Predict(Route)
    optimizer = dspy.GEPA(metric=metric, auto=args.auto, reflection_lm=reflection, num_threads=args.threads,
                          track_stats=True, reflection_minibatch_size=3, seed=0)  # fmt: skip
    trainset = [example(r) for r in dev]
    optimised = optimizer.compile(program, trainset=trainset, valset=trainset)
    instruction = optimised.signature.instructions
    stats = getattr(optimised, "detailed_results", None)
    record = {"arm": args.arm, "auto": args.auto, "instruction": instruction, "baseline_instruction": ROUTER_SYSTEM.strip(),
              "dev_ids": [r.id for r in dev], "held_ids": [r.id for r in held],
              "candidates": [{"score": s, "instruction": c["predict"].get("instructions") if isinstance(c, dict) else str(c)}
                             for c, s in zip(getattr(stats, "candidates", []), getattr(stats, "val_aggregate_scores", []), strict=False)]
              if stats else []}  # fmt: skip
    out_path.write_text(json.dumps(record, indent=1))
    print("found instruction:\n", instruction)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
