"""Jev variants side by side: the ask option and the input summaries, on both samples (free).

    uv run python scripts/compare_jev_variants.py

Fresh 600: Jev's final run and its three variants (`run_router.py --arm jev [--ask] [--params]
--sample fresh`), against Sonnet, under the answer key as is and the ask-first rule
(`toolpick.scoring.stipulated_right`). Development 600: the same variants on the first sample, against
Jev's final setup there (the customer-wording experiment, dev and held-out halves together), as the
check that a change isn't tuned to the fresh messages. Writes data/jev_variants.json for the deck.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp
from toolpick.scoring import stipulated_right

FRESH = {"Jev": "fresh/jev", "Jev + need more information": "fresh/jev_ask", "Jev + inputs": "fresh/jev_params",
         "Jev + inputs + need more information": "fresh/jev_ask_params", "Sonnet 5.5": "fresh/native_sonnet"}  # fmt: skip
DEV = {"Jev": ["experiments/jev_descriptions_dev", "experiments/jev_descriptions_held"],
       "Jev + need more information": ["experiments/jev_ask_first"], "Jev + inputs": ["experiments/jev_params_first"],
       "Jev + inputs + need more information": ["experiments/jev_ask_params_first"]}  # fmt: skip
OUT = ROOT / "data" / "jev_variants.json"


def rows_of(runs: list[str]) -> dict[str, dict]:
    return {r["id"]: r for run in runs for r in tp.scored_rows(run)}


def main() -> None:
    fresh = {k: rows_of([v]) for k, v in FRESH.items()}
    ids = sorted(set.intersection(*(set(r) for r in fresh.values())))
    son = fresh["Sonnet 5.5"]
    out = {"fresh": {}, "dev": {}, "n_fresh": len(ids)}
    print(f"fresh 600 (n = {len(ids)})")
    print(f"{'':38s} {'key':>6s} {'ask-1st':>7s} {'harm':>5s} {'hands':>5s} {'asks':>5s} {'$/1k':>6s} {'time':>5s}  vs Sonnet (key, ask-first)")
    for k, R in fresh.items():
        right = [R[i]["right"] for i in ids]
        strict = [stipulated_right(R[i]) for i in ids]
        asks = sum(R[i]["pick"] == tp.ASK or R[i]["kind"] == "text" for i in ids) / len(ids)
        s = tp.summarise([R[i] for i in ids])
        vs = (tp.mcnemar(right, [son[i]["right"] for i in ids]), tp.mcnemar(strict, [stipulated_right(son[i]) for i in ids]))
        out["fresh"][k] = {"correct": sum(right) / len(ids), "strict": sum(strict) / len(ids), "harmful": s["harmful"],
                           "handoff": s["handoff"], "asks": asks, "cost_per_1000": s["cost_per_1000"],
                           "latency_median_s": s["latency_median_s"], "input_tokens_median": s["input_tokens_median"],
                           "p_vs_sonnet": vs[0]["p"], "p_vs_sonnet_strict": vs[1]["p"],
                           "confidence_median": statistics.median(R[i]["confidence"] for i in ids) if k != "Sonnet 5.5" else None}  # fmt: skip
        o = out["fresh"][k]
        print(f"{k:38s} {o['correct']:6.1%} {o['strict']:7.1%} {o['harmful']:5.1%} {o['handoff']:5.0%} {asks:5.0%} "
              f"{o['cost_per_1000']:6.2f} {o['latency_median_s']:5.2f}  p={o['p_vs_sonnet']:.2g}, {o['p_vs_sonnet_strict']:.2g}")  # fmt: skip
    dev = {k: rows_of(v) for k, v in DEV.items()}
    dids = sorted(set.intersection(*(set(r) for r in dev.values())))
    base = [dev["Jev"][i]["right"] for i in dids]
    print(f"\ndevelopment 600 (n = {len(dids)}), against Jev's final setup")
    for k, R in dev.items():
        right = [R[i]["right"] for i in dids]
        strict = sum(stipulated_right(R[i]) for i in dids) / len(dids)
        p = tp.mcnemar(base, right)["p"] if k != "Jev" else None
        out["dev"][k] = {"correct": sum(right) / len(dids), "strict": strict, "p_vs_jev": p}
        print(f"{k:38s} {sum(right) / len(dids):6.1%} {strict:7.1%}" + (f"  p={p:.2g}" if p is not None else ""))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
