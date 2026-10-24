"""Score CLM's run against Jev and the LLMs with toolpick's own scoring (free, no API calls).

    uv run --no-sync python clm/score_clm.py [--run fresh/clm | --run experiments/clm_first]
    uv run --no-sync python clm/score_clm.py --file clm/cache/clm_names.jsonl     # a --variant diagnostic

Correct / benign / harmful, hand-offs, latency, exact McNemar against Jev and Sonnet on the same
messages (`toolpick.paired`), the ask-first rule (`toolpick.scoring.stipulated_right`), per-intent
misses and CLM's most common wrong picks. Uses the audited key, like every number in the talk.
"median s" includes Jev's argument-filling call (as in the README); "pick s" is the pick alone.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp
from toolpick.scoring import parse_run_name, stipulated_right

FRESH = {"CLM-8B (local)": "fresh/clm", "Jev": "fresh/jev", "Claude Sonnet 5.5": "fresh/native_sonnet",
         "Claude Haiku 5.5": "fresh/native_haiku", "DeepSeek V4.1 Flash": "fresh/native_deepseek"}  # fmt: skip
FIRST_JEV = ["experiments/jev_descriptions_dev", "experiments/jev_descriptions_held"]  # Jev's final setup, first 600


def rows_of(runs: list[str]) -> dict[str, dict]:
    return {r["id"]: r for run in runs for r in tp.scored_rows(run)}


def rows_from_file(path: str) -> dict[str, dict]:
    """``scored_rows`` for a run file outside runs/ (the ``run_clm.py --variant`` diagnostics)."""
    import json

    from toolpick.data import all_requests_by_id

    lab, by_id, out = tp.labels(), all_requests_by_id(), {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        req, kind = by_id[r["id"]], r.get("kind", "tool_call")
        o = tp.outcome(r["pick"], lab[req.intent], kind)
        out[r["id"]] = {"id": r["id"], "intent": req.intent, "gap": lab[req.intent]["gap"], "text": req.text, "pick": r["pick"],
                        "kind": kind, "outcome": o, "right": o == "right", "wrong_call": o in ("wrong", "bluffed"),
                        "verdict": tp.VERDICT[o], "confidence": r.get("confidence"), "latency_s": r["latency_s"],
                        "pick_latency_s": r["latency_s"], "input_tokens": r.get("input_tokens") or 0, "output_tokens": 0,
                        "cost_usd": 0.0, "top": r.get("top")}  # fmt: skip
    return out


def table(named: dict[str, dict[str, dict]], ids: list[str], ref: str) -> None:
    print(f"{'selector':22s} {'n':>4s} {'correct':>8s} {'benign':>7s} {'harmful':>8s} {'hand-off':>8s} {'ask-1st':>8s} "
          f"{'median s':>8s} {'p95 s':>6s} {'pick s':>6s}   vs {ref} (only this / only {ref}, p)")  # fmt: skip
    base = named[ref]
    for k, R in named.items():
        rows = [R[i] for i in ids]
        s = tp.summarise(rows)
        strict = sum(map(stipulated_right, rows)) / len(rows)
        line = (f"{k:22s} {s['n']:4d} {s['correct']:8.1%} {s['benign']:7.1%} {s['harmful']:8.1%} {s['handoff']:8.1%} "
                f"{strict:8.1%} {s['latency_median_s']:8.2f} {s['latency_p95_s']:6.2f} "
                f"{statistics.median(r['pick_latency_s'] for r in rows):6.2f}")  # fmt: skip
        if k != ref:
            m = tp.mcnemar([R[i]["right"] for i in ids], [base[i]["right"] for i in ids])
            ms = tp.mcnemar([stipulated_right(R[i]) for i in ids], [stipulated_right(base[i]) for i in ids])
            line += f"   {m['only_a']}/{m['only_b']}, p={m['p']:.2g}; ask-first p={ms['p']:.2g}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fresh/clm", help="a run under runs/")
    ap.add_argument("--file", default=None, help="score this .jsonl instead (a --variant diagnostic)")
    args = ap.parse_args()
    if args.file:
        clm, label = rows_from_file(args.file), args.file
    else:
        clm, label = rows_of([args.run]), args.run
        print(f"parse_run_name({args.run!r}) = {parse_run_name(args.run)}")
        for r in clm.values():  # the run file has the top-8 list; scored_rows drops it
            r["top"] = tp.load_run(args.run)[r["id"]].get("top")
    if not clm:
        raise SystemExit(f"no rows for {label}")
    fresh = all(i.startswith("F") for i in clm)
    if fresh:
        named = {k: rows_of([v]) for k, v in FRESH.items()}
        named["CLM-8B (local)"] = clm
    else:
        named = {"CLM-8B (local)": clm, "Jev": rows_of(FIRST_JEV)}
    ids = sorted(set.intersection(*(set(R) for R in named.values())))
    print(f"\n{label}: {len(clm)} rows; {len(ids)} messages answered by every selector below (audited key)\n")
    for ref in ("Jev", "Claude Sonnet 5.5"):
        if ref in named:
            table(named, ids, ref)
            print()
    jev = named["Jev"]
    for name in ("Jev", "Claude Sonnet 5.5"):
        if name in named:
            p = tp.paired(args.run, FRESH[name]) if not args.file and fresh and name in FRESH else None
            if p:
                print(f"CLM - {name}: {p['diff']:+.1%} (95% bootstrap CI {p['ci'][0]:+.1%} to {p['ci'][1]:+.1%})")
    agree = sum(clm[i]["pick"] == jev[i]["pick"] for i in ids) / len(ids)
    print(f"CLM picks the same tool as Jev on {agree:.1%} of messages")
    mix = Counter(clm[i]["outcome"] for i in ids)
    print("CLM outcomes:", {o: mix[o] for o in tp.OUTCOMES})
    conf = [clm[i]["confidence"] for i in ids]
    rc = [clm[i]["confidence"] for i in ids if clm[i]["right"]]
    wc = [clm[i]["confidence"] for i in ids if not clm[i]["right"]]
    print(f"CLM confidence: median {statistics.median(conf):.2f} (right {statistics.median(rc):.2f}, "
          f"not right {statistics.median(wc) if wc else float('nan'):.2f})")
    print(f"CLM input tokens per message (encoder, state only): median {statistics.median(r['input_tokens'] for r in clm.values()):.0f}")

    print("\nper intent (CLM right / n, Jev right), worst first:")
    by = {}
    for i in ids:
        d = by.setdefault(clm[i]["intent"], {"n": 0, "clm": 0, "jev": 0, "picks": Counter(), "gap": clm[i]["gap"]})
        d["n"] += 1
        d["clm"] += clm[i]["right"]
        d["jev"] += jev[i]["right"]
        if not clm[i]["right"]:
            d["picks"][clm[i]["pick"]] += 1
    for intent, d in sorted(by.items(), key=lambda kv: kv[1]["clm"] / kv[1]["n"]):
        if d["clm"] < d["n"]:
            top = ", ".join(f"{k.split('.')[-1]} {v}" for k, v in d["picks"].most_common(3))
            print(f"  {intent:26s} {'gap' if d['gap'] else '   '} CLM {d['clm']:3d}/{d['n']:3d}  Jev {d['jev']:3d}   misses: {top}")
    wrong = Counter((clm[i]["outcome"], clm[i]["pick"]) for i in ids if not clm[i]["right"])
    print("\nCLM's most common misses (outcome, pick):")
    for (o, p), c in wrong.most_common(12):
        print(f"  {c:4d}  {o:10s} {p}")
    picks = Counter(clm[i]["pick"] for i in ids)
    print(f"\nCLM used {len(picks)} of 91 options; the five most picked: "
          + ", ".join(f"{k.split('.')[-1]} {v}" for k, v in picks.most_common(5)))
    jpicks = Counter(jev[i]["pick"] for i in ids)
    print(f"Jev used {len(jpicks)}; FAQ tool picked by CLM {picks[tp.FAQ]} times, by Jev {jpicks[tp.FAQ]}")
    import math

    margins = [math.log(r["top"][0][1] / max(r["top"][1][1], 1e-12)) for r in (clm[i] for i in ids) if r.get("top")]
    if margins:
        close = sum(m < 1.0 for m in margins) / len(margins)
        print(f"top-1 vs top-2 logit margin (= 100 x cosine gap): median {statistics.median(margins):.2f}; "
              f"{close:.0%} of picks below 1.0 (a cosine gap under 0.01)")
    if not args.file:
        print("\nconfidence gate (hand off below t; toolpick.gate_curve): t, share acted on, wrong of acted, correct")
        for g in tp.gate_curve(args.run, thresholds=(0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
            print(f"  t={g['t']:.1f}  acted {g['acted']:5.1%}  wrong of acted {g['wrong_of_acted']:5.1%}  correct {g['right']:5.1%}")


if __name__ == "__main__":
    main()
