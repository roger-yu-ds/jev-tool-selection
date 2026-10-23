"""Sanity checks on runs/*.jsonl: the bugs that would make a result wrong without making it look wrong.

    uv run python scripts/check_runs.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp

names = {t["name"] for t in tp.tools()}
schema = {t["name"]: t["inputSchema"] for t in tp.tools()}
ids = {r.id for r in tp.requests()}
fresh_ids = {r.id for r in tp.requests("fresh")}


def check_labels() -> None:
    bad = []
    for v in ("audited", "first_pass"):
        for k, lab in tp.labels(v).items():
            for t in lab["acceptable"] + lab["partial"]:
                if t not in names:
                    bad.append((v, k, t))
            if set(lab["acceptable"]) & set(lab["partial"]):
                bad.append((v, k, "acceptable and partial overlap"))
    print("labels:", "ok" if not bad else bad)


def check_run(name: str) -> None:
    rows = [json.loads(x) for x in (tp.RUNS / f"{name}.jsonl").read_text().splitlines() if x.strip()]
    rid = [r["id"] for r in rows]
    dup = [k for k, c in Counter(rid).items() if c > 1]
    expected = fresh_ids if name.startswith("fresh/") else ids
    missing = sorted(expected - set(rid))
    unknown_ids = sorted(set(rid) - expected)
    unparsed = [r for r in rows if r["pick"] is None and r.get("kind", "tool_call") == "tool_call"]
    not_in_catalog = [r for r in rows if r["pick"] and r["pick"] not in names]
    truncated = [r for r in rows if r.get("finish_reason") == "length"]
    multi = [r for r in rows if len(r.get("tool_calls") or []) > 1]
    cached_rate = sum(1 for r in rows if (r.get("cached_tokens") or 0) > 0) / len(rows) if rows else 0
    providers = Counter(r.get("provider") for r in rows)
    models = Counter(r.get("model") or (r.get("response") or {}).get("model") for r in rows)
    reasoning = [((r.get("response") or {}).get("usage") or {}).get("completion_tokens_details") or {} for r in rows]
    reasoning_tok = [x.get("reasoning_tokens") for x in reasoning if x and x.get("reasoning_tokens")]
    lat = sorted(r["latency_s"] for r in rows)
    # arguments against schema (native only): required fields present, no unknown fields when additionalProperties is false
    arg_issues = Counter()
    for r in rows:
        for tc in r.get("tool_calls") or []:
            s = schema.get(tc["name"]) or {}
            args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}
            if "_unparsed" in args:
                arg_issues["unparseable JSON"] += 1
                continue
            for req in s.get("required", []):
                if req not in args:
                    arg_issues[f"missing required `{req}` for {tc['name'].split('.')[-1]}"] += 1
            if s.get("additionalProperties") is False:
                for k in args:
                    if k not in (s.get("properties") or {}):
                        arg_issues[f"unknown field `{k}` for {tc['name'].split('.')[-1]}"] += 1
    print(f"\n== {name}: {len(rows)} rows")
    for label, val in (("duplicate ids", dup), ("missing ids", missing[:5] + ([f"... {len(missing)}"] if len(missing) > 5 else [])),
                       ("unknown ids", unknown_ids), ("unparsed picks", [(r['id'], (r.get('raw') or '')[:60]) for r in unparsed][:5]),
                       ("picks not in catalogue", [(r['id'], r['pick']) for r in not_in_catalog][:5]),
                       ("truncated (finish_reason=length)", [r['id'] for r in truncated][:5]),
                       ("multiple tool calls", [(r['id'], [c['name'].split('.')[-1] for c in r['tool_calls']]) for r in multi][:4])):  # fmt: skip
        if val:
            print(f"  {label}: {val}" + (f" (total {len(unparsed)})" if label == "unparsed picks" else "")
                  + (f" (total {len(truncated)})" if label.startswith("truncated") else "")
                  + (f" (total {len(multi)})" if label.startswith("multiple") else ""))
    print(f"  cached on {cached_rate:.0%} of calls · providers {dict(providers)} · model {dict(models)}")
    if reasoning_tok:
        print(f"  reasoning tokens: median {sorted(reasoning_tok)[len(reasoning_tok) // 2]}, on {len(reasoning_tok)} calls")
    print(f"  latency p50 {lat[len(lat) // 2]:.2f}s p95 {lat[int(.95 * (len(lat) - 1))]:.2f}s max {lat[-1]:.1f}s · "
          f"cost ${sum(r.get('cost_usd') or 0 for r in rows):.3f}")
    if arg_issues:
        print("  argument issues:", dict(arg_issues.most_common(6)))
    picks = Counter(r["pick"] for r in rows)
    print(f"  top picks: {[(k and k.split('.')[-1], v) for k, v in picks.most_common(5)]}")


def check_jev_rerun() -> None:
    old = tp.RUNS / "archive" / "jev.jsonl"
    if not old.exists():
        return
    a = {json.loads(x)["id"]: json.loads(x)["pick"] for x in old.read_text().splitlines() if x.strip()}
    b = {r["id"]: r["pick"] for r in tp.load_run("jev").values()}
    same = sum(a[k] == b[k] for k in a if k in b)
    print(f"\njev rerun vs archived run: {same}/{len(a)} identical picks")


if __name__ == "__main__":
    check_labels()
    for n in tp.available_runs() + tp.available_runs("fresh"):
        check_run(n)
    check_jev_rerun()
