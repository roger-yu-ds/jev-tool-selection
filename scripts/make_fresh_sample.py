"""Draw a fresh, varied 600-message sample from Bitext: data/requests_fresh.jsonl.

    uv run python scripts/make_fresh_sample.py

The first 600 (data/requests.jsonl) came from a hash-selected 4% slice of Bitext. The fresh sample:

- excludes that whole slice, so no message can repeat;
- drops any message whose words overlap a first-sample message too closely (token Jaccard
  >= 0.6 after lowercasing), so close rewordings of the same template don't count as new;
- takes 22 or 23 messages per intent (all 27), and within an intent picks greedily for variety
  (farthest-point sampling on token Jaccard distance, seeded), so the set spans Bitext's phrasing
  styles rather than its most common template.

Placeholders ({{Order Number}} and so on) are filled the same way as the first sample. The tool
descriptions with customer wording (`toolpick.data.CUSTOMER_WORDS`) were written while looking at
the first 600 only, so this sample is a clean test of them.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolpick import requests

URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
       "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")  # fmt: skip
CACHE = ROOT / "data" / "cache" / "bitext_customer_support_0000.parquet"
OUT = ROOT / "data" / "requests_fresh.jsonl"
N = 600


def _h(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


FILL = {
    "{{Order Number}}": lambda h: f"ORD-{400000 + h % 99999}",
    "{{Account Type}}": lambda h: ("premium", "standard", "business")[h % 3],
    "{{Account Category}}": lambda h: ("Platinum", "Gold", "Basic")[h % 3],
    "{{Person Name}}": lambda h: ("Ana", "Ben", "Chloe", "Dev", "Eli")[h % 5],
    "{{Refund Amount}}": lambda h: str(20 + h % 180),
    "{{Currency Symbol}}": lambda h: "$",
    "{{Delivery City}}": lambda h: ("Ballarat", "Geelong", "Newcastle")[h % 3],
    "{{Delivery Country}}": lambda h: "Australia",
    "{{Invoice Number}}": lambda h: f"INV-2026-{h % 9000 + 1000}",
}


def fill(text: str) -> str:
    h = _h(text)
    for k, f in FILL.items():
        text = text.replace(k, f(h))
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def load_bitext() -> pd.DataFrame:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(httpx.get(URL, follow_redirects=True, timeout=120).content)
    return pd.read_parquet(CACHE)


def main() -> None:
    df = load_bitext()
    first = requests()
    first_tokens = [tokens(r.text) for r in first]
    intents = sorted({r.intent for r in first})
    in_first_slice = df["instruction"].map(lambda t: _h("nb24", t) % 1000 < 40)
    pool = df[~in_first_slice & df["intent"].isin(intents)].drop_duplicates("instruction")
    per = {k: N // len(intents) + (1 if i < N % len(intents) else 0) for i, k in enumerate(intents)}
    chosen = []
    for intent in intents:
        cands = []
        for t in pool[pool["intent"] == intent]["instruction"]:
            text = fill(t)
            tk = tokens(text)
            if max(jaccard(tk, f) for f in first_tokens) >= 0.6:
                continue
            cands.append((text, tk))
        cands.sort(key=lambda c: _h("fresh", c[0]))  # seeded order
        picked = [cands[0]]
        dist = [1 - jaccard(c[1], picked[0][1]) for c in cands]
        while len(picked) < per[intent]:
            i = max(range(len(cands)), key=lambda j: dist[j])
            picked.append(cands[i])
            dist = [min(d, 1 - jaccard(c[1], cands[i][1])) for d, c in zip(dist, cands, strict=True)]
        chosen += [(intent, text) for text, _ in picked]
        print(f"{intent:26s} pool {len(cands):4d} -> {len(picked)}")
    chosen.sort(key=lambda x: _h("fresh-order", x[1]))
    rows = [{"id": f"F{i:04d}", "intent": intent, "text": text} for i, (intent, text) in enumerate(chosen)]
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    max_overlap = max(max(jaccard(tokens(r["text"]), f) for f in first_tokens) for r in rows)
    print(f"wrote {OUT.name}: {len(rows)} messages; highest word overlap with any first-sample message: {max_overlap:.2f}")


if __name__ == "__main__":
    main()
