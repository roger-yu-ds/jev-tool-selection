"""Two cheap baselines: is this test tool selection, or intent classification? (free, no API calls)

    uv run python scripts/baseline_intent_nb.py

1. **Always hand off.** Pick hand_off_to_human for every message.
2. **Naive Bayes intent → tool.** A bag-of-words naive Bayes (words and word pairs, add-one smoothing)
   trained on the rest of Bitext guesses each message's intent; the tool is then the first acceptable
   tool for that intent in the answer key (hand_off_to_human for intents no tool covers).

Training data: every Bitext row of the 27 intents, minus the 1,200 messages in our two samples and minus
any row with token Jaccard >= 0.6 to a fresh message (the same near-duplicate rule as
scripts/make_fresh_sample.py). The baseline is supervised and reads the answer key, so it isn't a fair
rival to zero-shot selectors that only see tool descriptions; it shows how far "recognise the intent,
look up the tool" goes on template-generated data with a per-intent key. Writes data/baselines.json.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_fresh_sample import fill, jaccard, load_bitext, tokens

import toolpick as tp

OUT = ROOT / "data" / "baselines.json"


def features(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return words + [f"{a}_{b}" for a, b in pairwise(words)]


class NaiveBayes:
    """Multinomial naive Bayes: argmax over intents of log P(intent) + sum log P(feature | intent)."""

    def fit(self, texts: list[str], labels: list[str]) -> NaiveBayes:
        self.prior = Counter(labels)
        self.counts = {k: Counter() for k in self.prior}
        for text, label in zip(texts, labels, strict=True):
            self.counts[label].update(features(text))
        self.vocab = len(set().union(*self.counts.values()))
        self.totals = {k: sum(c.values()) for k, c in self.counts.items()}
        return self

    def predict(self, text: str) -> str:
        f = features(text)

        def score(k: str) -> float:
            denom = self.totals[k] + self.vocab
            return math.log(self.prior[k]) + sum(math.log((self.counts[k][x] + 1) / denom) for x in f)

        return max(self.prior, key=score)


def main() -> None:
    fresh, first = tp.requests("fresh"), tp.requests("first")
    lab = tp.labels()
    held = {r.text for r in fresh} | {r.text for r in first}
    df = load_bitext()
    df = df[df.intent.isin(lab)].drop_duplicates("instruction")
    df = df.assign(text=df.instruction.map(fill))
    df = df[~df.text.isin(held)]
    fresh_tokens = [tokens(r.text) for r in fresh]
    near = [max(jaccard(tokens(t), f) for f in fresh_tokens) >= 0.6 for t in df.text]
    train = df[[not x for x in near]]
    print(f"training rows: {len(train):,} (dropped {sum(near):,} near-duplicates of fresh messages)")

    nb = NaiveBayes().fit(list(train.text), list(train.intent))
    guess = [nb.predict(r.text) for r in fresh]
    intent_acc = sum(g == r.intent for g, r in zip(guess, fresh, strict=True)) / len(fresh)
    nb_right = [tp.outcome(lab[g]["acceptable"][0], lab[r.intent]) == "right" for g, r in zip(guess, fresh, strict=True)]
    hand_right = [tp.outcome(tp.HAND, lab[r.intent]) == "right" for r in fresh]
    confusions = Counter((r.intent, g) for g, r in zip(guess, fresh, strict=True) if g != r.intent).most_common(8)

    jev = {r["id"]: r["right"] for r in tp.scored_rows("fresh/jev")}
    son = {r["id"]: r["right"] for r in tp.scored_rows("fresh/native_sonnet")}
    ids = [r.id for r in fresh]
    nb_by_id = dict(zip(ids, nb_right, strict=True))
    out = {
        "always_hand_off": sum(hand_right) / len(fresh),
        "nb_intent_accuracy": intent_acc,
        "nb_tool_correct": sum(nb_right) / len(fresh),
        "nb_vs_jev": tp.mcnemar([nb_by_id[i] for i in ids], [jev[i] for i in ids]),
        "nb_vs_sonnet": tp.mcnemar([nb_by_id[i] for i in ids], [son[i] for i in ids]),
        "nb_top_confusions": [[a, b, n] for (a, b), n in confusions],
        "train_rows": len(train),
    }
    print(f"always hand off:          {out['always_hand_off']:.1%} correct")
    print(f"naive Bayes intent:       {intent_acc:.1%} of intents right")
    print(f"naive Bayes intent→tool:  {out['nb_tool_correct']:.1%} correct "
          f"(Jev {sum(jev.values()) / len(jev):.1%}, p = {out['nb_vs_jev']['p']:.2g}; "
          f"Sonnet {sum(son.values()) / len(son):.1%}, p = {out['nb_vs_sonnet']['p']:.2g})")  # fmt: skip
    print("most common intent confusions (true → guessed):", ", ".join(f"{a}→{b} ({n})" for (a, b), n in confusions))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
