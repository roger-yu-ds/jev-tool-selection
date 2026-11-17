"""Write the three README charts to docs/img/ from the committed runs (free, no API calls).

    uv run python scripts/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import toolpick as tp
from toolpick import charts

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
RUNS = {"Jev": "fresh/jev", "Claude Sonnet 5.5": "fresh/native_sonnet", "Claude Haiku 5.5": "fresh/native_haiku",
        "DeepSeek V4.1 Flash": "fresh/native_deepseek"}  # fmt: skip
KINDS = ["a tool can do it", "no tool can (hand off)"]


def main() -> None:
    scored = {name: pd.DataFrame(tp.scored_rows(run)) for name, run in RUNS.items()}
    table = pd.DataFrame({name: {"correct": d.right.mean(), "benign": (d.verdict == "benign").mean(), "harmful": d.wrong_call.mean(),
                                 "harmful_n": int(d.wrong_call.sum()), "cost": 1000 * d.cost_usd.mean()}
                          for name, d in scored.items()}).T.astype({"harmful_n": int})  # fmt: skip
    split = pd.DataFrame({name: {KINDS[0]: d[~d.gap].right.mean(), KINDS[1]: d[d.gap].right.mean()} for name, d in scored.items()}).T
    figures = {
        "outcomes.png": charts.outcome_bars(table, "Jev and Sonnet are level, and almost every miss is harmless",
                                            "Final setup, 600 fresh messages. Correct, benign and harmful share of each selector's answers"),
        "split.png": charts.split_bars(split, "Jev's lead comes from knowing when nothing fits",
                                       "Correct answers on the 289 messages a tool can handle and the 311 that none can"),
        "cost.png": charts.cost_scatter(table, "The same accuracy for a fraction of the price",
                                        "Correct against billed cost per 1,000 messages, final setup, 600 fresh messages"),
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    for name, chart in figures.items():
        chart.save(OUT / name, scale_factor=2)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
