"""Write notebooks/rerun_failures.ipynb: re-run sampled failures from the fresh run, to check they repeat.

    uv run python scripts/build_failures_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp

OUT = ROOT / "notebooks" / "rerun_failures.ipynb"
RUNS = {"jev": "fresh/jev", "deepseek": "fresh/native_deepseek", "haiku": "fresh/native_haiku", "sonnet": "fresh/native_sonnet"}
WHAT = {
    "handed_off": "handed off to a person, although a real tool could do the job (benign)",
    "asked": "asked the customer a question instead of calling anything, where that isn't the right first step (benign)",
    "defensible": "made a sensible read-only first step that isn't the right tool (benign)",
    "wrong": "called a tool that can't do the job, where a right tool exists (harmful)",
    "bluffed": "called a tool on a request no tool can handle (harmful)",
}

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Do the failures repeat?

Every model got some messages wrong in the final run (600 fresh messages, policy prompt, customer-worded tool
descriptions, LLMs as native tool-calling agents). This notebook takes a few of those failures for each
**model × failure type**, sends the *same* message again with the *same* setup, and shows whether the model makes the
same choice again.

All runs used temperature 0, but that doesn't guarantee identical answers: providers batch requests, Sonnet 5.5 always
reasons a little, and Jev is a separate service. If a failure repeats every time, it's a property of the model and setup;
if it flips, it's noise, and a single run overstates how sure we are about that message.

**How to use it:** run the setup cells, then run any model × failure-type cell as often as you like. Each run makes
`REPEATS` calls per sampled message and adds them to the summary at the bottom.

**Cost:** per call about US$0.0003 for Jev and DeepSeek, $0.001 for Haiku and $0.009 for Sonnet, plus about $0.19 the
first time Sonnet is called in any five-minute window (it writes the 76k-token prompt cache). A full pass over every
cell with the defaults is roughly $0.40. Set `LIVE = False` to browse the samples without calling anything.
""")

code("""
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import Markdown, display

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parents[1] / ".env")  # when this folder sits inside the course repo

import toolpick as tp
from toolpick.live import ask

LIVE = True        # False: show the samples, make no calls
REPEATS = 3        # calls per sampled message, per run of a cell
PER_COMBO = 3      # messages sampled for each model × failure type
SEED = 0           # change it to draw different messages

RUNS = {"jev": "fresh/jev", "deepseek": "fresh/native_deepseek", "haiku": "fresh/native_haiku", "sonnet": "fresh/native_sonnet"}
LAB = tp.labels()
pd.set_option("display.max_colwidth", 120)
print("keys:", {k: bool(os.environ.get(k)) for k in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY")})
""")

md("""
## The failures

Every incorrect answer in the final run, by model and type. The types:

- **handed_off**: handed off to a person although a real tool could do the job (benign)
- **asked**: asked the customer a question where that isn't the right first step (benign)
- **defensible**: a sensible read-only first step that isn't the right tool, e.g. the FAQ search (benign)
- **wrong**: a tool that can't do the job, where a right tool exists (harmful)
- **bluffed**: a tool on a request no tool can handle (harmful)
""")

code("""
rows = pd.DataFrame([r | {"model": m} for m, run in RUNS.items() for r in tp.scored_rows(run)])
failed = rows[~rows["right"]]
pd.crosstab(failed["model"], failed["outcome"], margins=True, margins_name="total")
""")

code("""
def short(pick):
    return "(asks a question)" if pick is None or pd.isna(pick) else pick.split(".")[-1]


def sample(model, outcome, k=PER_COMBO, seed=SEED):
    \"\"\"Up to k failures of one type for one model, spread across intents.\"\"\"
    pool = failed[(failed["model"] == model) & (failed["outcome"] == outcome)].sample(frac=1, random_state=seed)
    picked = pool.drop_duplicates("intent").head(k)
    if len(picked) < k:
        picked = pd.concat([picked, pool.drop(picked.index).head(k - len(picked))])
    return picked


def show(model, outcome):
    s = sample(model, outcome)
    view = s[["id", "intent", "text", "pick", "reply"]].assign(pick=s["pick"].map(short), reply=s["reply"].str.slice(0, 140))
    if model == "jev":
        view["confidence"] = s["confidence"].round(2)
    display(view.rename(columns={"pick": "original pick", "reply": "original reply"}).reset_index(drop=True))
    return s


RESULTS = []  # every re-run call made in this session


def rerun(model, outcome, repeats=REPEATS):
    \"\"\"Send each sampled message again `repeats` times with the final setup and compare with the original.\"\"\"
    s = sample(model, outcome)
    if not LIVE:
        print("LIVE is False: no calls made")
        return None
    out, spent = [], 0.0
    for _, r in s.iterrows():
        picks, outcomes = [], []
        for _ in range(repeats):
            new = ask(model, r["text"])
            spent += new["cost_usd"]
            o = tp.outcome(new["pick"], LAB[r["intent"]], new["kind"], [c["name"] for c in new.get("tool_calls") or []])
            picks.append(short(new["pick"]))
            outcomes.append(o)
            RESULTS.append({"model": model, "type": outcome, "id": r["id"], "intent": r["intent"], "original": short(r["pick"]),
                            "rerun": short(new["pick"]), "outcome": o, "same_pick": short(new["pick"]) == short(r["pick"]),
                            "still_wrong": o != "right", "confidence": new.get("confidence"), "reply": (new.get("text") or "")[:160]})
        out.append({"id": r["id"], "intent": r["intent"], "message": r["text"], "original": short(r["pick"]),
                    "re-runs": " · ".join(picks), "same pick": f"{sum(p == short(r['pick']) for p in picks)}/{repeats}",
                    "still wrong": f"{sum(o != 'right' for o in outcomes)}/{repeats}"})
    display(pd.DataFrame(out))
    print(f"{len(s) * repeats} calls, US${spent:.4f}")
    return pd.DataFrame(out)
""")

for model, run in RUNS.items():
    rows = tp.scored_rows(run)
    kinds = sorted({r["outcome"] for r in rows if not r["right"]}, key=tp.OUTCOMES.index)
    if not kinds:
        continue
    name = tp.MODEL_NAMES[model]
    mode = "answers one multiple-choice question" if model == "jev" else "runs as a native tool-calling agent"
    md(f"## {name}\n\n{name} {mode}. {sum(not r['right'] for r in rows)} of {len(rows)} answers were incorrect.")
    for kind in kinds:
        n = sum(r["outcome"] == kind for r in rows)
        md(f"### {name} · `{kind}` ({n} in the run)\n\n{name} {WHAT[kind]}.")
        code(f'_ = show("{model}", "{kind}")\n_ = rerun("{model}", "{kind}")')

md("""
## Summary of every re-run in this session

*same pick*: the re-run chose exactly what the original run chose. *still wrong*: the re-run is still incorrect
(possibly a different wrong answer). A failure that repeats on every call is systematic; one that flips is noise.
""")

code("""
if RESULTS:
    res = pd.DataFrame(RESULTS)
    summary = res.groupby(["model", "type"]).agg(calls=("id", "size"), messages=("id", "nunique"),
                                                 same_pick=("same_pick", "mean"), still_wrong=("still_wrong", "mean"))
    display(summary.style.format({"same_pick": "{:.0%}", "still_wrong": "{:.0%}"}))
    flipped = res[~res["still_wrong"]]
    if len(flipped):
        display(Markdown(f"**{len(flipped)} re-runs came out correct.** These are the failures that look like noise:"))
        display(flipped[["model", "type", "id", "intent", "original", "rerun", "reply"]])
else:
    print("no re-runs yet: run some of the cells above")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"}})
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT} ({len(cells)} cells)")
