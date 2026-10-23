"""Write notebooks/tool_selection.ipynb, the walkthrough for the audience.

    uv run python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "tool_selection.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Who picks the tool? A walkthrough

This notebook recomputes every number in the MeasureCamp talk from the committed runs, and shows you how to
make new calls if you have keys. It runs without any API key.

**The question.** An AI support assistant gets a customer message and 91 tools (90 copied verbatim from
official MCP servers, plus `hand_off_to_human`). Which tool should it call first? We compare four selectors:

- **Jev** (TypeSafe System One): a small model that answers a `choice` question with a probability per option.
- **DeepSeek V4.1 Flash, Claude Haiku 5.5, Claude Sonnet 5.5**, each in two modes: as a *router* (told to
  reply with one tool's name) and with *native tool calling* (the tools in the API's `tools` parameter).

The **final setup**, used for the talk's main results, runs on 600 *fresh* messages: every selector gets the same
routing policy, six tool descriptions get one line in customers' words, and the LLMs work as real agents (native
tool calling). The **development runs** on the first 600 messages show how we got there: plain and policy prompts,
the router mode, and a first-pass answer key that we later audited. The ranking depends on those choices, so this
notebook lets you see each one.
""")

code("""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))
import toolpick as tp

pd.set_option("display.max_colwidth", 90)
print(len(tp.requests("fresh")), "fresh messages ·", len(tp.tools()), "tools")
print("final runs:", tp.available_runs("fresh"))
print("development runs:", tp.available_runs())
""")

md("""
## 1. The data

Two samples of 600 from Bitext's customer-support dataset. The text is template-generated with deliberate typos;
the `intent` is Bitext's label, which we map to tools. The fresh sample shares no message with the first.
""")

code("""
from dataclasses import asdict

reqs = pd.DataFrame([asdict(r) for r in tp.requests("fresh")])
display(reqs.sample(8, random_state=1))
reqs["intent"].value_counts().to_frame("messages").T
""")

md("""
The tools. Every name, description and schema was taken from the vendor: by running the official package and
calling `tools/list`, by calling a live public Shopify store, or from official docs (Intercom). `data/REPORT.md`
has the provenance of each server. Note how uneven the prompt budget is.
""")

code("""
cat = pd.DataFrame(tp.catalog())[["qualified_name", "server", "description", "exactness"]]
display(cat.groupby("server").size().to_frame("tools").T)
tok = tp.tool_tokens_by_server()
print({k: f"{v / sum(tok.values()):.0%}" for k, v in sorted(tok.items(), key=lambda kv: -kv[1])}, "of the tool JSON")
cat[cat["server"] == "shopify_customer_account"]
""")

md("""
## 2. The labels

Per intent: the `acceptable` tools (correct) and the `partial` tools (a benign first step that doesn't finish the
job). **14 of the 27 intents have no real tool**; there the correct answer is `hand_off_to_human`. `ask_ok` marks
the intents where asking the customer a question is correct, because every correct tool needs details the message
doesn't give.

The default answer key is `audited`: every failure in the development runs was checked by hand, and each change has
its reason (the `changes` column). `first_pass` is what the study started with. Both live in `src/toolpick/data.py`.
""")

code("""
lab = tp.labels()
pd.DataFrame([{"intent": k, "no tool": v["gap"], "acceptable": ", ".join(t.split(".")[-1] for t in v["acceptable"]),
               "partial": ", ".join(t.split(".")[-1] for t in v["partial"]), "ask_ok": v["ask_ok"], "changes": " ".join(v["changes"])}
              for k, v in lab.items()]).sort_values(["no tool", "intent"])
""")

md("""
## 3. Scoring one decision

Every row of a run has the selector's pick (or no pick, if a native model replied in text). `tp.outcome` turns a
pick into one of six outcomes, which fold into three verdicts: **correct** (`right`), **benign** (`defensible`,
`handed_off`, `asked`: the customer waits, nothing breaks) and **harmful** (`wrong`, `bluffed`: a tool that can't
do what was asked, so it looks as if something happened).
""")

code("""
for pick in ["shopify_customer_account.get_order_status", "shopify_ucp.get_order", "hand_off_to_human",
             "shopify_storefront.search_shop_policies_and_faqs", None]:
    print(f"{str(pick):55s} track_order -> {tp.outcome(pick, lab['track_order']):12s}  cancel_order -> {tp.outcome(pick, lab['cancel_order'])}")
""")

md("""
## 4. The headline table

`scored_rows` joins a run with the answer key; `summarise` gives the rates. First the final setup on the fresh
messages, then the development runs. Change `version` to `"first_pass"` and watch the ranking move.
""")

code("""
def table(sample="fresh", version="audited"):
    rows = []
    for name in tp.available_runs(sample):
        s = tp.summarise(tp.scored_rows(name, version))
        rows.append({"run": name, "n": s["n"], "correct": s["correct"], "benign": s["benign"], "harmful": s["harmful"], "asked": s["asked"],
                     "handed_off": s["handoff"], "latency_s": s["latency_median_s"], "input_tokens": s["input_tokens_median"],
                     "$ per 1000": s["cost_per_1000"]})
    return pd.DataFrame(rows).set_index("run").sort_values("correct", ascending=False)


def show(t):
    return (t.style.format({"correct": "{:.1%}", "benign": "{:.1%}", "harmful": "{:.1%}", "asked": "{:.1%}", "handed_off": "{:.1%}",
                            "latency_s": "{:.2f}", "input_tokens": "{:,.0f}", "$ per 1000": "{:.2f}"})
             .bar(subset=["correct"], color="#8ab17d", vmin=0, vmax=1).bar(subset=["harmful"], color="#e9a3a3", vmin=0, vmax=1))


show(table("fresh"))
""")

code("""
pd.concat({"audited key": table("first")[["correct", "harmful"]], "first-pass key": table("first", "first_pass")[["correct", "harmful"]]},
          axis=1).style.format("{:.1%}")
""")

md("""
## 5. Paired comparisons

Two accuracies on the same messages aren't independent, so we don't compare them as if they were. `paired`
counts the messages where only one of the two was right (an exact McNemar test) and bootstraps the difference.
""")

code("""
pairs = []
todo = [("fresh/jev", f"fresh/native_{m}") for m in ("deepseek", "haiku", "sonnet")]
todo += [("jev" + s_, o + s_) for o in ("router_deepseek", "router_sonnet", "native_sonnet") for s_ in ("", "_policy")]
for a, other in todo:
        p = tp.paired(a, other)
        pairs.append({"jev vs": other, "n": p["n_pairs"], "jev": p["right_a"], "other": p["right_b"],
                      "only jev right": p["only_a"], "only other right": p["only_b"], "diff": p["diff"],
                      "95% CI": f"[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]", "p": p["p"]})
pd.DataFrame(pairs).style.format({"jev": "{:.1%}", "other": "{:.1%}", "diff": "{:+.1%}", "p": "{:.2g}"})
""")

md("""
## 6. Error analysis

Three views. First: when a selector is harmful, what does it reach for?
""")

code("""
def wrong_calls(name):
    rows = pd.DataFrame(tp.scored_rows(name))
    w = rows[rows["verdict"] == "harmful"]
    return (w.assign(pick=w["pick"].str.split(".").str[-1]).groupby(["intent", "pick"]).size()
             .sort_values(ascending=False).head(10).to_frame("n"))

display(wrong_calls("fresh/native_haiku"))
wrong_calls("fresh/native_sonnet")
""")

md("""
Second: Jev's errors and its confidence. When Jev is wrong it mostly hands off (benign). Its confidence tracks
whether it's right, which lets you send only the unsure messages to a bigger model.
""")

code("""
jev = pd.DataFrame(tp.scored_rows("fresh/jev"))
display(jev.groupby("outcome").size().to_frame("messages"))
jev["band"] = pd.cut(jev["confidence"], [0, .7, .9, 1.01], right=False, labels=["below 0.7", "0.7-0.9", "0.9+"])
jev.groupby("band", observed=True)["right"].agg(["size", "mean"]).rename(columns={"size": "messages", "mean": "correct"})
""")

md("""
Third: the native models often reply in text instead of calling a tool. On vague messages that's reasonable; the
router condition can't do it. Scoring the first turn only makes native accuracy a lower bound.
""")

code("""
nat = pd.DataFrame(tp.scored_rows("fresh/native_sonnet"))
asked = nat[nat["outcome"] == "asked"]
print(f"{len(asked)} of {len(nat)} replied in text")
asked[["intent", "text", "reply"]].sample(5, random_state=2)
""")

md("""
## 7. Browse disagreements

Pick two runs and look at the messages where they differ. This is the most useful thing to do with any eval:
the disagreements tell you whether the labels, the prompt or the model is at fault.
""")

code("""
a = pd.DataFrame(tp.scored_rows("fresh/jev")).set_index("id")
b = pd.DataFrame(tp.scored_rows("fresh/native_sonnet")).set_index("id")
both = a.join(b[["pick", "outcome", "reply"]], rsuffix="_sonnet", how="inner")
diff = both[both["pick"].fillna("-") != both["pick_sonnet"].fillna("-")]
print(len(diff), "disagreements")
diff[["intent", "text", "pick", "outcome", "pick_sonnet", "outcome_sonnet"]].head(12)
""")

md("""
## 8. Make your own calls (optional)

With `OPENROUTER_API_KEY` and/or `TYPESAFE_API_KEY` in `.env`, the scripts in `scripts/` run any arm on any
number of messages and append to `runs/`. A few messages cost cents:

```bash
uv run python scripts/run_router.py --arm jev --sample fresh --limit 5
uv run python scripts/run_native.py --arm haiku --sample fresh --limit 5
```

The cell below shows one Jev call end to end.
""")

code("""
import os
from dotenv import load_dotenv

load_dotenv(Path.cwd().parent / ".env")
if os.environ.get("TYPESAFE_API_KEY"):
    from toolpick.jev import Jev
    from toolpick.prompts import jev_instructions

    criteria = {t["name"]: t["description"] for t in tp.tools(customer_words=True)}
    out = Jev().choose("i wqant assistance checking purchase ORD-475206 status", jev_instructions(policy=True), criteria)
    a = out["answers"]["tool"]
    print(a["choice"], f"confidence {a['confidence']:.2f}", f"{out['latency_s']:.2f}s", f"${out['cost_usd']:.5f}")
    print(sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4])
else:
    print("no TYPESAFE_API_KEY: skipping the live call")
""")

md("""
## What to take away

1. Write the routing policy down and give it to every model; then you're measuring the model, not its guess at your policy.
2. Describe tools in your customers' words, not the vendor's.
3. Audit the answer key: check every failure by hand. Ours had nine mistakes.
4. Split wrong answers into benign and harmful. A hand-off and a bluff are not the same miss.
5. Compare on the same messages, with a paired test, and test the final setup on messages you didn't develop on.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"}})
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}")
