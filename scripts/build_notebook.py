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

Each LLM run exists under two prompts (*base* and *policy*), and everything is scored under two label
versions (*v1*, *v2*). The point of the talk is that the ranking depends on those choices, so you should see them.
""")

code("""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))
import toolpick as tp

pd.set_option("display.max_colwidth", 90)
print(len(tp.requests()), "messages ·", len(tp.tools()), "tools ·", len(tp.available_runs()), "runs:", tp.available_runs())
""")

md("""
## 1. The data

600 messages, a fixed sample of Bitext's customer-support dataset. The text is template-generated with deliberate
typos; the `intent` is Bitext's label, which we map to tools.
""")

code("""
from dataclasses import asdict

reqs = pd.DataFrame([asdict(r) for r in tp.requests()])
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

Per intent: `acceptable` tools (a clean fit), `partial` tools (a defensible first step that doesn't finish the job).
**15 of the 27 intents have no real tool**; there the only acceptable answer is `hand_off_to_human`.

`v2` is the revised version that also accepts the readings the LLMs made consistently: an FAQ search for a
how-to question on a gap intent, a hand-off on `payment_issue`, the refund-policy FAQ on `get_refund`. Both
versions, with every override and its reason, live in `src/toolpick/data.py`.
""")

code("""
lab = tp.labels("v1")
pd.DataFrame([{"intent": k, "no tool": v["gap"], "acceptable": ", ".join(t.split(".")[-1] for t in v["acceptable"]),
               "partial": ", ".join(t.split(".")[-1] for t in v["partial"]), "override": v["override"] or ""}
              for k, v in lab.items()]).sort_values(["no tool", "intent"])
""")

md("""
## 3. Scoring one decision

Every row of a run has the selector's pick (or no pick, if a native model replied in text). `tp.outcome` turns a
pick into one of six outcomes. Only `right` counts as right; `wrong` and `bluffed` together are the *wrong-call
rate*, the number a customer would notice.
""")

code("""
for pick in ["shopify_customer_account.get_order_status", "shopify_ucp.get_order", "hand_off_to_human",
             "shopify_storefront.search_shop_policies_and_faqs", None]:
    print(f"{str(pick):55s} track_order -> {tp.outcome(pick, lab['track_order']):12s}  cancel_order -> {tp.outcome(pick, lab['cancel_order'])}")
""")

md("""
## 4. The headline table

`scored_rows` joins a run with the labels; `summarise` gives the rates. Change `version` to `"v2"` and watch the
ranking move.
""")

code("""
def table(version="v1"):
    rows = []
    for name in tp.available_runs():
        s = tp.summarise(tp.scored_rows(name, version))
        rows.append({"run": name, "n": s["n"], "right": s["right"], "wrong_call": s["wrong_call"], "asked": s["asked"],
                     "handed_off": s["handoff"], "latency_s": s["latency_median_s"], "input_tokens": s["input_tokens_median"],
                     "$ per 1000": s["cost_per_1000"]})
    return pd.DataFrame(rows).set_index("run").sort_values("right", ascending=False)

(table("v1").style.format({"right": "{:.1%}", "wrong_call": "{:.1%}", "asked": "{:.1%}", "handed_off": "{:.1%}",
                           "latency_s": "{:.2f}", "input_tokens": "{:,.0f}", "$ per 1000": "{:.2f}"})
   .bar(subset=["right"], color="#8ab17d", vmin=0, vmax=1).bar(subset=["wrong_call"], color="#e9a3a3", vmin=0, vmax=1))
""")

code("""
table("v2")[["right", "wrong_call"]].style.format("{:.1%}")
""")

md("""
## 5. Paired comparisons

Two accuracies on the same messages aren't independent, so we don't compare them as if they were. `paired`
counts the messages where only one of the two was right (an exact McNemar test) and bootstraps the difference.
""")

code("""
pairs = []
for other in ["router_deepseek", "router_haiku", "router_sonnet", "native_deepseek", "native_haiku", "native_sonnet"]:
    for suffix in ["", "_policy"]:
        p = tp.paired("jev" + suffix, other + suffix)
        pairs.append({"jev vs": other + suffix, "n": p["n_pairs"], "jev": p["right_a"], "other": p["right_b"],
                      "only jev right": p["only_a"], "only other right": p["only_b"], "diff": p["diff"],
                      "95% CI": f"[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]", "p": p["p"]})
pd.DataFrame(pairs).style.format({"jev": "{:.1%}", "other": "{:.1%}", "diff": "{:+.1%}", "p": "{:.2g}"})
""")

md("""
## 6. Error analysis

Three views. First: when a selector is wrong, what does it reach for? Most LLM wrong calls are the store's FAQ
search on a question no tool can answer, which is exactly what the one-paragraph routing policy fixes.
""")

code("""
def wrong_calls(name):
    rows = pd.DataFrame(tp.scored_rows(name))
    w = rows[rows["wrong_call"]]
    return (w.assign(pick=w["pick"].str.split(".").str[-1]).groupby(["intent", "pick"]).size()
             .sort_values(ascending=False).head(10).to_frame("n"))

display(wrong_calls("router_sonnet"))
wrong_calls("router_sonnet_policy")
""")

md("""
Second: Jev's errors. Jev almost never calls the wrong tool; it hands off where a tool could have done the job.
That is a miss too (the customer waits for a person), so it counts as wrong, but it is a different kind of miss,
and Jev's confidence lets you choose how often to take it.
""")

code("""
jev = pd.DataFrame(tp.scored_rows("jev"))
display(jev[~jev["gap"]].groupby("outcome").size().to_frame("covered intents"))
pd.DataFrame(tp.gate_curve("jev")).style.format({"acted": "{:.1%}", "wrong_of_acted": "{:.1%}", "right": "{:.1%}"})
""")

md("""
Third: the native models often reply in text instead of calling a tool. On vague messages that's reasonable; the
router condition can't do it. Scoring the first turn only makes native accuracy a lower bound.
""")

code("""
nat = pd.DataFrame(tp.scored_rows("native_sonnet"))
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
a = pd.DataFrame(tp.scored_rows("jev")).set_index("id")
b = pd.DataFrame(tp.scored_rows("router_sonnet_policy")).set_index("id")
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
uv run python scripts/run_router.py --arm jev --limit 5
uv run python scripts/run_native.py --arm haiku --limit 5 --policy
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

    criteria = {t["name"]: t["description"] for t in tp.tools()}
    out = Jev().choose("i wqant assistance checking purchase ORD-475206 status", jev_instructions(policy=False), criteria)
    a = out["answers"]["tool"]
    print(a["choice"], f"confidence {a['confidence']:.2f}", f"{out['latency_s']:.2f}s", f"${out['cost_usd']:.5f}")
    print(sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4])
else:
    print("no TYPESAFE_API_KEY: skipping the live call")
""")

md("""
## What to take away

1. Use real tool text, sets of acceptable answers, and an explicit "no tool" class.
2. Report the wrong-call rate next to accuracy. A hand-off and a bluff are not the same miss.
3. Compare selectors on the same messages, with a paired test.
4. Write the routing policy down and put it in the prompt; then you're measuring the model, not its guess at your policy.
5. Publish the labels. They decide the winner.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"}})
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}")
