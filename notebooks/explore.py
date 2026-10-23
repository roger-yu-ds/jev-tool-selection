import marimo

__generated_with = "0.14.0"
app = marimo.App(width="full", app_title="Who picks the tool?")


@app.cell
def _():
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
    sys.path.insert(0, str(ROOT / "src"))
    import toolpick as tp

    alt.data_transformers.disable_max_rows()
    return ROOT, alt, mo, pd, tp


@app.cell
def _(mo):
    mo.md(
        r"""
        # Who picks the tool?

        **600 customer-support messages. 90 real MCP tools. Four ways of choosing one.**

        An AI support assistant gets a message like *"i wqant assistance checking purchase ORD-475206 status"*
        and has to decide what to do: call Shopify's `get_order_status`, search the store's FAQ, ask a question,
        or hand the customer to a person. This notebook lets you look at every one of those decisions, made by
        four selectors, in the final setup and in the development runs that led to it.

        **The selectors.** [Jev](https://typesafe.ai) (TypeSafe's System One: a small model that answers a
        multiple-choice question with a probability per option), and three LLMs: DeepSeek V4.1 Flash,
        Claude Haiku 5.5 and Claude Sonnet 5.5. During development each LLM ran in two **modes**:

        - **router**: the tool definitions pasted into the system prompt as JSON, and the instruction *"pick the
          ONE tool that should be called first; reply with its name"*. This is the like-for-like comparison with
          Jev, which answers the same question.
        - **native**: the tools in the API's `tools` parameter, a plain assistant prompt, `tool_choice: auto`.
          This is how an agent really works, and it can also reply in text instead of calling anything.

        **The final setup** (the default view, on 600 *fresh* messages the study never saw while it was
        being designed): every selector gets the same routing policy (who the assistant serves, when to search
        the FAQ, when to hand off), and six tool descriptions get one extra line in customers' words ("download
        my bill #123"), because real descriptions are written for developers. The LLMs run as real agents
        (native tool calling); Jev answers one multiple-choice question.

        **The development runs** (the *first 600* view) are how we got there: *base* (plain prompt) and *policy*
        prompts, the router mode (the LLM told to name one tool), and a DSPy GEPA-optimised prompt.

        **The answer key.** Each of the 27 Bitext intents has the tools that are *correct* and the tools that
        are a *benign first step*. 14 intents have no real tool at all: the correct answer there is
        `hand_off_to_human`. The *audited* key (the default) came from checking every failure by hand; the
        *first pass* is what the study started with. The table at the bottom lists every change and why.

        **Scoring.** Every answer is **correct** or **incorrect**, and an incorrect answer is either:

        - **benign**: nothing bad happens. The selector handed the customer to a person, asked a question back,
          or made a sensible first step that doesn't finish the job. The customer waits; the business loses a
          little automation.
        - **harmful**: the selector called a tool that can't do what was asked (an order lookup for "cancel my
          order"). The assistant looks as if it acted and nothing happened. This is what a customer would notice.

        The *Detail* toggle below splits these into the six underlying outcomes: `right`, `defensible` (a partial
        tool), `handed_off`, `asked` (no tool call), `wrong` (wrong tool where a tool exists) and `bluffed` (a tool
        on an intent no tool covers).
        """
    )


@app.cell
def _(mo, tp):
    sample = mo.ui.radio(options={"fresh 600 (final setup)": "fresh", "first 600 (development)": "first"}, value="fresh 600 (final setup)", label="Messages")
    label_version = mo.ui.radio(options={"audited (final)": "audited", "first pass": "first_pass"}, value="audited (final)", label="Answer key")
    prompt_filter = mo.ui.multiselect(options=["final", "final_ask", "final_params", "final_ask_params", "base", "policy", "gepa"], value=["final", "final_ask", "final_params", "final_ask_params", "base", "policy"], label="Prompt")
    detail = mo.ui.switch(value=False, label="Detail: six outcomes instead of correct / benign / harmful")
    mode_filter = mo.ui.multiselect(options=["jev", "router", "native"], value=["jev", "router", "native"], label="Mode")
    mo.hstack([sample, label_version, prompt_filter, mode_filter, detail], justify="start", gap=2, wrap=True)
    return detail, label_version, mode_filter, prompt_filter, sample


@app.cell
def _(label_version, mode_filter, pd, prompt_filter, sample, tp):
    runs_all = tp.available_runs(sample.value)
    rows_all = pd.DataFrame([r for name in runs_all for r in tp.scored_rows(name, label_version.value)])
    rows_all["selector"] = rows_all["model"].map(tp.MODEL_NAMES)
    rows_all["condition"] = rows_all["mode"] + " · " + rows_all["prompt"]
    rows_all["label"] = rows_all.apply(lambda r: f"{r.selector} ({r['mode']}, {r['prompt']})", axis=1)
    rows = rows_all[rows_all["prompt"].isin(prompt_filter.value) & rows_all["mode"].isin(mode_filter.value)]
    return rows, rows_all


@app.cell
def _(mo, rows):
    _g = rows.groupby(["run", "selector", "mode", "prompt"], sort=False)
    headline = _g.agg(n=("id", "size"), correct=("right", "mean"),
                      benign=("verdict", lambda s: (s == "benign").mean()), harmful=("wrong_call", "mean"),
                      asked=("outcome", lambda s: (s == "asked").mean()),
                      handed_off=("pick", lambda s: (s == "hand_off_to_human").mean()),
                      latency_median_s=("latency_s", "median"), input_tokens=("input_tokens", "median"),
                      cost_per_1000=("cost_usd", lambda s: 1000 * s.sum() / len(s))).reset_index()  # fmt: skip
    headline = headline.sort_values(["prompt", "mode", "correct"], ascending=[True, True, False])
    mo.vstack([
        mo.md("## Headline numbers\n\nEvery run under the filters above. *correct* + *benign* + *harmful* = 100%. *asked* and *handed_off* are parts of benign (and handing off is correct on intents no tool covers)."),
        mo.ui.table(headline.round(3), selection=None, pagination=False),
    ])
    return (headline,)


@app.cell
def _(alt, headline, mo):
    _base = alt.Chart(headline).encode(
        x=alt.X("harmful:Q", title="harmful (lower is better)", axis=alt.Axis(format="%")),
        y=alt.Y("correct:Q", title="correct", axis=alt.Axis(format="%")),
        color=alt.Color("selector:N", title="selector"),
        shape=alt.Shape("mode:N", title="mode"),
        tooltip=["run", alt.Tooltip("correct:Q", format=".1%"), alt.Tooltip("benign:Q", format=".1%"),
                 alt.Tooltip("harmful:Q", format=".1%"), "n"],
    )
    _chart = (_base.mark_point(size=160, filled=True, opacity=0.9)
              + _base.mark_text(dy=-12, fontSize=10).encode(text="prompt:N")).properties(width=520, height=360,
              title="Accuracy against harm: the top-left corner is where you want to be")  # fmt: skip
    mo.vstack([
        mo.md("### Correct against harmful\n\nOne point per run; the label is the prompt. Everything not on either axis is benign. The same selector moves up and left when the routing policy is added."),
        mo.ui.altair_chart(_chart),
    ])


@app.cell
def _(alt, detail, mo, rows, tp):
    _col = "outcome" if detail.value else "verdict"
    _mix = (rows.groupby(["label", "gap", _col]).size().rename("n").reset_index())
    _mix["share"] = _mix["n"] / _mix.groupby(["label", "gap"])["n"].transform("sum")
    _mix["kind"] = _mix["gap"].map({False: "covered intents (a real tool fits)", True: "gap intents (no real tool)"})
    if detail.value:
        _order, _colors = tp.OUTCOMES, ["#2a9d8f", "#8ab17d", "#e9c46a", "#a8a8a8", "#e76f51", "#c1121f"]
    else:
        _order, _colors = tp.VERDICTS, ["#2a9d8f", "#d6cfbf", "#c1121f"]
    _chart = alt.Chart(_mix).mark_bar().encode(
        y=alt.Y("label:N", title=None),
        x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
        color=alt.Color(f"{_col}:N", sort=_order, scale=alt.Scale(domain=_order, range=_colors), title=None),
        order=alt.Order("o:Q"),
        column=alt.Column("kind:N", title=None),
        tooltip=["label", _col, alt.Tooltip("share:Q", format=".1%"), "n"],
    ).transform_calculate(o=f"indexof({_order!r}, datum.{_col})").properties(width=380, height=22 * rows["label"].nunique())
    mo.vstack([
        mo.md("### Where each selector fails\n\nSplit by whether a real tool exists for the intent. On gap intents, correct means it handed off. Turn on *Detail* to see what the benign and harmful answers were."),
        mo.ui.altair_chart(_chart),
    ])


@app.cell
def _(alt, mo, rows):
    _pi = rows.groupby(["intent", "gap", "label"])["right"].mean().reset_index()
    _n = rows.groupby("intent")["id"].nunique().rename("n").reset_index()
    _pi = _pi.merge(_n, on="intent")
    _pi["intent_label"] = _pi["intent"] + _pi["gap"].map({True: " (no tool)", False: ""})
    _chart = alt.Chart(_pi).mark_rect().encode(
        y=alt.Y("intent_label:N", title=None, sort=alt.SortField("right", order="ascending")),
        x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-30)),
        color=alt.Color("right:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]), title="right"),
        tooltip=["intent", "label", alt.Tooltip("right:Q", format=".0%"), "n"],
    ).properties(width=28 * rows["label"].nunique(), height=18 * 27)
    mo.vstack([
        mo.md("### Per intent\n\nShare right, per intent and run. Rows are sorted by average difficulty. With 12–32 messages per intent, treat differences under about 20 points as noise."),
        mo.ui.altair_chart(_chart),
    ])


@app.cell
def _(mo, rows_all, tp):
    mo.md("## Browse the decisions")
    _runs = sorted(rows_all["run"].unique())
    _jev = next(r for r in _runs if r.endswith("jev"))
    _son = next((r for r in _runs if r.endswith("native_sonnet")), "(none)")
    run_pick = mo.ui.dropdown(options=_runs, value=_jev, label="Run")
    compare_pick = mo.ui.dropdown(options=["(none)", *_runs], value=_son, label="Compare with")
    intent_pick = mo.ui.multiselect(options=sorted(rows_all["intent"].unique()), label="Intent (empty = all)")
    outcome_pick = mo.ui.multiselect(options=tp.VERDICTS + tp.OUTCOMES, label="Verdict or outcome (empty = all)")
    text_search = mo.ui.text(placeholder="search the message text…", label="Text")
    only_disagree = mo.ui.checkbox(label="only where the two runs disagree")
    mo.hstack([run_pick, compare_pick, intent_pick, outcome_pick, text_search, only_disagree], wrap=True, gap=1.5, justify="start")
    return compare_pick, intent_pick, only_disagree, outcome_pick, run_pick, text_search


@app.cell
def _(compare_pick, intent_pick, mo, only_disagree, outcome_pick, rows_all, run_pick, text_search):
    _a = rows_all[rows_all["run"] == run_pick.value].set_index("id")
    _cols = ["intent", "gap", "text", "pick", "verdict", "outcome", "confidence", "reply", "arguments", "latency_s", "cost_usd"]
    _view = _a[_cols].copy()
    if compare_pick.value != "(none)":
        _b = rows_all[rows_all["run"] == compare_pick.value].set_index("id")
        _view["pick_b"] = _b["pick"]
        _view["verdict_b"] = _b["verdict"]
        _view["outcome_b"] = _b["outcome"]
        _view["reply_b"] = _b["reply"]
        if only_disagree.value:
            _view = _view[_view["pick"].fillna("(asked)") != _view["pick_b"].fillna("(asked)")]
    if intent_pick.value:
        _view = _view[_view["intent"].isin(intent_pick.value)]
    if outcome_pick.value:
        _view = _view[_view["outcome"].isin(outcome_pick.value) | _view["verdict"].isin(outcome_pick.value)]
    if text_search.value.strip():
        _view = _view[_view["text"].str.contains(text_search.value.strip(), case=False, regex=False)]
    _view["pick"] = _view["pick"].fillna("(asked: no tool call)")
    _view["reply"] = _view["reply"].str.slice(0, 160)
    if "reply_b" in _view:
        _view["reply_b"] = _view["reply_b"].str.slice(0, 160)
    browser = mo.ui.table(_view.reset_index(), selection="single", page_size=15)
    mo.vstack([mo.md(f"**{len(_view)} messages.** Click a row to see the full record below."), browser])
    return (browser,)


@app.cell
def _(browser, label_version, mo, run_pick, tp):
    _sel = browser.value
    if _sel is None or len(_sel) == 0:
        _out = mo.md("*Select a row above.*")
    else:
        _rid = _sel.iloc[0]["id"]
        _row = tp.load_run(run_pick.value)[_rid]
        _lab = tp.labels(label_version.value)[_sel.iloc[0]["intent"]]
        _items = [f"**{_rid}** · intent `{_sel.iloc[0]['intent']}` · {'no real tool' if _lab['gap'] else 'a real tool fits'}",
                  f"> {_sel.iloc[0]['text']}",
                  f"**Acceptable:** `{'`, `'.join(_lab['acceptable'])}`" + (f"  **Partial:** `{'`, `'.join(_lab['partial'])}`" if _lab['partial'] else ""),
                  f"**Research note:** {_lab['note']}" + ("".join(f"\n\n**Answer-key change:** {c}" for c in _lab["changes"]))]
        if "top" in _row:
            _items.append("**Jev's distribution:** " + ", ".join(f"`{k.split('.')[-1]}` {v:.2f}" for k, v in _row["top"][:6]))
        if _row.get("tool_calls"):
            _items.append("**Tool calls:**\n```json\n" + __import__("json").dumps(_row["tool_calls"], indent=1) + "\n```")
        if _row.get("text") or _row.get("raw"):
            _items.append("**Reply text:**\n\n> " + (_row.get("text") or _row.get("raw")).replace("\n", "\n> "))
        _items.append(f"latency {_row['latency_s']:.2f} s · input {_row.get('input_tokens')} (cached {_row.get('cached_tokens', 0)}) · "
                      f"output {_row.get('output_tokens', 0)} · ${_row.get('cost_usd', 0):.5f}")
        _out = mo.md("\n\n".join(_items))
    _out  # noqa: B018  (marimo shows the last expression of a cell)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Error analysis

        Three questions worth asking of any tool-selection eval, each with its own view below:

        1. **What does the selector reach for when it's wrong?** (the confusion table). A model that answers every
           unknown with the FAQ search is doing something systematic, and a one-line policy fixes it.
        2. **Does it know when it doesn't know?** (Jev's confidence gate). A probability per option lets you trade
           coverage for harmful answers with a threshold; an LLM's text reply doesn't.
        3. **Is the prompt the problem, or the model?** (the prompt toggle in the headline table). If every model gains
           15 points from three sentences, the eval was measuring the prompt.
        """
    )


@app.cell
def _(mo, rows, rows_all, run_pick):
    _r = rows_all[(rows_all["run"] == run_pick.value) & rows_all["outcome"].isin(["wrong", "bluffed"])]
    _conf = (_r.assign(pick=_r["pick"].str.split(".").str[-1]).groupby(["intent", "gap", "pick"]).size()
             .rename("n").reset_index().sort_values("n", ascending=False))  # fmt: skip
    mo.vstack([
        mo.md(f"### Harmful answers in `{run_pick.value}`: what it picked instead\n\n{len(_r)} harmful answers out of {len(rows_all[rows_all['run'] == run_pick.value])}."),
        mo.ui.table(_conf.head(25), selection=None, pagination=False),
    ])


@app.cell
def _(alt, label_version, mo, pd, tp):
    _rows = []
    for _name in ("fresh/jev", "jev", "jev_policy"):
        if _name in tp.available_runs() + tp.available_runs("fresh"):
            for g in tp.gate_curve(_name, label_version.value, thresholds=[i / 20 for i in range(20)]):
                _rows.append({**g, "run": _name})
    _df = pd.DataFrame(_rows).melt(id_vars=["t", "run"], var_name="metric", value_name="value")
    _chart = alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("t:Q", title="confidence threshold (below it, hand off instead of calling the tool)"),
        y=alt.Y("value:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("metric:N", title=None), strokeDash=alt.StrokeDash("run:N"),
        tooltip=["run", "metric", alt.Tooltip("t:Q"), alt.Tooltip("value:Q", format=".1%")],
    ).properties(width=560, height=300)
    mo.vstack([
        mo.md("### Jev's confidence as a dial\n\n`acted`: share of messages where Jev calls a real tool; `wrong_of_acted`: how many of those calls are harmful; `right`: share correct after gating."),
        mo.ui.altair_chart(_chart),
    ])


@app.cell
def _(alt, mo, rows):
    _asked = rows[rows["outcome"] == "asked"]
    _c = _asked.groupby(["label", "gap"]).size().rename("n").reset_index()
    _chart = alt.Chart(_c).mark_bar().encode(
        y=alt.Y("label:N", title=None), x=alt.X("n:Q", title="messages answered in text (no tool call)"),
        color=alt.Color("gap:N", title="no real tool"), tooltip=["label", "gap", "n"]).properties(width=500)  # fmt: skip
    _examples = _asked.sample(min(6, len(_asked)), random_state=1)[["label", "intent", "text", "reply"]] if len(_asked) else _asked
    mo.vstack([
        mo.md("### 'Asked': when the native models reply instead of calling a tool\n\nMany Bitext messages are vague (*'I need assistance buying an article'*). A native agent asks a question; a router is forced to pick. Incorrect but benign: nobody is misled, the customer just answers a question."),
        mo.ui.altair_chart(_chart),
        mo.ui.table(_examples.assign(reply=_examples["reply"].str.slice(0, 200)), selection=None, pagination=False) if len(_asked) else mo.md("*No native runs selected.*"),
    ])


@app.cell
def _(label_version, mo, pd, tp):
    _pairs = []
    _avail = tp.available_runs() + tp.available_runs("fresh")
    _todo = [("fresh/jev", f"fresh/native_{m}") for m in ("deepseek", "haiku", "sonnet")]
    _todo += [("jev" + s_, b + s_) for b in ("router_deepseek", "router_haiku", "router_sonnet", "native_deepseek", "native_haiku", "native_sonnet") for s_ in ("", "_policy")]
    for _a, _bb in _todo:
            if _a in _avail and _bb in _avail:
                p = tp.paired(_a, _bb, label_version.value)
                if p:
                    _pairs.append({"pair": f"{_a} vs {_bb}", "n": p["n_pairs"], "jev right": p["right_a"], "other right": p["right_b"],
                                   "only Jev right": p["only_a"], "only other right": p["only_b"], "diff": p["diff"],
                                   "ci_low": p["ci"][0], "ci_high": p["ci"][1], "p (McNemar)": p["p"]})  # fmt: skip
    mo.vstack([
        mo.md("### Paired comparisons with Jev\n\nSame messages, so the test counts how often only one of the two was right (exact McNemar) and bootstraps the difference."),
        mo.ui.table(pd.DataFrame(_pairs).round(4), selection=None, pagination=False),
    ])


@app.cell
def _(alt, mo, pd, tp):
    _t = tp.tool_tokens_by_server()
    _n = pd.Series([t["server"] for t in tp.catalog()]).value_counts()
    _df = pd.DataFrame({"server": list(_t), "chars": list(_t.values())})
    _df["tools"] = _df["server"].map(_n)
    _df["share"] = _df["chars"] / _df["chars"].sum()
    _chart = alt.Chart(_df).mark_bar().encode(
        y=alt.Y("server:N", sort="-x", title=None), x=alt.X("share:Q", axis=alt.Axis(format="%"), title="share of the tool JSON"),
        tooltip=["server", "tools", "chars", alt.Tooltip("share:Q", format=".1%")]).properties(width=500)  # fmt: skip
    _cat = pd.DataFrame([{"tool": t["qualified_name"], "server": t["server"], "description": t["description"][:200],
                          "source": t["source_url"], "exactness": t["exactness"]} for t in tp.catalog()])  # fmt: skip
    mo.vstack([
        mo.md("## The tools\n\n90 tools from 7 official servers, verbatim. Notion and Shopify's checkout schemas take most of the prompt budget; the four Shopify customer-account tools that answer most real questions are 1% of it."),
        mo.ui.altair_chart(_chart),
        mo.ui.table(_cat, selection=None, page_size=12),
    ])


@app.cell
def _(mo, pd, tp):
    _lab = tp.labels("audited")
    _first = tp.labels("first_pass")
    _short = lambda ts: ", ".join(t.split(".")[-1] for t in ts)
    _df = pd.DataFrame([{"intent": k, "no real tool": v["gap"], "correct": _short(v["acceptable"]), "benign first step": _short(v["partial"]),
                         "a question back is correct": v["ask_ok"], "added in the audit": _short(t for t in v["acceptable"] if t not in _first[k]["acceptable"]),
                         "why": " ".join(v["changes"]), "research note": v["note"][:200]} for k, v in _lab.items()])  # fmt: skip
    mo.vstack([
        mo.md("## The answer key\n\nEvery scoring decision, with its reason. The *audited* key came from checking every failure on the first 600 messages by hand; the *first pass* is what the study started with. Change a label in `src/toolpick/data.py` and every chart above re-scores."),
        mo.ui.table(_df, selection=None, page_size=27),
    ])


if __name__ == "__main__":
    app.run()
