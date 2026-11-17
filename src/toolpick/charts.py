"""The charts in the notebook and the README (Altair). Plumbing: you don't need it to follow the study."""

from __future__ import annotations

import altair as alt
import pandas as pd

BLUE, ORANGE, YELLOW, RED = "#2a78d6", "#eb6834", "#eda100", "#e34948"
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
VERDICT_COLOURS = {"correct": BLUE, "benign": YELLOW, "harmful": RED}
PCT = ".0%"


def finish(chart: alt.Chart, title: str, subtitle: str = "") -> alt.Chart:
    """The shared look: quiet axes, hairline grid, title top left."""
    return (
        chart.properties(title=alt.Title(title, subtitle=subtitle, anchor="start", offset=14))
        .configure(background=SURFACE, font="Liberation Sans, Arial, sans-serif", padding=14)
        .configure_title(fontSize=15, color=INK, subtitleColor=SECONDARY, subtitleFontSize=12)
        .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=SECONDARY, titleColor=SECONDARY,
                        labelFontSize=12, titleFontSize=12, titleFontWeight="normal", labelLimit=260)
        .configure_legend(labelColor=SECONDARY, titleColor=SECONDARY, labelFontSize=12, orient="top", title=None)
        .configure_view(stroke=None)
    )  # fmt: skip


def outcome_bars(table: pd.DataFrame, title: str, subtitle: str = "") -> alt.Chart:
    """One stacked bar per selector: correct, benign, harmful. ``table`` has one row per selector
    with those three shares, a ``harmful_n`` count, and the selector's name as the index."""
    order = list(table.index)
    long = table[list(VERDICT_COLOURS)].rename_axis("selector").reset_index().melt("selector", var_name="verdict", value_name="share")
    long["rank"] = long["verdict"].map({v: i for i, v in enumerate(VERDICT_COLOURS)})
    y = alt.Y("selector:N", sort=order, title=None, axis=alt.Axis(domain=False, ticks=False, labelPadding=8))
    bars = alt.Chart(long).mark_bar(size=22, stroke=SURFACE, strokeWidth=2).encode(
        y=y, x=alt.X("share:Q", stack="zero", title="share of 600 messages", axis=alt.Axis(format=PCT, values=[0, .25, .5, .75, 1]),
                     scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("verdict:N", sort=list(VERDICT_COLOURS), scale=alt.Scale(domain=list(VERDICT_COLOURS), range=list(VERDICT_COLOURS.values()))),
        order="rank:Q")  # fmt: skip
    side = table.rename_axis("selector").reset_index()
    side["label"] = side["correct"].map("{:.1%}".format)
    side["harm"] = side["harmful_n"].map(lambda n: f"{n} harmful")
    inside = alt.Chart(side).mark_text(align="left", dx=8, color="#ffffff", fontWeight="bold", fontSize=12).encode(
        y=y, x=alt.value(0), text="label:N")  # fmt: skip
    right = alt.Chart(side).mark_text(align="left", dx=8, color=SECONDARY, fontSize=12).encode(
        y=y, x=alt.X("one:Q"), text="harm:N").transform_calculate(one="1")  # fmt: skip
    return finish((bars + inside + right).properties(width=520, height=alt.Step(40)), title, subtitle)


def cost_scatter(table: pd.DataFrame, title: str, subtitle: str = "") -> alt.Chart:
    """Correct against billed cost per 1,000 messages (log scale). Index: selector; columns: correct, cost."""
    d = table.rename_axis("selector").reset_index()
    base = alt.Chart(d).encode(
        x=alt.X("cost:Q", scale=alt.Scale(type="log", domain=[0.1, 20]), title="billed US$ per 1,000 messages (log scale)",
                axis=alt.Axis(values=[0.1, 0.3, 1, 3, 10], format="$~g")),
        y=alt.Y("correct:Q", scale=alt.Scale(domain=[0.4, 1]), title="correct", axis=alt.Axis(format=PCT)))  # fmt: skip
    dots = base.mark_point(size=110, filled=True, color=BLUE, opacity=1, stroke=SURFACE, strokeWidth=2)
    names = base.mark_text(align="left", dx=10, dy=1, color=INK, fontSize=12).encode(text="selector:N")
    return finish((dots + names).properties(width=520, height=300), title, subtitle)


def split_bars(table: pd.DataFrame, title: str, subtitle: str = "") -> alt.Chart:
    """Two bars per selector: correct on requests a tool can handle, and on requests no tool can."""
    kinds = ["a tool can do it", "no tool can (hand off)"]
    long = table[kinds].rename_axis("selector").reset_index().melt("selector", var_name="request", value_name="correct")
    long["label"] = long["correct"].map("{:.1%}".format)
    y = alt.Y("selector:N", sort=list(table.index), title=None, axis=alt.Axis(domain=False, ticks=False, labelPadding=8))
    off = alt.YOffset("request:N", sort=kinds)
    x = alt.X("correct:Q", scale=alt.Scale(domain=[0, 1.1]), title="correct", axis=alt.Axis(format=PCT, values=[0, .25, .5, .75, 1]))
    bars = alt.Chart(long).mark_bar(size=14, cornerRadiusEnd=4).encode(
        y=y, yOffset=off, x=x, color=alt.Color("request:N", sort=kinds, scale=alt.Scale(domain=kinds, range=[BLUE, ORANGE])))  # fmt: skip
    text = alt.Chart(long).mark_text(align="left", dx=5, color=SECONDARY, fontSize=11).encode(y=y, yOffset=off, x=x, text="label:N")
    return finish((bars + text).properties(width=520, height=alt.Step(18)), title, subtitle)


def share_lines(lines: pd.DataFrame, here: float, title: str, subtitle: str = "") -> alt.Chart:
    """Overall accuracy as the share of no-tool requests changes. ``lines``: share, correct, selector."""
    names = list(dict.fromkeys(lines["selector"]))
    colour = alt.Color("selector:N", sort=names, scale=alt.Scale(domain=names, range=[BLUE, ORANGE][: len(names)]), legend=None)
    x = alt.X("share:Q", title="share of requests no tool can handle", axis=alt.Axis(format=PCT, values=[0, .25, .5, .75, 1]),
               scale=alt.Scale(domain=[0, 1.22]))
    y = alt.Y("correct:Q", title="correct", axis=alt.Axis(format=PCT), scale=alt.Scale(domain=[0.88, 1.0]))
    line = alt.Chart(lines).mark_line(strokeWidth=2).encode(x=x, y=y, color=colour)
    ends = lines[lines["share"] == lines["share"].max()]
    dots = alt.Chart(ends).mark_point(size=70, filled=True, opacity=1).encode(x=x, y=y, color=colour)
    labels = alt.Chart(ends).mark_text(align="left", dx=8, color=INK, fontSize=12).encode(x=x, y=y, text="selector:N")
    mark = pd.DataFrame({"share": [here], "note": [f"this study: {here:.0%}"]})
    rule = alt.Chart(mark).mark_rule(color=MUTED, strokeWidth=1).encode(x="share:Q")
    note = alt.Chart(mark).mark_text(align="left", dx=6, dy=6, baseline="top", color=SECONDARY, fontSize=12).encode(
        x="share:Q", y=alt.value(0), text="note:N")  # fmt: skip
    return finish((rule + line + dots + labels + note).properties(width=520, height=280), title, subtitle)


def ranked_bars(series: pd.Series, title: str, subtitle: str = "", x_title: str = "", fmt: str = PCT) -> alt.Chart:
    """Horizontal bars, largest first, one hue, value at the tip."""
    d = series.sort_values(ascending=False).rename_axis("name").reset_index(name="value")
    y = alt.Y("name:N", sort=list(d["name"]), title=None, axis=alt.Axis(domain=False, ticks=False, labelPadding=8))
    x = alt.X("value:Q", title=x_title, axis=alt.Axis(format=PCT), scale=alt.Scale(domain=[0, float(d["value"].max()) * 1.12]))
    bars = alt.Chart(d).mark_bar(size=18, color=BLUE, cornerRadiusEnd=4).encode(y=y, x=x)
    text = alt.Chart(d).mark_text(align="left", dx=5, color=SECONDARY, fontSize=12).encode(y=y, x=x, text=alt.Text("value:Q", format=fmt))
    return finish((bars + text).properties(width=440, height=alt.Step(30)), title, subtitle)
