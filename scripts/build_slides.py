"""Build the talk: slides/index.html, a self-contained deck with click-to-reveal steps.

    uv run python scripts/build_slides.py

Keys: → / space / click = next step or slide, ← = back, Home/End, `p` = print view (all slides
stacked; then Ctrl+P → PDF), `a` = jump to the appendix. Every number is computed from runs/.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp

OUT = ROOT / "slides" / "index.html"
e = html.escape
ARMS = ["jev", "deepseek", "haiku", "sonnet"]
SHORT = {"jev": "Jev", "deepseek": "DeepSeek", "haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5"}
COL = {"jev": "#1d4ed8", "deepseek": "#7c3aed", "haiku": "#d97706", "sonnet": "#dc2626"}


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def run_name(arm: str, mode: str, policy: bool) -> str:
    return ("jev" if arm == "jev" else f"{mode}_{arm}") + ("_policy" if policy else "")


def summ(arm: str, mode: str, policy: bool, version: str = "v1") -> dict:
    return tp.summarise(tp.scored_rows(run_name(arm, mode, policy), version))


# ---------------------------------------------------------------- figures
def bars(values: list[tuple[str, float, str]], *, width=720, height=None, fmt=pct, max_v=1.0, step_class="") -> str:
    """Horizontal bars; each bar can be its own reveal step."""
    rh, left = 54, 150
    height = height or rh * len(values) + 10
    out = [f'<svg viewBox="0 0 {width} {height}" class="fig">']
    for i, (label, v, color) in enumerate(values):
        y = 5 + i * rh
        w = v / max_v * (width - left - 90)
        cls = f' class="{step_class}"' if step_class else ""
        out.append(f'<g{cls}><text x="{left - 14}" y="{y + 33}" text-anchor="end" class="lbl">{e(label)}</text>'
                   f'<rect x="{left}" y="{y + 10}" width="{w:.1f}" height="34" rx="4" fill="{color}"/>'
                   f'<text x="{left + w + 12:.1f}" y="{y + 34}" class="val">{fmt(v)}</text></g>')  # fmt: skip
    out.append("</svg>")
    return "".join(out)


def morph_bars(a: list[tuple[str, float, str]], b: list[tuple[str, float, str]], cap_a: str, cap_b: str, *, width=720) -> str:
    """Bars that animate from a to b when the containing .morph step is turned on."""
    rh, left = 54, 150
    height = rh * len(a) + 10
    out = [f'<div class="morph"><svg viewBox="0 0 {width} {height}" class="fig">']
    for i, ((label, va, color), (_, vb, _)) in enumerate(zip(a, b, strict=True)):
        y = 5 + i * rh
        wa, wb = (v * (width - left - 90) for v in (va, vb))
        out.append(f'<text x="{left - 14}" y="{y + 33}" text-anchor="end" class="lbl">{e(label)}</text>'
                   f'<rect x="{left}" y="{y + 10}" height="34" rx="4" fill="{color}" style="--w1:{wa:.1f}px;--w2:{wb:.1f}px"/>'
                   f'<text class="val va" x="0" style="--x1:{left + wa + 12:.1f}px;--x2:{left + wb + 12:.1f}px" y="{y + 34}">{pct(va)}</text>'
                   f'<text class="val vb" x="0" style="--x1:{left + wa + 12:.1f}px;--x2:{left + wb + 12:.1f}px" y="{y + 34}">{pct(vb)}</text>')  # fmt: skip
    out.append(f'</svg><p class="cap"><span class="ca">{e(cap_a)}</span><span class="cb">{e(cap_b)}</span></p></div>')
    return "".join(out)


def scatter(points: list[dict], *, width=760, height=460) -> str:
    """right (y) against wrong-call rate (x); each point its own step via class."""
    L, B, T, R = 70, 50, 20, 20
    pw, ph = width - L - R, height - T - B
    xmax = 0.5

    def X(v):
        return L + min(v, xmax) / xmax * pw

    def Y(v):
        return T + (1 - v) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" class="fig">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" x2="{width - R}" y1="{Y(v):.0f}" y2="{Y(v):.0f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{Y(v) + 4:.0f}" text-anchor="end" class="tick">{pct(v)}</text>')  # fmt: skip
    for v in (0, .1, .2, .3, .4, .5):
        out.append(f'<text x="{X(v):.0f}" y="{height - B + 20}" text-anchor="middle" class="tick">{pct(v)}</text>')
    out.append(f'<text x="{L + pw / 2:.0f}" y="{height - 6}" text-anchor="middle" class="axis">wrong tool called (share of all messages) →</text>')
    out.append(f'<text transform="translate(16,{T + ph / 2:.0f}) rotate(-90)" text-anchor="middle" class="axis">right ↑</text>')
    for p in points:
        cls = p.get("cls", "")
        shape = p.get("shape", "circle")
        x, y = X(p["x"]), Y(p["y"])
        if shape == "circle":
            mark = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{p["color"]}" opacity=".92"/>'
        elif shape == "square":
            mark = f'<rect x="{x - 10:.1f}" y="{y - 10:.1f}" width="20" height="20" fill="{p["color"]}" opacity=".92"/>'
        else:
            mark = f'<polygon points="{x:.1f},{y - 13:.1f} {x + 12:.1f},{y + 9:.1f} {x - 12:.1f},{y + 9:.1f}" fill="{p["color"]}" opacity=".92"/>'
        dx, dy = p.get("dx", 16), p.get("dy", 5)
        out.append(f'<g class="{cls}">{mark}<text x="{x + dx:.1f}" y="{y + dy:.1f}" class="pt">{e(p["label"])}</text></g>')
    out.append("</svg>")
    return "".join(out)


def intents_grid() -> str:
    lab = tp.labels()
    cnt = {}
    for r in tp.requests():
        cnt[r.intent] = cnt.get(r.intent, 0) + 1
    items = sorted(lab, key=lambda k: (lab[k]["gap"], k))
    cells = []
    for k in items:
        gap = lab[k]["gap"]
        cells.append(f'<div class="cell {"gap" if gap else "cov"}"><span>{e(k.replace("_", " "))}</span><small>{cnt[k]}</small></div>')
    return '<div class="intents">' + "".join(cells) + "</div>"


# ---------------------------------------------------------------- slides
def build() -> str:
    jev = summ("jev", "router", False)
    jev_p, sn_p = summ("jev", "router", True), summ("sonnet", "router", True)
    nat = {a: summ(a, "native", False) for a in ARMS[1:]}
    nat_p = {a: summ(a, "native", True) for a in ARMS[1:]}
    v2 = {a: summ(a, "router", False, "v2") for a in ARMS}
    lab = tp.labels()
    n_gap = sum(lab[k]["gap"] for k in lab)
    gap_msgs = sum(lab[r.intent]["gap"] for r in tp.requests())
    pairs = {a: tp.paired("jev", run_name(a, "router", False)) for a in ARMS[1:]}
    pairs_p = {a: tp.paired("jev_policy", run_name(a, "router", True)) for a in ARMS[1:]}
    tok = tp.tool_tokens_by_server()
    gepa_path = ROOT / "data" / "gepa_deepseek.json"
    gepa = json.loads(gepa_path.read_text()) if gepa_path.exists() else None
    gepa_run = tp.summarise(tp.scored_rows("router_deepseek_gepa")) if "router_deepseek_gepa" in tp.available_runs() else None

    # the hook message: one where the four router picks differ most
    hook = next(r for r in tp.requests() if r.id == "R0186")
    hook_picks = [(SHORT[a], (tp.load_run(run_name(a, "router", False)).get(hook.id) or {}).get("pick") or "—") for a in ARMS]

    best_llm = max(ARMS[1:], key=lambda a: summ(a, "router", False)["right"])
    router_base = [(SHORT[a], summ(a, "router", False)["right"], COL[a]) for a in ARMS]
    router_v2 = [(SHORT[a], v2[a]["right"], COL[a]) for a in ARMS]
    router_pol = [(SHORT[a], summ(a, "router", True)["right"], COL[a]) for a in ARMS]
    pts = [{"x": summ(a, "router", False)["wrong_call"], "y": summ(a, "router", False)["right"], "label": f"{SHORT[a]} (router)",
            "color": COL[a], "cls": "step"} for a in ARMS]
    pts += [{"x": nat[a]["wrong_call"], "y": nat[a]["right"], "label": f"{SHORT[a]} (native)", "color": COL[a], "shape": "square",
             "cls": "step"} for a in ARMS[1:]]

    def stat(arm, mode, policy):
        s = summ(arm, mode, policy)
        return s

    cost_rows = "".join(
        f"<tr><td>{e(SHORT[a])}{' (router)' if a != 'jev' else ''}</td><td>{s['input_tokens_median']:,.0f}</td>"
        f"<td>{s['latency_median_s']:.2f} s</td><td>${s['cost_per_1000']:.2f}</td></tr>"
        for a, s in ((a, summ(a, "router", False)) for a in ARMS))
    cost_rows += "".join(
        f"<tr class='step'><td>{e(SHORT[a])} (native)</td><td>{s['input_tokens_median']:,.0f}</td>"
        f"<td>{s['latency_median_s']:.2f} s</td><td>${s['cost_per_1000']:.2f}</td></tr>" for a, s in nat.items())

    slides = []

    # 1 title
    slides.append("""<section class="title">
<p class="kicker">MeasureCamp Sydney · 24 October 2026</p>
<h1>Who picks the tool?</h1>
<p class="sub">Measuring tool selection on real MCP servers</p>
<p class="by">Roger Yu</p>
</section>""")

    # 2 hook
    picks_html = "".join(f'<div class="pick step"><span class="who" style="color:{COL[a]}">{e(SHORT[a])}</span>'
                         f'<code>{e(p.split(".")[-1])}</code></div>' for a, (_, p) in zip(ARMS, hook_picks, strict=True))
    slides.append(f"""<section>
<p class="kicker">A customer writes</p>
<blockquote class="big">“{e(hook.text)}”</blockquote>
<p class="kicker step">Four selectors, four answers</p>
<div class="picks">{picks_html}</div>
<p class="punch step">Which one is right?</p>
</section>""")

    # 3 setup
    slides.append("""<section>
<h2>The test</h2>
<div class="three">
<div class="step"><div class="num">600</div><p>customer-support messages<br><small>Bitext, 27 intents</small></p></div>
<div class="step"><div class="num">90</div><p>real tools from 7 official MCP servers<br><small>Shopify · Stripe · Intercom · GitHub · Notion, verbatim</small></p></div>
<div class="step"><div class="num">4</div><p>selectors<br><small>Jev · DeepSeek V4.1 Flash · Claude Haiku 5.5 · Claude Sonnet 5.5</small></p></div>
</div>
<p class="note step">Each picks <b>the one tool to call first</b>. Scored against a set of acceptable tools per intent, not one gold answer.</p>
</section>""")

    # 4 gaps
    slides.append(f"""<section>
<h2>Real servers cover {27 - n_gap} of 27 intents</h2>
{intents_grid()}
<p class="note step">No official tool cancels or edits a placed order, opens a complaint, hands off, or touches an account.
<b>{pct(gap_msgs / 600)} of messages have no tool.</b> The right answer there is <code>hand_off_to_human</code>, the one tool we added.</p>
</section>""")

    # 5 accuracy
    slides.append(f"""<section>
<h2>Right first pick <span class="dim">(router mode, plain prompt)</span></h2>
{bars(router_base, step_class="step")}
<p class="note step">Jev beats {e(SHORT[best_llm])} by {100 * pairs[best_llm]['diff']:.0f} points on the same 600 messages
(only Jev right {pairs[best_llm]['only_a']}, only {e(SHORT[best_llm])} right {pairs[best_llm]['only_b']}; McNemar p = {pairs[best_llm]['p']:.2g}).</p>
</section>""")

    # 6 two ways to fail
    slides.append(f"""<section>
<h2>Two ways to fail</h2>
{scatter(pts)}
<p class="note step">A <b>wrong call</b> is a tool that can't do the job: an order lookup for “cancel my order”.
Squares are the LLMs with real tool calling; they ask a clarifying question on {pct(min(s['asked'] for s in nat.values()))}–{pct(max(s['asked'] for s in nat.values()))} of messages.</p>
</section>""")

    # 7 the labels decide
    slides.append(f"""<section>
<h2>Change the ruler</h2>
{morph_bars(router_base, router_v2, "first-pass labels", "revised labels: an FAQ search on a how-to question counts")}
<p class="note step">Same picks. One defensible change to what counts as right, and the ranking flips: the LLMs read “how do I…?” literally; Jev hands off.</p>
</section>""")

    # 8 state the rule
    slides.append(f"""<section>
<h2>State the rule</h2>
{morph_bars(router_base, router_pol, "plain prompt", "+ three sentences of routing policy")}
<p class="note step">Every selector gains {min(100 * (summ(a, 'router', True)['right'] - summ(a, 'router', False)['right']) for a in ARMS):.0f}–{max(100 * (summ(a, 'router', True)['right'] - summ(a, 'router', False)['right']) for a in ARMS):.0f} points.
Jev {pct(jev_p['right'])} and Sonnet {pct(sn_p['right'])} are level (p = {pairs_p['sonnet']['p']:.2f}); Jev makes {pct(jev_p['wrong_call'], 1)} wrong calls, Sonnet {pct(sn_p['wrong_call'], 1)}.</p>
</section>""")

    # 9 cost
    slides.append(f"""<section>
<h2>What it costs to decide</h2>
<table class="cost"><thead><tr><th>selector</th><th>input tokens</th><th>median latency</th><th>per 1,000 picks</th></tr></thead>
<tbody>{cost_rows}</tbody></table>
<p class="note step">Jev sees names and descriptions ({jev['input_tokens_median'] / 1000:.0f}k tokens); the LLMs see full schemas. Prompt caching is on for every LLM run; these are billed prices.</p>
</section>""")

    # 10 checklist
    slides.append("""<section>
<h2>Measuring tool selection</h2>
<ol class="check">
<li class="step">Real tool text, not tools written to fit your test set.</li>
<li class="step">Sets of acceptable answers, and <b>“no tool”</b> as a class.</li>
<li class="step">Report <b>wrong calls</b> next to accuracy. A hand-off and a bluff are not the same miss.</li>
<li class="step">Paired comparisons on the same messages (McNemar), not two accuracies.</li>
<li class="step">Write the policy down and put it in the prompt. Then you're measuring the model.</li>
<li class="step">Publish the labels. They decide the winner.</li>
</ol>
<p class="punch step">The score is a labelling decision. Make it on purpose.</p>
</section>""")

    # appendix
    srv = sorted(tok.items(), key=lambda kv: -kv[1])
    tot = sum(tok.values())
    srv_rows = "".join(f"<tr><td>{e(s)}</td><td>{sum(1 for t in tp.catalog() if t['server'] == s)}</td><td>{pct(c / tot)}</td></tr>" for s, c in srv)
    gepa_html = ""
    if gepa and gepa_run:
        gepa_html = f"""<section class="appendix">
<h2>A4 · Can an optimiser find the policy?</h2>
<p>DSPy GEPA rewrote the router preamble for DeepSeek on a 200-message development split (Claude Sonnet 5.5 as the reflection model),
scored once on the 400 held-out messages.</p>
<table class="cost"><thead><tr><th>DeepSeek router prompt</th><th>right (held-out 400)</th><th>wrong calls</th></tr></thead><tbody>
<tr><td>plain</td><td>{pct(tp.summarise([r for r in tp.scored_rows('router_deepseek') if r['id'] in set(gepa['held_ids'])])['right'], 1)}</td><td>{pct(tp.summarise([r for r in tp.scored_rows('router_deepseek') if r['id'] in set(gepa['held_ids'])])['wrong_call'], 1)}</td></tr>
<tr><td>hand-written policy</td><td>{pct(tp.summarise([r for r in tp.scored_rows('router_deepseek_policy') if r['id'] in set(gepa['held_ids'])])['right'], 1)}</td><td>{pct(tp.summarise([r for r in tp.scored_rows('router_deepseek_policy') if r['id'] in set(gepa['held_ids'])])['wrong_call'], 1)}</td></tr>
<tr><td>GEPA-found</td><td>{pct(gepa_run['right'], 1)}</td><td>{pct(gepa_run['wrong_call'], 1)}</td></tr></tbody></table>
<details><summary>The found preamble</summary><pre>{e(gepa['instruction'])}</pre></details>
</section>"""

    slides.append(f"""<section class="appendix">
<h2>Appendix · A1 · The data</h2>
<ul>
<li><b>Messages:</b> 600 from Bitext's customer-support dataset (26,872 rows, 27 intents, hybrid synthetic with deliberate noise; CDLA-Sharing-1.0). 12–32 per intent.</li>
<li><b>Tools:</b> {len(tp.catalog())} from 7 official servers, by running the vendor's package or docker image and calling <code>tools/list</code> (Stripe 0.2.5, GitHub 2.0.3, Notion 2.5.2), by <code>tools/list</code> on a live public Shopify store (storefront, UCP, customer account), or from official docs (Intercom). Plus <code>hand_off_to_human</code>.</li>
<li><b>Stripe caveat:</b> the 22 per-task tools are the last per-task release (Oct 2025); the hosted server now exposes generic read/write tools.</li>
</ul>
<table class="cost small"><thead><tr><th>server</th><th>tools</th><th>share of tool JSON</th></tr></thead><tbody>{srv_rows}</tbody></table>
</section>""")

    slides.append(f"""<section class="appendix">
<h2>A2 · Labels and scoring</h2>
<ul>
<li>A research pass mapped each intent to <b>acceptable</b> tools (a clean fit) and <b>partial</b> tools (a defensible first step). Four author overrides, each with a stated reason. {n_gap} intents have no acceptable tool: only <code>hand_off_to_human</code> is right.</li>
<li><b>Revised labels (v2)</b> also accept the store-FAQ search on the {n_gap} gap intents, a hand-off on <code>payment_issue</code>, and the refund-policy FAQ on <code>get_refund</code>.</li>
<li><b>Outcomes:</b> right · defensible · handed off (a tool could have done it) · asked (no tool call) · wrong · bluffed. Only <em>right</em> scores. Wrong + bluffed = the wrong-call rate.</li>
<li><b>Statistics:</b> exact McNemar on paired right/wrong; 4,000-sample bootstrap for the difference; Wilson intervals for rates.</li>
<li><b>Prompts:</b> identical text for the three LLMs; Jev gets the same question as a <code>choice</code> over names and descriptions. Temperature 0. Sonnet 5.5 can't disable reasoning, so it ran at low effort.</li>
</ul>
</section>""")

    nat_rows = "".join(
        f"<tr><td>{e(SHORT[a])}</td><td>{pct(nat[a]['right'])} → {pct(nat_p[a]['right'])}</td><td>{pct(nat[a]['wrong_call'])} → {pct(nat_p[a]['wrong_call'])}</td>"
        f"<td>{pct(nat[a]['asked'])} → {pct(nat_p[a]['asked'])}</td></tr>" for a in ARMS[1:])
    slides.append(f"""<section class="appendix">
<h2>A3 · Router against native tool calling</h2>
<p>The router condition pastes the tool JSON into the prompt and asks for a name; it's the like-for-like test against Jev. Native tool calling puts the tools in the API's <code>tools</code> parameter with <code>tool_choice: auto</code>, which is how an agent really works, and it may answer in text instead.</p>
<table class="cost"><thead><tr><th>native</th><th>right (plain → policy)</th><th>wrong calls</th><th>asked a question</th></tr></thead><tbody>{nat_rows}</tbody></table>
<p>Scoring the first tool call only; a second turn after a clarifying question was out of scope, so native accuracy is a lower bound. Tool names were sent as <code>server__tool</code> (dots aren't allowed).</p>
</section>""")

    slides.append(gepa_html)

    slides.append("""<section class="appendix">
<h2>A5 · Caveats and links</h2>
<ul>
<li>“No tool” means none in this catalogue. Shopify's customer-account server may expose more once a customer signs in; a store can add servers.</li>
<li>Bitext messages are template-generated and often vague; real tickets are longer and messier.</li>
<li>Jev sees names and descriptions; the LLMs also see schemas. That is how each is used, but it isn't identical information.</li>
<li>The routing policy lists example gap intents, which partly hands the models the labels. The GEPA appendix is the version that only saw a development split.</li>
<li>Costs are OpenRouter's billed prices with caching; Jev at US$0.042 per million input tokens.</li>
</ul>
<p>Code, data and every run: <code>talks/measurecamp-2026</code> · marimo explorer: <code>uv run marimo edit notebooks/explore.py</code></p>
</section>""")

    body = "\n".join(s for s in slides if s)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Who picks the tool?</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#15161a; --dim:#6b6f78; --accent:#1d4ed8; --rule:#e6e2d8; }}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; background:var(--bg); color:var(--fg); font-family: "Inter", "Helvetica Neue", Arial, sans-serif; font-size: clamp(16px, 1.6vw, 24px); }}
.deck {{ position:relative; width:100vw; height:100vh; overflow:hidden; }}
section {{ position:absolute; inset:0; padding: 6vh 8vw; display:none; flex-direction:column; justify-content:center; gap: 1.2em; }}
section.active {{ display:flex; }}
h1 {{ font-size: 3.2em; margin:0; letter-spacing:-0.02em; line-height:1.05; }}
h2 {{ font-size: 1.9em; margin:0 0 .2em; letter-spacing:-0.01em; }}
.title {{ align-items:flex-start; }}
.sub {{ font-size:1.4em; color:var(--dim); margin:0; }}
.by {{ margin-top:2em; color:var(--dim); }}
.kicker {{ text-transform:uppercase; letter-spacing:.12em; font-size:.7em; color:var(--dim); margin:0; }}
.dim {{ color:var(--dim); font-weight:400; font-size:.7em; }}
blockquote.big {{ font-size:1.8em; margin:0; font-weight:500; line-height:1.25; max-width: 22em; }}
.picks {{ display:flex; gap:1em; flex-wrap:wrap; }}
.pick {{ flex:1 1 10em; border:1px solid var(--rule); border-radius:.6em; padding:.8em 1em; background:#fff; }}
.pick .who {{ display:block; font-weight:600; font-size:.8em; margin-bottom:.3em; }}
.pick code {{ font-size:.85em; }}
.punch {{ font-size:1.6em; font-weight:600; margin:.2em 0 0; }}
.three {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:2em; }}
.num {{ font-size:4em; font-weight:700; letter-spacing:-0.03em; line-height:1; color:var(--accent); }}
.three p {{ margin:.3em 0 0; }}
small {{ color:var(--dim); font-size:.75em; }}
.note {{ color:var(--fg); font-size:.95em; max-width: 36em; margin:0; }}
.fig {{ width:100%; max-height:58vh; }}
.fig .lbl {{ font-size:18px; fill:var(--fg); }}
.fig .val {{ font-size:18px; fill:var(--fg); font-weight:600; }}
.fig .tick {{ font-size:13px; fill:var(--dim); }}
.fig .axis {{ font-size:14px; fill:var(--dim); }}
.fig .pt {{ font-size:15px; fill:var(--fg); }}
.fig .grid {{ stroke:var(--rule); }}
.intents {{ display:grid; grid-template-columns: repeat(9, 1fr); gap:.4em; }}
.cell {{ border-radius:.4em; padding:.5em .6em; font-size:.62em; line-height:1.15; background:#dfe9d8; color:#1f3d1a; display:flex; flex-direction:column; gap:.2em; min-height:3.6em; }}
.cell.gap {{ background:#ece9e2; color:var(--dim); }}
.cell small {{ font-size:.85em; opacity:.7; }}
.morph rect {{ width: var(--w1); transition: width .7s ease; }}
.morph.on rect {{ width: var(--w2); }}
.morph .val {{ transition: opacity .4s, transform .7s ease; transform: translateX(var(--x1)); }}
.morph .va {{ opacity:1; }} .morph .vb {{ opacity:0; }}
.morph.on .val {{ transform: translateX(var(--x2)); }}
.morph.on .va {{ opacity:0; }} .morph.on .vb {{ opacity:1; }}
.morph .cb {{ display:none; }} .morph.on .ca {{ display:none; }} .morph.on .cb {{ display:inline; }}
.cap {{ text-align:center; color:var(--dim); font-size:.8em; margin:.2em 0 0; }}
table.cost {{ border-collapse:collapse; font-size:.95em; width:100%; max-width:40em; }}
table.cost th, table.cost td {{ text-align:left; padding:.45em .8em; border-bottom:1px solid var(--rule); }}
table.cost th {{ color:var(--dim); font-weight:500; font-size:.8em; text-transform:uppercase; letter-spacing:.08em; }}
table.small {{ font-size:.75em; }}
ol.check {{ font-size:1.05em; line-height:1.5; padding-left:1.2em; margin:0; max-width: 32em; }}
ol.check li {{ margin:.25em 0; }}
.appendix {{ justify-content:flex-start; gap:.8em; font-size:.85em; }}
.appendix h2 {{ font-size:1.5em; }}
.appendix ul {{ margin:0; padding-left:1.2em; line-height:1.45; }}
.appendix li {{ margin:.3em 0; }}
pre {{ white-space:pre-wrap; font-size:.7em; background:#fff; border:1px solid var(--rule); padding:.8em; border-radius:.4em; max-height:40vh; overflow:auto; }}
code {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; background:#f1eee6; padding:.05em .3em; border-radius:.25em; }}
.step {{ visibility:hidden; opacity:0; transition: opacity .25s; }}
.step.on {{ visibility:visible; opacity:1; }}
.counter {{ position:fixed; right:1.2em; bottom:.9em; font-size:.6em; color:var(--dim); }}
.hint {{ position:fixed; left:1.2em; bottom:.9em; font-size:.6em; color:var(--dim); }}
body.print .deck {{ height:auto; overflow:visible; }}
body.print section {{ position:relative; display:flex; height:100vh; page-break-after:always; border-bottom:1px solid var(--rule); }}
body.print .step {{ visibility:visible; opacity:1; }}
body.print .morph rect {{ width: var(--w2); }} body.print .morph .val {{ transform: translateX(var(--x2)); }} body.print .morph .va {{ opacity:0; }} body.print .morph .vb {{ opacity:1; }} body.print .morph .ca {{ display:none; }} body.print .morph .cb {{ display:inline; }}
body.print .counter, body.print .hint {{ display:none; }}
@media print {{ body {{ font-size:18px; }} .deck {{ height:auto; }} section {{ position:relative; display:flex !important; height:100vh; page-break-after:always; }} .step {{ visibility:visible !important; opacity:1 !important; }} .morph rect {{ width: var(--w2) !important; }} .morph .val {{ transform: translateX(var(--x2)) !important; }} .morph .va {{ opacity:0 !important; }} .morph .vb {{ opacity:1 !important; }} .counter, .hint {{ display:none; }} }}
</style></head>
<body>
<div class="deck">
{body}
</div>
<div class="counter"></div>
<div class="hint">→ next · ← back · p print · a appendix</div>
<script>
const slides = [...document.querySelectorAll('section')];
let i = Math.max(0, Math.min(slides.length - 1, parseInt(location.hash.slice(1) || '0', 10) || 0));
function show(k) {{ slides.forEach((s, j) => s.classList.toggle('active', j === k)); i = k; location.hash = k;
  document.querySelector('.counter').textContent = (k + 1) + ' / ' + slides.length; }}
function steps() {{ return [...slides[i].querySelectorAll('.step, .morph')]; }}
function next() {{ const s = steps().find(x => !x.classList.contains('on')); if (s) {{ s.classList.add('on'); return; }} if (i < slides.length - 1) show(i + 1); }}
function prev() {{ const on = steps().filter(x => x.classList.contains('on')); if (on.length) {{ on[on.length - 1].classList.remove('on'); return; }} if (i > 0) {{ show(i - 1); steps().forEach(x => x.classList.add('on')); }} }}
document.addEventListener('keydown', ev => {{
  if (['ArrowRight', ' ', 'PageDown'].includes(ev.key)) {{ ev.preventDefault(); next(); }}
  else if (['ArrowLeft', 'PageUp'].includes(ev.key)) {{ ev.preventDefault(); prev(); }}
  else if (ev.key === 'Home') show(0); else if (ev.key === 'End') show(slides.length - 1);
  else if (ev.key === 'p') document.body.classList.toggle('print');
  else if (ev.key === 'a') show(slides.findIndex(s => s.classList.contains('appendix')));
}});
document.querySelector('.deck').addEventListener('click', ev => {{ if (document.body.classList.contains('print')) return; if (ev.target.closest('a, details, summary')) return; next(); }});
show(i);
if (location.search.includes('steps')) steps().forEach(x => x.classList.add('on'));
</script>
</body></html>"""


REVIEW = ROOT / "slides" / "review.html"
FEEDBACK = """
<section class="appendix review" id="feedback">
<h2>Feedback</h2>
<p>Annotate any slide above (select text or click an element), or answer here. Queue, then press Send in the Lavish bar.</p>
<form data-lavish-question="deck" onsubmit="event.preventDefault(); const f = new FormData(this); const a = [...f.entries()].filter(([k, v]) => v).map(([k, v]) => k + ': ' + v).join(' | '); if (a) window.lavish?.queuePrompt('Deck decisions: ' + a, {tag: 'choice', text: 'Deck decisions', element: this, queueKey: 'deck'});">
<p><b>Length.</b> <label><input type="radio" name="length" value="10 slides is right"> 10 slides is right</label>
<label><input type="radio" name="length" value="cut to 8"> cut to 8</label>
<label><input type="radio" name="length" value="allow 12"> allow 12</label></p>
<p><b>Native tool calling.</b> <label><input type="radio" name="native" value="keep in slide 6 only"> keep it in slide 6 only</label>
<label><input type="radio" name="native" value="give it its own slide"> give it its own slide</label>
<label><input type="radio" name="native" value="appendix only"> appendix only</label></p>
<p><b>GEPA.</b> <label><input type="radio" name="gepa" value="appendix only"> appendix only</label>
<label><input type="radio" name="gepa" value="promote to main deck"> promote to the main deck</label></p>
<p><b>Vendor names.</b> <label><input type="radio" name="names" value="name all"> name Jev, DeepSeek, Claude and the MCP vendors</label>
<label><input type="radio" name="names" value="anonymise MCP vendors"> anonymise the MCP vendors</label></p>
<p><label>Anything else: <input type="text" name="other" size="60"></label></p>
<p><button type="submit">Queue answers</button></p>
</form>
</section>
"""


def build_review() -> str:
    """The same deck as one annotatable scroll page for Lavish: every step shown, no click handler."""
    page = build()
    page = page.replace('<html lang="en">', '<html lang="en" data-lavish-annotate="on">')
    page = page.replace("<body>", '<body class="print">')
    page = page.replace("document.querySelector('.deck').addEventListener('click'", "void(")
    page = page.replace('<div class="deck">', '<p style="padding:1em 8vw;color:#6b6f78;font-size:.8em">Review copy: every click-to-reveal step is shown. The live deck is slides/index.html.</p><div class="deck">')
    page = page.replace("</div>\n<div class=\"counter\">", FEEDBACK + "</div>\n<div class=\"counter\">")
    page = page.replace("body.print section {{ position:relative; display:flex; height:100vh;", "body.print section {{ position:relative; display:flex; min-height:70vh; height:auto;")
    return page


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build())
    REVIEW.write_text(build_review())
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB) and {REVIEW.name}")
