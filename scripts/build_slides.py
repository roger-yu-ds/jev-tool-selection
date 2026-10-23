"""Build the talk: slides/index.html, a self-contained deck with click-to-reveal steps.

    uv run python scripts/build_slides.py

Keys: → / space / click = next step or slide, ← = back, Home/End, `p` = print view (all slides
stacked; then Ctrl+P → PDF), `a` = jump to the appendix. Every number is computed from runs/: the
main slides from the final setup on the fresh 600 (runs/fresh/), the appendix from the development
runs on the first 600.
"""

from __future__ import annotations

import base64
import html
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp
import toolpick.data
import toolpick.prompts

OUT = ROOT / "slides" / "index.html"
e = html.escape
ARMS = ["jev", "deepseek", "haiku", "sonnet"]
SHORT = {"jev": "Jev", "deepseek": "DeepSeek", "haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5"}
COL = {"jev": "#1d4ed8", "deepseek": "#7c3aed", "haiku": "#d97706", "sonnet": "#dc2626"}


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


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


VCOL = {"correct": "#2a9d8f", "benign": "#d6cfbf", "harmful": "#c1121f"}


def stacked(rows: list[tuple[str, dict, str]], *, width=900) -> str:
    """One bar per selector split into correct / benign / harmful; each row is a reveal step."""
    rh, left = 52, 230
    height = rh * len(rows) + 46
    pw = width - left - 20
    out = [f'<svg viewBox="0 0 {width} {height}" class="fig">']
    lx = left
    for v in ("correct", "benign", "harmful"):
        out.append(f'<rect x="{lx}" y="4" width="14" height="14" fill="{VCOL[v]}"/><text x="{lx + 20}" y="16" class="tick">{v}</text>')
        lx += 120
    for i, (label, s, cls) in enumerate(rows):
        y = 34 + i * rh
        g = [f'<g class="{cls}"><text x="{left - 14}" y="{y + 24}" text-anchor="end" class="lbl">{e(label)}</text>']
        x = left
        for v in ("correct", "benign", "harmful"):
            w = s[v] * pw
            g.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="36" fill="{VCOL[v]}"/>')
            if w > 46:
                fill = "#fff" if v != "benign" else "#15161a"
                g.append(f'<text x="{x + w / 2:.1f}" y="{y + 24}" text-anchor="middle" class="val" style="fill:{fill}">{pct(s[v])}</text>')
            x += w
        out.append("".join(g) + "</g>")
    out.append("</svg>")
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


FRESH = {"jev": "fresh/jev", "deepseek": "fresh/native_deepseek", "haiku": "fresh/native_haiku", "sonnet": "fresh/native_sonnet"}
# US$ per million tokens: input, cached read, output (OpenRouter list prices for the providers used).
PRICES = {"deepseek": (0.14, 0.0042, 0.42), "haiku": (0.10, 0.01, 0.5), "sonnet": (2.0, 0.10, 10.0)}
CASCADE_T = 0.8  # chosen on the first 600 (development) messages, applied unchanged to the fresh 600


def fsumm(arm: str) -> dict:
    return tp.summarise(tp.scored_rows(FRESH[arm]))


def frows(arm: str) -> dict[str, dict]:
    return {r["id"]: r for r in tp.scored_rows(FRESH[arm])}


def question(reply: str, limit: int = 110) -> str:
    """The first question in a text reply, shortened: what the model actually asked."""
    text = re.sub(r"[*_`#]", "", " ".join(reply.split()))
    m = re.search(r"[^.?!:]*\?", text)
    q = (m.group(0) if m else text).strip(" -–")
    return q if len(q) <= limit else q[: limit - 1].rsplit(" ", 1)[0] + "…"


def say(arm: str, r: dict) -> str:
    """What a selector did, in words."""
    if r["kind"] == "text" or r["pick"] is None:
        q = question(r.get("reply") or "")
        return f"asks the customer: <i>“{e(q)}”</i>" if q else "asks the customer a question"
    p = r["pick"]
    if p == tp.HAND:
        return "hands off to a person"
    return f"calls <code>{e(p.split('.')[-1])}</code>"


def intents_grid() -> str:
    lab = tp.labels()
    cnt: dict[str, int] = {}
    for r in tp.requests("fresh"):
        cnt[r.intent] = cnt.get(r.intent, 0) + 1
    cells = []
    for k in sorted(lab, key=lambda k: (lab[k]["gap"], k)):
        cells.append(f'<div class="cell {"gap" if lab[k]["gap"] else "cov"}"><span>{e(k.replace("_", " "))}</span><small>{cnt.get(k, 0)}</small></div>')
    return '<div class="intents">' + "".join(cells) + "</div>"


def build() -> str:
    S = {a: fsumm(a) for a in ARMS}
    R = {a: frows(a) for a in ARMS}
    ids = sorted(set.intersection(*(set(r) for r in R.values())))
    lab = tp.labels()
    n_cov = sum(not v["gap"] for v in lab.values())
    fresh_reqs = tp.requests("fresh")
    gap_share = sum(lab[r.intent]["gap"] for r in fresh_reqs) / len(fresh_reqs)
    pairs = {a: tp.paired(FRESH["jev"], FRESH[a]) for a in ARMS[1:]}
    best = max(ARMS[1:], key=lambda a: S[a]["correct"])

    # hook: a message where the four answers differ most (verdicts first, then picks)
    def spread(i):
        return (len({R[a][i]["pick"] for a in ARMS}), len({R[a][i]["verdict"] for a in ARMS}), -len(R["jev"][i]["text"]))
    hook = max(ids, key=spread)
    hook_text = R["jev"][hook]["text"]

    # scoring examples from the fresh run
    def first(pred):
        return next((R[a][i], a) for i in ids for a in ARMS[::-1] if pred(R[a][i]))
    ex_ok, ex_ok_a = first(lambda r: r["verdict"] == "correct" and not r["gap"] and r["pick"] not in (None, tp.HAND))
    ex_benign, ex_benign_a = first(lambda r: r["verdict"] == "benign" and r["pick"] == tp.HAND)
    ex_harm, ex_harm_a = first(lambda r: r["verdict"] == "harmful" and "create" in (r["pick"] or ""))
    n_harm = sum(R[a][i]["verdict"] == "harmful" for a in ARMS for i in ids)

    # confidence and the Jev-first cascade (threshold fixed on the development sample)
    conf = {i: R["jev"][i]["confidence"] or 0 for i in ids}
    bands = [(0.9, 1.01, "0.9 or higher"), (0.7, 0.9, "0.7 – 0.9"), (0.0, 0.7, "below 0.7")]
    band_rows = []
    for lo, hi, label in bands:
        b = [i for i in ids if lo <= conf[i] < hi]
        band_rows.append((label, len(b) / len(ids), sum(R["jev"][i]["right"] for i in b) / max(1, len(b))))
    sent = [i for i in ids if conf[i] < CASCADE_T]
    casc_right = sum((R["sonnet"][i] if conf[i] < CASCADE_T else R["jev"][i])["right"] for i in ids) / len(ids)
    casc_harm = sum((R["sonnet"][i] if conf[i] < CASCADE_T else R["jev"][i])["wrong_call"] for i in ids) / len(ids)
    # Jev's pick is always paid for; its argument-filling call only when Jev's answer is used
    casc_cost = 1000 * sum(R["jev"][i]["pick_cost_usd"] + (R["sonnet"][i]["cost_usd"] if conf[i] < CASCADE_T else R["jev"][i]["fill_cost_usd"])
                           for i in ids) / len(ids)  # fmt: skip

    # where Jev and Sonnet disagree
    only_j = [i for i in ids if R["jev"][i]["right"] and not R["sonnet"][i]["right"]]
    only_s = [i for i in ids if R["sonnet"][i]["right"] and not R["jev"][i]["right"]]

    slides = []
    slides.append("""<section class="title">
<p class="kicker">MeasureCamp Sydney · 24 October 2026</p>
<h1>Who picks the tool?</h1>
<p class="sub">Measuring AI tool selection on real MCP servers</p>
<p class="by">Roger Yu</p>
</section>""")

    # about me: bullets drafted from career/career-facts.md (swap in the LinkedIn wording); photos in slides/img
    def photo(name: str) -> str:
        return "data:image/jpeg;base64," + base64.b64encode((ROOT / "slides" / "img" / f"{name}.jpg").read_bytes()).decode()
    slides.append(f"""<section>
<h2>About me</h2>
<div class="about">
<ul class="about-list">
<li>Data scientist turned AI engineer, based in Sydney</li>
<li>At THE ICONIC, built a multimodal search service that grew into a shared service used by many teams</li>
<li>Since then, AI roles at Heatseeker.ai and Dovetail, building and evaluating LLM features</li>
<li>Most interested in how to measure AI systems, which is where this talk comes from</li>
</ul>
<div class="photos">
<figure><img src="{photo('naomi_mochi')}" alt="A toddler sitting with a white Samoyed"><figcaption>Naomi and Mochi</figcaption></figure>
<figure><img src="{photo('cactus')}" alt="Pots of cacti and living stones"><figcaption>The cactus collection</figcaption></figure>
</div>
</div>
</section>""")

    picks = "".join(f'<div class="pick step"><span class="who" style="color:{COL[a]}">{e(SHORT[a])}</span>{say(a, R[a][hook])}</div>' for a in ARMS)
    slides.append(f"""<section>
<p class="kicker">A customer writes</p>
<blockquote class="big">“{e(hook_text)}”</blockquote>
<p class="kicker step">Four AI assistants, four answers</p>
<div class="picks">{picks}</div>
<p class="punch step">Which one is right? And how would you measure it?</p>
</section>""")

    slides.append("""<section>
<h2>The test</h2>
<div class="three">
<div class="step"><div class="num">600</div><p>customer-support messages<br><small>Bitext · 27 request types · typos and all</small></p></div>
<div class="step"><div class="num">90</div><p>real tools from 7 official MCP servers<br><small>Shopify · Stripe · Intercom · GitHub · Notion, copied word for word</small></p></div>
<div class="step"><div class="num">4</div><p>AI assistants<br><small>Jev · DeepSeek V4.1 Flash · Claude Haiku 5.5 · Claude Sonnet 5.5</small></p></div>
</div>
<p class="note step">Each one sees a message and the 90 tools, and decides the first thing to do: call a tool, ask the customer a question, or hand off to a person.</p>
</section>""")

    slides.append(f"""<section>
<h2>Real servers cover {n_cov} of 27 request types</h2>
{intents_grid()}
<p class="note step">No official tool cancels or edits a placed order, opens a complaint, or touches an account.
For {pct(gap_share)} of messages the right answer is <b>“hand off to a person”</b>.</p>
</section>""")

    slides.append(f"""<section>
<h2>Three kinds of answer</h2>
<div class="verdicts">
<div class="step v-ok"><h3>Correct</h3><p>“{e(ex_ok['text'])}”</p><p class="who">{e(SHORT[ex_ok_a])} {say(ex_ok_a, ex_ok)}</p></div>
<div class="step v-benign"><h3>Wrong, but benign</h3><p>“{e(ex_benign['text'])}”</p><p class="who">{e(SHORT[ex_benign_a])} {say(ex_benign_a, ex_benign)}: the customer waits, nothing breaks</p></div>
<div class="step v-harm"><h3>Wrong, and harmful</h3><p>“{e(ex_harm['text'])}”</p><p class="who">{e(SHORT[ex_harm_a])} {say(ex_harm_a, ex_harm)}: it creates a customer in Stripe, the payments system, instead of a store account</p></div>
</div>
<p class="note step">Benign: handing off, asking a question, or reading the customer's own records. Harmful: changing something it shouldn't, or using a tool built for something else, such as GitHub's issue tracker or the helpdesk's internal records.
In the final runs, harmful answers were rare: {n_harm} in {len(ids) * len(ARMS):,}.</p>
</section>""")

    vrows = [(SHORT[a], S[a], "step") for a in ARMS]
    slides.append(f"""<section>
<h2>Results</h2>
{stacked(vrows)}
<p class="note step">Jev against {e(SHORT[best])}: {pct(S['jev']['correct'])} and {pct(S[best]['correct'])} correct
(only Jev right on {pairs[best]['only_a']} messages, only {e(SHORT[best])} on {pairs[best]['only_b']}; p = {pairs[best]['p']:.2g}).</p>
</section>""")

    fills = list(tp.load_run("fill/jev_fresh").values())
    fill_tok = statistics.median(f["input_tokens"] + f["output_tokens"] for f in fills)
    fill_lat = statistics.median(f["latency_s"] for f in fills)
    fill_cost = 1000 * sum(f["cost_usd"] for f in fills) / len(fills)
    tok_ratio = S["haiku"]["input_tokens_median"] / S["deepseek"]["input_tokens_median"]

    def uncached(a):
        if a == "jev":
            return S[a]["cost_per_1000"]
        p_in, _, p_out = PRICES[a]
        return 1000 * (S[a]["input_tokens_median"] * p_in + S[a]["output_tokens_median"] * p_out) / 1e6
    cost_rows = "".join(
        f"<tr class='step'><td style='color:{COL[a]};font-weight:600'>{e(SHORT[a])}</td><td>{S[a]['input_tokens_median']:,.0f}</td>"
        f"<td>{S[a]['latency_median_s']:.1f} s</td><td>${S[a]['cost_per_1000']:.2f}</td><td>${uncached(a):,.2f}</td></tr>" for a in ARMS)
    slides.append(f"""<section>
<h2>What it costs</h2>
<table class="cost big"><thead><tr><th></th><th>tokens read per message</th><th>median time</th><th>per 1,000 messages</th><th>without caching</th></tr></thead>
<tbody>{cost_rows}</tbody></table>
<p class="note step">Jev only picks the tool, reading each tool's name and description (about 7,200 tokens). A second, small
call to DeepSeek then fills in the chosen tool's arguments; Jev's row includes it ({fill_tok:,.0f} tokens, {fill_lat:.1f} s,
${fill_cost:.3f} per 1,000). The LLMs pick and fill in one call, so they read every tool's full input schema: the same text for all three, which
Claude's tokeniser counts as about {tok_ratio:.1f} times as many tokens as DeepSeek's.</p>
<p class="note step">The tool list is the same on every call, so the providers cache it and bill repeat reads at 3–5% of the
normal price; Jev bills every token at $0.042 per million. That is why cached DeepSeek costs about the same as Jev. Without a
warm cache (quiet traffic, or tool lists that change per customer) the LLMs cost 25 to 500 times more.</p>
</section>""")

    band_html = "".join(f"<tr class='step'><td>{e(lbl)}</td><td>{pct(share)}</td><td><b>{pct(acc)}</b></td></tr>" for lbl, share, acc in band_rows)
    slides.append(f"""<section>
<h2>Jev knows when it's right</h2>
<table class="cost big"><thead><tr><th>Jev's confidence</th><th>share of messages</th><th>correct</th></tr></thead><tbody>{band_html}</tbody></table>
<p class="note step"><b>Use both.</b> Every message goes to Jev first. If Jev's confidence is {CASCADE_T} or higher, use its answer;
otherwise send the message on to Sonnet and use Sonnet's answer. Only {pct(len(sent) / len(ids))} of messages reach Sonnet.</p>
<p class="note step">Result: {pct(casc_right)} correct ({pct(S['sonnet']['correct'])} for Sonnet alone), {pct(casc_harm, 1)} harmful,
${casc_cost:.2f} per 1,000 messages instead of ${S['sonnet']['cost_per_1000']:.2f}. The {CASCADE_T} cut-off was chosen on the development messages, before this data.</p>
</section>""")

    def top_intents(sel):
        c: dict[str, int] = {}
        for i in sel:
            c[R["jev"][i]["intent"]] = c.get(R["jev"][i]["intent"], 0) + 1
        return ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:3])

    def examples(sel, k=3):
        seen, out = set(), []
        for i in sel:
            if R["jev"][i]["intent"] in seen:
                continue
            seen.add(R["jev"][i]["intent"])
            out.append(i)
            if len(out) == k:
                break
        return "".join(
            f'<div class="ex step"><p class="q">“{e(R["jev"][i]["text"])}”</p>'
            f'<p><span style="color:{COL["jev"]}">Jev</span> {say("jev", R["jev"][i])} · '
            f'<span style="color:{COL["sonnet"]}">Sonnet</span> {say("sonnet", R["sonnet"][i])}</p></div>' for i in out)

    def unneeded_handoffs(rows):
        covered = [r for r in rows if not r["gap"]]
        return sum(r["outcome"] == "handed_off" for r in covered) / len(covered)
    dev_ho = unneeded_handoffs(tp.scored_rows("jev_policy"))
    fresh_ho = unneeded_handoffs([R["jev"][i] for i in ids])
    son_ho = unneeded_handoffs([R["sonnet"][i] for i in ids])
    jev_miss = sum(not R["jev"][i]["right"] for i in ids)
    jev_ho_miss = sum(R["jev"][i]["outcome"] == "handed_off" for i in ids)
    ho = [R["jev"][i] for i in ids if R["jev"][i]["pick"] == tp.HAND]
    ho_ok = sorted(r["confidence"] for r in ho if r["right"])
    ho_bad = sorted(r["confidence"] for r in ho if not r["right"])
    slides.append(f"""<section>
<h2>They fail in different places</h2>
<div class="two">
<div><div class="num" style="color:{COL['jev']}">{len(only_j)}</div><p>messages only Jev got right<br><small>{e(top_intents(only_j))}</small></p>{examples(only_j)}</div>
<div><div class="num" style="color:{COL['sonnet']}">{len(only_s)}</div><p>messages only Sonnet got right<br><small>{e(top_intents(only_s))}</small></p>{examples(only_s)}</div>
</div>
<p class="note step">Jev wins where nothing can be done but pass the customer to a person; Sonnet asks a question instead.
Sonnet wins where a tool can do it; Jev hands off. Customer wording cut Jev's unneeded hand-offs from {pct(dev_ho)} (development messages) to {pct(fresh_ho)}
of the messages a tool could handle ({pct(son_ho)} for Sonnet), but they are still {jev_ho_miss} of its {jev_miss} mistakes.
They come with low confidence: a median of {ho_bad[len(ho_bad) // 2]:.2f}, against {ho_ok[len(ho_ok) // 2]:.2f} for its right
hand-offs, which is what the 0.8 cut-off catches.</p>
</section>""")

    def dev_correct(name, ids_=None):
        rows = tp.scored_rows(name)
        if ids_ is not None:
            rows = [r for r in rows if r["id"] in ids_]
        return tp.summarise(rows)["correct"]
    held_ids = set(json.loads((ROOT / "data" / "gepa_deepseek.json").read_text())["held_ids"])
    lessons = [
        ("Write down when to pass a customer to a person, and give it to every model.",
         f"Three sentences of policy took Sonnet from {pct(dev_correct('router_sonnet'))} to {pct(dev_correct('router_sonnet_policy'))} on the development messages."),
        ("Describe each tool the way customers talk.",
         ("“Download my bill”, not “This tool will fetch a list of Invoices from Stripe”. One extra line on six tools took Jev from "
          f"{pct(dev_correct('jev_policy', held_ids))} to {pct(dev_correct('experiments/jev_descriptions_held'))}.")),
        ("Check the answer key against every failure.",
         (f"We found nine mistakes in ours. With the original key, Jev led Sonnet {pct(tp.summarise(tp.scored_rows('jev', 'first_pass'))['correct'])} to "
          f"{pct(tp.summarise(tp.scored_rows('router_sonnet', 'first_pass'))['correct'])}; after the fixes and the policy, Sonnet led.")),
        ("Count harmless and harmful mistakes separately.",
         "Passing a customer to a person, or asking a question, is not the same as doing the wrong thing."),
        ("Test on messages you didn't build with.",
         "Every number on the earlier slides comes from 600 fresh messages, drawn after all of the above was fixed."),
    ]
    items = "".join(f'<li class="step"><b>{e(a)}</b><br><small>{e(b)}</small></li>' for a, b in lessons)
    slides.append(f"""<section>
<h2>What we learned about measuring this</h2>
<ol class="check">{items}</ol>
<p class="punch step">The score is a design decision. Make it on purpose.</p>
</section>""")

    # ---------------------------------------------------------------- appendix
    tok = tp.tool_tokens_by_server()
    tot = sum(tok.values())
    srv_rows = "".join(f"<tr><td>{e(s)}</td><td>{sum(1 for t in tp.catalog() if t['server'] == s)}</td><td>{pct(c / tot)}</td></tr>"
                       for s, c in sorted(tok.items(), key=lambda kv: -kv[1]))  # fmt: skip
    slides.append(f"""<section class="appendix">
<h2>Appendix · A1 · Data and tools</h2>
<ul>
<li><b>Messages:</b> from Bitext's customer-support dataset (26,872 rows; template-generated with deliberate typos; CDLA-Sharing-1.0). Bitext supplies only the customer messages and a label for the request type (27 types, such as “track order”); it knows nothing about tools. Which tools fit each request type is our answer key (A3). The main results use 600 <b>fresh</b> messages (22–23 per type), drawn after the study was designed, sharing no message and at most 58% of words with the 600 used during development.</li>
<li><b>Tools:</b> {len(tp.catalog())} tools copied word for word from 7 official MCP servers. An MCP server is how a company lets an AI assistant use its product: it publishes a list of tools, each with a name, a description and the arguments it takes. We collected the lists three ways. Stripe (22 tools), GitHub (10) and Notion (24) publish their servers as software anyone can run, so we ran the official version and asked it for its list. Shopify's servers run on every Shopify store, so we asked a live public store (20 tools across three servers). Intercom's needs a login, so we copied its 14 tools from its documentation. The 91st tool, <code>hand_off_to_human</code>, is ours.</li>
<li>Stripe's hosted server has since replaced these 22 tools with generic read and write tools; we used the last version with one tool per task.</li>
</ul>
<table class="cost small"><thead><tr><th>server</th><th>tools</th><th>share of the tool JSON</th></tr></thead><tbody>{srv_rows}</tbody></table>
</section>""")

    words = "".join(f"<li><code>{e(k.split('.')[-1])}</code>: {e(v)}</li>" for k, v in tp.data.CUSTOMER_WORDS.items())
    slides.append(f"""<section class="appendix">
<h2>A2 · The setup</h2>
<ul>
<li><b>Every selector chooses from the same 91 tools:</b> the 90 real ones plus <code>hand_off_to_human</code>, the one we added.</li>
<li><b>The LLMs work as real agents:</b> the tools go in the API's tools setting, the way an AI app normally gives a model its tools, and the model decides by itself whether to call one or reply in text. Least-random setting (temperature 0), prompt caching on. Tool names are written <code>server__tool</code> because the API doesn't allow dots. Scored on the model's first move: its first tool call, or a question back to the customer.</li>
<li><b>Jev:</b> one multiple-choice question whose 91 options are the tool names and descriptions; it returns a probability for every option.</li>
<li><b>Same routing policy for all four:</b></li>
</ul>
<pre>{e(tp.prompts.POLICY.strip())}</pre>
<ul><li><b>Customer wording.</b> Every tool comes with a short description written by the company that makes it, for example Stripe's <code>list_invoices</code>: “This tool will fetch a list of Invoices from Stripe.” We kept 84 of the 90 exactly as published. For six, we added one line at the end in the words customers use:<ul>{words}</ul></li></ul>
</section>""")

    first = tp.labels("first_pass")
    changes = "".join(f"<li><code>{e(k)}</code>: {e(' '.join(v['changes'][-2:]))}</li>" for k, v in lab.items()
                      if v["changes"] and (v["acceptable"] != first[k]["acceptable"] or v["ask_ok"]))  # fmt: skip
    changes += f"<li><b>Every request type:</b> {e(tp.data.AUDIT_READ_ONLY)}</li>"
    slides.append(f"""<section class="appendix">
<h2>A3 · The answer key</h2>
<p>Per request type: the tools that are <b>correct</b>; <b>benign</b> answers (handing off where a tool existed, asking a question, or reading the customer's own records or the store's public information); and <b>harmful</b> ones (a change that wasn't asked for, or a tool built for something else: GitHub, Notion, the helpdesk's internal conversations and contacts, Stripe's developer docs). Handing off is correct on the {27 - n_cov} types no tool covers and benign elsewhere; asking a question back is correct where the tool needs details the message lacks. Every failure in the development runs was checked by hand; these changes came out of it:</p>
<ul class="tight">{changes}</ul>
</section>""")

    def dev(name, version="audited", ids_=None):
        rows = tp.scored_rows(name, version)
        if ids_ is not None:
            rows = [r for r in rows if r["id"] in ids_]
        return tp.summarise(rows)["correct"] if rows else float("nan")
    g = json.loads((ROOT / "data" / "gepa_deepseek.json").read_text())
    held = set(g["held_ids"])
    journey = [
        ("Plain prompt", dev("jev"), dev("router_sonnet"), dev("native_sonnet")),
        ("+ routing policy", dev("jev_policy"), dev("router_sonnet_policy"), dev("native_sonnet_policy")),
        ("+ customer wording (held-out)", dev("experiments/jev_descriptions_held"), dev("experiments/sonnet_policy_descriptions_held"), None),
    ]
    jrows = "".join(f"<tr><td>{e(a)}</td><td>{pct(b)}</td><td>{pct(c)}</td><td>{'–' if d is None else pct(d)}</td></tr>" for a, b, c, d in journey)
    slides.append(f"""<section class="appendix">
<h2>A4 · How we got here (development runs, first 600)</h2>
<table class="cost"><thead><tr><th>correct, audited key</th><th>Jev</th><th>Sonnet, told to name a tool</th><th>Sonnet, real agent</th></tr></thead><tbody>{jrows}</tbody></table>
<ul>
<li>With the first-pass answer key and a plain prompt, Jev led ({pct(dev('jev', 'first_pass'))} against {pct(dev('router_sonnet', 'first_pass'))}). Auditing the key and adding the policy turned that into a Sonnet lead.</li>
<li>Jev over-used the hand-off because tool descriptions are written for developers; one line of customer wording fixed it. Tool order and the GitHub/Notion distractors made little difference (A6).</li>
<li>DSPy GEPA rewrote DeepSeek's prompt from 200 messages: {pct(dev('router_deepseek_gepa', 'first_pass', held))} correct on the 399 it never saw, against {pct(dev('router_deepseek_policy', 'first_pass', held))} for the hand-written policy (first-pass key).</li>
</ul>
</section>""")

    inv = next(t for t in tp.tools() if t["name"] == "stripe.list_invoices")["description"].strip()
    inv_words = tp.data.CUSTOMER_WORDS["stripe.list_invoices"]
    slides.append(f"""<section class="appendix">
<h2>A5 · The three versions, side by side</h2>
<div class="versions">
<div><h3>1 · Plain prompt</h3><p>What the LLM agents were told:</p><pre>{e(tp.prompts.NATIVE_SYSTEM)}</pre>
<p>Jev's question:</p><pre>{e(tp.prompts.JEV_INSTRUCTIONS)}</pre></div>
<div><h3>2 · + routing policy</h3><p>The same, with these three rules added for every model:</p><pre>{e(tp.prompts.POLICY.strip())}</pre></div>
<div><h3>3 · + customer wording</h3><p>The same prompts; six tool descriptions get one extra line. Stripe's <code>list_invoices</code>, before:</p>
<pre>{e(inv)}</pre><p>after:</p><pre>{e(inv)}
{e(inv_words)}</pre></div>
</div>
</section>""")

    variants = [
        ("Re-run, nothing changed", "baseline", "The same setup sent again, to see how much the answers move on their own."),
        ("Hand-off listed first", "handoff_first", "The hand-off option moved from last of the 91 options to first. Does position matter?"),
        ("Options shuffled", "shuffled", "All 91 options in random order."),
        ("GitHub and Notion removed", "no_distractors", "The 34 tools a support assistant wouldn't need, taken out. Do irrelevant tools confuse it?"),
        ("No hand-off sentence in the question", "no_handoff_line", "Removed “pick hand_off_to_human if no tool can do it” from Jev's question. Is the question nudging it?"),
        ("Narrower hand-off description", "handoff_desc", "Hand-off described by what it's for (complaints, account changes…) instead of “none of the other tools can do this”."),
        ("Customer wording on six tools", "descriptions", "One line in customers' words added to six tool descriptions, e.g. “download my bill #123” on Stripe's invoices tool."),
    ]  # fmt: skip
    dev_ids = set(json.loads((ROOT / "data" / "gepa_deepseek.json").read_text())["dev_ids"])
    need_handoff = sum(lab[tp.data.all_requests_by_id()[i].intent]["gap"] for i in dev_ids) / len(dev_ids)
    vrows_html = "".join(
        f"<tr><td><b>{e(lbl)}</b><br><small>{e(why)}</small></td><td>{pct(tp.summarise(tp.scored_rows(f'experiments/jev_{v}_dev'))['correct'])}</td>"
        f"<td>{pct(tp.summarise(tp.scored_rows(f'experiments/jev_{v}_dev'))['handoff'])}</td></tr>" for lbl, v, why in variants)
    slides.append(f"""<section class="appendix">
<h2>A6 · Why Jev handed off too often</h2>
<p>We changed one thing at a time and asked Jev again about the same 200 development messages. <b>Correct</b>: share of answers that were right. <b>Hands off</b>: share of messages Jev passed to a person; about {pct(need_handoff)} of these messages really need that, so anything far above it is over-handing-off.</p>
<table class="cost small vtable"><thead><tr><th>what we changed</th><th>correct</th><th>hands off</th></tr></thead><tbody>{vrows_html}</tbody></table>
<p>Only the customer wording made a big difference: Jev matches the customer's words to the tool descriptions, and “download my bill” doesn't match “fetch a list of Invoices from Stripe”. Re-running alone changes about 8 of 200 answers.</p>
</section>""")

    ds_r, ds_n = tp.summarise(tp.scored_rows("fresh/router_deepseek")), S["deepseek"]
    p_ds = tp.paired(FRESH["jev"], "fresh/router_deepseek")
    slides.append(f"""<section class="appendix">
<h2>A7 · Same model, two ways of asking</h2>
<table class="cost"><thead><tr><th>DeepSeek V4.1 Flash, fresh 600</th><th>correct</th><th>asks a question</th><th>hands off</th><th>median time</th><th>per 1,000</th></tr></thead><tbody>
<tr><td>as an agent (tools in the API, free to reply)</td><td>{pct(ds_n['correct'], 1)}</td><td>{pct(ds_n['asked'])}</td><td>{pct(ds_n['handoff'])}</td><td>{ds_n['latency_median_s']:.1f} s</td><td>${ds_n['cost_per_1000']:.2f}</td></tr>
<tr><td>as a picker (“name the one tool”, like Jev)</td><td>{pct(ds_r['correct'], 1)}</td><td>{pct(ds_r['asked'])}</td><td>{pct(ds_r['handoff'])}</td><td>{ds_r['latency_median_s']:.1f} s</td><td>${ds_r['cost_per_1000']:.2f}</td></tr>
</tbody></table>
<p>As an agent, DeepSeek ignores the policy's “hand off” rule and asks a clarifying question instead. Made to pick, it hands off like the others and scores {pct(ds_r['correct'])}, still behind Jev's {pct(S['jev']['correct'])} on the same messages (only Jev right {p_ds['only_a']}, only DeepSeek {p_ds['only_b']}; p = {p_ds['p']:.1g}); most of its misses are hand-offs on cancellation-fee and refund questions.</p>
</section>""")

    # argument checks: data/arguments.json, written by scripts/check_arguments.py
    def e2e(a):
        return [R[a][i]["right"] and R[a][i]["args_ok"] is not False for i in ids]
    argj_p = tp.mcnemar(e2e("jev"), e2e("sonnet"))["p"]
    argj = json.loads((ROOT / "data" / "arguments.json").read_text())
    arg_rows = "".join(
        f"<tr><td style='color:{COL[a]};font-weight:600'>{e(SHORT[a])}</td><td>{pct(argj[a]['right_tool'], 1)}</td>"
        f"<td>{argj[a]['args_ok']} of {argj[a]['calls_scored']}</td><td><b>{pct(argj[a]['right_end_to_end'], 1)}</b></td></tr>" for a in ARMS)
    slides.append(f"""<section class="appendix">
<h2>A8 · Right tool, right arguments?</h2>
<p>There is no answer key for argument values, so we checked what can be checked by rule, on every call to a correct tool
other than a hand-off: the arguments match the tool's schema; every order number, ID or email comes from the message (none made up);
an order number in the message is passed on; refund amounts are in cents. Free text (search queries, hand-off reasons) isn't scored.
Jev's arguments come from its second, filling call.</p>
<table class="cost small"><thead><tr><th></th><th>right tool</th><th>arguments pass</th><th>right tool and arguments</th></tr></thead><tbody>{arg_rows}</tbody></table>
<p>Jev's misses: when it picks the return or refund tool for “I want my money back”, the filling call must invent the order line or
payment it doesn't have (<code>line_item_id: "unknown"</code>); on those messages the LLMs mostly ask the customer or look the order up first. DeepSeek made up emails such as
<code>dev@example.com</code> from a first name. Jev and Sonnet are now level (p = {argj_p:.2g}).</p>
</section>""")

    # Jev with a need_more_information option, and the ask-first stipulation (both keys, every selector)
    from toolpick.scoring import stipulated_right
    ask_runs = [("Jev", "fresh/jev", "jev"), ("Jev + “need more information”", "fresh/jev_ask", "jev"),
                ("Jev + inputs in the descriptions", "fresh/jev_params", "jev"), ("Jev + both", "fresh/jev_ask_params", "jev"),
                ("DeepSeek", FRESH["deepseek"], "deepseek"),
                ("Haiku 5.5", FRESH["haiku"], "haiku"), ("Sonnet 5.5", FRESH["sonnet"], "sonnet")]  # fmt: skip
    AR = {run: {r["id"]: r for r in tp.scored_rows(run)} for _, run, _ in ask_runs}
    ask_rows = "".join(
        f"<tr><td style='color:{COL[c]};font-weight:600'>{e(lbl)}</td><td>{pct(sum(r['right'] for r in AR[run].values()) / len(AR[run]), 1)}</td>"
        f"<td>{pct(sum(map(stipulated_right, AR[run].values())) / len(AR[run]), 1)}</td></tr>" for lbl, run, c in ask_runs)
    ja = AR["fresh/jev_ask"]
    asks = [r for r in ja.values() if r["pick"] == tp.ASK]
    ask_ok_n = sum(r["right"] for r in asks)
    ask_conf = statistics.median(r["confidence"] for r in asks)
    dev_a = {r["id"]: r for r in tp.scored_rows("experiments/jev_ask_first")}
    dev_b = {r["id"]: r for n in ("dev", "held") for r in tp.scored_rows(f"experiments/jev_descriptions_{n}")}
    dev_ids = sorted(dev_a.keys() & dev_b.keys())
    jv = json.loads((ROOT / "data" / "jev_variants.json").read_text())  # scripts/compare_jev_variants.py
    dev_p, cost_p = jv["dev"]["Jev + inputs"]["correct"], jv["fresh"]["Jev + inputs"]["cost_per_1000"]
    p_ask = tp.mcnemar([stipulated_right(AR["fresh/jev"][i]) for i in ids], [stipulated_right(AR[FRESH["sonnet"]][i]) for i in ids])["p"]
    slides.append(f"""<section class="appendix">
<h2>A9 · Two changes to Jev's options</h2>
<p>The LLMs can always reply with a question; Jev couldn't. We added a 92nd option, <code>need_more_information</code>, which an agent
would turn into a question, and scored it exactly like an LLM's question. Separately, we added each tool's inputs (name, type,
required or not) to the descriptions Jev reads, as the LLMs see the full input schemas. <b>Ask-first rule</b>: on refund, order and payment
requests that give no order number or amount (like “I want my money back”), only a question, a hand-off or a lookup that needs
nothing from the customer counts as correct, for every model.</p>
<table class="cost small"><thead><tr><th>fresh 600</th><th>answer key as is</th><th>ask-first rule</th></tr></thead><tbody>{ask_rows}</tbody></table>
<p>Jev used the new option on {pct(len(asks) / len(ja))} of messages, at low confidence (median {ask_conf:.2f}). {ask_ok_n} were right
(refunds, orders, payment problems); the rest replaced a correct invoice lookup, refund check or hand-off. On the development
messages it also lowered Jev's score ({pct(sum(dev_b[i]['right'] for i in dev_ids) / len(dev_ids))} → {pct(sum(dev_a[i]['right'] for i in dev_ids) / len(dev_ids))}).
Listing the inputs didn't help either ({pct(jv["dev"]["Jev"]["correct"])} → {pct(dev_p)} on the development messages), and cost more (${cost_p:.2f} per 1,000).
Under the ask-first rule Sonnet leads Jev, but not significantly (p = {p_ask:.2f}).</p>
</section>""")

    slides.append("""<section class="appendix">
<h2>A10 · Caveats</h2>
<ul>
<li>Bitext messages are template-generated and often vague; real tickets are longer and messier.</li>
<li>The customer wording and the routing policy were written while looking at the development messages; the fresh 600 test them on new messages from the same dataset, not on real traffic.</li>
<li>“No tool” means none in this catalogue: Shopify's customer-account server may expose more once a customer signs in, and a store can add servers.</li>
<li>The answer key is a judgement. It's published with a reason for every decision (<code>src/toolpick/data.py</code>), and the explorer re-scores everything if you change it.</li>
<li>Each message is scored on its own, as the first message of a chat from a signed-in customer with no order in view. With the conversation history, asking versus acting and the arguments would change (A8, A9). Argument values are checked by rule, not against an answer key. Sonnet 5.5 can't disable reasoning, so it ran at low effort. Costs are OpenRouter's billed prices; Jev at US$0.042 per million input tokens.</li>
<li>Jev reads names and descriptions; the LLMs also read schemas. That is how each is used.</li>
</ul>
<p>Code, data and every run: <code>github.com/roger-yu-ds/jev-tool-selection</code> (private) · explorer: <code>uv run marimo edit notebooks/explore.py</code></p>
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
.about {{ display:grid; grid-template-columns: 1.1fr 1fr; gap:2.2em; align-items:center; }}
.about-list {{ margin:0; padding-left:1.1em; line-height:1.5; font-size:.95em; }}
.about-list li {{ margin:.45em 0; }}
.photos {{ display:grid; grid-template-columns: 3fr 4fr; gap:.8em; align-items:end; }}
.photos figure {{ margin:0; }}
.photos img {{ width:100%; border-radius:10px; display:block; }}
.photos figcaption {{ font-size:.55em; color:var(--dim); margin-top:.35em; }}
@media (max-width: 700px) {{ .about {{ grid-template-columns: 1fr; }} }}
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
.verdicts {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:1.2em; }}
.verdicts > div {{ border-radius:.6em; padding:1em 1.2em; background:#fff; border-top:6px solid; }}
.verdicts h3 {{ margin:0 0 .5em; font-size:1.1em; }}
.verdicts p {{ margin:.3em 0; font-size:.85em; }}
.verdicts .who {{ color:var(--dim); }}
.v-ok {{ border-color:#2a9d8f !important; }} .v-benign {{ border-color:#c9c0ac !important; }} .v-harm {{ border-color:#c1121f !important; }}
.two {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:3em; }}
.two p {{ margin:.3em 0 0; }}\n.vtable td {{ padding:.25em .8em; }} .vtable small {{ font-size:.85em; }}\n.versions {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:1em; }}\n.versions h3 {{ margin:0 0 .3em; font-size:1em; }}\n.versions p {{ margin:.3em 0; font-size:.85em; }}\n.versions pre {{ font-size:.62em; max-height:none; }}\n.ex {{ margin-top:.8em; padding:.5em .8em; background:#fff; border:1px solid var(--rule); border-radius:.5em; font-size:.72em; }}\n.ex .q {{ font-size:1.05em; font-weight:500; }}\nol.check small {{ font-size:.72em; line-height:1.35; display:inline-block; margin-top:.15em; }}
table.big {{ font-size:1.15em; }}
ul.tight li {{ margin:.1em 0; font-size:.9em; }}
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
<label><input type="radio" name="length" value="cut to 8"> cut to 8</label></p>
<p><b>Hook message.</b> <label><input type="radio" name="hook" value="keep"> keep</label>
<label><input type="radio" name="hook" value="pick another"> pick another</label></p>
<p><b>The journey (A4–A6).</b> <label><input type="radio" name="journey" value="appendix only"> appendix only, for questions</label>
<label><input type="radio" name="journey" value="one slide in the main deck"> one slide in the main deck</label></p>
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
