"""Build ledger/index.html: every decision in every run, filterable, in one self-contained page.

    uv run python scripts/build_ledger.py

Two views. *Decisions*: one row per (run, message) with the pick, the verdict under either answer
key, Jev's confidence, latency, tokens and cost; click a row for the full reply, tool-call
arguments, Jev's top options and the answer key for that intent. *Messages*: one row per message,
one column per run, so you can see where the selectors disagree. Filters apply to both; the summary
above the table recomputes from whatever is filtered, and the filtered rows export as CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toolpick as tp
from toolpick.data import all_requests_by_id
from toolpick.scoring import FILLS

OUT = ROOT / "ledger" / "index.html"
SAMPLE_OF = {"fresh": "fresh 600 (final)", "first": "first 600 (development)", "experiments": "Jev experiments"}


def run_meta(name: str) -> dict:
    if name.startswith("experiments/"):
        stem = name.split("/", 1)[1]
        model = "jev" if stem.startswith("jev_") else stem.split("_")[0]
        return {"sample": "experiments", "model": model, "mode": "jev" if model == "jev" else "router", "prompt": stem}
    m = tp.parse_run_name(name)
    return {"sample": m["sample"], "model": m["model"], "mode": m["mode"], "prompt": m["prompt"]}


def main() -> None:
    reqs = all_requests_by_id()
    lab_a, lab_f = tp.labels("audited"), tp.labels("first_pass")
    runs = tp.available_runs("fresh") + tp.available_runs() + sorted(
        f"experiments/{p.stem}" for p in (tp.RUNS / "experiments").glob("*.jsonl"))
    run_list, rows, used = [], [], set()
    for ri, name in enumerate(runs):
        meta = run_meta(name)
        run_list.append({"name": name, **meta})
        fills = tp.load_run(FILLS[name]) if name in FILLS else {}
        for rid, r in tp.load_run(name).items():
            req = reqs[rid]
            if rid in fills:  # Jev's argument-filling call: add its time, tokens and cost, show its arguments
                f = fills[rid]
                r = {**r, "latency_s": r["latency_s"] + f["latency_s"], "input_tokens": (r.get("input_tokens") or 0) + f["input_tokens"],
                     "output_tokens": (r.get("output_tokens") or 0) + f["output_tokens"], "cost_usd": r["cost_usd"] + f["cost_usd"],
                     "tool_calls": f["tool_calls"],
                     "text": f"(includes the argument-filling call to DeepSeek: {f['latency_s']:.2f} s, ${f['cost_usd']:.6f})"}  # fmt: skip
            used.add(rid)
            calls = [c["name"] for c in r.get("tool_calls") or []]
            kind = r.get("kind", "tool_call")
            oa = tp.outcome(r["pick"], lab_a[req.intent], kind, calls)
            of = tp.outcome(r["pick"], lab_f[req.intent], kind, calls)
            top = r.get("top") or (sorted(r["probabilities"].items(), key=lambda kv: -kv[1])[:5] if r.get("probabilities") else None)
            rows.append([ri, rid, r["pick"], oa, of, r.get("confidence"), r["latency_s"], r.get("input_tokens") or 0,
                         r.get("cached_tokens") or 0, r.get("output_tokens") or 0, round(r.get("cost_usd") or 0, 6),
                         (r.get("raw") or r.get("text") or "")[:400], [[c["name"], c["arguments"]] for c in r.get("tool_calls") or []] or None,
                         [[k, round(v, 3)] for k, v in top[:5]] if top else None, r.get("provider") or r.get("model")])  # fmt: skip
    msgs = {rid: [reqs[rid].intent, reqs[rid].text, "fresh" if rid.startswith("F") else "first"] for rid in sorted(used)}
    labels = {k: {"gap": v["gap"], "acc": v["acceptable"], "part": v["partial"], "ask": v["ask_ok"], "why": v["changes"],
                  "acc_first": lab_f[k]["acceptable"], "note": v["note"]} for k, v in lab_a.items()}  # fmt: skip
    data = {"runs": run_list, "rows": rows, "msgs": msgs, "labels": labels, "samples": SAMPLE_OF,
            "names": tp.MODEL_NAMES}  # fmt: skip
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", blob))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(rows):,} decisions, {len(run_list)} runs, {len(msgs):,} messages")


PAGE = r"""<!doctype html>
<html lang="en" data-lavish-annotate="on">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tool-selection ledger</title>
<style>
:root { --bg:#fbfaf7; --card:#fff; --fg:#15161a; --dim:#6b6f78; --rule:#e6e2d8; --accent:#1d4ed8;
        --ok:#2a9d8f; --benign:#c9c0ac; --harm:#c1121f; --okbg:#e3f2ef; --benignbg:#f1ede4; --harmbg:#f9e1e2; }
* { box-sizing:border-box; }
:where(.grid, .flex) > * { min-width:0; }
html, body { margin:0; background:var(--bg); color:var(--fg); font-family:"Inter","Helvetica Neue",Arial,sans-serif; font-size:14px; }
header { padding:22px 28px 10px; }
h1 { margin:0; font-size:26px; letter-spacing:-.01em; }
.sub { color:var(--dim); margin:4px 0 0; max-width:70em; line-height:1.45; }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--rule); padding:10px 28px; display:flex; flex-wrap:wrap; gap:10px 14px; align-items:flex-end; }
.bar label { display:flex; flex-direction:column; gap:3px; font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.06em; }
.bar select, .bar input { font:inherit; font-size:13px; padding:5px 7px; border:1px solid var(--rule); border-radius:6px; background:#fff; color:var(--fg); min-width:0; }
.bar select[multiple] { height:84px; min-width:180px; }
.bar input[type=search] { width:220px; }
.bar .chk { flex-direction:row; align-items:center; gap:6px; text-transform:none; letter-spacing:0; font-size:13px; color:var(--fg); }
button { font:inherit; font-size:13px; padding:6px 11px; border:1px solid var(--rule); border-radius:6px; background:#fff; cursor:pointer; }
button.primary { background:var(--fg); color:#fff; border-color:var(--fg); }
.tabs { display:flex; gap:6px; }
.tabs button[aria-selected=true] { background:var(--fg); color:#fff; border-color:var(--fg); }
main { padding:14px 28px 60px; }
.summary { display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap:10px; margin-bottom:14px; }
.run { background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:9px 11px; }
.run .nm { font-weight:600; font-size:13px; overflow-wrap:anywhere; }
.run .meta { color:var(--dim); font-size:11.5px; margin:2px 0 6px; }
.stack { display:flex; height:10px; border-radius:3px; overflow:hidden; background:var(--rule); }
.stack span { display:block; height:100%; }
.run .nums { display:flex; gap:10px; font-size:12px; margin-top:5px; flex-wrap:wrap; }
.count { color:var(--dim); margin:4px 0 8px; font-size:13px; }
.tablewrap { overflow-x:auto; background:var(--card); border:1px solid var(--rule); border-radius:8px; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:6px 9px; border-bottom:1px solid var(--rule); vertical-align:top; font-size:13px; }
th { position:sticky; top:0; background:#f6f3ec; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); font-weight:600; cursor:pointer; white-space:nowrap; }
td.msg { max-width:30em; overflow-wrap:anywhere; }
td code, .detail code { font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:12px; background:#f1eee6; padding:1px 4px; border-radius:4px; overflow-wrap:anywhere; }
tr.rowsel { cursor:pointer; } tr.rowsel:hover td { background:#faf8f2; }
.pill { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11.5px; font-weight:600; white-space:nowrap; }
.v-correct { background:var(--okbg); color:#1c6b61; } .v-benign { background:var(--benignbg); color:#6a6150; } .v-harmful { background:var(--harmbg); color:#9a0e18; }
td.cell { font-size:12px; white-space:nowrap; } td.cell.correct { background:var(--okbg); } td.cell.benign { background:var(--benignbg); } td.cell.harmful { background:var(--harmbg); }
.pager { display:flex; gap:8px; align-items:center; margin:10px 0; }
.detail { position:fixed; right:0; top:0; bottom:0; width:min(560px, 100vw); background:#fff; border-left:1px solid var(--rule); box-shadow:-8px 0 24px rgba(0,0,0,.06); padding:20px 22px; overflow:auto; z-index:10; display:none; }
.detail.open { display:block; }
.detail h3 { margin:0 0 6px; font-size:16px; }
.detail blockquote { margin:8px 0; font-size:16px; line-height:1.35; }
.detail pre { white-space:pre-wrap; background:#f8f6f0; border:1px solid var(--rule); border-radius:6px; padding:8px; font-size:12px; overflow-wrap:anywhere; }
.detail dl { display:grid; grid-template-columns: 9em 1fr; gap:4px 10px; margin:10px 0; font-size:13px; }
.detail dt { color:var(--dim); }
.detail dd { margin:0; overflow-wrap:anywhere; }
.legend { display:flex; gap:12px; font-size:12px; color:var(--dim); align-items:center; flex-wrap:wrap; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
@media (max-width: 700px) { header, .bar, main { padding-left:16px; padding-right:16px; } .bar input[type=search] { width:100%; } }
</style></head>
<body>
<header>
  <h1>Tool-selection ledger</h1>
  <p class="sub">Every decision from every run: which tool each selector picked for each customer message, and how it scored. Filter by sample, run, model, intent, verdict or text; the cards above the table recompute from the filtered rows. Click a decision for the full reply, tool-call arguments, Jev's top options and the answer key. Select text or click any row to annotate it in Lavish.</p>
</header>
<div class="bar" id="bar">
  <div class="tabs" role="tablist">
    <button role="tab" aria-selected="true" data-view="decisions">Decisions</button>
    <button role="tab" aria-selected="false" data-view="messages">Messages × runs</button>
  </div>
  <label>Sample<select id="f-sample"></select></label>
  <label>Runs (ctrl-click)<select id="f-run" multiple></select></label>
  <label>Answer key<select id="f-key"><option value="a">audited (final)</option><option value="f">first pass</option></select></label>
  <label>Intent<select id="f-intent"><option value="">all</option></select></label>
  <label>Request type<select id="f-gap"><option value="">all</option><option value="cov">a real tool exists</option><option value="gap">no real tool</option></select></label>
  <label>Verdict<select id="f-verdict"><option value="">all</option><option>correct</option><option>benign</option><option>harmful</option><option value="incorrect">incorrect (benign + harmful)</option></select></label>
  <label>Outcome<select id="f-outcome"><option value="">all</option><option>right</option><option>defensible</option><option>handed_off</option><option>asked</option><option>wrong</option><option>bluffed</option></select></label>
  <label>Pick contains<input id="f-pick" type="search" placeholder="e.g. hand_off, invoices"></label>
  <label>Message contains<input id="f-text" type="search" placeholder="search the customer's words"></label>
  <label>Jev confidence ≤<input id="f-conf" type="number" min="0" max="1" step="0.05" placeholder="any" style="width:80px"></label>
  <label class="chk"><input id="f-dis" type="checkbox"> only messages where the selected runs disagree</label>
  <button id="reset">Reset</button>
  <button id="csv" class="primary">Export CSV</button>
</div>
<main>
  <div class="legend"><span><span class="sw" style="background:var(--ok)"></span>correct</span><span><span class="sw" style="background:var(--benign)"></span>benign (handed off, asked, read-only first step)</span><span><span class="sw" style="background:var(--harm)"></span>harmful (wrong write, or wrong system)</span></div>
  <div class="summary" id="summary"></div>
  <div class="count" id="count"></div>
  <div class="pager" id="pager-top"></div>
  <div class="tablewrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>
  <div class="pager" id="pager"></div>
</main>
<aside class="detail" id="detail" aria-live="polite"></aside>
<script>
const D = __DATA__;
const VERDICT = {right:"correct", defensible:"benign", handed_off:"benign", asked:"benign", wrong:"harmful", bluffed:"harmful"};
const PAGE = 100;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const short = p => p == null ? "(asks a question)" : p.split(".").pop();
const runLabel = r => {
  const nm = D.names[r.model] || r.model;
  if (r.sample === "experiments") return `${nm} · ${r.prompt}`;
  if (r.model === "jev") return `Jev · ${r.prompt}`;
  return `${nm} · ${r.mode} · ${r.prompt}`;
};
// rows as objects
const R = D.rows.map(a => ({run:a[0], id:a[1], pick:a[2], oa:a[3], of:a[4], conf:a[5], lat:a[6], tin:a[7], tcache:a[8], tout:a[9], cost:a[10], reply:a[11], calls:a[12], top:a[13], prov:a[14]}));
let state = {view:"decisions", page:0, sort:null, dir:1};

function fill() {
  const s = $("f-sample");
  for (const [k, v] of Object.entries(D.samples)) s.add(new Option(v, k));
  const intents = Object.keys(D.labels).sort();
  for (const k of intents) $("f-intent").add(new Option(k + (D.labels[k].gap ? " (no tool)" : ""), k));
  fillRuns();
}
function fillRuns() {
  const sel = $("f-run"); sel.innerHTML = "";
  D.runs.forEach((r, i) => { if (r.sample === $("f-sample").value) { const o = new Option(runLabel(r), i); o.selected = true; sel.add(o); } });
}
function filters() {
  const runs = new Set([...$("f-run").selectedOptions].map(o => +o.value));
  return {runs, key: $("f-key").value, intent: $("f-intent").value, gap: $("f-gap").value, verdict: $("f-verdict").value,
          outcome: $("f-outcome").value, pick: $("f-pick").value.trim().toLowerCase(), text: $("f-text").value.trim().toLowerCase(),
          conf: $("f-conf").value === "" ? null : +$("f-conf").value, dis: $("f-dis").checked};
}
function outcomeOf(r, key) { return key === "a" ? r.oa : r.of; }
function filtered() {
  const f = filters();
  let rows = R.filter(r => f.runs.has(r.run));
  if (f.dis) {
    const picks = new Map();
    for (const r of rows) { const s = picks.get(r.id) || new Set(); s.add(r.pick ?? "(asked)"); picks.set(r.id, s); }
    rows = rows.filter(r => picks.get(r.id).size > 1);
  }
  return rows.filter(r => {
    const m = D.msgs[r.id], lab = D.labels[m[0]], o = outcomeOf(r, f.key), v = VERDICT[o];
    if (f.intent && m[0] !== f.intent) return false;
    if (f.gap === "gap" && !lab.gap) return false;
    if (f.gap === "cov" && lab.gap) return false;
    if (f.verdict === "incorrect" ? v === "correct" : (f.verdict && v !== f.verdict)) return false;
    if (f.outcome && o !== f.outcome) return false;
    if (f.pick && !(r.pick || "(asks a question)").toLowerCase().includes(f.pick)) return false;
    if (f.text && !m[1].toLowerCase().includes(f.text)) return false;
    if (f.conf !== null && (r.conf == null || r.conf > f.conf)) return false;
    return true;
  });
}
function pct(x) { return (100 * x).toFixed(1) + "%"; }
function summary(rows, key) {
  const by = new Map();
  for (const r of rows) { const a = by.get(r.run) || []; a.push(r); by.set(r.run, a); }
  $("summary").innerHTML = [...by.entries()].sort((a, b) => a[0] - b[0]).map(([ri, rs]) => {
    const n = rs.length, c = {correct:0, benign:0, harmful:0};
    let cost = 0, lat = [];
    for (const r of rs) { c[VERDICT[outcomeOf(r, key)]]++; cost += r.cost; lat.push(r.lat); }
    lat.sort((x, y) => x - y);
    const run = D.runs[ri];
    return `<div class="run" data-run="${esc(run.name)}"><div class="nm">${esc(runLabel(run))}</div>
      <div class="meta">${esc(run.name)} · ${n} decisions · median ${lat[Math.floor(lat.length / 2)].toFixed(2)} s · $${(1000 * cost / n).toFixed(2)} / 1,000</div>
      <div class="stack"><span style="width:${100*c.correct/n}%;background:var(--ok)"></span><span style="width:${100*c.benign/n}%;background:var(--benign)"></span><span style="width:${100*c.harmful/n}%;background:var(--harm)"></span></div>
      <div class="nums"><span><b>${pct(c.correct/n)}</b> correct</span><span>${pct(c.benign/n)} benign</span><span>${pct(c.harmful/n)} harmful</span></div></div>`;
  }).join("") || "<p class='count'>No runs selected.</p>";
}
function pager(total) {
  const pages = Math.max(1, Math.ceil(total / PAGE));
  state.page = Math.min(state.page, pages - 1);
  const html = `<button ${state.page === 0 ? "disabled" : ""} data-p="-1">← previous</button><span>page ${state.page + 1} of ${pages}</span><button ${state.page >= pages - 1 ? "disabled" : ""} data-p="1">next →</button>`;
  $("pager").innerHTML = html; $("pager-top").innerHTML = html;
}
const COLS = [
  ["id", r => r.id], ["run", r => runLabel(D.runs[r.run])], ["intent", r => D.msgs[r.id][0]], ["message", r => D.msgs[r.id][1]],
  ["pick", r => short(r.pick)], ["verdict", r => VERDICT[outcomeOf(r, $("f-key").value)]], ["outcome", r => outcomeOf(r, $("f-key").value)],
  ["Jev conf", r => r.conf], ["latency s", r => r.lat], ["input tok", r => r.tin], ["output tok", r => r.tout], ["cost $", r => r.cost],
];
function renderDecisions(rows) {
  if (state.sort !== null) { const get = COLS[state.sort][1]; rows = [...rows].sort((a, b) => { const x = get(a), y = get(b); return (x > y ? 1 : x < y ? -1 : 0) * state.dir; }); }
  pager(rows.length);
  const key = $("f-key").value;
  $("tbl").tHead.innerHTML = "<tr>" + COLS.map((c, i) => `<th data-sort="${i}">${c[0]}${state.sort === i ? (state.dir > 0 ? " ↑" : " ↓") : ""}</th>`).join("") + "</tr>";
  const page = rows.slice(state.page * PAGE, (state.page + 1) * PAGE);
  $("tbl").tBodies[0].innerHTML = page.map(r => {
    const m = D.msgs[r.id], o = outcomeOf(r, key), v = VERDICT[o], idx = R.indexOf(r);
    return `<tr class="rowsel" data-i="${idx}" id="d-${esc(D.runs[r.run].name)}-${r.id}"><td><code>${r.id}</code></td><td>${esc(runLabel(D.runs[r.run]))}</td><td>${esc(m[0])}</td><td class="msg">${esc(m[1])}</td>
      <td><code>${esc(short(r.pick))}</code></td><td><span class="pill v-${v}">${v}</span></td><td>${o}</td><td>${r.conf == null ? "" : r.conf.toFixed(2)}</td>
      <td>${r.lat.toFixed(2)}</td><td>${r.tin.toLocaleString()}</td><td>${r.tout}</td><td>${r.cost.toFixed(5)}</td></tr>`;
  }).join("");
  $("count").textContent = `${rows.length.toLocaleString()} decisions match.`;
}
function renderMessages(rows) {
  const key = $("f-key").value;
  const runs = [...new Set(rows.map(r => r.run))].sort((a, b) => a - b);
  const by = new Map();
  for (const r of rows) { const m = by.get(r.id) || {}; m[r.run] = r; by.set(r.id, m); }
  const ids = [...by.keys()].sort();
  pager(ids.length);
  $("tbl").tHead.innerHTML = "<tr><th>id</th><th>intent</th><th>message</th>" + runs.map(ri => `<th title="${esc(D.runs[ri].name)}">${esc(runLabel(D.runs[ri]))}</th>`).join("") + "</tr>";
  $("tbl").tBodies[0].innerHTML = ids.slice(state.page * PAGE, (state.page + 1) * PAGE).map(id => {
    const m = D.msgs[id], cells = by.get(id);
    return `<tr id="m-${id}"><td><code>${id}</code></td><td>${esc(m[0])}</td><td class="msg">${esc(m[1])}</td>` + runs.map(ri => {
      const r = cells[ri]; if (!r) return "<td class='cell'></td>";
      const v = VERDICT[outcomeOf(r, key)];
      return `<td class="cell ${v} rowsel" data-i="${R.indexOf(r)}" title="${v}">${esc(short(r.pick))}</td>`;
    }).join("") + "</tr>";
  }).join("");
  $("count").textContent = `${ids.length.toLocaleString()} messages match (${rows.length.toLocaleString()} decisions). Click a cell for details.`;
}
function render() {
  const rows = filtered();
  summary(rows, $("f-key").value);
  (state.view === "decisions" ? renderDecisions : renderMessages)(rows);
}
function detail(i) {
  const r = R[i], m = D.msgs[r.id], lab = D.labels[m[0]], run = D.runs[r.run];
  const v = VERDICT[r.oa], vf = VERDICT[r.of];
  const tools = a => a.length ? a.map(t => `<code>${esc(t.split(".").pop())}</code>`).join(" ") : "–";
  $("detail").innerHTML = `<button id="close" style="float:right">Close</button>
    <h3><code>${r.id}</code> · ${esc(runLabel(run))}</h3>
    <blockquote>“${esc(m[1])}”</blockquote>
    <dl><dt>intent</dt><dd>${esc(m[0])} ${lab.gap ? "(no real tool)" : ""}</dd>
    <dt>pick</dt><dd><code>${esc(r.pick ?? "(asks a question)")}</code></dd>
    <dt>audited key</dt><dd><span class="pill v-${v}">${v}</span> ${r.oa}</dd>
    <dt>first-pass key</dt><dd><span class="pill v-${vf}">${vf}</span> ${r.of}</dd>
    ${r.conf != null ? `<dt>Jev confidence</dt><dd>${r.conf.toFixed(2)}</dd>` : ""}
    <dt>latency</dt><dd>${r.lat.toFixed(2)} s</dd>
    <dt>tokens</dt><dd>${r.tin.toLocaleString()} in (${r.tcache.toLocaleString()} cached), ${r.tout} out</dd>
    <dt>cost</dt><dd>$${r.cost.toFixed(5)}</dd><dt>served by</dt><dd>${esc(r.prov || "")}</dd><dt>run file</dt><dd><code>runs/${esc(run.name)}.jsonl</code></dd></dl>
    ${r.top ? `<h3>Jev's top options</h3><pre>${r.top.map(([k, p]) => p.toFixed(3) + "  " + k).join("\n")}</pre>` : ""}
    ${r.calls ? `<h3>Tool calls</h3><pre>${esc(JSON.stringify(r.calls, null, 1))}</pre>` : ""}
    ${r.reply ? `<h3>Reply text</h3><pre>${esc(r.reply)}</pre>` : ""}
    <h3>Answer key for ${esc(m[0])}</h3>
    <dl><dt>correct</dt><dd>${tools(lab.acc)}${lab.ask ? " · a question back" : ""}</dd><dt>benign first step</dt><dd>${tools(lab.part)}</dd>
    <dt>first pass</dt><dd>${tools(lab.acc_first)}</dd><dt>why</dt><dd>${esc(lab.why.join(" ")) || "–"}</dd><dt>research note</dt><dd>${esc(lab.note)}</dd></dl>
    <p><button id="flag" class="primary">Queue: this label looks wrong</button></p>`;
  $("detail").classList.add("open");
  $("close").onclick = () => $("detail").classList.remove("open");
  $("flag").onclick = () => {
    const text = `Check the answer key for ${r.id} (${m[0]}): "${m[1]}" — ${runLabel(run)} picked ${r.pick ?? "(asks a question)"}, scored ${r.oa}.`;
    if (window.lavish?.queuePrompt) window.lavish.queuePrompt(text, {tag: "label-check", text: `Label check ${r.id}`, queueKey: `label-${r.id}-${run.name}`, data: {id: r.id, run: run.name, intent: m[0], pick: r.pick, outcome: r.oa}});
    $("flag").textContent = "Queued: press Send in the Lavish bar";
  };
}
function csv() {
  const key = $("f-key").value;
  const head = ["run", "id", "intent", "gap", "message", "pick", "verdict", "outcome", "outcome_first_pass", "jev_confidence", "latency_s", "input_tokens", "cached_tokens", "output_tokens", "cost_usd", "reply"];
  const q = s => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const lines = [head.join(",")].concat(filtered().map(r => {
    const m = D.msgs[r.id];
    return [D.runs[r.run].name, r.id, m[0], D.labels[m[0]].gap, m[1], r.pick ?? "", VERDICT[outcomeOf(r, key)], r.oa, r.of, r.conf ?? "", r.lat, r.tin, r.tcache, r.tout, r.cost, r.reply].map(q).join(",");
  }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], {type: "text/csv"}));
  a.download = "tool-selection-ledger.csv"; a.click();
}
fill();
document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => {
  state.view = b.dataset.view; state.page = 0;
  document.querySelectorAll("[data-view]").forEach(x => x.setAttribute("aria-selected", x === b));
  render();
});
$("f-sample").onchange = () => { fillRuns(); state.page = 0; render(); };
["f-run", "f-key", "f-intent", "f-gap", "f-verdict", "f-outcome", "f-dis"].forEach(id => $(id).onchange = () => { state.page = 0; render(); });
["f-pick", "f-text", "f-conf"].forEach(id => $(id).oninput = () => { state.page = 0; render(); });
$("reset").onclick = () => { document.querySelectorAll(".bar input[type=search], .bar input[type=number]").forEach(x => x.value = ""); ["f-intent", "f-gap", "f-verdict", "f-outcome"].forEach(id => $(id).value = ""); $("f-dis").checked = false; fillRuns(); state.page = 0; render(); };
$("csv").onclick = csv;
document.addEventListener("click", ev => {
  const p = ev.target.closest("[data-p]"); if (p) { state.page += +p.dataset.p; render(); return; }
  const th = ev.target.closest("th[data-sort]"); if (th) { const i = +th.dataset.sort; state.dir = state.sort === i ? -state.dir : 1; state.sort = i; render(); return; }
  const tr = ev.target.closest(".rowsel"); if (tr && !ev.target.closest("#detail")) detail(+tr.dataset.i);
});
render();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
