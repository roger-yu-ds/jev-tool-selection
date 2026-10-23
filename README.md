# Who picks the tool? Measuring tool selection on real MCP servers

Code, data and results behind a MeasureCamp Sydney talk (24 October 2026). 600 customer-support
messages, 90 tools copied word for word from seven official MCP servers, and four ways of choosing
the tool: TypeSafe's Jev, and DeepSeek V4.1 Flash, Claude Haiku 5.5 and Claude Sonnet 5.5 as a
router (told to name one tool) and with native tool calling (the tools in the API's `tools`
parameter). Every run is scored against per-intent sets of acceptable tools, and every number in
the talk can be recomputed from `runs/`.

## Run it

```bash
uv sync                                   # Python 3.12, pandas, altair, marimo, jupyter (+ dspy, langfuse extras)
uv run marimo edit notebooks/explore.py   # the interactive explorer: dataset browser, error analysis, prompt toggles
uv run jupyter lab notebooks/             # the walkthrough notebook
```

Nothing above needs an API key: the runs are committed. To make new calls, put keys in `.env`:

```
OPENROUTER_API_KEY=...    # DeepSeek and Claude, through OpenRouter
TYPESAFE_API_KEY=...      # Jev
LANGFUSE_PUBLIC_KEY=...   # optional: scripts/log_langfuse.py logs every call as a trace
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://127.0.0.1:3010
```

```bash
uv run python scripts/run_router.py --arm jev --limit 5            # Jev on the first 5 messages
uv run python scripts/run_router.py --arm deepseek --policy        # the LLM as a router, policy prompt
uv run python scripts/run_native.py --arm sonnet                   # native tool calling
uv run python scripts/optimise_prompt.py --arm deepseek            # DSPy GEPA on the router preamble
uv run python scripts/log_langfuse.py                              # one Langfuse trace per call
```

Runs resume: a script skips the request ids already in its `runs/<name>.jsonl`.

## Results (22 October 2026)

Strict scoring (`right` = an acceptable tool for the intent; on the 15 intents no real tool covers,
only `hand_off_to_human`). `v2` also accepts the store-FAQ search on a how-to question, a hand-off on
`payment_issue` and the refund-policy FAQ on `get_refund`. *Wrong calls* = a tool that can't do the
job. *Asked* = the model replied in text instead of calling a tool (native runs). Sonnet as a router
ran on every other message.

| run | n | right (v1) | right (v2) | wrong calls | asked | handed off | median latency | input tokens | billed $/1,000 |
|---|---|---|---|---|---|---|---|---|---|
| `jev` | 600 | 73.8% | 79.2% | 2.3% | 0% | 74% | 0.29 s | 6,921 | 0.29 |
| `jev_policy` | 600 | 82.2% | 88.3% | 1.0% | 0% | 68% | 0.30 s | 7,061 | 0.30 |
| `router_deepseek` | 600 | 66.3% | 80.7% | 16.0% | 0% | 55% | 0.80 s | 43,831 | 0.36 |
| `router_deepseek_policy` | 600 | 78.3% | 85.2% | 4.8% | 0% | 64% | 0.78 s | 43,975 | 0.37 |
| `router_deepseek_gepa` (held-out 399) | 399 | 84.2% | 91.5% | 12.0% | 0% | 44% | 1.07 s | 44,718 | 0.22 |
| `router_haiku` | 600 | 47.0% | 61.2% | 43.5% | 0% | 20% | 1.23 s | 69,857 | 0.74 |
| `router_haiku_policy` | 600 | 73.3% | 78.2% | 15.0% | 0% | 50% | 1.20 s | 70,068 | 0.78 |
| `router_sonnet` | 300 | 62.7% | 87.0% | 24.0% | 1% | 41% | 1.60 s | 69,858 | 8.10 |
| `router_sonnet_policy` | 300 | 85.0% | 93.0% | 6.0% | 0% | 54% | 1.54 s | 70,069 | 8.04 |
| `native_deepseek` | 600 | 28.7% | 42.8% | 28.5% | 40% | 3% | 2.64 s | 51,321 | 0.29 |
| `native_deepseek_policy` | 600 | 33.7% | 44.8% | 22.0% | 42% | 8% | 3.62 s | 51,465 | 0.29 |
| `native_haiku` | 600 | 37.2% | 48.0% | 21.5% | 38% | 12% | 2.40 s | 75,803 | 0.83 |
| `native_haiku_policy` | 600 | 70.0% | 74.8% | 6.0% | 17% | 48% | 2.81 s | 76,014 | 0.84 |
| `native_sonnet` | 600 | 43.5% | 63.0% | 27.3% | 24% | 17% | 3.00 s | 75,804 | 8.80 |
| `native_sonnet_policy` | 600 | 77.7% | 81.8% | 6.8% | 10% | 51% | 3.21 s | 76,015 | 9.07 |

Paired tests (exact McNemar on the same messages): Jev beats every LLM on the plain prompt
(p ≤ 0.003); with the policy prompt Sonnet-as-router edges Jev (85.0% vs 82.2% on 300, p = 0.049) and
Jev beats the rest. The GEPA-found DeepSeek preamble (`scripts/optimise_prompt.py`, 200-message dev
split, scored once on the other 399) beats the hand-written policy by 7.5 points (p = 0.004) and is
level with Jev (p = 0.47), with 15x Jev's wrong-call rate. `scripts/check_runs.py` lists the sanity
checks run on every file. All runs together cost about US$19 in billed LLM calls and US$0.35 of Jev.

## Layout

| Path | What |
|---|---|
| `data/requests.jsonl` | the 600 messages (id, Bitext intent, text) |
| `data/catalog.json`, `data/REPORT.md` | the 90 real tools and how each was retrieved |
| `data/intent_map.json` | per intent: acceptable tools, partial tools, notes (the research pass) |
| `src/toolpick/data.py` | loading, the label overrides and the revised (v2) labels, with reasons |
| `src/toolpick/scoring.py` | outcomes, summaries, McNemar, bootstrap, the confidence gate |
| `src/toolpick/prompts.py` | every prompt: router, native, Jev, and the routing policy |
| `runs/*.jsonl` | one row per call: pick, usage, cost, latency, timestamp, full API response |
| `notebooks/explore.py` | marimo explorer |
| `notebooks/tool_selection.ipynb` | the walkthrough |
| `slides/` | the talk |

## Licences and provenance

Messages: Bitext customer-support dataset (CDLA-Sharing-1.0). Tool text: Stripe (`@stripe/mcp`
0.2.5, MIT), GitHub (`github-mcp-server` 2.0.3, MIT), Notion (`@notionhq/notion-mcp-server` 2.5.2,
MIT); Shopify (live `tools/list` on a public store) and Intercom (docs summaries) are the vendors'
own text, reproduced here for research. See `data/REPORT.md`.
