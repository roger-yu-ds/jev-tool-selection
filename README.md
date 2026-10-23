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
uv run python scripts/run_router.py --arm jev --sample fresh --limit 5   # Jev, final setup, 5 fresh messages
uv run python scripts/run_native.py --arm sonnet --sample fresh         # Sonnet as a real agent, final setup
uv run python scripts/run_router.py --arm deepseek --policy             # a development run: router mode, first 600
uv run python scripts/jev_experiments.py --split dev                    # why Jev handed off too often
uv run python scripts/optimise_prompt.py --arm deepseek            # DSPy GEPA on the router preamble
uv run python scripts/log_langfuse.py                              # one Langfuse trace per call
```

Runs resume: a script skips the request ids already in its `runs/<name>.jsonl`.

## Results

### Final setup, fresh 600 messages (24 October talk)

Every selector gets the same routing policy (`src/toolpick/prompts.py`), six tool descriptions carry one
line in customers' words (`toolpick.data.CUSTOMER_WORDS`), the LLMs run as real agents (native tool
calling, `tool_choice: auto`), and everything is scored with the audited answer key. The 600 messages
(`data/requests_fresh.jsonl`) were drawn after the setup was fixed and share no message with the 600
used during development.

| selector | correct | benign | harmful | asked a question | median time | input tokens | billed $ / 1,000 |
|---|---|---|---|---|---|---|---|
| Jev | 95.3% | 4.5% | 0.2% | 0% | 0.3 s | 7,207 | 0.30 |
| DeepSeek V4.1 Flash | 47.2% | 52.5% | 0.3% | 36% | 2.0 s | 51,609 | 0.28 |
| Claude Haiku 5.5 | 82.2% | 17.8% | 0.0% | 13% | 1.7 s | 76,221 | 0.85 |
| Claude Sonnet 5.5 | 93.3% | 6.7% | 0.0% | 4% | 1.8 s | 76,222 | 9.07 |

*Correct*: an acceptable tool, or a question back where the tool needs details the message lacks.
*Benign*: handed off where a tool existed, asked a question, or read the customer's own records.
*Harmful*: a write that wasn't asked for, or a tool in the wrong system (3 of 2,400 answers).

Paired (exact McNemar, same messages): Jev against Sonnet 95.3% vs 93.3%
(only Jev right 37, only Sonnet 25, p = 0.16: a tie);
against Haiku p = 5e-13; against DeepSeek p = 2e-68. DeepSeek asks the customer a clarifying
question on a third of messages instead of following the policy and handing off; that is benign, not harmful.

### Development runs, first 600 messages (how we got there)

| development run (first 600, audited key) | correct | harmful |
|---|---|---|
| `jev` (n=600) | 77.7% | 0.0% |
| `jev_policy` (n=600) | 85.5% | 0.0% |
| `router_deepseek` (n=600) | 73.5% | 0.3% |
| `router_deepseek_policy` (n=600) | 86.0% | 0.0% |
| `router_haiku` (n=600) | 56.8% | 7.2% |
| `router_haiku_policy` (n=600) | 83.5% | 0.7% |
| `router_sonnet` (n=300) | 70.3% | 0.3% |
| `router_sonnet_policy` (n=300) | 92.7% | 0.0% |
| `native_deepseek` (n=600) | 43.0% | 0.5% |
| `native_deepseek_policy` (n=600) | 49.7% | 0.3% |
| `native_haiku` (n=600) | 53.3% | 0.0% |
| `native_haiku_policy` (n=600) | 85.2% | 0.2% |
| `native_sonnet` (n=600) | 59.0% | 0.0% |
| `native_sonnet_policy` (n=600) | 92.2% | 0.0% |

Three things moved the results during development, all in the appendix of the deck: auditing the answer key
by hand (every failure checked; `toolpick.data.AUDIT`), writing the routing policy into every prompt, and
adding customer wording to six tool descriptions after finding that Jev over-used the hand-off because real
descriptions are written for developers (`scripts/jev_experiments.py`, results in `runs/experiments/`).
DSPy GEPA (`scripts/optimise_prompt.py`) rewrote DeepSeek's router prompt from 200 messages and scored
84.2% on the 399 it never saw (first-pass key). All runs together: about US$28 of billed calls.

## Layout

| Path | What |
|---|---|
| `data/requests_fresh.jsonl` | the fresh 600 messages used for the final results (`scripts/make_fresh_sample.py`) |
| `data/requests.jsonl` | the first 600 messages, used during development |
| `data/catalog.json`, `data/REPORT.md` | the 90 real tools and how each was retrieved |
| `data/intent_map.json` | per intent: acceptable tools, partial tools, notes (the research pass) |
| `src/toolpick/data.py` | loading, the customer wording, and the answer key (first pass and audited, every change with its reason) |
| `src/toolpick/scoring.py` | outcomes, summaries, McNemar, bootstrap, the confidence gate |
| `src/toolpick/prompts.py` | every prompt: router, native, Jev, and the routing policy |
| `runs/fresh/*.jsonl` | the final runs: one row per call with pick, usage, cost, latency, timestamp and full API response |
| `runs/*.jsonl`, `runs/experiments/` | the development runs and the Jev experiments |
| `notebooks/explore.py` | marimo explorer |
| `notebooks/tool_selection.ipynb` | the walkthrough |
| `slides/` | the talk |

## Licences and provenance

Messages: Bitext customer-support dataset (CDLA-Sharing-1.0). Tool text: Stripe (`@stripe/mcp`
0.2.5, MIT), GitHub (`github-mcp-server` 2.0.3, MIT), Notion (`@notionhq/notion-mcp-server` 2.5.2,
MIT); Shopify (live `tools/list` on a public store) and Intercom (docs summaries) are the vendors'
own text, reproduced here for research. See `data/REPORT.md`.
