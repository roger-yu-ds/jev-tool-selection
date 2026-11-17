# Results in detail

Every table here is recomputed from `runs/` by the code in `src/toolpick/`. The walkthrough with charts and
commentary is [`notebooks/tool_selection.ipynb`](../notebooks/tool_selection.ipynb); this page is the reference:
the full tables, the variants that didn't make the headline, and the commands that produced each one.

Contents: [final setup](#final-setup-fresh-600-messages) (headline, CLM-8B, arguments, the "need more
information" option) · [development runs](#development-runs-first-600-messages-how-the-setup-got-there) ·
[commands](#commands)

## Final setup, fresh 600 messages

Every selector gets the same routing policy (`src/toolpick/prompts.py`), six tool descriptions carry one
line in customers' words (`toolpick.data.CUSTOMER_WORDS`), the LLMs run as real agents (native tool
calling, `tool_choice: auto`), and everything is scored with the audited answer key. The 600 messages
(`data/requests_fresh.jsonl`) were drawn after the setup was fixed and share no message with the 600
used during development.

| selector | correct | benign | harmful | asked a question | median time | input tokens | billed $ / 1,000 |
|---|---|---|---|---|---|---|---|
| Jev (pick + fill) | 95.3% | 4.5% | 0.2% | 0% | 1.4 s | 7,736 | 0.34 |
| DeepSeek V4.1 Flash | 47.2% | 52.5% | 0.3% | 36% | 2.0 s | 51,609 | 0.28 |
| Claude Haiku 5.5 | 82.2% | 17.8% | 0.0% | 13% | 1.7 s | 76,221 | 0.85 |
| Claude Sonnet 5.5 | 93.3% | 6.7% | 0.0% | 4% | 1.8 s | 76,222 | 9.07 |
| CLM-8B, zero-shot (local, Q8_0) | 35.3% | 47.0% | 17.7% | 0% | 1.1 s (pick only) | 182 | 0 (local) |

*Correct*: an acceptable tool, or a question back where the tool needs details the message lacks.
*Benign*: handed off where a tool existed, asked a question, or read the customer's own records.
*Harmful*: a write that wasn't asked for, or a tool in the wrong system (3 of 2,400 answers).
*Jev (pick + fill)*: Jev only picks the tool, so a second call to DeepSeek V4.1 Flash fills in the chosen
tool's arguments (`scripts/run_jev_fill.py`, `runs/fill/jev_fresh.jsonl`), and Jev's time, tokens and cost
include it. On its own Jev's pick takes 0.26 s (median), reads 7,207 tokens and costs $0.30 per 1,000; the
filling call adds 1.1 s, about 600 tokens and $0.035.

*CLM-8B* (Stanford/NVIDIA "Contrastive Language Model", 2026-09-23, Apache-2.0; `contrastive-lm` 0.1.0 with
the released `CLM-v0.1-8B` head) answers Jev's exact final-setup question through its System-One-compatible
API, served locally: Qwen3-8B Q8_0 in llama.cpp with last-token pooling (16 of 36 layers on the GTX 1080 Ti),
validated against HF transformers (0.6B parity cosine ≥ 0.999) and against the README's reference playground
request. Setup, validation and caveats: `clm/NOTES.md`; runner `scripts/run_clm.py`; scorer `clm/score_clm.py`.
Zero-shot it lands below always-hand-off (59.2%): it uses 10 of the 91 tools, never the FAQ tool, and its
harmful picks are mostly `request_return` writes; picks are near-ties (median top-two margin 0.47 logits).
Against Jev: only CLM right 5, only Jev right 365 (p ≈ 5e-101); first 600: 42.2% correct. The README's best
numbers come from fine-tuned heads, so this measures the released head on a new task, not the architecture.

Paired (exact McNemar, same messages): Jev against Sonnet 95.3% vs 93.3%
(only Jev right 37, only Sonnet 25, p = 0.16: a tie);
against Haiku p = 5e-13; against DeepSeek p = 2e-68. DeepSeek asks the customer a clarifying
question on a third of messages instead of following the policy and handing off; that is benign, not harmful.

**Arguments** (`scripts/check_arguments.py`, rules in `src/toolpick/arguments.py`; free, no API calls).
There is no answer key for argument values, so the check is by rule, on every call to a correct tool
other than a hand-off: the arguments match the tool's schema; every order number, ID or email comes from
the message; an order number in the message is passed on; Stripe refund amounts are in cents. Free text
(queries, hand-off reasons) isn't scored. The tools' parameters are not only IDs: free-text queries, enums
(`policy_types`, return reasons), integers (amounts, limits) and nested objects (carts, checkouts, UCP's
`meta`, which the harness supplies).

| selector | right tool | arguments pass | right tool and arguments |
|---|---|---|---|
| Jev (arguments from the filling call) | 95.3% | 219 of 232 | 93.2% |
| DeepSeek V4.1 Flash | 47.2% | 180 of 185 | 46.3% |
| Claude Haiku 5.5 | 82.2% | 200 of 200 | 82.2% |
| Claude Sonnet 5.5 | 93.3% | 215 of 215 | 93.3% |

Jev's 13 misses are all "I want my money back" messages where Jev picked `request_return` or
`create_refund` and the filling call, forced to call the tool, made up the order line or payment
(`line_item_id: "unknown"`); the LLMs asked the customer or looked the order up first. DeepSeek made up
emails from first names (`dev@example.com`). Right tool and arguments, Jev against Sonnet: 34 vs 35
messages only one got right, p = 1.0. A variant filling call allowed to ask a question instead
(`--ask`, `runs/fill/jev_fresh_ask.jsonl`) asked on 208 of 600 messages, overriding 136 of Jev's correct
hand-offs, so it isn't used.

**Jev with a "need more information" option** (`run_router.py --arm jev --ask`, `runs/fresh/jev_ask.jsonl`).
The LLMs can always reply with a question; Jev couldn't. A 92nd option, `need_more_information`
(`toolpick.data.ASK_TOOL`), stands for "ask the customer first" and is scored exactly like an LLM's question:
correct on refund, order and payment requests (`ASK_OK`), benign elsewhere. The *ask-first rule*
(`toolpick.scoring.stipulated_right`, applied to every selector) is the stricter stipulation: on those requests,
when the message gives no order number, invoice number or amount ("I want my money back"), only a question,
a hand-off the key accepts, or a tool that needs nothing from the customer is correct.

| fresh 600 | answer key as is | ask-first rule |
|---|---|---|
| Jev | 95.3% | 90.3% |
| Jev + need_more_information | 87.5% | 85.8% |
| Jev + inputs in the descriptions (`--params`) | 93.3% | 90.5% |
| Jev + both | 84.5% | 83.5% |
| DeepSeek V4.1 Flash | 47.2% | 47.0% |
| Claude Haiku 5.5 | 82.2% | 81.8% |
| Claude Sonnet 5.5 | 93.3% | 93.0% |

Jev chose the new option on 16% of messages at low confidence (median 0.56): 41 on refund, order and payment
requests (correct), the rest in place of a correct invoice lookup, refund check or hand-off. The option lowered
Jev's score on the development messages too (`runs/experiments/jev_ask_first.jsonl`, 96.8% → 91.0%), so it isn't
used for the headline. Adding each tool's inputs to the descriptions Jev reads (`--params`,
`toolpick.data.inputs_text`: name, type, required, a short description, fields or allowed values; 11.7k instead of
7.2k tokens) didn't help either: 93.3% on the fresh messages, 96.8% → 95.0% on the development messages (p = 0.007),
at $0.52 per 1,000. `scripts/compare_jev_variants.py` prints all four Jev versions on both samples. Under the ask-first rule, Sonnet against Jev is 93.0% vs 90.3% (p = 0.10). The LLM runs
are unchanged: they could already reply with a question. The filling call for this run
(`run_jev_fill.py --run fresh/jev_ask`) writes the question; its cost and time are included.

### Asking about an unclear message (a second scoring)

On a request no tool can handle, the answer key accepts only a hand-off, so every question back is a miss.
Claude Opus 5.5 (`scripts/judge_asks.py`, `src/toolpick/judge.py`) read each of the 180 different messages the
agents asked about, alone, and labelled it *unclear* (29), *no tool* (149) or *tool* (2). Under
`toolpick.right_if_asking_when_unclear` a question on an unclear message also counts as correct; a hand-off
stays correct. Labels and reasons: `runs/judge/messages_opus.jsonl`. The README's deep dive has the method,
its weaknesses and examples.

| fresh 600 | questions on no-tool requests | on an unclear message | answer key as is | asking about an unclear message counts |
|---|---|---|---|---|
| Jev | 0 | 0 | 95.3% | 95.3% |
| Claude Sonnet 5.5 | 15 | 8 | 93.3% | 94.7% |
| Claude Haiku 5.5 | 62 | 23 | 82.2% | 86.0% |
| DeepSeek V4.1 Flash | 174 | 29 | 47.2% | 52.0% |

Jev against Sonnet under this scoring: 29 vs 25 messages only one got right, p = 0.68. The judging cost
US$0.77 (180 calls; the 9,728-token prompt written to the cache once and read on the other 179). A first
version of the judge that also saw the agent's reply is kept in `runs/judge/calibration_replies.jsonl`: on
15 messages judged beside two different replies, the label changed with the reply on 2.

## Development runs, first 600 messages (how the setup got there)

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

Three things moved the results during development: auditing the answer key
by hand (every failure checked; `toolpick.data.AUDIT`), writing the routing policy into every prompt, and
adding customer wording to six tool descriptions after finding that Jev over-used the hand-off because real
descriptions are written for developers (`scripts/jev_experiments.py`, results in `runs/experiments/`).
DSPy GEPA (`scripts/optimise_prompt.py`) rewrote DeepSeek's router prompt from 200 messages and scored
84.2% on the 399 it never saw (first-pass key). All runs together: about US$31 of billed calls.

## Commands

Free, from the committed runs:

```bash
uv run python scripts/check_runs.py              # sanity checks on every run file
uv run python scripts/check_arguments.py         # the rule-based argument check
uv run python scripts/compare_jev_variants.py    # the four Jev versions side by side
uv run python scripts/build_ledger.py            # ledger/index.html: every decision, filterable
```

With keys in `.env` (see `.env.example`). Runs resume: a script skips the request ids already in its
`runs/<name>.jsonl`.

```bash
uv run python scripts/run_router.py --arm jev --sample fresh --limit 5   # Jev, final setup, 5 fresh messages
uv run python scripts/run_native.py --arm sonnet --sample fresh         # Sonnet as a real agent, final setup
uv run python scripts/run_jev_fill.py                                   # fill in the arguments of Jev's picks (DeepSeek)
uv run python scripts/check_arguments.py                                # rule-based argument check (free)
uv run python scripts/run_router.py --arm jev --ask --sample fresh     # Jev with a need_more_information option
uv run python scripts/run_router.py --arm jev --params --sample fresh  # Jev with each tool's inputs in its description
uv run python scripts/compare_jev_variants.py                          # the Jev versions side by side (free)
uv run python scripts/run_router.py --arm deepseek --policy             # a development run: router mode, first 600
uv run python scripts/jev_experiments.py --split dev                    # why Jev handed off too often
uv run python scripts/optimise_prompt.py --arm deepseek            # DSPy GEPA on the router preamble
uv run python scripts/log_langfuse.py                              # one Langfuse trace per call
uv run python scripts/judge_asks.py --dry-run                      # the judge: plan and cost estimate (free); drop --dry-run to run it (ANTHROPIC_API_KEY)
```
