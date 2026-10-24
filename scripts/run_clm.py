"""CLM-8B as a tool selector, run locally: the same `choice` question Jev answers in the final setup.

    clm/serve_encoder.sh                   # 1. Qwen3-8B (GGUF Q8_0) in llama.cpp: /v1/embeddings on :8090
    clm/serve_clm.sh                       # 2. clm-serve (contrastive-lm 0.1.0, CLM-v0.1-8B head): :8700
    uv run python scripts/run_clm.py --sample fresh --limit 20     # 3. this, in the toolpick env

CLM (Contrastive-LM/CLM-v0.1-8B) is a frozen Qwen3-8B encoder plus two small projection heads trained
with InfoNCE. clm-serve speaks TypeSafe's System One wire format, so the request is Jev's request body
from `run_router.py --arm jev --sample fresh`: the customer's message as the state, one `choice`
question whose options are the 91 tool names -> descriptions (`tools(customer_words=True)`) and whose
instructions are `jev_instructions(policy=True)`. The differences are CLM's, not ours: the state head
reads `message + "\\n\\n" + instructions` and the action head reads each option's description only
(the option key, the tool name, is not embedded); the answer is a softmax over 100 x cosine.

The first call embeds the 91 descriptions, which clm-serve then keeps in its vector cache, so the script
makes one warm-up call first (its time and encoder tokens are printed, not written as a row). Each
row's ``input_tokens`` is what the encoder read for that call (the state), ``cost_usd`` is 0 (local),
and ``latency_s`` is the client-side time of the HTTP call, as for Jev.

Rows go to runs/fresh/clm.jsonl (`--sample fresh`) or runs/experiments/clm_first.jsonl (`--sample
first`), in the schema of Jev's rows. Re-running skips ids already done.

`--variant names|plain_descriptions|short_question|message_only --out clm/cache/<name>.jsonl` changes one
thing from that setup (see VARIANTS), to see why CLM picks what it picks; score with clm/score_clm.py --file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolpick import HAND, RUNS, requests, tools
from toolpick.prompts import JEV_INSTRUCTIONS, jev_instructions

OUT = {"fresh": "fresh/clm", "first": "experiments/clm_first"}
# Diagnostics (`--variant`, with `--out`): each changes one thing from the final setup.
VARIANTS = {
    "final": "Jev's final setup: customer-worded descriptions, routing policy + question as instructions",
    "names": "each option's text is 'tool_name: description' (CLM otherwise never sees the tool names)",
    "plain_descriptions": "the real descriptions without the six 'Customers say: ...' lines",
    "short_question": "instructions = the question alone, without the 800-character routing policy",
    "message_only": "no instructions: the state head reads the customer's message alone",
}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["id"] for x in path.read_text().splitlines() if x.strip()}


class CLM:
    """POST /v1/systemone on a local clm-serve: Jev's payload, CLM's model name."""

    def __init__(self, url: str, model: str = "clm-latest", timeout_s: float = 900.0) -> None:
        self.url, self.model = url.rstrip("/"), model
        self._client = httpx.Client(timeout=timeout_s)

    def choose(self, text: str, instructions: str, criteria: dict[str, str], question_id: str = "tool") -> dict:
        payload = {"state": text, "model": self.model,
                   "questions": {question_id: {"type": "choice", "instructions": instructions, "criteria": criteria}}}  # fmt: skip
        started = time.perf_counter()
        resp = self._client.post(f"{self.url}/v1/systemone", json=payload)
        latency = time.perf_counter() - started
        resp.raise_for_status()
        server_ms = resp.headers.get("X-CLM-Latency-Ms")
        return {**resp.json(), "latency_s": latency, "server_ms": float(server_ms) if server_ms else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=list(OUT), default="fresh")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8700")
    ap.add_argument("--model", default="clm-latest", help="the clm-serve model (the reference head is clm-latest)")
    ap.add_argument("--label", default="clm-v0.1-8b (Qwen3-8B Q8_0 GGUF, llama.cpp, local)",
                    help="written as each row's model: the head and the encoder that produced it")
    ap.add_argument("--out", default=None,
                    help="run name under runs/ (default: fresh/clm or experiments/clm_first), or a path ending in .jsonl")
    ap.add_argument("--variant", choices=list(VARIANTS), default="final",
                    help="final = Jev's final setup (the default; the only one written to runs/ by default). The others "
                         "change one thing, to see why CLM picks what it picks; they need --out")
    args = ap.parse_args()
    if args.variant != "final" and not args.out:
        ap.error("a --variant other than final needs --out (keep diagnostics out of runs/fresh/clm.jsonl)")

    tool_list = tools(customer_words=args.variant != "plain_descriptions")
    assert HAND in {t["name"] for t in tool_list}
    criteria = {t["name"]: (f"{t['name']}: " if args.variant == "names" else "") + t["description"] for t in tool_list}
    instructions = {"short_question": JEV_INSTRUCTIONS, "message_only": None}.get(args.variant, jev_instructions(policy=True))
    reqs = requests(args.sample)[: args.limit] if args.limit else requests(args.sample)
    name = args.out or OUT[args.sample]
    out = Path(name) if name.endswith(".jsonl") else RUNS / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in reqs if r.id not in done_ids(out)]
    print(f"{name}: {len(todo)} to run", flush=True)
    if not todo:
        return
    clm = CLM(args.url, args.model)
    warm = clm.choose("warm-up", instructions, criteria)  # embeds and caches the 91 option descriptions
    print(f"warm-up: {warm['latency_s']:.1f}s, {warm['usage']['input_tokens']} encoder tokens "
          f"(the 91 descriptions + one state)", flush=True)
    t0 = time.perf_counter()
    for i, r in enumerate(todo, 1):
        ts = datetime.now(tz=UTC).isoformat()
        body = clm.choose(r.text, instructions, criteria)
        a = body["answers"]["tool"]
        top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:8]
        row = {"id": r.id, "pick": a["choice"], "kind": "tool_call", "confidence": a["confidence"],
               "top": [[k, round(v, 5)] for k, v in top], "latency_s": round(body["latency_s"], 4),
               "server_ms": body["server_ms"], "input_tokens": body["usage"]["input_tokens"],
               "output_tokens": body["usage"].get("output_tokens", 0), "cost_usd": 0.0, "ts": ts, "model": args.label,
               **({"variant": args.variant} if args.variant != "final" else {}),
               "response": {k: v for k, v in body.items() if k not in ("latency_s", "server_ms")}}  # fmt: skip
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if i % 25 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            print(f"{i}/{len(todo)}  {el / i:.2f}s per message  eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
    print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
