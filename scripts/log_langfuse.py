"""Log every call in runs/*.jsonl to Langfuse, one trace per call.

    uv run python scripts/log_langfuse.py --check            # auth only
    uv run python scripts/log_langfuse.py                    # backfill every run (idempotent)
    uv run python scripts/log_langfuse.py --runs jev native_sonnet_policy

Needs LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST in .env. Langfuse v4 takes
traces over OpenTelemetry, so each call becomes one OTel span (type `generation`) exported to
`/api/public/otel/v1/traces` with its measured start and end time, model, usage (uncached input,
cached input, output), billed cost, input, output and metadata; scores go to the ingestion API,
which still accepts them.

Per trace: name `tool-select`, session `<mode>/<prompt>` (e.g. native/policy), tags [model, mode,
prompt], input = the customer's message, output = the pick, metadata (intent, request id, run);
the generation carries the pick, the tool-call arguments, Jev's confidence and top tools, the
provider and the prompt name (the prompts themselves are stored once in prompt management);
scores `right_v1`, `right_v2`, `wrong_call` (0/1) and `outcome_v1` (categorical). Trace ids are
deterministic (a hash of run + request id), so re-running updates rather than duplicates.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parents[1] / ".env")  # when this folder sits inside the course repo, its .env works too
sys.path.insert(0, str(ROOT / "src"))

from toolpick import RUNS, labels, outcome, parse_run_name, requests, tools
from toolpick.prompts import jev_instructions, native_system, router_system
from toolpick.scoring import MODEL_NAMES, WRONG_CALL

MODEL_IDS = {"jev": "jev", "deepseek": "deepseek/deepseek-v4.1-flash", "haiku": "anthropic/claude-haiku-5.5",
             "sonnet": "anthropic/claude-sonnet-5.5"}  # fmt: skip


def creds() -> tuple[str, str, str]:
    host = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "http://127.0.0.1:3010").rstrip("/")
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST in .env")
    return host, pk, sk


def client() -> httpx.Client:
    host, pk, sk = creds()
    return httpx.Client(base_url=host, auth=(pk, sk), timeout=120)


def tid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def ensure_prompts(c: httpx.Client) -> None:
    tools_json = json.dumps(tools(), separators=(",", ":"))
    texts = {"router/base": router_system(tools_json, False), "router/policy": router_system(tools_json, True),
             "native/base": native_system(False), "native/policy": native_system(True),
             "jev/base": jev_instructions(False), "jev/policy": jev_instructions(True)}  # fmt: skip
    gepa = ROOT / "data" / "gepa_deepseek.json"
    if gepa.exists():
        from toolpick.prompts import ROUTER_TOOLS_HEADER

        texts["router/gepa"] = json.loads(gepa.read_text())["instruction"] + "\n\n" + ROUTER_TOOLS_HEADER + tools_json
    for name, text in texts.items():
        r = c.get(f"/api/public/v2/prompts/{name.replace('/', '%2F')}")
        if r.status_code == 200 and r.json().get("prompt") == text:
            continue
        c.post("/api/public/v2/prompts", json={"name": name, "type": "text", "prompt": text, "labels": ["production"]}).raise_for_status()


# -- OTel export with fixed trace ids and measured timestamps -----------------------------------
class FixedIds:
    """An OpenTelemetry IdGenerator whose next trace id is set by the caller."""

    def __init__(self) -> None:
        self.next_trace = 0

    def generate_trace_id(self) -> int:
        return self.next_trace

    def generate_span_id(self) -> int:  # one span per trace here, so derive it from the trace id
        return (self.next_trace >> 64) ^ (self.next_trace & ((1 << 64) - 1)) or 1

    def is_trace_id_random(self) -> bool:
        return False


def tracer_and_provider():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    host, pk, sk = creds()
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    exporter = OTLPSpanExporter(endpoint=f"{host}/api/public/otel/v1/traces", headers={"Authorization": f"Basic {auth}"}, timeout=60)
    ids = FixedIds()
    provider = TracerProvider(resource=Resource.create({"service.name": "toolpick"}), id_generator=ids)
    provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=20000, max_export_batch_size=100, export_timeout_millis=60000))
    import langfuse

    # Langfuse parses its own `langfuse.*` attributes from spans whose scope is the SDK's.
    return provider.get_tracer("langfuse-sdk", langfuse.__version__), provider, ids


def j(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def emit_span(tracer, ids: FixedIds, run: str, row: dict, req, lab1: dict, lab2: dict) -> tuple[str, dict]:
    meta_run = parse_run_name(run)
    model = meta_run["model"]
    trace_id = tid("measurecamp", run, row["id"])
    kind = row.get("kind", "tool_call")
    calls = [c["name"] for c in row.get("tool_calls") or []]
    o1 = outcome(row["pick"], lab1[req.intent], kind, calls)
    o2 = outcome(row["pick"], lab2[req.intent], kind, calls)
    start = datetime.fromisoformat(row["ts"]) if row.get("ts") else datetime.now(tz=UTC) - timedelta(seconds=row["latency_s"])
    end = start + timedelta(seconds=row["latency_s"])
    usage = {"input": (row.get("input_tokens") or 0) - (row.get("cached_tokens") or 0),
             "input_cached_tokens": row.get("cached_tokens") or 0, "output": row.get("output_tokens") or 0}  # fmt: skip
    if model == "jev":
        output = {"choice": row["pick"], "confidence": row.get("confidence"), "top": row.get("top")}
    elif "raw" in row:
        output = row.get("raw")
    else:
        output = {"pick": row["pick"], "text": row.get("text"), "tool_calls": row.get("tool_calls")}
    attrs = {
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": row.get("model") or MODEL_IDS[model],
        "langfuse.observation.input": j([{"role": "user", "content": req.text}]),
        "langfuse.observation.output": j(output),
        "langfuse.observation.usage_details": j(usage),
        "langfuse.observation.cost_details": j({"total": float(row.get("cost_usd") or 0)}),
        "langfuse.observation.prompt.name": f"{meta_run['mode']}/{meta_run['prompt']}",
        "langfuse.observation.metadata": j({"request_id": req.id, "intent": req.intent, "gap_intent": lab1[req.intent]["gap"],
                                            "kind": kind, "outcome_v1": o1, "outcome_v2": o2, "provider": row.get("provider"),
                                            "finish_reason": row.get("finish_reason"), "jev_confidence": row.get("confidence"),
                                            "jev_top": row.get("top"), "tool_calls": row.get("tool_calls"),
                                            "timestamps": "measured" if row.get("ts") else "approximate (backfilled; duration is measured)"}),
        "langfuse.trace.name": "tool-select",
        "session.id": f"{meta_run['mode']}/{meta_run['prompt']}",
        "gen_ai.request.model": row.get("model") or MODEL_IDS[model],
        "gen_ai.usage.input_tokens": int(row.get("input_tokens") or 0),
        "gen_ai.usage.output_tokens": int(row.get("output_tokens") or 0),
        "langfuse.trace.tags": [model, meta_run["mode"], meta_run["prompt"], "measurecamp-2026"],
        "langfuse.trace.input": req.text,
        "langfuse.trace.output": row["pick"] or "(text reply)",
        "langfuse.trace.metadata": j({"intent": req.intent, "request_id": req.id, "run": run, "model_name": MODEL_NAMES[model]}),
        "langfuse.environment": "measurecamp-2026",
    }
    ids.next_trace = int(trace_id, 16)
    span = tracer.start_span(f"select:{model}", start_time=int(start.timestamp() * 1e9), attributes=attrs)
    span.end(end_time=int(end.timestamp() * 1e9))
    scores = {"right_v1": float(o1 == "right"), "right_v2": float(o2 == "right"), "wrong_call": float(o1 in WRONG_CALL),
              "outcome_v1": o1}  # fmt: skip
    return trace_id, scores


def send_scores(c: httpx.Client, items: list[tuple[str, dict]]) -> int:
    now = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    events = []
    for trace_id, scores in items:
        for name, val in scores.items():
            events.append({"id": str(uuid.uuid4()), "timestamp": now, "type": "score-create",
                           "body": {"id": tid("score", trace_id, name), "traceId": trace_id, "name": name, "value": val,
                                    "dataType": "CATEGORICAL" if isinstance(val, str) else "NUMERIC"}})  # fmt: skip
    errors = 0
    for i in range(0, len(events), 200):
        r = c.post("/api/public/ingestion", json={"batch": events[i : i + 200]})
        r.raise_for_status()
        errs = r.json().get("errors", [])
        if errs and errors == 0:
            print("  first score error:", errs[0].get("error") or errs[0].get("message"))
        errors += len(errs)
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--runs", nargs="*", help="run names to log (default: all)")
    args = ap.parse_args()
    c = client()
    r = c.get("/api/public/projects")
    r.raise_for_status()
    print("project:", [p["name"] for p in r.json()["data"]], flush=True)
    if args.check:
        return
    ensure_prompts(c)
    tracer, provider, ids = tracer_and_provider()
    reqs = {r.id: r for r in requests()}
    lab1, lab2 = labels("v1"), labels("v2")
    paths = [RUNS / f"{n}.jsonl" for n in args.runs] if args.runs else sorted(RUNS.glob("*.jsonl"))
    for path in paths:
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        items = [emit_span(tracer, ids, path.stem, row, reqs[row["id"]], lab1, lab2) for row in rows]
        provider.force_flush()
        print(f"{path.stem}: {len(rows)} spans exported, {send_scores(c, items)} score errors", flush=True)
    provider.shutdown()


if __name__ == "__main__":
    main()
