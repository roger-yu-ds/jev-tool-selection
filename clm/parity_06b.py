"""Encoder parity: llama.cpp's pooled embeddings against HF transformers fp32, on Qwen3-0.6B.

CLM's reference encoder is vLLM serving Qwen3-8B with ``--runner pooling``: the token ids of the raw
text (the Qwen3 tokenizer adds no BOS/EOS, so vLLM's ``add_special_tokens=True`` equals the training
recipe's ``add_special_tokens=False``), the hidden state of the last token after the final RMSNorm,
L2-normalised. This computes exactly that with transformers in fp32 for Qwen3-0.6B (same tokenizer,
same architecture; 0.6B fits in RAM at fp32) and compares it with what llama-server returns from
``/v1/embeddings``, the call CLM's Embedder makes. The strings are real CLM inputs from the study:
state texts (message + routing policy + question, built with ``clm.schema.state_text``) and tool
descriptions (the action texts).

    uv run python parity_06b.py --url http://127.0.0.1:8091 --label q8_0-gpu [--save cache/ref06b.npz]

Negative controls show the check would catch a mismatch: mean pooling instead of last-token, and an
appended <|endoftext|> (what an EOS-adding tokenizer would do).
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
TALK = HERE.parent
HF_06B = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")


def _load(name: str, path: Path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def study_strings(n_states: int = 6) -> list[tuple[str, str]]:
    """(kind, text): CLM state texts for the first fresh messages, then a spread of action texts."""
    from clm.schema import state_text

    data = _load("tp_data", TALK / "src/toolpick/data.py")  # by path: toolpick/__init__ needs jsonschema
    prompts = _load("tp_prompts", TALK / "src/toolpick/prompts.py")
    ins = prompts.jev_instructions(policy=True)
    msgs = [json.loads(x) for x in (TALK / "data/requests_fresh.jsonl").read_text().splitlines()[:n_states]]
    crit = {t["name"]: t["description"] for t in data.tools(customer_words=True)}
    picks = ["hand_off_to_human", "shopify_customer_account.request_return", "stripe.list_invoices",
             "intercom.list_companies", "notion.API-post-search", "shopify_ucp.lookup_locations"]  # short to 1.3k chars
    return [("state", state_text(m["text"], ins)) for m in msgs] + [("action", crit[k]) for k in picks]


def hf_reference(texts: list[str]) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_06B)
    model = AutoModel.from_pretrained(HF_06B, dtype=torch.float32).eval()
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    last, mean, with_eos, ids_all = [], [], [], []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            assert ids == tok(t, add_special_tokens=True)["input_ids"], "tokenizer adds special tokens"
            h = model(input_ids=torch.tensor([ids])).last_hidden_state[0]
            last.append(h[-1].numpy())
            mean.append(h.mean(0).numpy())
            with_eos.append(model(input_ids=torch.tensor([ids + [eos]])).last_hidden_state[0, -1].numpy())
            ids_all.append(ids)
    l2 = lambda x: np.stack(x) / np.linalg.norm(np.stack(x), axis=1, keepdims=True)
    return {"last": l2(last), "mean": l2(mean), "with_eos": l2(with_eos), "ids": ids_all}


def hf_bf16_control(texts: list[str], ref_last: np.ndarray) -> np.ndarray:
    """Cosine of HF bf16 against HF fp32: the rounding noise of a bf16 encoder (vLLM's default dtype
    for Qwen3), for scale against the GGUF quantisation error."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_06B)
    model = AutoModel.from_pretrained(HF_06B, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        v = np.stack([model(input_ids=torch.tensor([tok(t, add_special_tokens=False)["input_ids"]]))
                      .last_hidden_state[0, -1].float().numpy() for t in texts])  # fmt: skip
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.sum(v * ref_last, axis=1)


def server_embeddings(url: str, texts: list[str]) -> tuple[np.ndarray, int]:
    """The request CLM's Embedder sends (base64), decoded and L2-normalised the way it does."""
    r = requests.post(f"{url}/v1/embeddings", json={"model": "qwen3", "input": texts, "encoding_format": "base64",
                                                     "truncate_prompt_tokens": 2048}, timeout=600)
    r.raise_for_status()
    j = r.json()
    out = [None] * len(texts)
    for d in j["data"]:
        v = np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32)
        out[d["index"]] = v / (np.linalg.norm(v) + 1e-12)
    return np.stack(out), int(j["usage"]["prompt_tokens"])


def server_tokens(url: str, text: str) -> list[int]:
    r = requests.post(f"{url}/tokenize", json={"content": text, "add_special": True, "parse_special": True}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--label", default="llama.cpp")
    ap.add_argument("--ref", default=str(HERE / "cache/ref06b.npz"), help="HF reference cache (computed if missing)")
    ap.add_argument("--bf16-control", action="store_true", help="only print HF bf16 vs HF fp32 (no server needed)")
    args = ap.parse_args()
    items = study_strings()
    texts = [t for _, t in items]
    if args.bf16_control:
        z = np.load(args.ref, allow_pickle=False)
        c = hf_bf16_control(texts, z["last"])
        print(f"HF bf16 vs HF fp32 last-token, Qwen3-0.6B: min {c.min():.6f}  mean {c.mean():.6f}")
        return
    ref_path = Path(args.ref)
    if ref_path.exists():
        z = np.load(ref_path, allow_pickle=False)
        ref = {"last": z["last"], "mean": z["mean"], "with_eos": z["with_eos"],
               "ids": json.loads(str(z["ids"]))}  # fmt: skip
    else:
        ref = hf_reference(texts)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref_path, last=ref["last"], mean=ref["mean"], with_eos=ref["with_eos"], ids=json.dumps(ref["ids"]))
    emb, n_tok = server_embeddings(args.url, texts)
    tok_ok = [server_tokens(args.url, t) == ids for t, ids in zip(texts, ref["ids"], strict=True)]
    cos = np.sum(emb * ref["last"], axis=1)
    print(f"{args.label}: {len(texts)} strings, {n_tok} tokens (HF: {sum(map(len, ref['ids']))}), "
          f"token ids identical {sum(tok_ok)}/{len(texts)}")
    for (kind, t), c, n in zip(items, cos, ref["ids"], strict=True):
        print(f"  {kind:6s} {len(n):4d} tok  cos {c:.6f}  {t[:60]!r}")
    print(f"  cosine vs HF fp32 last-token: min {cos.min():.6f}  mean {cos.mean():.6f}")
    print(f"  negative controls (HF vs HF): mean pooling min {np.sum(ref['mean'] * ref['last'], 1).min():.4f}, "
          f"appended <|endoftext|> min {np.sum(ref['with_eos'] * ref['last'], 1).min():.4f}; "
          f"llama.cpp vs HF mean pooling max {np.sum(emb * ref['mean'], 1).max():.4f}")
    off = ref["last"] @ ref["last"].T
    print(f"  (scale: two different study strings, HF vs HF, mean cosine {off[~np.eye(len(texts), dtype=bool)].mean():.4f})")


if __name__ == "__main__":
    main()
