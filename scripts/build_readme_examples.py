"""Rewrite the examples in the README's deep dive from the committed runs (free, no API calls).

    uv run python scripts/build_readme_examples.py

For each chosen message: what Jev, Sonnet, Haiku and DeepSeek did with it, the judge's label, and
each agent's reply in full. The block between the two markers in README.md is replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import toolpick as tp

README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- examples:start -->", "<!-- examples:end -->"
RUNS = {"Jev": "fresh/jev", "Claude Sonnet 5.5": "fresh/native_sonnet", "Claude Haiku 5.5": "fresh/native_haiku",
        "DeepSeek V4.1 Flash": "fresh/native_deepseek"}  # fmt: skip
GROUPS = {
    "The judge called these unclear, so a question counts as correct": ["F0177", "F0336", "F0185", "F0058"],
    "The judge called these clear requests no tool can do, so a question stays a miss": ["F0229", "F0118", "F0352", "F0363"],
    "The judge said a tool can do this one": ["F0073"],
}
LABEL = {"unclear": "unclear", "no_tool": "clear, and no tool can do it", "tool": "clear, and a tool can do it"}


def short(name: str) -> str:
    return name.split(".")[-1]


def did(selector: str, row: dict) -> str:
    """One line on what a selector did with the message."""
    if row["outcome"] == "asked":
        return "replied with a question"
    if selector == "Jev":
        return f"picked `{short(row['pick'])}` (confidence {row['confidence']:.2f})"
    return f"called `{short(row['pick'])}`"


def scored(row: dict) -> tuple[str, str]:
    """(as scored, under the ask-when-unclear rule) in words."""
    now = "correct" if row["right"] else f"miss ({row['outcome'].replace('_', ' ')})"
    return now, "correct" if tp.right_if_asking_when_unclear(row) else now


def quote(text: str) -> str:
    return "\n".join(f"> {line}".rstrip() for line in text.strip().splitlines())


def example(rid: str, rows: dict[str, dict[str, dict]]) -> str:
    req = next(r for r in tp.requests("fresh") if r.id == rid)
    judged = tp.judgements()[rid]
    out = [f"<details>\n<summary><b>“{req.text}”</b> · judge: {LABEL[judged['label']]}</summary>\n",
           f"Message `{rid}`, Bitext intent `{req.intent}`. The judge: *{judged['reason']}*\n",
           "| selector | what it did | as scored | if asking about an unclear message counts |", "|---|---|---|---|"]  # fmt: skip
    for selector in RUNS:
        row = rows[selector][rid]
        out.append(f"| {selector} | {did(selector, row)} | {' | '.join(scored(row))} |")
    for selector in RUNS:
        row = rows[selector][rid]
        if row["outcome"] == "asked":
            out.append(f"\n**{selector}** replied:\n\n{quote(row['reply'])}")
        elif selector != "Jev" and row["arguments"]:
            out.append(f"\n**{selector}** called `{short(row['pick'])}` with `{json.dumps(row['arguments'], ensure_ascii=False)}`")
    return "\n".join(out) + "\n\n</details>"


def main() -> None:
    rows = {selector: {r["id"]: r for r in tp.scored_rows(run)} for selector, run in RUNS.items()}
    parts = []
    for title, ids in GROUPS.items():
        parts.append(f"**{title}**\n\n" + "\n\n".join(example(rid, rows) for rid in ids))
    text = README.read_text()
    head, rest = text.split(START)
    README.write_text(head + START + "\n\n" + "\n\n".join(parts) + "\n\n" + END + rest.split(END)[1])
    print(f"wrote {sum(len(v) for v in GROUPS.values())} examples into {README.name}")


if __name__ == "__main__":
    main()
