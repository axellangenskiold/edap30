"""Prepare data: load from data/raw/ or call student.collect_data.generate_samples,
clean, deduplicate, and write train and eval splits.

Supported raw file formats in data/raw/:
    .txt    Plain text. The whole file becomes one sample.
    .jsonl  One JSON object per line. Each object is either:
              {"text": "..."}                                   (plain text)
              {"instruction": "...", "output": "..."}           (instruction)
              {"instruction": "...", "input": "...",
               "output": "..."}                                 (instruction
                                                                 with input)
    .json   A JSON array of the same kinds of objects as .jsonl.

Instruction-style records are converted to plain text using an Alpaca-style
template (see `instruction_to_text`).

Usage:
    python -m framework.prepare_data --raw-dir data/raw --out-dir data
    python -m framework.prepare_data --topic "recipes" --n-samples 600
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from .utils import read_jsonl, write_jsonl, get_logger

log = get_logger("prepare_data")


def instruction_to_text(instruction: str, output: str, input_text: str = "") -> str:
    """Format an instruction record as a single training string.

    Uses the Alpaca-style template, which is widely recognised and works well
    for causal language models.
    """
    instruction = (instruction or "").strip()
    output = (output or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def record_to_text(record: Dict) -> str | None:
    """Turn a raw record into a plain text string, or return None if the
    record doesn't match any supported shape.

    Supported shapes:
      {"text": "..."}                                   plain text
      {"instruction": "...", "output": "..."}           instruction
      {"instruction": "...", "input": "...",
       "output": "..."}                                 instruction with input
    """
    if not isinstance(record, dict):
        return None
    if "text" in record:
        text = record.get("text")
        return text.strip() if isinstance(text, str) else None
    if "instruction" in record and "output" in record:
        return instruction_to_text(
            instruction=record.get("instruction", ""),
            output=record.get("output", ""),
            input_text=record.get("input", ""),
        )
    return None


def load_raw(raw_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    skipped = 0
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix == ".jsonl":
            for record in read_jsonl(p):
                text = record_to_text(record)
                if text:
                    rows.append({"text": text})
                else:
                    skipped += 1
        elif p.suffix == ".json":
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                log.warning("Could not parse %s as JSON (%s); skipping.", p, e)
                continue
            if not isinstance(payload, list):
                log.warning("%s is not a JSON array; skipping.", p)
                continue
            for record in payload:
                text = record_to_text(record)
                if text:
                    rows.append({"text": text})
                else:
                    skipped += 1
        elif p.suffix == ".txt":
            text = p.read_text(encoding="utf-8").strip()
            if text:
                rows.append({"text": text})
    if skipped:
        log.warning("Skipped %d records that did not match a supported shape "
                    "(missing 'text' or both 'instruction' and 'output').",
                    skipped)
    return rows


def clean(rows: List[Dict], min_chars: int = 40) -> List[Dict]:
    seen, out = set(), []
    for r in rows:
        text = (r.get("text") or "").strip()
        if len(text) < min_chars or text in seen:
            continue
        seen.add(text)
        out.append({"text": text})
    return out


def split(rows: List[Dict], eval_frac: float, seed: int):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    k = max(1, int(len(rows) * eval_frac))
    return rows[k:], rows[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--topic", type=str, default=None,
                    help="If provided, call student.collect_data instead of loading raw files.")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.topic:
        log.info("Generating %d samples via student.collect_data for topic=%r",
                 args.n_samples, args.topic)
        from student.collect_data import generate_samples
        rows = generate_samples(args.topic, args.n_samples)
    else:
        log.info("Loading raw files from %s", args.raw_dir)
        rows = load_raw(args.raw_dir)

    log.info("Raw rows: %d", len(rows))
    rows = clean(rows)
    log.info("After cleaning: %d", len(rows))
    if len(rows) < 500:
        log.warning("Fewer than 500 samples; assignment requires at least 500.")

    train, evl = split(rows, args.eval_frac, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "eval.jsonl", evl)
    log.info("Wrote %d train / %d eval rows to %s",
             len(train), len(evl), args.out_dir)


if __name__ == "__main__":
    main()
