"""Evaluate the base model, a scratch-trained model, and the LoRA adapter.

Uses student.metrics.compute_cross_entropy and compute_perplexity as the core.
Writes outputs/metrics.json plus text files with sample generations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .utils import (
    read_jsonl, get_logger, device_auto, import_model_module,
)

from student.metrics import compute_cross_entropy, compute_perplexity

log = get_logger("evaluate", log_file="outputs/training_logs/evaluate.log")


# Perplexity drivers: pull (sum_nll, n_tokens) from each batch and let the
# student's compute_perplexity do the aggregation.
@torch.no_grad()
def ppl_hf(model, tokenizer, rows, device, max_len: int = 512) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for row in rows:
        enc = tokenizer(row["text"], truncation=True, max_length=max_len,
                        return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        # The HF model returns logits aligned with input_ids, so we shift here.
        out = model(**enc)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        sum_nll, n = compute_cross_entropy(logits, targets)
        total_nll += sum_nll
        total_tokens += n
    return compute_perplexity(total_nll, total_tokens)


@torch.no_grad()
def ppl_custom(model, tokenizer, rows, device, block_size: int) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for row in rows:
        ids = tokenizer.encode(row["text"])
        if len(ids) < 2:
            continue
        ids = torch.tensor(ids, dtype=torch.long, device=device)
        for start in range(0, len(ids) - 1, block_size):
            chunk = ids[start : start + block_size + 1]
            if chunk.size(0) < 2:
                continue
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            logits, _ = model(x)
            sum_nll, n = compute_cross_entropy(logits, y)
            total_nll += sum_nll
            total_tokens += n
    return compute_perplexity(total_nll, total_tokens)


# Generation helpers.
def gen_hf(model, tok, prompt: str, max_new: int = 80) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                         top_k=50, temperature=0.9,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def gen_custom(model, tok, prompt: str, max_new: int = 80) -> str:
    device = next(model.parameters()).device
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new,
                         temperature=0.9, top_k=50)
    return tok.decode(out[0].tolist(), skip_special_tokens=True)


# Loaders.
def load_custom(ckpt_path: Path, device):
    state = torch.load(ckpt_path, map_location=device)
    mod = import_model_module(state["model_module"])
    # Reconstruct model using saved config; vocab_size lives inside it.
    cfg = state["model_config"]
    model = mod.build_model(cfg["vocab_size"], cfg).to(device)
    model.load_state_dict(state["model_state"])
    tok = AutoTokenizer.from_pretrained(state["tokenizer"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    block_size = cfg.get("block_size", 128)
    return model, tok, block_size


def load_lora(base_model: str, adapter_dir: Path, device):
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    return model, tok


# Main.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--base-model", type=str, default="gpt2")
    ap.add_argument("--scratch-ckpt", type=Path,
                    default=Path("checkpoints/scratch/best.pt"))
    ap.add_argument("--reference-ckpt", type=Path,
                    default=Path("checkpoints/reference/best.pt"))
    ap.add_argument("--lora-ckpt", type=Path,
                    default=Path("checkpoints/lora"))
    ap.add_argument("--prompts", nargs="+",
                    default=["In summary,", "The key idea is",
                             "Today I learned"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = device_auto()

    rows = list(read_jsonl(args.eval_file))
    log.info("Eval rows: %d", len(rows))

    metrics: dict = {"eval_file": str(args.eval_file), "n_rows": len(rows)}

    # Base model.
    log.info("Evaluating base model: %s", args.base_model)
    btok = AutoTokenizer.from_pretrained(args.base_model)
    if btok.pad_token_id is None:
        btok.pad_token = btok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    metrics["base"] = {"perplexity": ppl_hf(base, btok, rows, device)}
    (args.out_dir / "samples_base.txt").write_text(
        "\n\n".join(f"=== PROMPT: {p}\n{gen_hf(base, btok, p)}"
                    for p in args.prompts),
        encoding="utf-8",
    )

    # Reference toy RNN model.
    if args.reference_ckpt.exists():
        log.info("Evaluating reference model: %s", args.reference_ckpt)
        m, t, bs = load_custom(args.reference_ckpt, device)
        metrics["reference"] = {"perplexity": ppl_custom(m, t, rows, device, bs)}
        (args.out_dir / "samples_reference.txt").write_text(
            "\n\n".join(f"=== PROMPT: {p}\n{gen_custom(m, t, p)}"
                        for p in args.prompts),
            encoding="utf-8",
        )

    # Scratch model (the student's custom LM).
    if args.scratch_ckpt.exists():
        log.info("Evaluating scratch model: %s", args.scratch_ckpt)
        m, t, bs = load_custom(args.scratch_ckpt, device)
        metrics["scratch"] = {"perplexity": ppl_custom(m, t, rows, device, bs)}
        (args.out_dir / "samples_scratch.txt").write_text(
            "\n\n".join(f"=== PROMPT: {p}\n{gen_custom(m, t, p)}"
                        for p in args.prompts),
            encoding="utf-8",
        )
    else:
        log.warning("Scratch checkpoint not found: %s", args.scratch_ckpt)

    # LoRA adapter.
    if args.lora_ckpt.exists() and any(args.lora_ckpt.iterdir()):
        log.info("Evaluating LoRA adapter: %s", args.lora_ckpt)
        m, t = load_lora(args.base_model, args.lora_ckpt, device)
        metrics["lora"] = {"perplexity": ppl_hf(m, t, rows, device)}
        (args.out_dir / "samples_lora.txt").write_text(
            "\n\n".join(f"=== PROMPT: {p}\n{gen_hf(m, t, p)}"
                        for p in args.prompts),
            encoding="utf-8",
        )
    else:
        log.warning("LoRA adapter dir empty or missing: %s", args.lora_ckpt)

    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    log.info("Wrote %s", args.out_dir / "metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
