"""LoRA fine-tune the base model using student.lora_config.get_lora_config()."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model

from .data import CausalLMDataset
from .utils import set_seed, get_logger, device_auto, count_params


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["input_ids"].size(0)
        total += out.loss.item() * bs
        n += bs
    model.train()
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    log = get_logger("train_lora",
                     log_file="outputs/training_logs/lora.log")
    log.info("Config: %s", cfg)

    set_seed(cfg.get("seed", 42))
    device = device_auto()
    log.info("Device: %s", device)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"])

    # Get the LoRA config from the student module.
    from student.lora_config import get_lora_config
    lora_cfg = get_lora_config()
    log.info("LoRA config: %s", lora_cfg)

    model = get_peft_model(base, lora_cfg).to(device)
    model.print_trainable_parameters()
    log.info("Trainable params: %.3fM", count_params(model) / 1e6)

    train_ds = CausalLMDataset(cfg["train_file"], tok, cfg["max_len"])
    eval_ds  = CausalLMDataset(cfg["eval_file"],  tok, cfg["max_len"])
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True, collate_fn=collate)
    eval_ld  = DataLoader(eval_ds, batch_size=cfg["batch_size"],
                          shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    out_dir = Path(cfg.get("ckpt_dir", "checkpoints/lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    step = 0
    for epoch in range(cfg["epochs"]):
        for batch in train_ld:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            if step % cfg.get("log_every", 50) == 0:
                log.info("epoch=%d step=%d loss=%.4f",
                         epoch, step, loss.item())
            step += 1

        el = eval_loss(model, eval_ld, device)
        ppl = math.exp(min(20, el))
        log.info("epoch=%d eval_loss=%.4f ppl=%.2f", epoch, el, ppl)

        model.save_pretrained(out_dir / f"epoch_{epoch}")
        tok.save_pretrained(out_dir / f"epoch_{epoch}")
        if el < best:
            best = el
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            log.info("New best eval_loss=%.4f -> saved %s", el, out_dir)


if __name__ == "__main__":
    main()
