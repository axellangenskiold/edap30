"""Train a custom causal LM from scratch.

Uses whichever model module the config points to:
    model_module: student.model     -> student's CustomLM
    model_module: reference.toy_rnn -> reference ToyRNN

The module must expose `build_model(vocab_size, model_cfg)` and
`config_to_dict(model)`, plus a model class whose forward returns
(logits, loss).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import TokenBlocksDataset
from .utils import (
    set_seed, get_logger, device_auto, count_params, import_model_module,
)


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.1):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    model.train()
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    run_name = cfg.get("run_name", args.config.stem)
    log = get_logger(
        f"train[{run_name}]",
        log_file=f"outputs/training_logs/{run_name}.log",
    )
    log.info("Config: %s", cfg)

    set_seed(cfg.get("seed", 42))
    device = device_auto()
    log.info("Device: %s", device)

    # Tokenizer.
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Data.
    block_size = cfg["model"].get("block_size", cfg.get("block_size", 128))
    train_ds = TokenBlocksDataset(cfg["train_file"], tok, block_size)
    eval_ds  = TokenBlocksDataset(cfg["eval_file"],  tok, block_size)
    log.info("Tokens: train=%d  eval=%d", len(train_ds.data), len(eval_ds.data))

    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True, drop_last=True)
    eval_ld  = DataLoader(eval_ds, batch_size=cfg["batch_size"],
                          shuffle=False, drop_last=False)

    # Model (from student or reference).
    mod = import_model_module(cfg["model_module"])
    model = mod.build_model(tok.vocab_size, cfg["model"]).to(device)
    log.info("Model: %s  params=%.2fM",
             type(model).__name__, count_params(model) / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * max(1, len(train_ld))
    warmup = cfg.get("warmup_steps", max(50, total_steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, warmup, total_steps)
    )

    out_dir = Path(cfg.get("ckpt_dir", f"checkpoints/{run_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    step = 0
    for epoch in range(cfg["epochs"]):
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % cfg.get("log_every", 50) == 0:
                log.info("epoch=%d step=%d loss=%.4f lr=%.2e",
                         epoch, step, loss.item(), sched.get_last_lr()[0])
            step += 1

        el = eval_loss(model, eval_ld, device)
        ppl = math.exp(min(20, el))
        log.info("epoch=%d eval_loss=%.4f ppl=%.2f", epoch, el, ppl)

        state = {
            "model_state": model.state_dict(),
            "model_module": cfg["model_module"],
            "model_config": mod.config_to_dict(model),
            "tokenizer": cfg["tokenizer"],
            "eval_loss": el,
            "epoch": epoch,
        }
        torch.save(state, out_dir / f"epoch_{epoch}.pt")
        if el < best:
            best = el
            torch.save(state, out_dir / "best.pt")
            log.info("New best eval_loss=%.4f -> saved best.pt", el)


if __name__ == "__main__":
    main()
