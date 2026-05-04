"""Toy RNN language model. REFERENCE IMPLEMENTATION.

READ THIS FILE before implementing student/model.py. Do NOT modify it.

This is a small LSTM-based causal language model. It demonstrates the exact
interface the framework expects from any language model: a class with a
two-return `forward()` and a `generate()` method.

Your job in student/model.py is to implement a Transformer/GPT-style model
with the same interface. The training, evaluation, and generation code in
framework/ works with any model that follows this contract.

Interface contract:
    class YourModel(nn.Module):
        def __init__(self, vocab_size: int, **kwargs): ...
        def forward(self, input_ids, targets=None) -> (logits, loss_or_None): ...
        def generate(self, input_ids, max_new_tokens, temperature, top_k) -> ids: ...
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyRNNConfig:
    vocab_size: int
    n_embd: int = 128
    n_hidden: int = 256
    n_layer: int = 1
    dropout: float = 0.1


class ToyRNN(nn.Module):
    """LSTM-based causal language model.

    Architecture:
        tokens -> embedding -> LSTM -> dropout -> linear -> logits
    """

    def __init__(self, config: ToyRNNConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_hidden,
            num_layers=config.n_layer,
            dropout=config.dropout if config.n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.n_hidden, config.vocab_size)

    # Required: forward returns (logits, loss_or_None).
    def forward(
        self,
        input_ids: torch.Tensor,              # shape (B, T)
        targets: torch.Tensor | None = None,  # shape (B, T) or None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed(input_ids)             # (B, T, E)
        x, _ = self.lstm(x)                   # (B, T, H)
        x = self.drop(x)
        logits = self.head(x)                 # (B, T, V)

        loss = None
        if targets is not None:
            # cross_entropy expects (N, V) and (N,); flatten the time dim.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    # Required: sample a continuation from a prompt.
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,              # shape (1, T0)
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self(out)
            next_logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok], dim=1)
        return out


# Factory function used by the framework. Any model module exposes one of these.
def build_model(vocab_size: int, model_cfg: dict) -> ToyRNN:
    """Construct a ToyRNN from a config dict. The framework calls this."""
    cfg = ToyRNNConfig(
        vocab_size=vocab_size,
        n_embd=model_cfg.get("n_embd", 128),
        n_hidden=model_cfg.get("n_hidden", 256),
        n_layer=model_cfg.get("n_layer", 1),
        dropout=model_cfg.get("dropout", 0.1),
    )
    return ToyRNN(cfg)


def config_to_dict(model: ToyRNN) -> dict:
    """For checkpointing. Framework calls this to save the model config."""
    return asdict(model.config)
