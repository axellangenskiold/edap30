"""Your custom language model goes here.

Study `reference/toy_rnn.py` first. Your model must expose the same interface:

    class CustomLM(nn.Module):
        forward(input_ids, targets=None) -> (logits, loss_or_None)
        generate(input_ids, max_new_tokens, temperature, top_k) -> ids

You must also provide `build_model(vocab_size, model_cfg)` and
`config_to_dict(model)` at the bottom of this file, exactly like toy_rnn does.
The framework calls these to construct and checkpoint your model.

Recommended architecture: a small GPT-style Transformer (token embedding +
positional embedding + Transformer blocks + LayerNorm + LM head). You are
free to use `torch.nn.TransformerEncoderLayer` or implement attention
yourself; both are fine for this assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CustomLMConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CustomLM(nn.Module):
    """TODO: implement your custom causal language model.

    Suggested layers:
      - token embedding       nn.Embedding(vocab_size, n_embd)
      - positional embedding  nn.Embedding(block_size, n_embd)
      - N transformer blocks  (self-attention + MLP, with residual and LN)
      - final LayerNorm
      - output head           nn.Linear(n_embd, vocab_size)

    Tip: parameter tying (self.head.weight = self.tok_emb.weight) trains
    faster on small data.
    """

    def __init__(self, config: CustomLMConfig):
        super().__init__()
        self.config = config
        # TODO: define your layers here.
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output head.
        self.head.weight = self.tok_emb.weight
        
        # raise NotImplementedError("Implement CustomLM.__init__")

    def forward(
        self,
        input_ids: torch.Tensor,              # shape (B, T)
        targets: torch.Tensor | None = None,  # shape (B, T) or None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits, loss). See toy_rnn.py for the pattern.

        TODO:
          1. Compute hidden states from input_ids.
          2. Project to logits over the vocabulary.
          3. If targets is not None, compute cross-entropy loss with
             ignore_index=-100.
        """
        
        B, T = input_ids.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(T, device=input_ids.device)        # (T,)
        tok = self.tok_emb(input_ids)                         # (B, T, E)
        pos = self.pos_emb(pos)                               # (T, E)
        x = self.drop(tok + pos)                              # (B, T, E)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.blocks(x, mask=causal_mask, is_causal=True)  # (B, T, E)

        x = self.ln_f(x)                                      # (B, T, E)
        logits = self.head(x)                                 # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample a continuation. You can copy this from toy_rnn.py almost
        verbatim, but remember to truncate input_ids to the last
        `config.block_size` tokens before each forward pass; unlike RNNs,
        Transformers have a fixed context length.
        """
    
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            idx_cond = out[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok], dim=1)
        return out


# Factory and checkpoint helpers (the framework calls these).
def build_model(vocab_size: int, model_cfg: dict) -> CustomLM:
    """Construct a CustomLM from a config dict."""
    cfg = CustomLMConfig(
        vocab_size=vocab_size,
        block_size=model_cfg.get("block_size", 256),
        n_layer=model_cfg.get("n_layer", 4),
        n_head=model_cfg.get("n_head", 4),
        n_embd=model_cfg.get("n_embd", 256),
        dropout=model_cfg.get("dropout", 0.1),
    )
    return CustomLM(cfg)


def config_to_dict(model: CustomLM) -> dict:
    return asdict(model.config)
