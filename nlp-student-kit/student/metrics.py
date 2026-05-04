"""Evaluation metrics.

You implement TWO small functions. Both are called by framework/evaluate.py.

    compute_cross_entropy(logits, targets)         -> (sum_nll, n_tokens)
    compute_perplexity(total_nll, total_tokens)    -> float

These two primitives together define perplexity on the evaluation set.
Keep them decoupled: one computes per-batch NLL, the other aggregates.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def compute_cross_entropy(
    logits: torch.Tensor,   # shape (B, T, V)
    targets: torch.Tensor,  # shape (B, T); -100 marks ignored positions
    ignore_index: int = -100,
) -> tuple[float, int]:
    """Sum of per-token negative log-likelihoods and the number of valid tokens.

    TODO:
      - Flatten logits to (B*T, V) and targets to (B*T,).
      - Use F.cross_entropy with reduction="sum" and the given ignore_index.
      - Count the number of target positions that are NOT ignore_index.
      - Return the two as (float, int).

    Note: use "sum", not "mean". The aggregator does the averaging, so
    per-batch token counts can differ without biasing the result.
    """
    raise NotImplementedError("Implement compute_cross_entropy")


def compute_perplexity(total_nll: float, total_tokens: int) -> float:
    """Aggregate a running NLL sum into a perplexity value.

    TODO:
      - Return exp(total_nll / total_tokens).
      - Guard against total_tokens == 0 (return float("inf") or similar).
      - Clamp the exponent to avoid overflow on very bad models, e.g. min(20, x).

    Returns:
        Perplexity as a Python float.
    """
    raise NotImplementedError("Implement compute_perplexity")
