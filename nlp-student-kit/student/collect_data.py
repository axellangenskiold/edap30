"""Data collection via the provided LLM API.

You only need to implement ONE function: `generate_samples`.
The framework handles cleaning, deduplication, and the 80/20 split.

Invoke your implementation with:
    python -m framework.prepare_data --topic "your_topic" --n-samples 600

If you'd rather collect data from existing documents, put .txt or .jsonl
files in data/raw/ and run `make data` instead. You can leave this file
as `NotImplementedError`.
"""
from __future__ import annotations

from typing import List, Dict


def generate_samples(topic: str, n_samples: int) -> List[Dict[str, str]]:
    """Generate `n_samples` training texts for `topic` via the LLM API.

    Args:
        topic:      short description of your chosen domain,
                    for example "cooking recipes" or "Shakespearean sonnets".
        n_samples:  number of text samples to produce (>= 500 is required).

    Returns:
        A list of dicts shaped like:
            [{"text": "sample 1..."}, {"text": "sample 2..."}, ...]

    TODO:
      1. Design a system prompt that pins down style, length, and topic.
      2. Vary the user prompt across calls so samples are diverse.
      3. Call the LLM API. Use either the `anthropic` or `openai` client library.
         Read the API key from an environment variable; never hard-code it.
      4. Parse each response into one text sample (or several, if you prompt
         the model to produce a batch).
      5. Return the full list. The framework takes it from there.

    Hints:
      - Budget your API calls. Batching multiple samples per call is cheaper.
      - Save raw API responses to data/raw/ so re-splitting is reproducible.
      - Basic filtering (length > 40 chars, no duplicates) happens in the
        framework, so don't worry about it here.
    """
    raise NotImplementedError(
        "Implement generate_samples(), or put .txt/.jsonl files in data/raw/ "
        "and run `make data` instead."
    )
