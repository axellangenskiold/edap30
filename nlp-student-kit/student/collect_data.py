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

import json
import math
import os
from pathlib import Path
from typing import List, Dict


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


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
    kit_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[2]
    _load_env_file(project_root / ".env")

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY not set. Add it to .env at the project root."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    batch_size = max(1, n_samples // 10)
    n_calls = math.ceil(n_samples / batch_size)

    system_prompt = (
        f"You generate diverse training data about: {topic}. "
        f"Each sample is 80-200 words, self-contained, factual, and "
        f"stylistically varied across samples. "
        f"Output JSON only — no commentary, no markdown."
    )

    samples: List[Dict[str, str]] = []
    raw_responses: List[str] = []

    for _ in range(n_calls):
        prompt = get_random_prompt(topic, batch_size)
        user_message = (prompt)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        content = response.choices[0].message.content
        raw_responses.append(content)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for item in payload.get("samples", []):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                samples.append({"text": item["text"]})

        if len(samples) >= n_samples:
            break

    raw_dir = kit_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_topic = topic.replace(" ", "_").replace("/", "_")
    (raw_dir / f"{safe_topic}_raw.json").write_text(
        json.dumps(raw_responses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return samples[:n_samples]


def get_random_prompt(topic: str, n_samples: int) -> str:
    """Return a random prompt string for the given topic, to encourage diversity."""
        
    prompts = [
        f"Write a detailed how-to guide related to {topic}.",
        f"Give me an explanation of a concept in {topic}.",
        f"Give an instruction on how to do {topic}.",
        f"Provide a step-by-step tutorial on a specific aspect of {topic}.",
        f"Describe a common problem and solution in {topic}.",
    ]
    
    json_format = """
        The output should be an array of objects, each with a "text" field containing the sample text, like this:
        [{{"text": "sample 1..."}}, {{"text": "sample 2..."}}, ...]
        OR
        [{{"instruction": "...", "output": "..."}},
            {{"instruction": "...", "input": "...", "output": "..."}},
            ...]
        """
        
    import random
        
    return f"{random.choice(prompts)}\n\n{json_format}"