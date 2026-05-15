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
import random
from pathlib import Path
from typing import List, Dict


def load_env_file(env_path: Path) -> None:
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


def get_random_prompt(topic: str, n_samples: int) -> str:
    prompts = [
        (
            f"I need to generate data to fine tune an LLM on {topic}. "
            f"The data should be structured as a JSON object with a "
            f"\"samples\" array, where each entry has a \"text\" field, "
            f"like this:\n"
            f"{{\"samples\": [\n"
            f"  {{\"text\": \"{{Thing related to {topic}}}: "
            f"{{detailed explanation, materials, and steps}}\"}},\n"
            f"  {{\"text\": \"...\"}}\n"
            f"]}}\n\n"
            f"I need {n_samples} samples. Use real, concrete examples "
            f"drawn from {topic}; every example must be unique and cover "
            f"a different sub-topic, technique, or scenario."
        ),
        (
            f"I need to generate data to fine tune an LM on {topic}. "
            f"The data should be structured as a JSON object with a "
            f"\"samples\" array, where each entry has \"input\" and "
            f"\"output\" fields, like this:\n"
            f"{{\"samples\": [\n"
            f"  {{\"input\": \"{{question asked by user about "
            f"{topic}}}\", \"output\": \"{{detailed answer about "
            f"{topic}}}\"}},\n"
            f"  {{\"input\": \"...\", \"output\": \"...\"}}\n"
            f"]}}\n\n"
            f"I need {n_samples} unique entries. Use real, concrete "
            f"examples drawn from {topic}; cover a wide range of "
            f"questions and scenarios so no two entries overlap."
        ),
        (
            f"I'm fine tuning a GPT-2 style model. I need you to "
            f"generate data for it. I'm fine tuning the model on "
            f"{topic}, so all data should be centered around {topic}. "
            f"The structure should be a JSON object with a \"samples\" "
            f"array, where each entry has \"instruction\" and \"output\" "
            f"fields, like this:\n"
            f"{{\"samples\": [\n"
            f"  {{\"instruction\": \"{{question or task about "
            f"{topic}}}\", \"output\": \"{{detailed answer about "
            f"{topic}}}\"}},\n"
            f"  {{\"instruction\": \"...\", \"output\": \"...\"}}\n"
            f"]}}\n\n"
            f"I need {n_samples} unique instructions. The returned "
            f"object must contain only the JSON; no commentary, no "
            f"markdown fences. Use a wide range of {topic}-related "
            f"tasks so every instruction is distinct."
        ),
    ]
    return random.choice(prompts)


def generate_samples(topic: str, n_samples: int) -> List[Dict[str, str]]:
    """Generate `n_samples` training texts for `topic` via the LLM API.

    Splits `n_samples` into 10 batches and makes only 10 calls to the
    LLM. Each call uses a random prompt from `get_random_prompt` so the
    style varies across batches. Returns the combined results list.
    """
    project_root = Path(__file__).resolve().parents[2]
    load_env_file(project_root / ".env")

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY not set. Add it to .env at the project root."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    n_calls = 10
    batch_size = math.ceil(n_samples / n_calls)

    system_prompt = (
        f"You generate diverse training data about: {topic}. "
        f"Each sample is 80-200 words, self-contained, factual, and "
        f"stylistically varied across samples. "
        f"Output JSON only - no commentary, no markdown."
    )

    results: List[Dict[str, str]] = []

    for _ in range(n_calls):
        user_message = get_random_prompt(topic, batch_size)

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

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for item in payload.get("samples", []):
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                results.append({"text": item["text"]})
            elif isinstance(item.get("instruction"), str) and isinstance(item.get("output"), str):
                rec = {"instruction": item["instruction"], "output": item["output"]}
                if isinstance(item.get("input"), str) and item["input"].strip():
                    rec["input"] = item["input"]
                results.append(rec)
            elif isinstance(item.get("input"), str) and isinstance(item.get("output"), str):
                results.append({"instruction": item["input"], "output": item["output"]})

    return results[:n_samples]
