"""LoRA configuration.

You implement ONE function: `get_lora_config()`.
The framework attaches the adapter to the base model and handles the
training loop.
"""
from __future__ import annotations

from peft import LoraConfig, TaskType


def get_lora_config() -> LoraConfig:
    """Return a peft.LoraConfig for fine-tuning a GPT-2-like base model.

    TODO:
      - Choose values for r (rank), lora_alpha, lora_dropout.
      - Pick target_modules. For GPT-2 family: ["c_attn"].
                             For Llama family:  ["q_proj", "v_proj"].
      - Use task_type=TaskType.CAUSAL_LM, bias="none".

    Starting point (feel free to tune):
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["c_attn"]

    Returns:
        A peft.LoraConfig instance.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
