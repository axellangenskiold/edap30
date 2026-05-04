"""Dataset classes. Do not edit."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .utils import read_jsonl


class TokenBlocksDataset(Dataset):
    """Concatenates all texts and samples contiguous blocks of length
    block_size+1 (inputs plus a one-position-shifted target). Used by the
    from-scratch trainer.
    """

    def __init__(self, path: str, tokenizer, block_size: int):
        ids: list[int] = []
        eos = tokenizer.eos_token_id
        for row in read_jsonl(path):
            ids.extend(tokenizer.encode(row["text"]))
            ids.append(eos)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size - 1)

    def __getitem__(self, i: int):
        chunk = self.data[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class CausalLMDataset(Dataset):
    """Pads or truncates each example to max_len. Used by the LoRA trainer
    with Hugging Face models, so we return the dict that HF expects.
    """

    def __init__(self, path: str, tokenizer, max_len: int):
        self.rows = list(read_jsonl(path))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        enc = self.tok(
            self.rows[i]["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention = enc["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention == 0] = -100
        return {"input_ids": input_ids,
                "attention_mask": attention,
                "labels": labels}
