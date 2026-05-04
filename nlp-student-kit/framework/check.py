"""`make check` target: verify that the student modules can be imported and
their expected functions exist. Catches typos before students launch training.
"""
from __future__ import annotations

import sys
import importlib


def main() -> int:
    failures: list[str] = []

    def require(name: str, path: str, attr: str):
        try:
            m = importlib.import_module(path)
        except Exception as e:
            failures.append(f"  [{name}] cannot import {path}: {e!r}")
            return
        if not hasattr(m, attr):
            failures.append(f"  [{name}] {path} is missing `{attr}`")
            return
        print(f"  [{name}] OK -> {path}.{attr}")

    print("Checking student modules...")
    require("model",        "student.model",        "build_model")
    require("model",        "student.model",        "CustomLM")
    require("model",        "student.model",        "config_to_dict")
    require("collect_data", "student.collect_data", "generate_samples")
    require("lora_config",  "student.lora_config",  "get_lora_config")
    require("metrics",      "student.metrics",      "compute_cross_entropy")
    require("metrics",      "student.metrics",      "compute_perplexity")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f)
        print("\nFix the above, then rerun `make check`.")
        return 1

    print("\nAll student modules present. (Runtime behavior is still up to you.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
