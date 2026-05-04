# Student Kit: Language Model + LoRA Assignment

Everything you need to complete the assignment. Run `make help` for commands.

Only `make setup` touches the network. After that, the whole pipeline runs
offline on the local machine or server.

## Layout

```
student_kit/
    Makefile                  you talk to the kit through here
    README.md                 this file
    requirements.txt

    reference/                READ-ONLY study material
        toy_rnn.py            complete LSTM LM that shows the required interface

    student/                  YOU WORK HERE
        model.py              TODO: your custom language model
        collect_data.py       TODO: LLM-API data generation
        lora_config.py        TODO: LoRA hyperparameters
        metrics.py            TODO: cross-entropy + perplexity

    framework/                DON'T TOUCH: training and eval plumbing
    configs/                  YAML hyperparameters (you can tune these)
    data/raw/                 drop your raw dataset here
    checkpoints/              filled by training
    outputs/                  filled by evaluation
```

## Quick Start

```bash
# 1. one-time install (needs network)
make setup

# 2. confirm the pipeline works end-to-end using the reference toy RNN
#    (drop any .jsonl/.txt files into data/raw/ first, or use the provided example)
make data
make train-reference
make eval

# 3. now implement the four TODOs in student/
make check                   # verify your code imports cleanly

# 4. run your full pipeline
make all                     # data + train-scratch + train-lora + eval
```

## The Four Things You Implement

Each file under `student/` has a docstring with the exact signature, inputs,
outputs, and hints. In summary:

| File                      | What to implement                                | Reference              |
| ------------------------- | ------------------------------------------------ | ---------------------- |
| `student/model.py`        | `CustomLM`, your own GPT-style LM                | `reference/toy_rnn.py` |
| `student/collect_data.py` | `generate_samples(topic, n)` via LLM API         | docstring              |
| `student/lora_config.py`  | `get_lora_config()` returning `LoraConfig`       | PEFT docs              |
| `student/metrics.py`      | `compute_cross_entropy` and `compute_perplexity` | docstring              |

The framework calls these functions with fixed argument shapes and expects
specific return types. As long as you respect the signatures, everything
else works automatically.

## Dataset Format

`make data` reads everything in `data/raw/` and produces an 80/20 split
into `data/train.jsonl` and `data/eval.jsonl`. The following file formats
are supported.

**Plain text JSONL** (one JSON object per line):

```json
{"text": "Your domain-specific sample here."}
{"text": "Another sample..."}
```

**Plain text JSON array** (a single `.json` file):

```json
[
  { "text": "Your domain-specific sample here." },
  { "text": "Another sample..." }
]
```

**Instruction-tuning data** (in either `.jsonl` or `.json` form). Records
follow the Alpaca shape and may optionally include an `input` field:

```json
[
  {
    "instruction": "Explain LoRA in simple terms.",
    "output": "LoRA stands for Low-Rank Adaptation..."
  },
  {
    "instruction": "Translate the following sentence into French.",
    "input": "The cat sits on the mat.",
    "output": "Le chat est assis sur le tapis."
  }
]
```

Instruction records are converted into a single training string using the
Alpaca template before training:

```
### Instruction:
{instruction}

### Input:
{input}            (only if non-empty)

### Response:
{output}
```

You can also drop a plain `.txt` file in `data/raw/`; it becomes one sample.

If you prefer to generate samples via the provided LLM API, implement
`student/collect_data.py` and replace `make data` with:

```bash
. .venv/bin/activate
python -m framework.prepare_data --topic "your_topic" --n-samples 600
```

## Windows users

Use WSL or Git Bash; the Makefile relies on bash features.

## Submission

When you're done, keep the same folder layout, make sure `checkpoints/` and
`outputs/metrics.json` are in place, add `report.pdf` at the top level, and
zip the folder as described in the assignment PDF.
