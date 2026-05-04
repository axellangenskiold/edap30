# Student Kit Makefile (conda edition)
# Run `make help` to see all available commands.
#
# Requires conda (Anaconda or Miniconda). Install from:
#   https://docs.anaconda.com/miniconda/

SHELL       := /bin/bash
ENV_NAME    ?= nlppy3.11
PY_VERSION  ?= 3.11
BASE_MODEL  ?= gpt2

# Resolve conda even when `conda` is a shell function and not on PATH.
# Falls back gracefully if CONDA_EXE isn't set.
CONDA_EXE   ?= $(shell command -v conda 2>/dev/null || echo conda)
CONDA_BASE  := $(shell $(CONDA_EXE) info --base 2>/dev/null)

# `conda activate` only works inside a sourced shell hook, so every recipe
# sources conda.sh first. The "&& conda activate $(ENV_NAME) && <cmd>" form
# keeps everything inside one recipe line.
ACT         := source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(ENV_NAME)

.DEFAULT_GOAL := help

.PHONY: help check-conda setup check data train-reference train-scratch \
        train-lora train eval all clean clean-all

help:
	@echo ""
	@echo "Student Kit: Language Model + LoRA Assignment  (conda)"
	@echo ""
	@echo "Requires conda. Current env: $(ENV_NAME)  (Python $(PY_VERSION))"
	@echo ""
	@echo "Recommended workflow:"
	@echo "  1.  make setup              # one-time: create conda env + deps + base model"
	@echo "  2.  make train-reference    # sanity-check pipeline with the toy RNN"
	@echo "  3.  Implement the TODOs in student/"
	@echo "  4.  make check              # verify your student code imports"
	@echo "  5.  make all                # data + train scratch + train lora + eval"
	@echo ""
	@echo "All targets:"
	@echo "  make setup              Create conda env '$(ENV_NAME)', install deps, cache base model"
	@echo "  make check              Quick import check of student modules"
	@echo "  make data               Build data/train.jsonl + data/eval.jsonl"
	@echo "  make train-reference    Train the toy RNN reference model"
	@echo "  make train-scratch      Train your custom LM from scratch"
	@echo "  make train-lora         LoRA fine-tune the base model"
	@echo "  make train              train-scratch + train-lora"
	@echo "  make eval               Evaluate base + scratch + lora on eval set"
	@echo "  make all                data -> train -> eval"
	@echo "  make clean              Remove outputs and checkpoints"
	@echo "  make clean-all          Also remove the conda env"
	@echo ""
	@echo "Overridable variables:"
	@echo "  ENV_NAME    (default: nlppy3.11)"
	@echo "  PY_VERSION  (default: 3.11)"
	@echo "  BASE_MODEL  (default: gpt2)"
	@echo ""
	@echo "Example:"
	@echo "  make setup ENV_NAME=my-lm PY_VERSION=3.11"
	@echo ""

# Guard: make sure conda is available.
check-conda:
	@command -v $(CONDA_EXE) >/dev/null 2>&1 || { \
	  echo "ERROR: conda not found. Install Miniconda or Anaconda first."; \
	  echo "  https://docs.anaconda.com/miniconda/"; \
	  exit 1; \
	}
	@test -n "$(CONDA_BASE)" || { \
	  echo "ERROR: could not resolve conda base directory."; \
	  exit 1; \
	}

# One-time setup.
setup: check-conda
	@echo ">>> Creating conda env '$(ENV_NAME)' (Python $(PY_VERSION))"
	@if $(CONDA_EXE) env list | awk '{print $$1}' | grep -qx "$(ENV_NAME)"; then \
	  echo "    env '$(ENV_NAME)' already exists; reusing."; \
	else \
	  $(CONDA_EXE) create -y -n $(ENV_NAME) python=$(PY_VERSION) pip; \
	fi
	@echo ">>> Installing requirements into '$(ENV_NAME)'"
	$(ACT) && pip install --upgrade pip && pip install -r requirements.txt
	@echo ">>> Caching base model: $(BASE_MODEL)"
	$(ACT) && python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
	  AutoTokenizer.from_pretrained('$(BASE_MODEL)'); \
	  AutoModelForCausalLM.from_pretrained('$(BASE_MODEL)'); \
	  print('Cached $(BASE_MODEL)')"
	@echo ""
	@echo ">>> Done. To activate the env manually:"
	@echo "        conda activate $(ENV_NAME)"

# Sanity check and data.
check: check-conda
	$(ACT) && python -m framework.check

data: check-conda
	$(ACT) && python -m framework.prepare_data \
	  --raw-dir data/raw --out-dir data

# Training.
train-reference: check-conda
	$(ACT) && python -m framework.train_scratch --config configs/reference.yaml

train-scratch: check-conda
	$(ACT) && python -m framework.train_scratch --config configs/scratch.yaml

train-lora: check-conda
	$(ACT) && python -m framework.train_lora --config configs/lora.yaml

train: train-scratch train-lora

# Evaluation.
eval: check-conda
	$(ACT) && python -m framework.evaluate \
	  --eval-file data/eval.jsonl --out-dir outputs

# Full pipeline.
all: data train eval

# Cleanup.
clean:
	rm -rf outputs/*.txt outputs/*.json outputs/training_logs
	rm -rf checkpoints/scratch/*.pt checkpoints/reference/*.pt
	rm -rf checkpoints/lora/epoch_* checkpoints/lora/adapter_* \
	       checkpoints/lora/tokenizer* checkpoints/lora/special_tokens* \
	       checkpoints/lora/vocab* checkpoints/lora/merges*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean check-conda
	@echo ">>> Removing conda env '$(ENV_NAME)'"
	-$(CONDA_EXE) env remove -y -n $(ENV_NAME)
