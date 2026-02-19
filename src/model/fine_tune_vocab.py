#!/usr/bin/env python3
"""
Stage 1: Extend vocabulary with semantic ID tokens and train embeddings only.
CPU-friendly defaults using Transformers backend; Unsloth optional.
"""

import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from .prompts import SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("finetune-qwen3-vocab")


@dataclass
class VocabConfig:
    model_name: str = "unsloth/Qwen3-0.6B"
    train_path: Path = Path("data/output/conversations_train.jsonl")
    output_dir: Path = Path("model/artifacts/qwen3_vocab_stage1")
    max_seq_length: int = 512
    num_semantic_tokens: int = 1024
    use_4bit: bool = False
    force_cpu_load: bool = True
    batch_size: int = 1
    max_steps: int = 200
    learning_rate: float = 5e-4
    warmup_steps: int = 20
    logging_steps: int = 10


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


def env_backend(force_cpu: bool) -> str:
    if force_cpu:
        return env_str("FT_BACKEND", "transformers").lower()
    return env_str("FT_BACKEND", "unsloth").lower()


def build_sid_tokens(num_semantic_tokens: int) -> List[str]:
    tokens = ["<|sid_start|>", "<|sid_end|>", "<|rec|>"]
    tokens.extend([f"<|sid_{i}|>" for i in range(num_semantic_tokens)])
    return tokens


def build_text(example: dict) -> dict:
    history = example.get("history", [])
    target = example.get("target", "")

    if isinstance(history, list):
        history_text = ", ".join(history)
    else:
        history_text = str(history)

    prompt = f"User bought: {history_text}\n<|rec|>"
    example["text"] = f"{SYSTEM_PROMPT}\n\n{prompt} {target}".strip()
    return example


def freeze_non_embeddings(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False
    embeddings = model.get_input_embeddings()
    embeddings.weight.requires_grad = True


def main() -> None:
    config = VocabConfig()
    config.model_name = env_str("FT_MODEL_NAME", config.model_name)
    config.max_seq_length = env_int("FT_MAX_SEQ_LEN", config.max_seq_length)
    config.batch_size = env_int("FT_BATCH_SIZE", config.batch_size)
    config.max_steps = env_int("FT_MAX_STEPS", config.max_steps)
    config.learning_rate = env_float("FT_LEARNING_RATE", config.learning_rate)
    config.warmup_steps = env_int("FT_WARMUP_STEPS", config.warmup_steps)
    config.logging_steps = env_int("FT_LOGGING_STEPS", config.logging_steps)
    config.use_4bit = env_bool("FT_USE_4BIT", config.use_4bit)
    config.force_cpu_load = env_bool("FT_FORCE_CPU", config.force_cpu_load)

    backend = env_backend(config.force_cpu_load)

    num_threads = os.getenv("FT_NUM_THREADS")
    if num_threads:
        torch.set_num_threads(int(num_threads))
    num_interop = os.getenv("FT_INTEROP_THREADS")
    if num_interop:
        torch.set_num_interop_threads(int(num_interop))

    logger.info("Loading dataset from %s", config.train_path)
    dataset = load_dataset("json", data_files=str(config.train_path), split="train")
    dataset = dataset.map(
        build_text,
        remove_columns=[c for c in dataset.column_names if c != "text"]
    )

    logger.info("Loading base model %s (backend=%s)", config.model_name, backend)
    device_map = "cpu" if config.force_cpu_load else None
    if backend == "unsloth":
        from unsloth import FastLanguageModel, add_new_tokens, is_bfloat16_supported

        dtype = "bfloat16" if is_bfloat16_supported() else None
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=config.use_4bit,
            device_map=device_map,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.float32,
            device_map=device_map,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not config.force_cpu_load:
        logger.info("Moving model to CPU for add_new_tokens to avoid OOM")
        model = model.to("cpu")

    new_tokens = build_sid_tokens(config.num_semantic_tokens)
    logger.info("Adding %s new tokens", len(new_tokens))
    if backend == "unsloth":
        num_added = add_new_tokens(model, tokenizer, new_tokens)
    else:
        num_added = tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
    logger.info("Tokenizer size increased by %s", num_added)

    if backend == "unsloth" and device == "cuda" and not config.force_cpu_load:
        logger.info("Moving model back to GPU after add_new_tokens")
        model = model.to("cuda")

    freeze_non_embeddings(model)

    sft_kwargs = {
        "dataset_text_field": "text",
        "max_length": config.max_seq_length,
        "per_device_train_batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "max_steps": config.max_steps,
        "logging_steps": config.logging_steps,
        "output_dir": str(config.output_dir),
        "report_to": "none",
    }
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    if config.force_cpu_load:
        if "use_cpu" in sft_params:
            sft_kwargs["use_cpu"] = True
        elif "no_cuda" in sft_params:
            sft_kwargs["no_cuda"] = True

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": sft_config,
    }
    if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir / "final"))
    tokenizer.save_pretrained(str(config.output_dir / "final"))

    metadata_path = config.output_dir / "vocab_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump({"num_added_tokens": num_added}, handle)

    logger.info(
        "Stage 1 completed. Model saved to %s", config.output_dir / "final"
    )


if __name__ == "__main__":
    main()
