#!/usr/bin/env python3
"""
Stage 2: Full fine-tuning using the extended vocabulary checkpoint.
CPU-friendly defaults using Transformers + PEFT; Unsloth optional.
"""

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from datasets import load_dataset
import torch
from trl import SFTConfig, SFTTrainer

from .prompts import SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("finetune-qwen3-full")


STAGE1_DEFAULT_MODEL_PATH = Path("model/artifacts/qwen3_vocab_stage1/final")


@dataclass
class FullConfig:
    model_path: Path = STAGE1_DEFAULT_MODEL_PATH
    train_path: Path = Path("data/output/conversations_train.jsonl")
    val_path: Path = Path("data/output/conversations_val.jsonl")
    output_dir: Path = Path("model/artifacts/qwen3_full")
    max_seq_length: int = 512
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    warmup_steps: int = 20
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_4bit: bool = False
    force_cpu_load: bool = True


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


def main() -> None:
    config = FullConfig()
    config.max_seq_length = env_int("FT_MAX_SEQ_LEN", config.max_seq_length)
    config.batch_size = env_int("FT_BATCH_SIZE", config.batch_size)
    config.learning_rate = env_float("FT_LEARNING_RATE", config.learning_rate)
    config.num_train_epochs = env_int("FT_NUM_EPOCHS", config.num_train_epochs)
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

    if not config.model_path.exists():
        raise FileNotFoundError(
            f"Stage-1 model not found at {config.model_path}. "
            "Run finetune_qwen3_vocab.py first or update FullConfig.model_path."
        )

    logger.info("Loading datasets")
    train_dataset = load_dataset(
        "json", data_files=str(config.train_path), split="train"
    )
    val_dataset = load_dataset(
        "json", data_files=str(config.val_path), split="train"
    )

    train_dataset = train_dataset.map(
        build_text,
        remove_columns=[c for c in train_dataset.column_names if c != "text"]
    )
    val_dataset = val_dataset.map(
        build_text,
        remove_columns=[c for c in val_dataset.column_names if c != "text"]
    )

    logger.info("Loading stage-1 model from %s (backend=%s)", config.model_path, backend)
    device_map = "cpu" if config.force_cpu_load else None
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if backend == "unsloth":
        from unsloth import FastLanguageModel, is_bfloat16_supported

        dtype = "bfloat16" if is_bfloat16_supported() else None
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(config.model_path),
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=config.use_4bit,
            device_map=device_map,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=True,
        )
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path), use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(config.model_path),
            dtype=torch.float32,
            device_map=device_map,
        )

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    sft_kwargs = {
        "dataset_text_field": "text",
        "max_length": config.max_seq_length,
        "per_device_train_batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "num_train_epochs": config.num_train_epochs,
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
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": sft_config,
    }
    if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir / "final"))
    tokenizer.save_pretrained(str(config.output_dir / "final"))

    logger.info(
        "Stage 2 completed. Model saved to %s", config.output_dir / "final"
    )


if __name__ == "__main__":
    main()
