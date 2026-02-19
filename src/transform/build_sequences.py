#!/usr/bin/env python3
"""
Build semantic ID sequences and conversation datasets for LLM fine-tuning.

Inputs:
- data/output/item_id_to_sid.jsonl (item_id -> sid_tuple)
- input data from DataLoader

Outputs (data/output):
- semantic_sequences.jsonl
- conversations_train.jsonl
- conversations_val.jsonl
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from load.get_data import DataLoader, SemanticConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build-sequences")


@dataclass
class BuildConfig:
    item_id_to_sid_path: Path = Path("data/output/item_id_to_sid.jsonl")
    output_dir: Path = Path("data/output")
    codebook_size: int = 256
    num_levels: int = 4
    max_history_len: int = 20
    val_split: float = 0.05
    random_seed: int = 42


class SequencesBuilder:
    def __init__(self, config: Optional[BuildConfig] = None) -> None:
        self.config = config or BuildConfig()

    def load_item_id_to_sid(self, path: Path) -> Dict[str, List[int]]:
        logger.info("Loading item_id_to_sid from %s", path)
        mapping: Dict[str, List[int]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                mapping[str(record["item_id"])] = record["sid_tuple"]
        logger.info("Loaded %s item->sid mappings", len(mapping))
        return mapping

    @staticmethod
    def parse_sequence(raw_sequence: object) -> List[str]:
        if raw_sequence is None:
            return []
        if isinstance(raw_sequence, list):
            return [str(item) for item in raw_sequence]
        if isinstance(raw_sequence, str):
            text = raw_sequence.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    pass
            if "," in text:
                return [part.strip() for part in text.split(",") if part.strip()]
            return [part for part in text.split() if part]
        return [str(raw_sequence)]

    @staticmethod
    def sid_tuple_to_tokens(
        sid_tuple: Sequence[int],
        codebook_size: int,
        num_levels: int
    ) -> List[int]:
        tokens: List[int] = []
        for level in range(num_levels):
            value = int(sid_tuple[level]) if level < len(sid_tuple) else 0
            tokens.append(level * codebook_size + value)
        return tokens

    @staticmethod
    def format_sid_tokens(tokens: Sequence[int]) -> str:
        inner = "".join([f"<|sid_{token}|>" for token in tokens])
        return f"<|sid_start|>{inner}<|sid_end|>"

    def build_semantic_sequences(
        self,
        sequences_map: Dict[str, object],
        item_id_to_sid: Dict[str, List[int]],
    ) -> List[dict]:
        output: List[dict] = []
        missing_items = 0

        for user_id, raw_sequence in sequences_map.items():
            item_ids = self.parse_sequence(raw_sequence)
            if not item_ids:
                continue

            semantic_tokens: List[str] = []
            for item_id in item_ids:
                sid_tuple = item_id_to_sid.get(str(item_id))
                if sid_tuple is None:
                    missing_items += 1
                    continue
                tokens = self.sid_tuple_to_tokens(
                    sid_tuple,
                    self.config.codebook_size,
                    self.config.num_levels,
                )
                semantic_tokens.append(self.format_sid_tokens(tokens))

            if len(semantic_tokens) < 2:
                continue

            output.append(
                {
                    "user_id": str(user_id),
                    "sequence": semantic_tokens,
                }
            )

        logger.info("Built %s semantic sequences", len(output))
        if missing_items:
            logger.info(
                "Skipped %s items not found in sid mapping", missing_items
            )
        return output

    def build_conversations(self, semantic_sequences: Iterable[dict]) -> List[dict]:
        conversations: List[dict] = []
        for row in semantic_sequences:
            tokens = row["sequence"]
            user_id = row["user_id"]

            for idx in range(1, len(tokens)):
                start_idx = max(0, idx - self.config.max_history_len)
                history = tokens[start_idx:idx]
                target = tokens[idx]

                prompt = f"User bought: {', '.join(history)}\n<|rec|>"
                text = f"{prompt} {target}"

                conversations.append(
                    {
                        "user_id": user_id,
                        "history": history,
                        "target": target,
                        "prompt": prompt,
                        "text": text,
                    }
                )

        logger.info("Built %s conversation rows", len(conversations))
        return conversations

    @staticmethod
    def save_jsonl(records: Iterable[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def run(self) -> None:
        data_loader_config = SemanticConfig()
        data_loader = DataLoader(data_loader_config)

        item_id_to_sid = self.load_item_id_to_sid(
            self.config.item_id_to_sid_path
        )
        sequences_map = data_loader.fetch_user_sequences()

        semantic_sequences = self.build_semantic_sequences(
            sequences_map,
            item_id_to_sid
        )
        semantic_sequences_path = self.config.output_dir / "semantic_sequences.jsonl"
        self.save_jsonl(semantic_sequences, semantic_sequences_path)
        logger.info("Saved semantic sequences to %s", semantic_sequences_path)

        conversations = self.build_conversations(semantic_sequences)
        random.Random(self.config.random_seed).shuffle(conversations)

        val_size = max(1, int(len(conversations) * self.config.val_split)) if conversations else 0
        val_rows = conversations[:val_size]
        train_rows = conversations[val_size:]

        train_path = self.config.output_dir / "conversations_train.jsonl"
        val_path = self.config.output_dir / "conversations_val.jsonl"

        self.save_jsonl(train_rows, train_path)
        self.save_jsonl(val_rows, val_path)

        logger.info("Saved %s train rows to %s", len(train_rows), train_path)
        logger.info("Saved %s val rows to %s", len(val_rows), val_path)
