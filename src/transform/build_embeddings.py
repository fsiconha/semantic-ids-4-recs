import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@dataclass
class EmbedConfig:
    batch_size: int = 100
    limit_rows: Optional[int] = None
    output_path: str = "data/output/items_with_embeddings.jsonl"
    model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    verify_consistency: bool = False


class EmbeddingsBuilder:
    def __init__(self, embed_config: Optional[EmbedConfig] = None) -> None:
        self.embed_config = embed_config or EmbedConfig()
        self.embeddings_client = SentenceTransformer(
            self.embed_config.model_name
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        effective_batch_size = (
            min(self.embed_config.batch_size, len(texts)) if texts else 1
        )
        embeddings = self.embeddings_client.encode(
            texts,
            batch_size=effective_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.tolist()

    @staticmethod
    def chunked(
        items: List[Tuple[str, str]],
        batch_size: int,
    ) -> Iterable[List[Tuple[str, str]]]:
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    @staticmethod
    def build_item_texts(items_map: dict) -> List[Tuple[str, str]]:
        items = [
            (item_id, desc) for item_id, desc in items_map.items() if desc
        ]
        return [
            (item_id, f"Descricao: {desc.strip()}")
            for item_id, desc in items
        ]

    def generate_embeddings(
        self,
        items: List[Tuple[str, str]],
        batch_size: int,
    ) -> List[Tuple[str, str, List[float]]]:
        output: List[Tuple[str, str, List[float]]] = []
        total_items = len(items)
        start_time = time.time()

        for _, batch in enumerate(self.chunked(items, batch_size), start=1):
            batch_ids = [item_id for item_id, _ in batch]
            batch_texts = [text for _, text in batch]

            batch_embeddings = self.embed_documents(batch_texts)
            for item_id, text, embedding in zip(
                batch_ids, batch_texts, batch_embeddings
            ):
                output.append((item_id, text, embedding))

            processed = len(output)
            if processed == total_items or processed % (batch_size * 5) == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                logging.info(
                    "Processed %s/%s items (%.1f items/sec)",
                    processed,
                    total_items,
                    rate,
                )

        return output

    def verify_embedding_consistency(
        self,
        items: List[Tuple[str, str]],
        batch_size: int,
    ) -> None:
        if not items:
            logging.warning("No items available for consistency check")
            return

        check_batch_size = min(batch_size, len(items))
        batch = items[:check_batch_size]
        batch_texts = [text for _, text in batch]

        batch_embeddings = self.embed_documents(batch_texts)
        single_embedding = self.embed_documents([batch_texts[0]])[0]

        are_similar = np.allclose(
            batch_embeddings[0],
            single_embedding,
            rtol=1e-6,
            atol=1e-6,
        )
        diff = np.abs(
            np.array(batch_embeddings[0]) - np.array(single_embedding)
        )

        logging.info("Embeddings are similar: %s", are_similar)
        logging.info("Max difference: %.2e", diff.max())
        logging.info("Mean difference: %.2e", diff.mean())

        if not are_similar:
            logging.warning("Embeddings differ more than expected")
        else:
            logging.info("Embedding consistency verified")

    @staticmethod
    def save_embeddings(
        records: List[Tuple[str, str, List[float]]],
        output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for item_id, text, embedding in records:
                handle.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "text": text,
                            "embedding": embedding,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    def run(self, items_map: dict) -> None:
        items = self.build_item_texts(items_map)
        if self.embed_config.limit_rows is not None:
            items = items[: self.embed_config.limit_rows]
            logging.info("Limiting to %s items", len(items))

        if not items:
            logging.warning("No items found to embed")
            return

        logging.info(
            "Embedding %s items with model %s (batch_size=%s)",
            len(items),
            self.embed_config.model_name,
            self.embed_config.batch_size,
        )

        if self.embed_config.verify_consistency:
            self.verify_embedding_consistency(
                items,
                self.embed_config.batch_size,
            )

        records = self.generate_embeddings(items, self.embed_config.batch_size)

        output_path = Path(self.embed_config.output_path)
        self.save_embeddings(records, output_path)

        embedding_array = np.array(
            [embedding for _, _, embedding in records], dtype=np.float32
        )
        logging.info("Saved embeddings to %s", output_path)
        logging.info("Embeddings shape: %s", embedding_array.shape)
