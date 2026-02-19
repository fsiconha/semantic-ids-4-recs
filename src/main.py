import logging
import os

# from load/get_data import DataLoader, SemanticConfig
from quantizer.train_rqvae import run_rqvae_pipeline
from transform.build_sequences import SequencesBuilder
from transform.build_embeddings import EmbeddingsBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def run_semantic_pipeline() -> None:
    def env_true(name: str, default: str = "true") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}

    def env_default(name: str, default: str) -> str:
        return os.getenv(name, default)

    run_embeddings = env_true("RUN_EMBEDDINGS", default="true")
    run_rqvae = env_true("RUN_RQVAE", default="true")
    run_sequences = env_true("RUN_BUILD_SEQUENCES", default="true")
    run_finetune_vocab = env_true("RUN_FINETUNE_VOCAB", default="false")
    run_finetune_full = env_true("RUN_FINETUNE_FULL", default="false")

    # semantic_config = SemanticConfig()
    # data_loader = DataLoader(semantic_config)
    items_dict = data_loader.fetch_items()

    total = len(items_dict)
    logging.info("Gerados %s itens unicos", total)

    if run_embeddings:
        embeddings_builder = EmbeddingsBuilder()
        embeddings_builder.run(items_dict)

    if run_rqvae:
        run_rqvae_pipeline()

    if run_sequences:
        sequences_builder = SequencesBuilder()
        sequences_builder.run()

    if run_finetune_vocab:
        from finetune_qwen3_vocab import main as finetune_vocab

        logging.info(
            "Stage-1 CPU settings: model=%s seq=%s batch=%s steps=%s lr=%s 4bit=%s force_cpu=%s",
            env_default("FT_MODEL_NAME", "unsloth/Qwen3-0.6B"),
            env_default("FT_MAX_SEQ_LEN", "512"),
            env_default("FT_BATCH_SIZE", "1"),
            env_default("FT_MAX_STEPS", "200"),
            env_default("FT_LEARNING_RATE", "5e-4"),
            env_default("FT_USE_4BIT", "false"),
            env_default("FT_FORCE_CPU", "true"),
        )
        finetune_vocab()

    if run_finetune_full:
        from finetune_qwen3_full import STAGE1_DEFAULT_MODEL_PATH, main as finetune_full

        logging.info(
            "Stage-2 CPU settings: seq=%s batch=%s epochs=%s lr=%s 4bit=%s force_cpu=%s",
            env_default("FT_MAX_SEQ_LEN", "512"),
            env_default("FT_BATCH_SIZE", "1"),
            env_default("FT_NUM_EPOCHS", "1"),
            env_default("FT_LEARNING_RATE", "1e-4"),
            env_default("FT_USE_4BIT", "false"),
            env_default("FT_FORCE_CPU", "true"),
        )
        logging.info("Stage-2 base model path: %s", STAGE1_DEFAULT_MODEL_PATH)
        finetune_full()


if __name__ == "__main__":
    run_semantic_pipeline()
