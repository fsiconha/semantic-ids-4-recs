#!/usr/bin/env python3
"""
Train a Residual Quantized VAE (RQ-VAE) for hierarchical semantic IDs.
"""

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency
    KMeans = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train-rqvae")


@dataclass
class RQVAEConfig:
    """Configuration for RQ-VAE training."""

    # Data settings
    data_dir: Path = field(default_factory=lambda: Path("data"))
    embeddings_path: Optional[Path] = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "rqvae")
    val_split: float = 0.05

    # Model parameters
    item_embedding_dim: int = 768
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    codebook_embedding_dim: int = 32
    codebook_quantization_levels: int = 4
    codebook_size: int = 256
    commitment_weight: float = 0.25
    use_rotation_trick: bool = True

    # EMA VQ (optional)
    use_ema_vq: bool = False
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5

    # Training parameters
    batch_size: int = 4096
    gradient_accumulation_steps: int = 1
    num_epochs: int = 50
    max_lr: float = 3e-4
    min_lr: float = 1e-6
    scheduler_type: str = "cosine"
    warmup_steps: int = 200

    # Optimization options
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_kmeans_init: bool = True
    reset_unused_codes: bool = True
    steps_per_codebook_reset: int = 200
    codebook_usage_threshold: float = 0.02

    # Logging
    steps_per_train_log: int = 20
    steps_per_val_log: int = 200

    # Runtime
    num_workers: int = 2
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.embeddings_path is None:
            self.embeddings_path = self.data_dir / "output" / "items_with_embeddings.jsonl"

    def log_config(self) -> None:
        logger.info("=== RQ-VAE Configuration ===")
        logger.info("Data:")
        logger.info("  embeddings_path: %s", self.embeddings_path)
        logger.info("  checkpoint_dir: %s", self.checkpoint_dir)
        logger.info("  val_split: %s", self.val_split)
        logger.info("Model:")
        logger.info("  item_embedding_dim: %s", self.item_embedding_dim)
        logger.info("  encoder_hidden_dims: %s", self.encoder_hidden_dims)
        logger.info("  codebook_embedding_dim: %s", self.codebook_embedding_dim)
        logger.info("  codebook_quantization_levels: %s", self.codebook_quantization_levels)
        logger.info("  codebook_size: %s", self.codebook_size)
        logger.info("  commitment_weight: %s", self.commitment_weight)
        logger.info("  use_rotation_trick: %s", self.use_rotation_trick)
        logger.info("Training:")
        logger.info("  batch_size: %s", self.batch_size)
        logger.info("  gradient_accumulation_steps: %s", self.gradient_accumulation_steps)
        logger.info("  num_epochs: %s", self.num_epochs)
        logger.info("  scheduler_type: %s", self.scheduler_type)
        logger.info("  max_lr: %s", self.max_lr)
        logger.info("  min_lr: %s", self.min_lr)
        logger.info("============================")


class EmbeddingDatasetLoader(Dataset):
    def __init__(self, embeddings_path: Path, limit: Optional[int] = None) -> None:
        logger.info("Loading embeddings from %s", embeddings_path)
        embeddings: List[List[float]] = []
        item_ids: List[str] = []

        with embeddings_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= limit:
                    break
                record = json.loads(line)
                embeddings.append(record["embedding"])
                item_ids.append(record["item_id"])

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.item_ids = item_ids

        logger.info("Loaded %s embeddings of dimension %s", len(self.embeddings), self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tensor:
        return self.embeddings[idx]


class QuantizationOutput(NamedTuple):
    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Optional[Tensor]
    commitment_loss: Tensor


class BaseVectorQuantizer(nn.Module):
    """Base class for vector quantization with shared functionality."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_size = config.codebook_size
        self.commitment_weight = config.commitment_weight
        self.use_rotation_trick = config.use_rotation_trick

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)

        self.register_buffer("usage_count", torch.zeros(self.codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

    @staticmethod
    def l2norm(t: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        return F.normalize(t, p=2, dim=dim, eps=eps)

    @staticmethod
    def rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        w = BaseVectorQuantizer.l2norm(u + q, dim=-1).detach()
        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()

        if e.ndim == 2:
            e_expanded = e.unsqueeze(1)
            result = e_expanded - 2 * (e_expanded @ w_col @ w_row) + 2 * (e_expanded @ u_col @ q_row)
            return result.squeeze(1)

        return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])
        norm_src = BaseVectorQuantizer.l2norm(src_flat)
        norm_tgt = BaseVectorQuantizer.l2norm(tgt_flat)
        rotated = BaseVectorQuantizer.rotation_trick(norm_src, norm_tgt, src_flat)
        return rotated.reshape(src.shape)

    def find_nearest_codes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        input_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_embedding_dim)
        distances = torch.cdist(flat_x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(input_shape)
        return indices.view(input_shape[:-1]), quantized

    def apply_gradient_estimator(self, x: Tensor, quantized: Tensor) -> Tensor:
        if self.training and x.requires_grad:
            if self.use_rotation_trick:
                return self.rotate_to(x, quantized)
            return x + (quantized - x).detach()
        return quantized

    def update_usage(self, indices: Tensor) -> None:
        flat = indices.flatten()
        counts = torch.bincount(flat, minlength=self.codebook_size)
        self.usage_count += counts
        self.update_count += 1

    def reset_usage_count(self) -> None:
        self.usage_count.zero_()
        self.update_count.zero_()

    def get_usage_rate(self) -> float:
        if self.update_count.item() == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()


class VectorQuantizer(BaseVectorQuantizer):
    """Vector quantization layer with learnable codebook."""

    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)
        quantized_st = self.apply_gradient_estimator(x, quantized)

        codebook_loss = F.mse_loss(x.detach(), quantized)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        if self.training:
            self.update_usage(indices)

        return QuantizationOutput(quantized_st, quantized, indices, loss, codebook_loss, commitment_loss)

    def reset_unused_codes(self, batch_data: Tensor) -> None:
        if self.update_count.item() == 0:
            return

        unused = (self.usage_count == 0).nonzero().squeeze(-1)
        if len(unused) > 0 and batch_data.shape[0] >= len(unused):
            batch_flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            random_idx = torch.randperm(batch_flat.shape[0], device=batch_flat.device)[: len(unused)]
            self.embedding.weight.data[unused] = batch_flat[random_idx].detach()

        self.reset_usage_count()


class EMAVectorQuantizer(BaseVectorQuantizer):
    """Vector quantization layer with EMA codebook updates."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__(config)
        self.decay = config.ema_decay
        self.epsilon = config.ema_epsilon

        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    @staticmethod
    def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
        moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)

    @staticmethod
    def laplace_smoothing(x: Tensor, n_categories: int, epsilon: float = 1e-5) -> Tensor:
        return (x + epsilon) / (x.sum() + n_categories * epsilon) * x.sum()

    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)

        if self.training:
            flat_x = x.reshape(-1, self.codebook_embedding_dim)
            flat_indices = indices.flatten()

            encodings = F.one_hot(flat_indices, self.codebook_size).float()
            self.ema_inplace(self.ema_cluster_size, encodings.sum(0), self.decay)
            dw = encodings.T @ flat_x
            self.ema_inplace(self.ema_w, dw, self.decay)

            cluster_size = self.laplace_smoothing(self.ema_cluster_size, self.codebook_size, self.epsilon)
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

            self.update_usage(indices)

        quantized_st = self.apply_gradient_estimator(x, quantized)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_weight * commitment_loss

        return QuantizationOutput(quantized_st, quantized, indices, loss, None, commitment_loss)


class RQVAE(nn.Module):
    """Residual Quantized VAE for semantic ID generation."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config

        dims = [config.item_embedding_dim] + config.encoder_hidden_dims + [config.codebook_embedding_dim]
        encoder_layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            encoder_layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
        encoder_layers.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*encoder_layers)

        dims_rev = [config.codebook_embedding_dim] + config.encoder_hidden_dims[::-1] + [config.item_embedding_dim]
        decoder_layers: List[nn.Module] = []
        for i in range(len(dims_rev) - 2):
            decoder_layers.extend([nn.Linear(dims_rev[i], dims_rev[i + 1]), nn.SiLU()])
        decoder_layers.append(nn.Linear(dims_rev[-2], dims_rev[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

        quantizer_class = EMAVectorQuantizer if config.use_ema_vq else VectorQuantizer
        self.vq_layers = nn.ModuleList([quantizer_class(config) for _ in range(config.codebook_quantization_levels)])

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], dict]:
        z = self.encode(x)
        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices: List[Tensor] = []
        vq_loss = 0.0
        codebook_losses: List[Tensor] = []
        commitment_losses: List[Tensor] = []

        for vq_layer in self.vq_layers:
            vq_output = vq_layer(residual)
            residual = residual - vq_output.quantized.detach()
            quantized_out = quantized_out + vq_output.quantized_st
            all_indices.append(vq_output.indices)
            vq_loss = vq_loss + vq_output.loss

            if vq_output.codebook_loss is not None:
                codebook_losses.append(vq_output.codebook_loss)
            commitment_losses.append(vq_output.commitment_loss)

        x_recon = self.decode(quantized_out)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_losses": codebook_losses,
            "commitment_losses": commitment_losses,
            "indices": all_indices,
            "residual": residual,
        }

        return x_recon, all_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            z = self.encode(x)
            residual = z
            indices_list = []
            for vq_layer in self.vq_layers:
                vq_output = vq_layer(residual)
                indices_list.append(vq_output.indices)
                residual = residual - vq_output.quantized
            return torch.stack(indices_list, dim=-1)

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        flat = semantic_ids.reshape(-1, semantic_ids.shape[-1])
        unique = torch.unique(flat, dim=0)
        return unique.shape[0] / flat.shape[0]

    def calculate_codebook_usage(self) -> List[float]:
        return [vq_layer.get_usage_rate() for vq_layer in self.vq_layers]

    def kmeans_init(self, data_loader: DataLoader, device: str) -> None:
        if KMeans is None:
            logger.warning("sklearn not available, skipping k-means init")
            return

        logger.info("Initializing codebooks with k-means")
        first_batch = next(iter(data_loader)).to(device)
        with torch.no_grad():
            z = self.encode(first_batch)
            residual = z
            for level, vq_layer in enumerate(self.vq_layers):
                residual_np = residual.detach().cpu().numpy().reshape(-1, self.config.codebook_embedding_dim)
                kmeans = KMeans(n_clusters=self.config.codebook_size, n_init=10, random_state=0)
                kmeans.fit(residual_np)
                vq_layer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).to(device)
                logger.info("Level %s: initialized %s codes", level, self.config.codebook_size)

                if level < self.config.codebook_quantization_levels - 1:
                    vq_output = vq_layer(residual)
                    residual = residual - vq_output.quantized


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gradient_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)


def train_rqvae(
    model: RQVAE,
    data_loader: DataLoader,
    config: RQVAEConfig,
    device: str,
    val_loader: Optional[DataLoader] = None,
) -> None:
    model = model.to(device)
    if config.use_kmeans_init:
        model.kmeans_init(data_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=0.01)

    steps_per_epoch = max(1, len(data_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs

    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_lr
        )
    elif config.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(config.min_lr / config.max_lr, 1e-8),
            total_iters=config.warmup_steps,
        )
        cosine_steps = max(1, total_steps - config.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=config.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
        )
    else:
        scheduler = None

    global_step = 0
    best_loss = float("inf")

    for epoch in tqdm(range(config.num_epochs), desc="epochs", unit="epoch"):
        model.train()
        for batch_idx, batch in enumerate(
            tqdm(data_loader, desc=f"train {epoch + 1}", unit="batch", leave=False)
        ):
            if batch_idx % config.gradient_accumulation_steps == 0:
                t0 = time.time()
                optimizer.zero_grad()
                loss_accum = 0.0

            batch = batch.to(device)
            _, _, loss_dict = model(batch)
            loss = loss_dict["loss"] / config.gradient_accumulation_steps
            loss_accum += loss_dict["loss"].detach()
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                grad_norm_before = get_gradient_norm(model)
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                    grad_norm_after = get_gradient_norm(model)
                else:
                    grad_norm_after = grad_norm_before

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                batch_time_ms = (time.time() - t0) * 1000
                avg_loss = loss_accum / config.gradient_accumulation_steps
                samples_per_second = (batch.shape[0] * config.gradient_accumulation_steps) / (batch_time_ms / 1000)

                if global_step == 1 or global_step % config.steps_per_train_log == 0:
                    codebook_usage = model.calculate_codebook_usage()
                    semantic_ids = torch.stack(loss_dict["indices"], dim=-1)
                    unique_ids = model.calculate_unique_ids_proportion(semantic_ids)
                    usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])
                    logger.info(
                        "Step %05d | Epoch %03d | loss %.4e | recon %.4e | vq %.4e | usage %s | unique %.1f%% | "
                        "lr %.2e | grad %.2f/%.2f | %.0f ms | %.0f samples/s",
                        global_step,
                        epoch + 1,
                        avg_loss.item(),
                        loss_dict["recon_loss"].item(),
                        loss_dict["vq_loss"].item(),
                        usage_str,
                        unique_ids * 100,
                        optimizer.param_groups[0]["lr"],
                        grad_norm_before,
                        grad_norm_after,
                        batch_time_ms,
                        samples_per_second,
                    )

                if (
                    config.reset_unused_codes
                    and global_step % config.steps_per_codebook_reset == 0
                    and isinstance(model.vq_layers[0], VectorQuantizer)
                ):
                    codebook_usage = model.calculate_codebook_usage()
                    if any(u < config.codebook_usage_threshold for u in codebook_usage):
                        reset_batch = next(iter(data_loader)).to(device)
                        with torch.no_grad():
                            z = model.encode(reset_batch)
                            residual = z
                            for level, vq_layer in enumerate(model.vq_layers):
                                if codebook_usage[level] < config.codebook_usage_threshold:
                                    vq_layer.reset_unused_codes(residual)
                                vq_out = vq_layer(residual)
                                residual = residual - vq_out.quantized

                if val_loader is not None and global_step % config.steps_per_val_log == 0:
                    val_loss = evaluate(model, val_loader, device)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        save_checkpoint(model, config, global_step, epoch, best_loss)

    save_checkpoint(model, config, global_step, config.num_epochs, best_loss, final=True)


def evaluate(model: RQVAE, val_loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", unit="batch", leave=False):
            batch = batch.to(device)
            _, _, loss_dict = model(batch)
            losses.append(loss_dict["loss"].item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(
    model: RQVAE,
    config: RQVAEConfig,
    global_step: int,
    epoch: int,
    best_loss: float,
    final: bool = False,
) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = "final_model.pth" if final else f"checkpoint_{global_step:06d}.pth"
    path = config.checkpoint_dir / name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "best_loss": best_loss,
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


def export_item_id_to_sid(
    model: RQVAE,
    dataset: EmbeddingDatasetLoader,
    output_path: Path,
    batch_size: int,
    device: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    total_items = len(dataset)
    logger.info("Exporting semantic IDs for %s items", total_items)

    with output_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for start in tqdm(
                range(0, total_items, batch_size),
                desc="export",
                unit="batch",
            ):
                end = min(start + batch_size, total_items)
                batch_embeddings = dataset.embeddings[start:end].to(device)
                batch_ids = dataset.item_ids[start:end]

                semantic_ids = model.encode_to_semantic_ids(batch_embeddings)
                semantic_ids_list = semantic_ids.cpu().numpy().tolist()

                for item_id, sid_tuple in zip(batch_ids, semantic_ids_list):
                    handle.write(
                        json.dumps(
                            {"item_id": item_id, "sid_tuple": sid_tuple},
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

    logger.info("Semantic ID map saved to %s", output_path)


def run_rqvae_pipeline() -> None:
    config = RQVAEConfig()
    config.log_config()

    if sys.platform == "darwin":
        config.num_workers = 0
        config.pin_memory = False
        logger.info("macOS detected: forcing num_workers=0 and pin_memory=False")

    dataset = EmbeddingDatasetLoader(config.embeddings_path)
    inferred_dim = dataset.embeddings.shape[1]
    if config.item_embedding_dim != inferred_dim:
        logger.info(
            "Adjusting item_embedding_dim from %s to %s based on data",
            config.item_embedding_dim,
            inferred_dim,
        )
        config.item_embedding_dim = inferred_dim
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    device = get_device()
    logger.info("Device: %s", device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=config.pin_memory and device == "cuda",
        drop_last=False,
    )

    model = RQVAE(config)
    train_rqvae(model, train_loader, config, device, val_loader)

    export_path = config.data_dir / "output" / "item_id_to_sid.jsonl"
    export_item_id_to_sid(
        model,
        dataset,
        export_path,
        batch_size=config.batch_size,
        device=device,
    )
