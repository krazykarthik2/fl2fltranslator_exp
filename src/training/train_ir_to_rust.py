"""Stage 2 training script: IR → Rust."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.model.ir_to_rust_model import IRToRustModel
from src.model.transformer import TransformerConfig
from src.data.dataset import TranslationDataset, DataCollator
from src.training.train_c_to_ir import get_lr_scheduler


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    n_epochs: int = 20
    warmup_steps: int = 4000
    max_src_len: int = 512
    max_tgt_len: int = 512
    src_vocab_size: int = 8000
    tgt_vocab_size: int = 8000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    label_smoothing: float = 0.1
    save_dir: str = "checkpoints/ir_to_rust"
    data_dir: str = "dataset/samples"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    val_split: float = 0.1
    log_interval: int = 10


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        os.makedirs(config.save_dir, exist_ok=True)

        model_cfg = TransformerConfig(
            vocab_size=config.src_vocab_size,
            max_seq_len=config.max_src_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.model = IRToRustModel(model_cfg, tgt_vocab_size=config.tgt_vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                          betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = get_lr_scheduler(self.optimizer, config.d_model, config.warmup_steps)
        self.collator = DataCollator(pad_idx=0)
        self.global_step = 0

    def _build_dataloaders(self, dataset: TranslationDataset):
        n_val = max(1, int(len(dataset) * self.config.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_dl = DataLoader(train_ds, batch_size=self.config.batch_size,
                              shuffle=True, collate_fn=self.collator, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=self.config.batch_size,
                            shuffle=False, collate_fn=self.collator)
        return train_dl, val_dl

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (src, tgt, src_mask, tgt_mask) in enumerate(dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            tgt_in = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            logits = self.model(
                src, tgt_in,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask[:, :-1],
            )
            loss = self.model.compute_loss(logits, tgt_labels,
                                            label_smoothing=self.config.label_smoothing)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            if batch_idx % self.config.log_interval == 0:
                print(f"  step={self.global_step} loss={loss.item():.4f}")
        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for src, tgt, src_mask, tgt_mask in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            tgt_in = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            logits = self.model(src, tgt_in,
                                src_padding_mask=src_mask,
                                tgt_padding_mask=tgt_mask[:, :-1])
            loss = self.model.compute_loss(logits, tgt_labels,
                                            label_smoothing=self.config.label_smoothing)
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        path = os.path.join(self.config.save_dir, f"epoch_{epoch:03d}_loss_{loss:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  Loaded checkpoint from {path}")

    def train(self, dataset: Optional[TranslationDataset] = None) -> None:
        if dataset is None:
            from src.data.dataset import load_dataset_from_dir
            dataset = load_dataset_from_dir(
                self.config.data_dir, src_ext=".ir", tgt_ext=".rs",
                max_src_len=self.config.max_src_len,
                max_tgt_len=self.config.max_tgt_len,
            )
        train_dl, val_dl = self._build_dataloaders(dataset)
        print(f"Training on {len(train_dl.dataset)} samples, "
              f"validating on {len(val_dl.dataset)} samples")
        for epoch in range(1, self.config.n_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.n_epochs}")
            train_loss = self.train_epoch(train_dl)
            val_loss = self.validate(val_dl)
            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            self.save_checkpoint(epoch, val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IR→Rust model (Stage 2)")
    parser.add_argument("--data-dir", default="dataset/samples")
    parser.add_argument("--save-dir", default="checkpoints/ir_to_rust")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = TrainingConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    if args.device:
        cfg.device = args.device

    trainer = Trainer(cfg)
    trainer.train()
