"""Stage 2 model: IR (S-expression) → Rust source."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer import EncoderDecoder, TransformerConfig


class IRToRustModel(nn.Module):
    """Encoder-decoder transformer for IR → Rust translation (no auxiliary heads)."""

    def __init__(self, config: TransformerConfig, tgt_vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.seq2seq = EncoderDecoder(config, tgt_vocab_size=tgt_vocab_size or config.vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits of shape (B, T_tgt, tgt_vocab_size)."""
        return self.seq2seq(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        tgt_labels: torch.Tensor,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """Cross-entropy loss with label smoothing."""
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            tgt_labels.reshape(B * T),
            ignore_index=self.config.pad_idx,
            label_smoothing=label_smoothing,
        )

    @classmethod
    def from_config(
        cls,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
    ) -> "IRToRustModel":
        config = TransformerConfig(
            vocab_size=src_vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        return cls(config, tgt_vocab_size=tgt_vocab_size)

    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self.seq2seq.generate(src, max_len, bos_idx, eos_idx, temperature)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"IRToRustModel(params={n:,})"


if __name__ == "__main__":
    model = IRToRustModel.from_config(src_vocab_size=8000, tgt_vocab_size=8000)
    print(model)
    src = torch.randint(0, 8000, (2, 32))
    tgt = torch.randint(0, 8000, (2, 24))
    logits = model(src, tgt)
    print("logits:", logits.shape)
