"""Stage 1 model: C source → IR (S-expression)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer import EncoderDecoder, TransformerConfig
from src.model.multitask_head import MultiTaskHead


class CToIRModel(nn.Module):
    """Encoder-decoder transformer with multi-task auxiliary heads for C → IR translation."""

    def __init__(self, config: TransformerConfig, tgt_vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.seq2seq = EncoderDecoder(config, tgt_vocab_size=tgt_vocab_size or config.vocab_size)
        self.aux_head = MultiTaskHead(config.d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        logits : (B, T_tgt, tgt_vocab_size)
        aux_predictions : dict with keys ownership/mutability/lifetime/unsafe
        """
        if tgt_mask is None:
            tgt_mask = EncoderDecoder._causal_mask(tgt.size(1), tgt.device)

        memory = self.seq2seq.encode(src, src_mask=src_mask, src_padding_mask=src_padding_mask)
        aux_preds = self.aux_head(memory)

        dec_out = self.seq2seq.decode(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
        )
        logits = self.seq2seq.output_proj(dec_out)
        return logits, aux_preds

    def compute_loss(
        self,
        logits: torch.Tensor,
        tgt_labels: torch.Tensor,
        aux_predictions: Dict[str, torch.Tensor],
        aux_labels: Optional[Dict[str, torch.Tensor]] = None,
        label_smoothing: float = 0.1,
        aux_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss.

        Returns
        -------
        total_loss, main_loss, aux_loss
        """
        B, T, V = logits.shape
        main_loss = F.cross_entropy(
            logits.reshape(B * T, V),
            tgt_labels.reshape(B * T),
            ignore_index=self.config.pad_idx,
            label_smoothing=label_smoothing,
        )

        aux_loss = torch.tensor(0.0, device=logits.device)
        if aux_labels:
            n_aux = 0
            for key, preds in aux_predictions.items():
                if key in aux_labels:
                    labels = aux_labels[key]  # (B, T)
                    _, T_enc, C = preds.shape
                    aux_loss = aux_loss + F.cross_entropy(
                        preds.reshape(-1, C),
                        labels.reshape(-1),
                        ignore_index=-1,
                    )
                    n_aux += 1
            if n_aux:
                aux_loss = aux_loss / n_aux

        total_loss = main_loss + aux_weight * aux_loss
        return total_loss, main_loss, aux_loss

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
    ) -> "CToIRModel":
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
        return f"CToIRModel(params={n:,})"


if __name__ == "__main__":
    model = CToIRModel.from_config(src_vocab_size=8000, tgt_vocab_size=8000)
    print(model)
    src = torch.randint(0, 8000, (2, 32))
    tgt = torch.randint(0, 8000, (2, 24))
    logits, aux = model(src, tgt)
    print("logits:", logits.shape)
    print("aux keys:", list(aux.keys()))
