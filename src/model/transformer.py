"""Transformer encoder-decoder in PyTorch (~40M parameters)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 8000
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    pad_idx: int = 0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if hasattr(self, "pe") and seq_len > self.pe.size(1):
            d_model = x.size(2)
            pe = torch.zeros(seq_len, d_model, device=x.device)
            pos = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe, persistent=False)
            
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        Q = self.q_proj(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            # attn_mask: (T_q, T_k) or (B, n_heads, T_q, T_k)
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_k) — True where padding
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_out = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x = self.norm1(x + self.dropout(self_out))
        cross_out = self.cross_attn(x, memory, memory, key_padding_mask=memory_padding_mask)
        x = self.norm2(x + self.dropout(cross_out))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.embed(src) * math.sqrt(self.embed.embedding_dim))
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_padding_mask=src_padding_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.embed(tgt) * math.sqrt(self.embed.embedding_dim))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask,
                      memory_padding_mask=memory_padding_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """Full encoder-decoder transformer."""

    def __init__(self, config: TransformerConfig, tgt_vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        tgt_vocab = tgt_vocab_size or config.vocab_size
        self.encoder = TransformerEncoder(config)
        # Decoder needs its own config if tgt vocab differs
        dec_config = TransformerConfig(
            vocab_size=tgt_vocab,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
        )
        self.decoder = TransformerDecoder(dec_config)
        self.output_proj = nn.Linear(config.d_model, tgt_vocab)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask filled with -inf (causal / autoregressive)."""
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(src, src_mask=src_mask, src_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, tgt_mask=tgt_mask,
                            tgt_padding_mask=tgt_padding_mask,
                            memory_padding_mask=memory_padding_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        memory = self.encode(src, src_mask=src_mask, src_padding_mask=src_padding_mask)
        dec_out = self.decode(tgt, memory, tgt_mask=tgt_mask,
                              tgt_padding_mask=tgt_padding_mask,
                              memory_padding_mask=src_padding_mask)
        return self.output_proj(dec_out)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy decoding."""
        self.eval()
        device = src.device
        memory = self.encode(src)
        B = src.size(0)
        tgt = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = self._causal_mask(tgt.size(1), device)
            dec_out = self.decode(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(dec_out[:, -1, :]) / temperature
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if (next_tok == eos_idx).all():
                break
        return tgt


if __name__ == "__main__":
    cfg = TransformerConfig()
    model = EncoderDecoder(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    src = torch.randint(0, cfg.vocab_size, (2, 32))
    tgt = torch.randint(0, cfg.vocab_size, (2, 24))
    out = model(src, tgt)
    print("Output shape:", out.shape)
