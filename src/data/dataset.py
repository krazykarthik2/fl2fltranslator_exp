"""Dataset classes for seq2seq translation training."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.tokenizer.c_tokenizer import CTokenizer


class TranslationDataset(Dataset):
    """Tokenized paired (src, tgt) translation dataset."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        tokenizer: Optional[CTokenizer] = None,
    ):
        self.tokenizer = tokenizer or CTokenizer()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.samples = self._encode_pairs(pairs)

    def _encode_text(self, text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
        tokens = self.tokenizer.tokenize(text)
        bos = vocab.get("<BOS>", 2)
        eos = vocab.get("<EOS>", 3)
        ids = CTokenizer.encode(tokens, vocab)
        # truncate to max_len - 2 to leave room for BOS/EOS
        ids = ids[: max_len - 2]
        return [bos] + ids + [eos]

    def _encode_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[List[int], List[int]]]:
        encoded = []
        for src_text, tgt_text in pairs:
            src_ids = self._encode_text(src_text, self.src_vocab, self.max_src_len)
            tgt_ids = self._encode_text(tgt_text, self.tgt_vocab, self.max_tgt_len)
            encoded.append((src_ids, tgt_ids))
        return encoded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self.samples[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    def __repr__(self) -> str:
        return f"TranslationDataset(size={len(self)})"


class DataCollator:
    """Pads a batch of (src, tgt) pairs to uniform length."""

    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_list, tgt_list = zip(*batch)
        src_padded = self._pad(src_list)
        tgt_padded = self._pad(tgt_list)
        src_mask = (src_padded == self.pad_idx)   # (B, T_src) — True where pad
        tgt_mask = (tgt_padded == self.pad_idx)   # (B, T_tgt)
        return src_padded, tgt_padded, src_mask, tgt_mask

    def _pad(self, seqs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), self.pad_idx, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    def __repr__(self) -> str:
        return f"DataCollator(pad_idx={self.pad_idx})"


def load_dataset_from_dir(
    data_dir: str,
    src_ext: str,
    tgt_ext: str,
    src_vocab: Optional[Dict[str, int]] = None,
    tgt_vocab: Optional[Dict[str, int]] = None,
    max_src_len: int = 512,
    max_tgt_len: int = 512,
) -> TranslationDataset:
    """Load paired files from *data_dir*.

    Expects files like ``001.c`` / ``001.ir`` in *data_dir*
    (or separate subdirectories named after *src_ext* / *tgt_ext*).
    Falls back to flat directory pairing by sorted stem.
    """
    src_dir = os.path.join(data_dir, src_ext.lstrip("."))
    tgt_dir = os.path.join(data_dir, tgt_ext.lstrip("."))
    # Special case for .rs -> rust folder
    if tgt_ext == ".rs" and not os.path.isdir(tgt_dir):
        tgt_dir = os.path.join(data_dir, "rust")

    if os.path.isdir(src_dir) and os.path.isdir(tgt_dir):
        src_files = sorted(f for f in os.listdir(src_dir) if f.endswith(src_ext))
        pairs = []
        for sf in src_files:
            stem = sf[: -len(src_ext)]
            tf = stem + tgt_ext
            if os.path.exists(os.path.join(tgt_dir, tf)):
                with open(os.path.join(src_dir, sf), encoding="utf-8") as fh:
                    src_text = fh.read()
                with open(os.path.join(tgt_dir, tf), encoding="utf-8") as fh:
                    tgt_text = fh.read()
                pairs.append((src_text, tgt_text))
    else:
        # Flat directory: expect paired .src_ext / .tgt_ext files with same stem
        src_files = sorted(f for f in os.listdir(data_dir) if f.endswith(src_ext))
        pairs = []
        for sf in src_files:
            stem = sf[:-len(src_ext)]
            tf = stem + tgt_ext
            if os.path.exists(os.path.join(data_dir, tf)):
                with open(os.path.join(data_dir, sf), encoding="utf-8") as fh:
                    src_text = fh.read()
                with open(os.path.join(data_dir, tf), encoding="utf-8") as fh:
                    tgt_text = fh.read()
                pairs.append((src_text, tgt_text))

    if src_vocab is None:
        tokenizer = CTokenizer()
        corpus = [p[0] for p in pairs]
        src_vocab = CTokenizer.build_vocab(corpus)
    if tgt_vocab is None:
        tokenizer = CTokenizer()
        corpus = [p[1] for p in pairs]
        tgt_vocab = CTokenizer.build_vocab(corpus)

    return TranslationDataset(
        pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
