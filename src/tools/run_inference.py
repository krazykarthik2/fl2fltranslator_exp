r"""Small inference helper used by `modelctl.bat`.

Usage example:
  python src\tools\run_inference.py c2rust path\to\input.c --checkpoint-dir checkpoints\c_to_rust
"""
from __future__ import annotations

import argparse
import os
import glob
from typing import Optional

import torch

from src.data.dataset import load_dataset_from_dir
from src.tokenizer.c_tokenizer import CTokenizer


def find_latest_checkpoint(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    files = sorted(glob.glob(os.path.join(path, "*.pt")), key=os.path.getmtime)
    return files[-1] if files else None


def load_model_and_vocabs(stage: str, checkpoints_path: str):
    from src.model.c_to_rust_model import CToRustModel
    from src.training.train_c_to_rust import TrainingConfig
    import sys
    sys.modules["__main__"].TrainingConfig = TrainingConfig
    src_ext, tgt_ext = ".c", ".rs"
    ck_dir = checkpoints_path or "checkpoints/c_to_rust"
    ModelClass = CToRustModel

    ckpt_path = find_latest_checkpoint(ck_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found at {ck_dir}")

    # In PyTorch 2.6+, weights_only=True is default and it blocks TrainingConfig global in checkpoints
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older torch versions without weights_only param
        ckpt = torch.load(ckpt_path, map_location="cpu")

    # Build vocabs
    src_vocab = ckpt.get("src_vocab")
    tgt_vocab = ckpt.get("tgt_vocab")

    if src_vocab is None or tgt_vocab is None:
        print("Warning: Vocabulary not found in checkpoint. Falling back to dataset-based vocabulary.")
        ds = load_dataset_from_dir("dataset/samples", src_ext=src_ext, tgt_ext=tgt_ext)
        src_vocab = ds.src_vocab
        tgt_vocab = ds.tgt_vocab

    cfg = ckpt.get("config")
    
    # Architectural defaults
    src_vocab_size = 8000
    tgt_vocab_size = 8000
    kwargs = {}

    if cfg is not None:
        src_vocab_size = getattr(cfg, "src_vocab_size", 8000)
        tgt_vocab_size = getattr(cfg, "tgt_vocab_size", 8000)
        
        # Architecture parameters
        for attr in ["d_model", "n_heads", "n_layers", "d_ff", "dropout"]:
            if hasattr(cfg, attr):
                kwargs[attr] = getattr(cfg, attr)
        
        # Max sequence length mapping
        if hasattr(cfg, "max_src_len"):
            kwargs["max_seq_len"] = cfg.max_src_len
    
    # Validation
    if src_vocab and len(src_vocab) > src_vocab_size:
        raise ValueError(f"Source vocabulary size ({len(src_vocab)}) exceeds model capacity ({src_vocab_size})")
    if tgt_vocab and len(tgt_vocab) > tgt_vocab_size:
        raise ValueError(f"Target vocabulary size ({len(tgt_vocab)}) exceeds model capacity ({tgt_vocab_size})")

    model = ModelClass.from_config(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size,
        **kwargs
    )
        
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab


def encode_text(text: str, vocab: dict) -> torch.Tensor:
    tok = CTokenizer()
    ids = [vocab.get("<BOS>", 2)] + CTokenizer.encode(tok.tokenize(text), vocab) + [vocab.get("<EOS>", 3)]
    return torch.tensor([ids], dtype=torch.long)


def decode_ids(ids: torch.Tensor, vocab: dict) -> str:
    inv = {v: k for k, v in vocab.items()}
    # drop BOS and everything after EOS
    out_ids = ids.tolist()
    toks = []
    for i in out_ids:
        if i == vocab.get("<BOS>", 2):
            continue
        if i == vocab.get("<EOS>", 3):
            break
        toks.append(inv.get(i, "<UNK>"))
    return " ".join(toks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["c2rust"])
    parser.add_argument("input", help="Path to input file containing source text")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw", action="store_true", help="Output only the translated text")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        src_text = fh.read()

    model, src_vocab, tgt_vocab = load_model_and_vocabs(args.stage, args.checkpoint_dir)
    device = torch.device(args.device)
    model.to(device)

    src_ids = encode_text(src_text, src_vocab).to(device)
    bos = tgt_vocab.get("<BOS>", 2)
    eos = tgt_vocab.get("<EOS>", 3)

    if not args.raw:
        print(f"\n{'='*60}")
        print(f" STAGE: {args.stage.upper()}")
        print(f" INPUT: {args.input}")
        print(f"{'='*60}\n")

    with torch.no_grad():
        # Model with auxiliary heads: show latent-space analysis
        if not args.raw:
            # Mock a target input for forward pass to get aux preds
            logits, aux_preds = model(src_ids, torch.tensor([[bos]], device=device))
        
        # Generate the main output
        out = model.generate(src_ids, max_len=args.max_len, bos_idx=bos, eos_idx=eos)
        
        if not args.raw:
            # Print aux traits for first few tokens
            print("--- Latent-Space Auxiliary Traits (Ownership/Mutability/Lifetime/Unsafe) ---")
            from src.model.multitask_head import OwnershipClassifier, MutabilityClassifier, LifetimeClassifier, UnsafeClassifier
            
            # Get tokens for display
            tok_obj = CTokenizer()
            src_tokens = tok_obj.tokenize(src_text)
            
            # Encoder output for aux heads
            memory = model.seq2seq.encode(src_ids)
            aux_results = model.aux_head(memory)
            
            for i, t in enumerate(src_tokens[:20]): # Show first 20 tokens
                if i + 1 >= memory.size(1): break
                idx = i + 1 # skip BOS
                
                own = aux_results["ownership"][0, idx].argmax().item()
                mut = aux_results["mutability"][0, idx].argmax().item()
                life = aux_results["lifetime"][0, idx].argmax().item()
                uns = aux_results["unsafe"][0, idx].argmax().item()
                
                print(f"  {t:12} | own: {OwnershipClassifier.LABELS[own]:12} | mut: {MutabilityClassifier.LABELS[mut]:10} | life: {LifetimeClassifier.LABELS[life]:10} | uns: {UnsafeClassifier.LABELS[uns]}")
            if len(src_tokens) > 20:
                print("  ...")
            print()

    # out: (B, T)
    out_tokens = decode_ids(out[0], tgt_vocab)
    
    if not args.raw:
        print("--- Translated Output ---")
        print(out_tokens)
        print(f"\n{'='*60}\n")
    else:
        # Raw output
        print(out_tokens)


if __name__ == "__main__":
    main()
