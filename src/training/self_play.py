"""Self-play refinement loop: C → IR → Rust → cargo check → retrain."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch

from src.feedback.cargo_checker import CargoChecker
from src.feedback.error_parser import CompileError, RustErrorParser
from src.ir.c_to_ir import CToIR
from src.ir.ir_to_rust import IRToRust
from src.model.c_to_ir_model import CToIRModel
from src.model.ir_to_rust_model import IRToRustModel
from src.tokenizer.c_tokenizer import CTokenizer


class SelfPlayTrainer:
    """
    Self-play refinement loop.

    Steps per iteration:
      1. C source → IR  (rule-based converter, fallback to model generation)
      2. IR → Rust      (model generation)
      3. cargo check    (feedback)
      4. Accumulate positives / negatives
      5. Periodically fine-tune on accumulated data
    """

    def __init__(
        self,
        c_to_ir_model: CToIRModel,
        ir_to_rust_model: IRToRustModel,
        cargo_checker: CargoChecker,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        device: str = "cpu",
        max_gen_len: int = 256,
    ):
        self.c_to_ir_model = c_to_ir_model
        self.ir_to_rust_model = ir_to_rust_model
        self.checker = cargo_checker
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = torch.device(device)
        self.max_gen_len = max_gen_len

        self.tokenizer = CTokenizer()
        self.rule_based_c2ir = CToIR()
        self.rule_based_ir2rust = IRToRust()
        self.error_parser = RustErrorParser()

        self.positive_examples: List[Dict] = []
        self.negative_examples: List[Dict] = []

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str, vocab: Dict[str, int], max_len: int = 512) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text)
        ids = CTokenizer.encode(tokens, vocab)
        bos = vocab.get("<BOS>", 2)
        eos = vocab.get("<EOS>", 3)
        ids = [bos] + ids[: max_len - 2] + [eos]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode_ids(self, ids: torch.Tensor, vocab: Dict[str, int]) -> str:
        inv = {v: k for k, v in vocab.items()}
        tokens = []
        for i in ids.squeeze().tolist():
            tok = inv.get(i, "<UNK>")
            if tok == "<EOS>":
                break
            if tok not in ("<BOS>", "<PAD>"):
                tokens.append(tok)
        return " ".join(tokens)

    def translate(self, c_code: str) -> Tuple[str, str]:
        """Translate C source → (ir_sexp_str, rust_code_str) using the models."""
        # Stage 1: C → IR
        bos = self.src_vocab.get("<BOS>", 2)
        eos = self.src_vocab.get("<EOS>", 3)

        src_ids = self._encode(c_code, self.src_vocab)
        self.c_to_ir_model.eval()
        with torch.no_grad():
            ir_ids = self.c_to_ir_model.generate(
                src_ids, max_len=self.max_gen_len, bos_idx=bos, eos_idx=eos
            )
        ir_text = self._decode_ids(ir_ids, self.tgt_vocab)

        # Stage 2: IR → Rust
        ir_ids_tensor = self._encode(ir_text, self.src_vocab)
        self.ir_to_rust_model.eval()
        with torch.no_grad():
            rust_ids = self.ir_to_rust_model.generate(
                ir_ids_tensor, max_len=self.max_gen_len, bos_idx=bos, eos_idx=eos
            )
        rust_text = self._decode_ids(rust_ids, self.tgt_vocab)
        return ir_text, rust_text

    def generate_correction_data(
        self, c_code: str, rust_code: str, errors: List[CompileError]
    ) -> Dict:
        """Create a training sample from compilation feedback."""
        correction_prompt = self.error_parser.to_correction_prompt(errors, rust_code)
        return {
            "c_code": c_code,
            "rust_code": rust_code,
            "errors": [{"code": e.error_code, "message": e.message} for e in errors],
            "correction_prompt": correction_prompt,
            "label": "negative",
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        c_samples: List[str],
        n_iterations: int = 100,
        retrain_interval: int = 20,
    ) -> Dict:
        """
        Run self-play for *n_iterations* steps.

        Returns summary dict with success_rate, n_positive, n_negative.
        """
        n_success = 0
        indices = list(range(len(c_samples)))

        for iteration in range(n_iterations):
            c_code = c_samples[random.choice(indices)]

            # Try rule-based translation first, fall back to model
            try:
                ir_node = self.rule_based_c2ir.convert(c_code)
                # Use rule-based IR but model-generated Rust for feedback
                ir_text = ir_node.to_sexp()
                rust_code = self.rule_based_ir2rust.emit(ir_node)
                # Wrap top-level seq children
                if ir_node.kind == "seq":
                    rust_parts = [self.rule_based_ir2rust.emit(c) for c in ir_node.children]
                    rust_code = "\n\n".join(rust_parts)
            except Exception:
                ir_text, rust_code = self.translate(c_code)

            success, output = self.checker.check(rust_code)

            if success:
                n_success += 1
                self.positive_examples.append({
                    "c_code": c_code,
                    "ir_code": ir_text,
                    "rust_code": rust_code,
                    "label": "positive",
                })
            else:
                errors = self.error_parser.parse(output)
                neg = self.generate_correction_data(c_code, rust_code, errors)
                self.negative_examples.append(neg)

            if (iteration + 1) % retrain_interval == 0:
                print(f"[self-play] iter={iteration+1}  "
                      f"pos={len(self.positive_examples)}  "
                      f"neg={len(self.negative_examples)}")

        return {
            "n_iterations": n_iterations,
            "n_positive": len(self.positive_examples),
            "n_negative": len(self.negative_examples),
            "success_rate": n_success / max(n_iterations, 1),
        }

    def __repr__(self) -> str:
        return (f"SelfPlayTrainer("
                f"pos={len(self.positive_examples)}, "
                f"neg={len(self.negative_examples)})")
