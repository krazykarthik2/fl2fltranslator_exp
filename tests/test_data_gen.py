"""Tests for synthetic data generation and dataset loading."""
import os
import tempfile
import pytest
from src.data.synthetic_gen import SyntheticCGenerator
from src.data.dataset import TranslationDataset, DataCollator
from src.tokenizer.c_tokenizer import CTokenizer


class TestSyntheticGenerator:
    def test_generate_function(self):
        gen = SyntheticCGenerator(seed=42)
        fn = gen.generate_function()
        assert isinstance(fn, str)
        assert len(fn) > 0
        # Should contain a function definition
        assert "(" in fn and ")" in fn and "{" in fn and "}" in fn

    def test_generate_batch(self):
        gen = SyntheticCGenerator(seed=42)
        batch = gen.generate_batch(10)
        assert len(batch) == 10
        for fn in batch:
            assert isinstance(fn, str)
            assert len(fn) > 0

    def test_generate_reproducible(self):
        gen1 = SyntheticCGenerator(seed=0)
        gen2 = SyntheticCGenerator(seed=0)
        batch1 = gen1.generate_batch(5)
        batch2 = gen2.generate_batch(5)
        assert batch1 == batch2

    def test_generate_dataset(self):
        gen = SyntheticCGenerator(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.generate_dataset(5, tmpdir)
            files = [f for f in os.listdir(tmpdir) if f.endswith(".c")]
            assert len(files) == 5
            for fname in files:
                path = os.path.join(tmpdir, fname)
                with open(path) as f:
                    content = f.read().strip()
                assert len(content) > 0

    def test_diversity(self):
        gen = SyntheticCGenerator(seed=99)
        batch = gen.generate_batch(50)
        # Not all functions should be identical
        unique = set(batch)
        assert len(unique) > 1


class TestTranslationDataset:
    def setup_method(self):
        src1 = "int add(int a, int b) { return a + b; }"
        src2 = "int sub(int x, int y) { return x - y; }"
        tgt1 = "(fn (name add) (params) (return (binop + (ident a) (ident b))))"
        tgt2 = "(fn (name sub) (params) (return (binop - (ident x) (ident y))))"
        pairs = [(src1, tgt1), (src2, tgt2)]
        tokenizer = CTokenizer()
        vocab = CTokenizer.build_vocab([src1, src2, tgt1, tgt2])
        self.dataset = TranslationDataset(pairs, src_vocab=vocab, tgt_vocab=vocab)

    def test_dataset_loading(self):
        assert len(self.dataset) == 2

    def test_dataset_item(self):
        import torch
        src, tgt = self.dataset[0]
        assert src.dtype == torch.long
        assert tgt.dtype == torch.long
        assert src.dim() == 1
        assert tgt.dim() == 1

    def test_dataset_bos_eos(self):
        src, tgt = self.dataset[0]
        # First token should be BOS (2), last should be EOS (3)
        assert src[0].item() == 2
        assert src[-1].item() == 3

    def test_data_collator(self):
        import torch
        collator = DataCollator(pad_idx=0)
        items = [self.dataset[0], self.dataset[1]]
        src_padded, tgt_padded, src_mask, tgt_mask = collator(items)
        assert src_padded.shape[0] == 2
        assert tgt_padded.shape[0] == 2

    def test_dataset_from_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            c_dir = os.path.join(tmpdir, "c")
            ir_dir = os.path.join(tmpdir, "ir")
            os.makedirs(c_dir)
            os.makedirs(ir_dir)
            for name, c_src, ir_src in [
                ("add", "int add(int a, int b) { return a + b; }",
                 "(fn (name add) (return (binop + (ident a) (ident b))))"),
                ("sub", "int sub(int x) { return -x; }",
                 "(fn (name sub) (return (unop - (ident x))))"),
            ]:
                with open(os.path.join(c_dir, name + ".c"), "w") as f:
                    f.write(c_src)
                with open(os.path.join(ir_dir, name + ".ir"), "w") as f:
                    f.write(ir_src)
            from src.data.dataset import load_dataset_from_dir
            ds = load_dataset_from_dir(tmpdir, src_ext=".c", tgt_ext=".ir")
            assert len(ds) == 2
