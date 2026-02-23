"""Tests for the transformer model."""
import pytest
import torch
from src.model.transformer import TransformerConfig, EncoderDecoder
from src.model.multitask_head import MultiTaskHead
from src.model.c_to_ir_model import CToIRModel
from src.model.ir_to_rust_model import IRToRustModel


@pytest.fixture
def small_config():
    """A tiny config for fast tests."""
    return TransformerConfig(
        vocab_size=200,
        max_seq_len=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )


class TestTransformerModel:
    def test_model_shape(self, small_config):
        model = EncoderDecoder(small_config)
        src = torch.randint(0, small_config.vocab_size, (2, 16))
        tgt = torch.randint(0, small_config.vocab_size, (2, 12))
        out = model(src, tgt)
        assert out.shape == (2, 12, small_config.vocab_size)

    def test_model_parameters_small(self, small_config):
        model = EncoderDecoder(small_config)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0
        print(f"Small model params: {n:,}")

    def test_model_parameters_full(self):
        cfg = TransformerConfig(vocab_size=8000)
        model = EncoderDecoder(cfg)
        n = sum(p.numel() for p in model.parameters())
        # Should be roughly 40M
        assert 30_000_000 < n < 60_000_000, f"Expected ~40M params, got {n:,}"

    def test_causal_mask(self, small_config):
        model = EncoderDecoder(small_config)
        mask = model._causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)
        # Upper triangle should be -inf
        assert mask[0, 1] == float("-inf")
        # Diagonal and below should be 0
        assert mask[1, 0] == 0.0

    def test_encode_decode_separate(self, small_config):
        model = EncoderDecoder(small_config)
        model.eval()
        src = torch.randint(0, small_config.vocab_size, (2, 16))
        tgt = torch.randint(0, small_config.vocab_size, (2, 10))
        memory = model.encode(src)
        assert memory.shape == (2, 16, small_config.d_model)
        dec = model.decode(tgt, memory)
        assert dec.shape == (2, 10, small_config.d_model)

    def test_generate(self, small_config):
        model = EncoderDecoder(small_config)
        model.eval()
        src = torch.randint(0, small_config.vocab_size, (1, 8))
        out = model.generate(src, max_len=10, bos_idx=2, eos_idx=3)
        assert out.shape[0] == 1
        assert out.shape[1] >= 2  # at least BOS + one token


class TestMultiTaskHead:
    def test_multitask_head(self):
        head = MultiTaskHead(d_model=64)
        enc_out = torch.randn(2, 16, 64)
        preds = head(enc_out)
        assert "ownership" in preds
        assert "mutability" in preds
        assert "lifetime" in preds
        assert "unsafe" in preds
        assert preds["ownership"].shape == (2, 16, 4)
        assert preds["mutability"].shape == (2, 16, 2)
        assert preds["lifetime"].shape == (2, 16, 4)
        assert preds["unsafe"].shape == (2, 16, 2)


class TestCToIRModel:
    def test_c_to_ir_model_forward(self, small_config):
        model = CToIRModel(small_config, tgt_vocab_size=200)
        src = torch.randint(0, 200, (2, 16))
        tgt = torch.randint(0, 200, (2, 12))
        logits, aux = model(src, tgt)
        assert logits.shape == (2, 12, 200)
        assert "ownership" in aux

    def test_c_to_ir_model_loss(self, small_config):
        model = CToIRModel(small_config, tgt_vocab_size=200)
        src = torch.randint(0, 200, (2, 16))
        tgt = torch.randint(0, 200, (2, 12))
        logits, aux = model(src, tgt[:, :-1])
        labels = tgt[:, 1:]
        total, main, aux_loss = model.compute_loss(logits, labels, aux)
        assert total.item() > 0

    def test_c_to_ir_model_from_config(self):
        model = CToIRModel.from_config(src_vocab_size=500, tgt_vocab_size=500,
                                        d_model=64, n_heads=4, n_layers=2, d_ff=128)
        assert model is not None

    def test_generate(self, small_config):
        model = CToIRModel(small_config, tgt_vocab_size=200)
        model.eval()
        src = torch.randint(0, 200, (1, 8))
        out = model.generate(src, max_len=10, bos_idx=2, eos_idx=3)
        assert out.shape[0] == 1


class TestIRToRustModel:
    def test_ir_to_rust_model_forward(self, small_config):
        model = IRToRustModel(small_config, tgt_vocab_size=200)
        src = torch.randint(0, 200, (2, 16))
        tgt = torch.randint(0, 200, (2, 12))
        logits = model(src, tgt)
        assert logits.shape == (2, 12, 200)

    def test_ir_to_rust_model_loss(self, small_config):
        model = IRToRustModel(small_config, tgt_vocab_size=200)
        src = torch.randint(0, 200, (2, 16))
        tgt = torch.randint(0, 200, (2, 12))
        logits = model(src, tgt[:, :-1])
        labels = tgt[:, 1:]
        loss = model.compute_loss(logits, labels)
        assert loss.item() > 0
