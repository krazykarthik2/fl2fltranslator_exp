"""Tests for the C tokenizer."""
import pytest
from src.tokenizer.c_tokenizer import CTokenizer, C_KEYWORDS


class TestCTokenizer:
    def setup_method(self):
        self.tok = CTokenizer()

    def test_basic_tokenize(self):
        tokens = self.tok.tokenize("int x = 5;")
        assert "int" in tokens
        assert "x" in tokens
        assert "=" in tokens
        assert "5" in tokens
        assert ";" in tokens

    def test_keywords(self):
        src = "if (x) return 0; else while (y) break;"
        tokens = self.tok.tokenize(src)
        for kw in ("if", "return", "else", "while", "break"):
            assert kw in tokens

    def test_operators(self):
        tokens = self.tok.tokenize("a == b && c != d || e <= f")
        assert "==" in tokens
        assert "&&" in tokens
        assert "!=" in tokens
        assert "||" in tokens
        assert "<=" in tokens

    def test_string_literal(self):
        tokens = self.tok.tokenize('"hello world"')
        assert '"hello world"' in tokens

    def test_comment_stripped(self):
        tokens = self.tok.tokenize("int x; // comment\n int y;")
        assert "x" in tokens
        assert "y" in tokens
        # Comment text should not appear
        assert "comment" not in tokens

    def test_build_vocab(self):
        corpus = ["int add(int a, int b) { return a + b; }"]
        vocab = CTokenizer.build_vocab(corpus)
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert "<BOS>" in vocab
        assert "<EOS>" in vocab
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1
        assert vocab["<BOS>"] == 2
        assert vocab["<EOS>"] == 3
        assert "int" in vocab
        assert "return" in vocab

    def test_encode_decode(self):
        src = "int x = 5;"
        tokens = self.tok.tokenize(src)
        vocab = CTokenizer.build_vocab([src])
        ids = CTokenizer.encode(tokens, vocab)
        assert len(ids) == len(tokens)
        decoded = CTokenizer.decode(ids, vocab)
        assert decoded == tokens

    def test_unknown_token(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "int": 4}
        ids = CTokenizer.encode(["int", "mystery_token"], vocab)
        assert ids[0] == 4
        assert ids[1] == 1  # <UNK>

    def test_empty_source(self):
        tokens = self.tok.tokenize("")
        assert tokens == []

    def test_hex_literal(self):
        tokens = self.tok.tokenize("int x = 0xFF;")
        assert "0xFF" in tokens

    def test_function_signature(self):
        tokens = self.tok.tokenize("int add(int a, int b)")
        assert tokens == ["int", "add", "(", "int", "a", ",", "int", "b", ")"]
