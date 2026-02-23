"""C source code tokenizer using regex-based lexing."""
from __future__ import annotations

import re
from typing import Dict, List

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while",
}

# Token patterns ordered from most to least specific
_TOKEN_PATTERNS: List[tuple[str, str]] = [
    ("COMMENT",     r"//[^\n]*|/\*.*?\*/"),
    ("STRING",      r'"(?:[^"\\]|\\.)*"'),
    ("CHAR",        r"'(?:[^'\\]|\\.)'"),
    ("FLOAT",       r"\b\d+\.\d*(?:[eE][+-]?\d+)?[fFlL]?\b|\b\d+[eE][+-]?\d+[fFlL]?\b"),
    ("HEX",         r"\b0[xX][0-9a-fA-F]+[uUlL]*\b"),
    ("OCT",         r"\b0[0-7]+[uUlL]*\b"),
    ("INT",         r"\b\d+[uUlL]*\b"),
    ("IDENT",       r"\b[A-Za-z_][A-Za-z0-9_]*\b"),
    ("OP3",         r"<<=|>>="),
    ("OP2",         r"->|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|[+\-*/%&|^]=|\.\.\.",),
    ("OP1",         r"[+\-*/%&|^~!<>=?:.,;]"),
    ("PAREN",       r"[(){}\[\]]"),
    ("WHITESPACE",  r"\s+"),
    ("UNKNOWN",     r"."),
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS),
    re.DOTALL,
)


class CTokenizer:
    """Regex-based C tokenizer."""

    def tokenize(self, source: str) -> List[str]:
        """Tokenize C source into a list of token strings (comments/whitespace stripped)."""
        tokens: List[str] = []
        for m in _MASTER_RE.finditer(source):
            kind = m.lastgroup
            text = m.group()
            if kind in ("WHITESPACE", "COMMENT"):
                continue
            tokens.append(text)
        return tokens

    @classmethod
    def build_vocab(cls, corpus: List[str]) -> Dict[str, int]:
        """Build vocabulary from a list of C source strings.

        Special tokens occupy indices 0-3:
          <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
        """
        tokenizer = cls()
        freq: Dict[str, int] = {}
        for src in corpus:
            for tok in tokenizer.tokenize(src):
                freq[tok] = freq.get(tok, 0) + 1

        vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for tok in sorted(freq, key=lambda t: -freq[t]):
            if tok not in vocab:
                vocab[tok] = len(vocab)
        return vocab

    @staticmethod
    def encode(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
        """Map token strings → integer ids."""
        unk = vocab.get("<UNK>", 1)
        return [vocab.get(t, unk) for t in tokens]

    @staticmethod
    def decode(ids: List[int], vocab: Dict[str, int]) -> List[str]:
        """Map integer ids → token strings."""
        inv = {v: k for k, v in vocab.items()}
        return [inv.get(i, "<UNK>") for i in ids]

    def __repr__(self) -> str:  # pragma: no cover
        return "CTokenizer()"


if __name__ == "__main__":
    src = "int add(int a, int b) { return a + b; }"
    tok = CTokenizer()
    tokens = tok.tokenize(src)
    print("Tokens:", tokens)
    vocab = CTokenizer.build_vocab([src])
    ids = CTokenizer.encode(tokens, vocab)
    print("IDs:   ", ids)
    print("Decoded:", CTokenizer.decode(ids, vocab))
