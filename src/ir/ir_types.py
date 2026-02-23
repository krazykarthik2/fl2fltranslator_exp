"""S-expression IR node types and IRNode dataclass."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# IR node kind constants
# ---------------------------------------------------------------------------
FN = "fn"
PARAMS = "params"
BODY = "body"
LET = "let"
ASSIGN = "assign"
IF = "if"
WHILE = "while"
FOR = "for"
RETURN = "return"
CALL = "call"
BIN_OP = "binop"
UN_OP = "unop"
DEREF = "deref"
ADDR_OF = "addr_of"
INDEX = "index"
MEMBER = "member"
CAST = "cast"
LITERAL = "literal"
IDENT = "ident"
TYPE = "type"
PTR = "ptr"
MUT = "mut"
CONST = "const"
STRUCT = "struct"
FIELD = "field"
BLOCK = "block"
SEQ = "seq"
NAME = "name"
PARAM = "param"


# ---------------------------------------------------------------------------
# IRNode
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    """A node in the S-expression intermediate representation."""

    kind: str
    children: List["IRNode"] = field(default_factory=list)
    value: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_sexp(self) -> str:
        """Serialize this node to an S-expression string."""
        if not self.children and self.value is not None:
            return f"({self.kind} {self.value})"
        parts = [self.kind]
        if self.value is not None:
            parts.append(self.value)
        parts.extend(child.to_sexp() for child in self.children)
        return "(" + " ".join(parts) + ")"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_sexp(cls, sexp: str) -> "IRNode":
        """Parse an S-expression string back to an IRNode tree."""
        tokens = cls._tokenize_sexp(sexp.strip())
        if not tokens:
            raise ValueError("Empty S-expression")
        node, _ = cls._parse_tokens(tokens, 0)
        return node

    @staticmethod
    def _tokenize_sexp(sexp: str) -> List[str]:
        """Split S-expression text into atomic tokens: '(', ')', or atoms."""
        return re.findall(r'\(|\)|[^\s()]+', sexp)

    @classmethod
    def _parse_tokens(cls, tokens: List[str], pos: int) -> Tuple["IRNode", int]:
        """Recursively parse tokens starting at *pos*; returns (node, next_pos)."""
        if tokens[pos] != "(":
            raise ValueError(f"Expected '(' at position {pos}, got {tokens[pos]!r}")
        pos += 1  # consume '('

        if pos >= len(tokens) or tokens[pos] == ")":
            raise ValueError("Empty parenthesised expression")

        kind = tokens[pos]
        pos += 1

        value: Optional[str] = None
        children: List[IRNode] = []

        while pos < len(tokens) and tokens[pos] != ")":
            tok = tokens[pos]
            if tok == "(":
                child, pos = cls._parse_tokens(tokens, pos)
                children.append(child)
            else:
                # bare atom — treat as value if first, otherwise wrap as leaf node
                if value is None and not children:
                    value = tok
                    pos += 1
                else:
                    # additional bare atom → create a leaf node
                    children.append(IRNode(kind=tok))
                    pos += 1

        if pos >= len(tokens):
            raise ValueError("Unmatched '(' in S-expression")
        pos += 1  # consume ')'

        return cls(kind=kind, children=children, value=value), pos

    def __repr__(self) -> str:
        return f"IRNode(kind={self.kind!r}, value={self.value!r}, children={self.children!r})"
