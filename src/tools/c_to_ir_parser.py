"""Deterministic C to IR parser."""
from __future__ import annotations

import sys
import os
from typing import List

# Ensure src is in path
sys.path.append(os.getcwd())

from src.tokenizer.c_tokenizer import CTokenizer

class CToIRParser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0) -> str:
        if self.pos + offset >= len(self.tokens):
            return ""
        return self.tokens[self.pos + offset]

    def consume(self, expected=None) -> str:
        tok = self.peek()
        if expected and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok} at pos {self.pos}")
        self.pos += 1
        return tok

    def parse_type(self) -> str:
        t = self.consume()
        while self.peek() == "*":
            self.consume("*")
            # In the sample IR, pointers are represented as (ptr (mut) (type ...))
            t = f"(ptr (mut) (type {t}))"
        if not t.startswith("("):
            t = f"(type {t})"
        return t

    def parse_expr(self) -> str:
        return self.parse_comparison()

    def parse_comparison(self) -> str:
        left = self.parse_additive()
        while self.peek() in ("<", ">", "==", "!=", "<=", ">="):
            op = self.consume()
            right = self.parse_additive()
            left = f"(binop {op} {left} {right})"
        return left

    def parse_additive(self) -> str:
        left = self.parse_primary()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self.parse_primary()
            left = f"(binop {op} {left} {right})"
        return left

    def parse_primary(self) -> str:
        tok = self.consume()
        if tok.isdigit():
            return f"(literal {tok})"
        if tok == "(":
            e = self.parse_expr()
            self.consume(")")
            return e
        
        # Check for indexing or unary op
        if self.peek() == "[":
            self.consume("[")
            idx = self.parse_expr()
            self.consume("]")
            res = f"(index (ident {tok}) {idx})"
        else:
            res = f"(ident {tok})"
            
        # Check for post-unop
        if self.peek() in ("++", "--"):
            op = self.consume()
            kind = "p++" if op == "++" else "p--"
            return f"(unop {kind} {res})"
            
        return res

    def parse_stmt(self) -> str:
        tok = self.peek()
        if tok == "while":
            self.consume("while")
            self.consume("(")
            cond = self.parse_expr()
            self.consume(")")
            body = self.parse_stmt()
            return f"(while {cond} {body})"
        elif tok == "for":
            self.consume("for")
            self.consume("(")
            init = self.parse_stmt_raw() # let or assign
            # init has ; at end, need to strip or handle
            init_clean = init.strip().rstrip(";")
            cond = self.parse_expr()
            self.consume(";")
            inc = self.parse_expr()
            self.consume(")")
            body = self.parse_stmt()
            return f"(for (seq {init_clean}) {cond} {inc} {body})"
        elif tok == "if":
            self.consume("if")
            self.consume("(")
            cond = self.parse_expr()
            self.consume(")")
            body = self.parse_stmt()
            return f"(if {cond} {body})"
        elif tok == "{":
            return self.parse_block()
        else:
            return self.parse_stmt_raw()

    def parse_stmt_raw(self) -> str:
        tok = self.peek()
        if tok in ("int", "char", "void", "float"):
            typ = self.parse_type()
            res = []
            while True:
                ident = self.consume()
                expr = None
                if self.peek() == "=":
                    self.consume("=")
                    expr = self.parse_expr()
                res.append(f"(let {typ} (ident {ident}) {expr if expr else ''})".replace(" )", ")"))
                if self.peek() == ",":
                    self.consume(",")
                else:
                    break
            self.consume(";")
            return " ".join(res)
        else:
            lhs = self.parse_expr()
            if self.peek() == "=":
                self.consume("=")
                rhs = self.parse_expr()
                self.consume(";")
                return f"(assign = {lhs} {rhs})"
            if self.peek() == ";":
                self.consume(";")
            return lhs

    def parse_block(self) -> str:
        self.consume("{")
        stmts = []
        while self.peek() != "}":
            stmts.append(self.parse_stmt())
        self.consume("}")
        return f"(block {' '.join(stmts)})"

    def parse_function(self) -> str:
        ret_type = self.parse_type()
        name = self.consume()
        self.consume("(")
        params = []
        while self.peek() != ")":
            ptype = self.parse_type()
            pname = self.consume()
            params.append(f"(param {ptype} (ident {pname}))")
            if self.peek() == ",":
                self.consume(",")
        self.consume(")")
        block = self.parse_block()
        return f"(fn (name {name}) (ret_type {ret_type}) (params {' '.join(params)}) {block})"

def main():
    if len(sys.argv) < 2:
        print("Usage: python c_to_ir_parser.py <input.c>")
        return
    
    with open(sys.argv[1], "r") as f:
        src = f.read()
    
    tok = CTokenizer()
    tokens = tok.tokenize(src)
    parser = CToIRParser(tokens)
    try:
        ir = parser.parse_function()
        print(ir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
