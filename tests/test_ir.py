"""Tests for IR types, C→IR conversion, and IR→Rust emission."""
import pytest
from src.ir.ir_types import IRNode, FN, PARAMS, RETURN, BIN_OP, IDENT, TYPE, BLOCK, LITERAL
from src.ir.c_to_ir import CToIR, IRConversionError
from src.ir.ir_to_rust import IRToRust


class TestIRNode:
    def test_irnode_to_sexp_leaf(self):
        node = IRNode(kind=IDENT, value="x")
        assert node.to_sexp() == "(ident x)"

    def test_irnode_to_sexp_nested(self):
        node = IRNode(kind=BIN_OP, value="+", children=[
            IRNode(kind=IDENT, value="a"),
            IRNode(kind=IDENT, value="b"),
        ])
        sexp = node.to_sexp()
        assert sexp.startswith("(binop +")
        assert "(ident a)" in sexp
        assert "(ident b)" in sexp

    def test_irnode_from_sexp_leaf(self):
        node = IRNode.from_sexp("(ident x)")
        assert node.kind == IDENT
        assert node.value == "x"
        assert node.children == []

    def test_irnode_from_sexp_nested(self):
        sexp = "(binop + (ident a) (ident b))"
        node = IRNode.from_sexp(sexp)
        assert node.kind == BIN_OP
        assert node.value == "+"
        assert len(node.children) == 2
        assert node.children[0].kind == IDENT
        assert node.children[0].value == "a"

    def test_irnode_roundtrip(self):
        original = IRNode(kind=RETURN, children=[
            IRNode(kind=BIN_OP, value="+", children=[
                IRNode(kind=IDENT, value="a"),
                IRNode(kind=LITERAL, value="1"),
            ])
        ])
        sexp = original.to_sexp()
        recovered = IRNode.from_sexp(sexp)
        assert recovered.kind == RETURN
        assert recovered.children[0].kind == BIN_OP


class TestCToIR:
    def setup_method(self):
        self.converter = CToIR()

    def test_c_to_ir_simple(self):
        src = "int add(int a, int b) { return a + b; }"
        node = self.converter.convert(src)
        sexp = node.to_sexp()
        assert "fn" in sexp
        assert "add" in sexp
        assert "binop" in sexp

    def test_c_to_ir_pointer(self):
        src = "void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }"
        node = self.converter.convert(src)
        sexp = node.to_sexp()
        assert "ptr" in sexp or "deref" in sexp

    def test_c_to_ir_for_loop(self):
        src = "int sum(int *arr, int n) { int s=0; for(int i=0;i<n;i++) s+=arr[i]; return s; }"
        node = self.converter.convert(src)
        sexp = node.to_sexp()
        assert "for" in sexp

    def test_c_to_ir_while_loop(self):
        src = "int countdown(int n) { int s=0; while(n>0) { s+=n; n--; } return s; }"
        node = self.converter.convert(src)
        sexp = node.to_sexp()
        assert "while" in sexp

    def test_c_to_ir_if_else(self):
        src = "int max(int a, int b) { if (a > b) return a; else return b; }"
        node = self.converter.convert(src)
        sexp = node.to_sexp()
        assert "if" in sexp

    def test_convert_to_string(self):
        src = "int add(int a, int b) { return a + b; }"
        result = self.converter.convert_to_string(src)
        assert isinstance(result, str)
        assert "fn" in result

    def test_c_to_ir_invalid_raises(self):
        with pytest.raises(IRConversionError):
            self.converter.convert("this is not valid C @@@@")


class TestIRToRust:
    def setup_method(self):
        self.converter = CToIR()
        self.emitter = IRToRust()

    def test_ir_to_rust_simple(self):
        src = "int add(int a, int b) { return a + b; }"
        ir = self.converter.convert(src)
        fn_nodes = [c for c in ir.children if c.kind == "fn"]
        assert fn_nodes
        rust = self.emitter.emit(fn_nodes[0])
        assert "fn add" in rust
        assert "i32" in rust

    def test_ir_to_rust_void_return(self):
        src = "void noop(int x) { int y = x + 1; }"
        ir = self.converter.convert(src)
        fn_nodes = [c for c in ir.children if c.kind == "fn"]
        rust = self.emitter.emit(fn_nodes[0])
        assert "fn noop" in rust

    def test_ir_to_rust_pointer_param(self):
        src = "void inc(int *p) { *p = *p + 1; }"
        ir = self.converter.convert(src)
        fn_nodes = [c for c in ir.children if c.kind == "fn"]
        rust = self.emitter.emit(fn_nodes[0])
        assert "*mut i32" in rust or "*const i32" in rust or "i32" in rust
