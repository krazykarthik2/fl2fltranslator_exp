"""C source → S-expression IR converter using pycparser."""
from __future__ import annotations

from typing import Optional

try:
    import pycparser
    from pycparser import c_ast, c_generator
    _PYCPARSER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYCPARSER_AVAILABLE = False

from src.ir.ir_types import (
    IRNode,
    FN, PARAMS, BODY, LET, ASSIGN, IF, WHILE, FOR, RETURN, CALL,
    BIN_OP, UN_OP, DEREF, ADDR_OF, INDEX, MEMBER, CAST, LITERAL,
    IDENT, TYPE, PTR, MUT, CONST, STRUCT, FIELD, BLOCK, SEQ,
    NAME, PARAM,
)


class IRConversionError(Exception):
    """Raised when C source cannot be converted to IR."""


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class _IRVisitor:
    """Walk a pycparser AST and produce IRNode trees."""

    def visit(self, node) -> IRNode:
        method = "visit_" + type(node).__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node) -> IRNode:  # pragma: no cover
        children = [self.visit(c) for _, c in (node.children() if node else [])]
        return IRNode(kind=type(node).__name__.lower(), children=children)

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def _build_type(self, type_node) -> IRNode:
        """Convert a pycparser type node to an IRNode."""
        if isinstance(type_node, c_ast.TypeDecl):
            return self._build_type(type_node.type)
        if isinstance(type_node, c_ast.IdentifierType):
            name = " ".join(type_node.names)
            return IRNode(kind=TYPE, value=name)
        if isinstance(type_node, c_ast.PtrDecl):
            quals = getattr(type_node.type, "quals", [])
            mutability = IRNode(kind=CONST if "const" in quals else MUT)
            inner = self._build_type(type_node.type)
            return IRNode(kind=PTR, children=[mutability, inner])
        if isinstance(type_node, c_ast.ArrayDecl):
            inner = self._build_type(type_node.type)
            dim = type_node.dim
            dim_val = None
            if dim is not None:
                dim_val = self._expr_value(dim)
            arr = IRNode(kind="array", children=[inner])
            if dim_val:
                arr.value = dim_val
            return arr
        if isinstance(type_node, c_ast.Struct):
            return IRNode(kind=STRUCT, value=type_node.name or "anon")
        if isinstance(type_node, c_ast.FuncDecl):
            return IRNode(kind="fn_ptr")
        return IRNode(kind=TYPE, value="unknown")

    def _expr_value(self, node) -> str:
        """Best-effort stringify of a simple expression node."""
        gen = c_generator.CGenerator()
        try:
            return gen.visit(node)
        except Exception:
            return "?"

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def visit_FileAST(self, node) -> IRNode:
        children = [self.visit(c) for _, c in node.children()]
        return IRNode(kind=SEQ, children=children)

    def visit_FuncDef(self, node) -> IRNode:
        fn_name = node.decl.name
        ret_type = self._build_type(node.decl.type.type)

        # params
        param_list = node.decl.type.args
        params_children: list = []
        if param_list:
            for _, p in param_list.children():
                ptype = self._build_type(p.type)
                pname = IRNode(kind=IDENT, value=p.name or "_")
                params_children.append(IRNode(kind=PARAM, children=[ptype, pname]))
        params_node = IRNode(kind=PARAMS, children=params_children)

        body = self.visit(node.body)

        return IRNode(kind=FN, children=[
            IRNode(kind=NAME, value=fn_name),
            IRNode(kind="ret_type", children=[ret_type]),
            params_node,
            body,
        ])

    def visit_Decl(self, node) -> IRNode:
        if isinstance(node.type, (c_ast.FuncDecl,)):
            return IRNode(kind="fn_decl", value=node.name)

        type_node = self._build_type(node.type)
        name_node = IRNode(kind=IDENT, value=node.name or "_")
        children = [type_node, name_node]
        if node.init is not None:
            children.append(self.visit(node.init))
        return IRNode(kind=LET, children=children)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def visit_Compound(self, node) -> IRNode:
        stmts = [self.visit(c) for _, c in node.children()] if node.block_items else []
        return IRNode(kind=BLOCK, children=stmts)

    def visit_Return(self, node) -> IRNode:
        children = [self.visit(node.expr)] if node.expr else []
        return IRNode(kind=RETURN, children=children)

    def visit_If(self, node) -> IRNode:
        cond = self.visit(node.cond)
        then = self.visit(node.iftrue)
        children = [cond, then]
        if node.iffalse:
            children.append(self.visit(node.iffalse))
        return IRNode(kind=IF, children=children)

    def visit_While(self, node) -> IRNode:
        cond = self.visit(node.cond)
        body = self.visit(node.stmt)
        return IRNode(kind=WHILE, children=[cond, body])

    def visit_DoWhile(self, node) -> IRNode:
        body = self.visit(node.stmt)
        cond = self.visit(node.cond)
        return IRNode(kind="do_while", children=[body, cond])

    def visit_For(self, node) -> IRNode:
        children = []
        children.append(self.visit(node.init) if node.init else IRNode(kind="empty"))
        children.append(self.visit(node.cond) if node.cond else IRNode(kind="empty"))
        children.append(self.visit(node.next) if node.next else IRNode(kind="empty"))
        children.append(self.visit(node.stmt))
        return IRNode(kind=FOR, children=children)

    def visit_Assignment(self, node) -> IRNode:
        lhs = self.visit(node.lvalue)
        rhs = self.visit(node.rvalue)
        return IRNode(kind=ASSIGN, value=node.op, children=[lhs, rhs])

    def visit_Break(self, node) -> IRNode:
        return IRNode(kind="break")

    def visit_Continue(self, node) -> IRNode:
        return IRNode(kind="continue")

    def visit_Switch(self, node) -> IRNode:
        cond = self.visit(node.cond)
        body = self.visit(node.stmt)
        return IRNode(kind="switch", children=[cond, body])

    def visit_Case(self, node) -> IRNode:
        expr = self.visit(node.expr)
        stmts = [self.visit(s) for s in (node.stmts or [])]
        return IRNode(kind="case", children=[expr] + stmts)

    def visit_Default(self, node) -> IRNode:
        stmts = [self.visit(s) for s in (node.stmts or [])]
        return IRNode(kind="default", children=stmts)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def visit_BinaryOp(self, node) -> IRNode:
        left = self.visit(node.left)
        right = self.visit(node.right)
        return IRNode(kind=BIN_OP, value=node.op, children=[left, right])

    def visit_UnaryOp(self, node) -> IRNode:
        if node.op == "*":
            return IRNode(kind=DEREF, children=[self.visit(node.expr)])
        if node.op == "&":
            return IRNode(kind=ADDR_OF, children=[self.visit(node.expr)])
        return IRNode(kind=UN_OP, value=node.op, children=[self.visit(node.expr)])

    def visit_Constant(self, node) -> IRNode:
        return IRNode(kind=LITERAL, value=node.value)

    def visit_ID(self, node) -> IRNode:
        return IRNode(kind=IDENT, value=node.name)

    def visit_FuncCall(self, node) -> IRNode:
        fn_name = self._expr_value(node.name)
        args = []
        if node.args:
            args = [self.visit(a) for _, a in node.args.children()]
        # Mark malloc/free/calloc/realloc as unsafe
        is_unsafe = fn_name in ("malloc", "free", "calloc", "realloc", "memcpy", "memmove")
        node_out = IRNode(kind=CALL, value=fn_name, children=args)
        if is_unsafe:
            node_out = IRNode(kind="unsafe", children=[node_out])
        return node_out

    def visit_ArrayRef(self, node) -> IRNode:
        arr = self.visit(node.name)
        idx = self.visit(node.subscript)
        return IRNode(kind=INDEX, children=[arr, idx])

    def visit_StructRef(self, node) -> IRNode:
        obj = self.visit(node.name)
        field_name = node.field.name
        return IRNode(kind=MEMBER, value=field_name, children=[obj])

    def visit_Cast(self, node) -> IRNode:
        ttype = self._build_type(node.to_type.type)
        expr = self.visit(node.expr)
        return IRNode(kind=CAST, children=[ttype, expr])

    def visit_ExprList(self, node) -> IRNode:
        exprs = [self.visit(e) for _, e in node.children()]
        return IRNode(kind=SEQ, children=exprs)

    def visit_TernaryOp(self, node) -> IRNode:
        cond = self.visit(node.cond)
        then = self.visit(node.iftrue)
        else_ = self.visit(node.iffalse)
        return IRNode(kind="ternary", children=[cond, then, else_])

    def visit_NamedInitializer(self, node) -> IRNode:
        val = self.visit(node.expr)
        return IRNode(kind="named_init", children=val.children, value=val.value)

    def visit_InitList(self, node) -> IRNode:
        exprs = [self.visit(e) for _, e in node.children()]
        return IRNode(kind="init_list", children=exprs)

    def visit_Typename(self, node) -> IRNode:
        return self._build_type(node.type)

    def visit_EmptyStatement(self, node) -> IRNode:
        return IRNode(kind="empty")

    def visit_DeclList(self, node) -> IRNode:
        decls = [self.visit(d) for _, d in node.children()]
        return IRNode(kind=SEQ, children=decls)

    def visit_Typedef(self, node) -> IRNode:
        return IRNode(kind="typedef", value=node.name)

    def visit_Struct(self, node) -> IRNode:
        children = []
        if node.decls:
            for _, d in node.decls.children() if hasattr(node.decls, "children") else []:
                children.append(self.visit(d))
        return IRNode(kind=STRUCT, value=node.name or "anon", children=children)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Minimal fake headers so pycparser can parse standard C
_FAKE_LIBC_HEADER = """
typedef unsigned long size_t;
typedef int ptrdiff_t;
void *malloc(size_t size);
void free(void *ptr);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void *memcpy(void *dest, const void *src, size_t n);
void *memmove(void *dest, const void *src, size_t n);
int printf(const char *fmt, ...);
int scanf(const char *fmt, ...);
"""


class CToIR:
    """Convert C source code to an S-expression IR tree."""

    def convert(self, c_source: str) -> IRNode:
        """Parse *c_source* and return an IRNode tree."""
        if not _PYCPARSER_AVAILABLE:
            raise IRConversionError("pycparser is not installed")
        try:
            full_source = _FAKE_LIBC_HEADER + "\n" + c_source
            parser = pycparser.CParser()
            ast = parser.parse(full_source, filename="<input>")
            visitor = _IRVisitor()
            result = visitor.visit(ast)
            # Filter out nodes from the fake header (fn_decl nodes for stdlib funcs)
            result.children = [
                c for c in result.children
                if not (c.kind == "fn_decl")
            ]
            return result
        except pycparser.c_parser.ParseError as e:
            raise IRConversionError(f"Parse error: {e}") from e
        except Exception as e:
            raise IRConversionError(f"Conversion error: {e}") from e

    def convert_to_string(self, c_source: str) -> str:
        """Convenience wrapper that returns the S-expression string."""
        return self.convert(c_source).to_sexp()

    def __repr__(self) -> str:
        return "CToIR()"


if __name__ == "__main__":
    converter = CToIR()
    src = "int add(int a, int b) { return a + b; }"
    print(converter.convert_to_string(src))
