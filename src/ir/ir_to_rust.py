"""IR (S-expression) → Rust code emitter."""
from __future__ import annotations

from src.ir.ir_types import (
    IRNode,
    FN, PARAMS, BODY, LET, ASSIGN, IF, WHILE, FOR, RETURN, CALL,
    BIN_OP, UN_OP, DEREF, ADDR_OF, INDEX, MEMBER, CAST, LITERAL,
    IDENT, TYPE, PTR, MUT, CONST, STRUCT, FIELD, BLOCK, SEQ,
    NAME, PARAM,
)

# C type → Rust primitive mapping
_C_TO_RUST_TYPE: dict[str, str] = {
    "int":              "i32",
    "unsigned int":     "u32",
    "unsigned":         "u32",
    "long":             "i64",
    "unsigned long":    "u64",
    "short":            "i16",
    "unsigned short":   "u16",
    "char":             "i8",
    "unsigned char":    "u8",
    "float":            "f32",
    "double":           "f64",
    "void":             "()",
    "size_t":           "usize",
    "ptrdiff_t":        "isize",
    "long long":        "i64",
    "unsigned long long": "u64",
}

# Operator pass-through (C → Rust operators are mostly identical)
_OP_MAP: dict[str, str] = {
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "&": "&", "|": "|", "^": "^", "<<": "<<", ">>": ">>",
    "&&": "&&", "||": "||",
    "p++": "/* p++ */", "p--": "/* p-- */",
}


class IRToRust:
    """Emit Rust source code from an IRNode tree."""

    def emit(self, node: IRNode) -> str:
        """Dispatch to the appropriate emit_* method."""
        method = f"emit_{node.kind}"
        emitter = getattr(self, method, self._emit_generic)
        return emitter(node)

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def emit_seq(self, node: IRNode) -> str:
        return "\n\n".join(self.emit(c) for c in node.children)

    def emit_fn(self, node: IRNode) -> str:
        name = self._child_value(node, NAME) or "unknown"
        ret_type_node = self._find_child(node, "ret_type")
        params_node = self._find_child(node, PARAMS)
        body_node = self._find_child(node, BLOCK)

        params_str = self.emit_params(params_node) if params_node else ""
        ret_str = ""
        if ret_type_node and ret_type_node.children:
            t = self.emit_type(ret_type_node.children[0])
            if t != "()":
                ret_str = f" -> {t}"

        body_str = self.emit_body(body_node) if body_node else "{ }"
        return f"fn {name}({params_str}){ret_str} {body_str}"

    def emit_params(self, node: IRNode) -> str:
        parts = []
        for child in (node.children if node else []):
            if child.kind == PARAM:
                type_node = child.children[0] if child.children else None
                name_node = child.children[1] if len(child.children) > 1 else None
                ptype = self.emit_type(type_node) if type_node else "_"
                pname = name_node.value if (name_node and name_node.value) else "_"
                parts.append(f"{pname}: {ptype}")
        return ", ".join(parts)

    def emit_body(self, node: IRNode) -> str:
        if node is None:
            return "{ }"
        stmts = []
        for child in node.children:
            s = self.emit_stmt(child)
            stmts.append(s)
        inner = "\n    ".join(stmts)
        return "{\n    " + inner + "\n}" if inner else "{ }"

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def emit_stmt(self, node: IRNode) -> str:
        if node.kind == RETURN:
            if node.children:
                return f"return {self.emit_expr(node.children[0])};"
            return "return;"
        if node.kind == LET:
            return self._emit_let(node)
        if node.kind == ASSIGN:
            lhs = self.emit_expr(node.children[0]) if node.children else "_"
            rhs = self.emit_expr(node.children[1]) if len(node.children) > 1 else "_"
            op = node.value or "="
            return f"{lhs} {op} {rhs};"
        if node.kind == IF:
            return self._emit_if(node)
        if node.kind == WHILE:
            return self._emit_while(node)
        if node.kind == FOR:
            return self._emit_for(node)
        if node.kind == BLOCK:
            return self.emit_body(node)
        if node.kind == SEQ:
            return "\n    ".join(self.emit_stmt(c) for c in node.children)
        if node.kind == "break":
            return "break;"
        if node.kind == "continue":
            return "continue;"
        if node.kind == "unsafe":
            inner = self.emit_stmt(node.children[0]) if node.children else ";"
            return f"unsafe {{ {inner} }}"
        if node.kind == "empty":
            return ""
        # expression statement
        return self.emit_expr(node) + ";"

    def _emit_let(self, node: IRNode) -> str:
        type_node = node.children[0] if node.children else None
        name_node = node.children[1] if len(node.children) > 1 else None
        init_node = node.children[2] if len(node.children) > 2 else None
        pname = name_node.value if (name_node and name_node.value) else "_"
        ptype = self.emit_type(type_node) if type_node else "_"
        if init_node:
            return f"let mut {pname}: {ptype} = {self.emit_expr(init_node)};"
        return f"let mut {pname}: {ptype};"

    def _emit_if(self, node: IRNode) -> str:
        cond = self.emit_expr(node.children[0]) if node.children else "true"
        then = self.emit_stmt(node.children[1]) if len(node.children) > 1 else "{ }"
        if not then.startswith("{"):
            then = f"{{ {then} }}"
        out = f"if {cond} {then}"
        if len(node.children) > 2:
            else_ = self.emit_stmt(node.children[2])
            if not else_.startswith("{") and not else_.startswith("if"):
                else_ = f"{{ {else_} }}"
            out += f" else {else_}"
        return out

    def _emit_while(self, node: IRNode) -> str:
        cond = self.emit_expr(node.children[0]) if node.children else "true"
        body = self.emit_stmt(node.children[1]) if len(node.children) > 1 else "{ }"
        if not body.startswith("{"):
            body = f"{{ {body} }}"
        return f"while {cond} {body}"

    def _emit_for(self, node: IRNode) -> str:
        # Convert C-style for loop to a Rust while loop
        init = self.emit_stmt(node.children[0]) if node.children else ""
        cond = self.emit_expr(node.children[1]) if len(node.children) > 1 else "true"
        step = self.emit_stmt(node.children[2]) if len(node.children) > 2 else ""
        body_node = node.children[3] if len(node.children) > 3 else IRNode(kind=BLOCK)
        body = self.emit_body(body_node) if body_node.kind == BLOCK else f"{{ {self.emit_stmt(body_node)} }}"
        # Inject step into body
        step_clean = step.rstrip(";")
        if step_clean:
            body = body.rstrip("}").rstrip() + f"\n    {step_clean};\n}}"
        init_clean = init.rstrip(";")
        return f"{init_clean};\nwhile {cond} {body}"

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def emit_expr(self, node: IRNode) -> str:
        if node.kind == LITERAL:
            return node.value or "0"
        if node.kind == IDENT:
            return node.value or "_"
        if node.kind == BIN_OP:
            op = _OP_MAP.get(node.value or "+", node.value or "+")
            lhs = self.emit_expr(node.children[0]) if node.children else "_"
            rhs = self.emit_expr(node.children[1]) if len(node.children) > 1 else "_"
            return f"({lhs} {op} {rhs})"
        if node.kind == UN_OP:
            op = node.value or "!"
            operand = self.emit_expr(node.children[0]) if node.children else "_"
            if op in ("p++", "++"):
                return f"{{ let _v = {operand}; {operand} += 1; _v }}"
            if op in ("p--", "--"):
                return f"{{ let _v = {operand}; {operand} -= 1; _v }}"
            if op == "-":
                return f"(-{operand})"
            if op == "!":
                return f"(!{operand})"
            return f"{op}{operand}"
        if node.kind == DEREF:
            inner = self.emit_expr(node.children[0]) if node.children else "_"
            return f"(*{inner})"
        if node.kind == ADDR_OF:
            inner = self.emit_expr(node.children[0]) if node.children else "_"
            return f"(&mut {inner})"
        if node.kind == INDEX:
            arr = self.emit_expr(node.children[0]) if node.children else "_"
            idx = self.emit_expr(node.children[1]) if len(node.children) > 1 else "0"
            return f"{arr}[{idx} as usize]"
        if node.kind == MEMBER:
            obj = self.emit_expr(node.children[0]) if node.children else "_"
            field = node.value or "field"
            return f"{obj}.{field}"
        if node.kind == CAST:
            t = self.emit_type(node.children[0]) if node.children else "_"
            e = self.emit_expr(node.children[1]) if len(node.children) > 1 else "_"
            return f"({e} as {t})"
        if node.kind == CALL:
            fn_name = node.value or "unknown"
            args = ", ".join(self.emit_expr(c) for c in node.children)
            return f"{fn_name}({args})"
        if node.kind == "unsafe":
            inner = self.emit_expr(node.children[0]) if node.children else "()"
            return f"unsafe {{ {inner} }}"
        if node.kind == ASSIGN:
            lhs = self.emit_expr(node.children[0]) if node.children else "_"
            rhs = self.emit_expr(node.children[1]) if len(node.children) > 1 else "_"
            op = node.value or "="
            return f"{{ {lhs} {op} {rhs} }}"
        if node.kind == "ternary":
            cond = self.emit_expr(node.children[0]) if node.children else "true"
            then = self.emit_expr(node.children[1]) if len(node.children) > 1 else "()"
            else_ = self.emit_expr(node.children[2]) if len(node.children) > 2 else "()"
            return f"(if {cond} {{ {then} }} else {{ {else_} }})"
        if node.kind == SEQ:
            parts = [self.emit_expr(c) for c in node.children]
            return ", ".join(parts)
        if node.kind == "init_list":
            parts = [self.emit_expr(c) for c in node.children]
            return "[" + ", ".join(parts) + "]"
        # Fallback
        return f"/* {node.kind} */()"

    # ------------------------------------------------------------------
    # Types
    # ------------------------------------------------------------------

    def emit_type(self, node: IRNode) -> str:
        if node.kind == TYPE:
            c_name = node.value or "void"
            return _C_TO_RUST_TYPE.get(c_name, c_name)
        if node.kind == PTR:
            mut_node = node.children[0] if node.children else None
            inner_node = node.children[1] if len(node.children) > 1 else None
            is_mut = mut_node is not None and mut_node.kind == MUT
            inner = self.emit_type(inner_node) if inner_node else "()"
            qualifier = "mut" if is_mut else "const"
            return f"*{qualifier} {inner}"
        if node.kind == "array":
            inner = self.emit_type(node.children[0]) if node.children else "()"
            size = node.value or "_"
            return f"[{inner}; {size}]"
        if node.kind == STRUCT:
            return node.value or "AnonStruct"
        return node.kind

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_child(self, node: IRNode, kind: str) -> IRNode | None:
        for c in node.children:
            if c.kind == kind:
                return c
        return None

    def _child_value(self, node: IRNode, kind: str) -> str | None:
        c = self._find_child(node, kind)
        return c.value if c else None

    def _emit_generic(self, node: IRNode) -> str:  # pragma: no cover
        return f"/* {node.kind} */()"

    def __repr__(self) -> str:
        return "IRToRust()"


if __name__ == "__main__":
    from src.ir.c_to_ir import CToIR
    converter = CToIR()
    emitter = IRToRust()
    src = "int add(int a, int b) { return a + b; }"
    ir = converter.convert(src)
    for child in ir.children:
        print(emitter.emit(child))
