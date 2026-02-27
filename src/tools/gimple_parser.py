"""Deterministic GIMPLE to S-expression IR parser."""
import sys
import re
import os

def clean_tok(tok):
    return tok.strip().strip("()")

def parse_gimple(lines):
    ir_parts = []
    
    re_func = re.compile(r'^(\w+)\s+(\w+)\s*\((.*)\)$')
    re_decl = re.compile(r'^\s*([\w\* ]+)\s+(\w+);$')
    re_assign = re.compile(r'^\s*([^=]+)\s*=\s*(.+);$')
    re_goto = re.compile(r'^\s*goto\s+(<[^>]+>);$')
    re_label = re.compile(r'^\s*(<[^>]+>):$')
    re_if = re.compile(r'^\s*if\s+\((.+)\)\s+goto\s+(<[^>]+>);\s+else\s+goto\s+(<[^>]+>);$')

    stack = []
    current_block = []

    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Handle blocks
        if line == "{":
            stack.append(current_block)
            current_block = []
            continue
        if line == "}":
            block_content = " ".join(current_block)
            current_block = stack.pop()
            current_block.append(f"(block {block_content})")
            continue

        # Function header
        m = re_func.match(line)
        if m:
            ret, name, params = m.groups()
            param_list = []
            if params.strip():
                for p in params.split(","):
                    p = p.strip()
                    parts = p.split()
                    if len(parts) >= 2:
                        pname = parts[-1]
                        ptype = " ".join(parts[:-1])
                        param_list.append(f"(param (type {ptype}) (ident {pname}))")
            ir_parts.append(f"(fn (name {name}) (ret_type (type {ret})) (params {' '.join(param_list)})")
            continue

        # Declaration
        m = re_decl.match(line)
        if m:
            typ, name = m.groups()
            current_block.append(f"(let (type {typ.strip()}) (ident {name}))")
            continue

        # If
        m = re_if.match(line)
        if m:
            cond, g1, g2 = m.groups()
            # Simplify condition parsing
            current_block.append(f"(if (expr {cond}) (goto (label {g1})) (goto (label {g2})))")
            continue

        # Goto
        m = re_goto.match(line)
        if m:
            dest = m.group(1)
            current_block.append(f"(goto (label {dest}))")
            continue

        # Label
        m = re_label.match(line)
        if m:
            name = m.group(1)
            current_block.append(f"(label {name})")
            continue

        # Assignment
        m = re_assign.match(line)
        if m:
            lhs, rhs = m.groups()
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # Simple expression heuristics
            if "+" in rhs:
                op_parts = rhs.split("+")
                rhs_ir = f"(binop + (ident {op_parts[0].strip()}) (ident {op_parts[1].strip()}))"
            elif "*" in rhs and not rhs.startswith("*"):
                op_parts = rhs.split("*")
                rhs_ir = f"(binop * (ident {op_parts[0].strip()}) (ident {op_parts[1].strip()}))"
            elif rhs.startswith("*"):
                rhs_ir = f"(deref (ident {rhs[1:].strip()}))"
            else:
                rhs_ir = f"(ident {rhs})"
            
            if lhs.startswith("*"):
                lhs_ir = f"(deref (ident {lhs[1:].strip()}))"
            else:
                lhs_ir = f"(ident {lhs})"
                
            current_block.append(f"(assign {lhs_ir} {rhs_ir})")
            continue
            
    # Close function
    final_ir = " ".join(ir_parts) + " " + " ".join(current_block) + ")"
    return final_ir

def main():
    if len(sys.argv) < 2:
        print("Usage: python gimple_parser.py <input.gimple>")
        return
    
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()
        
    ir = parse_gimple(lines)
    print(ir)

if __name__ == "__main__":
    main()
