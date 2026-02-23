"""Synthetic C code generator for training data."""
from __future__ import annotations

import os
import random as _random_module
from random import Random
from typing import List, Optional

# Module-level RNG used by template functions (seeded per-generator via _rng)
_rng = Random()

# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
_INT_VARS = ["a", "b", "c", "d", "i", "j", "k", "n", "m", "x", "y", "z",
             "val", "tmp", "res", "count", "sum", "idx", "len", "num"]
_PTR_VARS = ["p", "q", "ptr", "head", "node", "buf", "arr", "data", "cur", "next"]
_FN_NAMES = ["compute", "process", "transform", "calculate", "update",
             "find", "search", "sort", "copy", "merge", "split",
             "check", "verify", "apply", "reduce"]
_INT_TYPES = ["int", "long", "short", "unsigned int"]
_STRUCT_NAMES = ["Node", "Pair", "Entry", "Point", "Rect", "Stack", "Queue"]


def _rvar(pool: List[str]) -> str:
    return _rng.choice(pool)


def _rint(lo: int = 0, hi: int = 100) -> int:
    return _rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

def _tmpl_add() -> str:
    a, b = _rng.sample(_INT_VARS[:8], 2)
    t = _rng.choice(["int", "long"])
    fn = _rvar(_FN_NAMES)
    return f"{t} {fn}_{a}_{b}({t} {a}, {t} {b}) {{ return {a} + {b}; }}"


def _tmpl_subtract() -> str:
    a, b = _rng.sample(_INT_VARS[:8], 2)
    t = _rng.choice(["int", "long"])
    fn = _rvar(_FN_NAMES)
    return f"{t} {fn}({t} {a}, {t} {b}) {{ return {a} - {b}; }}"


def _tmpl_multiply() -> str:
    a, b = _rng.sample(_INT_VARS[:8], 2)
    t = _rng.choice(["int", "long"])
    fn = _rvar(_FN_NAMES)
    return f"{t} {fn}({t} {a}, {t} {b}) {{ return {a} * {b}; }}"


def _tmpl_max() -> str:
    a, b = _rng.sample(_INT_VARS[:8], 2)
    t = _rng.choice(["int", "long"])
    return f"{t} max_{a}_{b}({t} {a}, {t} {b}) {{ if ({a} > {b}) return {a}; return {b}; }}"


def _tmpl_min() -> str:
    a, b = _rng.sample(_INT_VARS[:8], 2)
    t = _rng.choice(["int", "long"])
    return f"{t} min_{a}_{b}({t} {a}, {t} {b}) {{ if ({a} < {b}) return {a}; return {b}; }}"


def _tmpl_abs() -> str:
    a = _rvar(_INT_VARS[:8])
    t = _rng.choice(["int", "long"])
    return f"{t} abs_{a}({t} {a}) {{ if ({a} < 0) return -{a}; return {a}; }}"


def _tmpl_swap() -> str:
    a, b = _rng.sample(_PTR_VARS[:6], 2)
    t = _rng.choice(["int", "long"])
    tmp = _rvar(["t", "tmp", "temp"])
    return (f"void swap_{a}_{b}({t} *{a}, {t} *{b}) "
            f"{{ {t} {tmp} = *{a}; *{a} = *{b}; *{b} = {tmp}; }}")


def _tmpl_sum_array() -> str:
    arr = _rvar(_PTR_VARS)
    n = _rvar(["n", "len", "size"])
    s = _rvar(["s", "sum", "total"])
    i = _rvar(["i", "j", "k"])
    t = _rng.choice(["int", "long"])
    return (f"{t} sum_array({t} *{arr}, int {n}) "
            f"{{ {t} {s} = 0; for(int {i} = 0; {i} < {n}; {i}++) {s} += {arr}[{i}]; return {s}; }}")


def _tmpl_factorial() -> str:
    n = _rvar(["n", "x", "k"])
    return (f"int factorial(int {n}) "
            f"{{ if ({n} <= 1) return 1; return {n} * factorial({n} - 1); }}")


def _tmpl_fibonacci() -> str:
    n = _rvar(["n", "x"])
    return (f"int fibonacci(int {n}) "
            f"{{ if ({n} <= 1) return {n}; return fibonacci({n} - 1) + fibonacci({n} - 2); }}")


def _tmpl_power() -> str:
    b, e = _rng.sample(["base", "exp", "n", "p", "x", "y"], 2)
    return (f"long power(long {b}, int {e}) "
            f"{{ long r = 1; while ({e} > 0) {{ r *= {b}; {e}--; }} return r; }}")


def _tmpl_gcd() -> str:
    a, b = _rng.sample(_INT_VARS[:6], 2)
    t = _rvar(["tmp", "t", "temp"])
    return (f"int gcd(int {a}, int {b}) "
            f"{{ while ({b} != 0) {{ int {t} = {b}; {b} = {a} % {b}; {a} = {t}; }} return {a}; }}")


def _tmpl_strlen() -> str:
    s = _rvar(["s", "str", "p"])
    n = _rvar(["n", "len", "i"])
    return (f"int my_strlen(const char *{s}) "
            f"{{ int {n} = 0; while ({s}[{n}]) {n}++; return {n}; }}")


def _tmpl_memset() -> str:
    buf, val, n = "buf", "val", "n"
    i = _rvar(["i", "j"])
    return (f"void my_memset(char *{buf}, char {val}, int {n}) "
            f"{{ for(int {i} = 0; {i} < {n}; {i}++) {buf}[{i}] = {val}; }}")


def _tmpl_bubble_sort() -> str:
    arr, n = _rvar(_PTR_VARS), _rvar(["n", "size", "len"])
    i, j, t = "i", "j", "tmp"
    return (f"void bubble_sort(int *{arr}, int {n}) "
            f"{{ for(int {i}=0; {i}<{n}-1; {i}++) "
            f"for(int {j}=0; {j}<{n}-{i}-1; {j}++) "
            f"if ({arr}[{j}] > {arr}[{j}+1]) {{ "
            f"int {t}={arr}[{j}]; {arr}[{j}]={arr}[{j}+1]; {arr}[{j}+1]={t}; }} }}")


def _tmpl_count_positive() -> str:
    arr, n = _rvar(_PTR_VARS), _rvar(["n", "size"])
    c, i = _rvar(["cnt", "count", "c"]), _rvar(["i", "j"])
    return (f"int count_positive(int *{arr}, int {n}) "
            f"{{ int {c}=0; for(int {i}=0;{i}<{n};{i}++) if({arr}[{i}]>0) {c}++; return {c}; }}")


def _tmpl_find_max_array() -> str:
    arr, n = _rvar(_PTR_VARS), _rvar(["n", "len"])
    m, i = _rvar(["m", "mx", "max_val"]), _rvar(["i", "j"])
    return (f"int find_max(int *{arr}, int {n}) "
            f"{{ int {m}={arr}[0]; for(int {i}=1;{i}<{n};{i}++) if({arr}[{i}]>{m}) {m}={arr}[{i}]; return {m}; }}")


def _tmpl_increment_ptr() -> str:
    p, n, i = _rvar(_PTR_VARS), _rvar(["n", "len"]), _rvar(["i", "j"])
    return (f"void increment_all(int *{p}, int {n}) "
            f"{{ for(int {i}=0;{i}<{n};{i}++) {p}[{i}]++; }}")


def _tmpl_dot_product() -> str:
    a, b, n = "a", "b", _rvar(["n", "len"])
    s, i = _rvar(["s", "sum"]), _rvar(["i", "j"])
    return (f"int dot_product(int *{a}, int *{b}, int {n}) "
            f"{{ int {s}=0; for(int {i}=0;{i}<{n};{i}++) {s}+={a}[{i}]*{b}[{i}]; return {s}; }}")


def _tmpl_is_palindrome() -> str:
    s, n = _rvar(["s", "str"]), _rvar(["n", "len"])
    i, j = "i", "j"
    return (f"int is_palindrome(const char *{s}, int {n}) "
            f"{{ int {i}=0, {j}={n}-1; "
            f"while({i}<{j}) {{ if({s}[{i}]!={s}[{j}]) return 0; {i}++; {j}--; }} return 1; }}")


def _tmpl_clamp() -> str:
    v, lo, hi = _rvar(["v", "val", "x"]), _rvar(["lo", "min_v", "a"]), _rvar(["hi", "max_v", "b"])
    t = _rng.choice(["int", "float", "double"])
    return (f"{t} clamp({t} {v}, {t} {lo}, {t} {hi}) "
            f"{{ if ({v} < {lo}) return {lo}; if ({v} > {hi}) return {hi}; return {v}; }}")


def _tmpl_linear_search() -> str:
    arr, n, target = _rvar(_PTR_VARS), _rvar(["n", "len"]), _rvar(["target", "key", "val"])
    i = _rvar(["i", "j"])
    return (f"int linear_search(int *{arr}, int {n}, int {target}) "
            f"{{ for(int {i}=0;{i}<{n};{i}++) if({arr}[{i}]=={target}) return {i}; return -1; }}")


def _tmpl_reverse_array() -> str:
    arr, n, i, j = _rvar(_PTR_VARS), _rvar(["n", "len"]), "i", "j"
    t = _rvar(["t", "tmp"])
    return (f"void reverse_array(int *{arr}, int {n}) "
            f"{{ int {i}=0, {j}={n}-1; "
            f"while({i}<{j}) {{ int {t}={arr}[{i}]; {arr}[{i}]={arr}[{j}]; {arr}[{j}]={t}; {i}++; {j}--; }} }}")


_TEMPLATES = [
    _tmpl_add, _tmpl_subtract, _tmpl_multiply, _tmpl_max, _tmpl_min,
    _tmpl_abs, _tmpl_swap, _tmpl_sum_array, _tmpl_factorial, _tmpl_fibonacci,
    _tmpl_power, _tmpl_gcd, _tmpl_strlen, _tmpl_memset, _tmpl_bubble_sort,
    _tmpl_count_positive, _tmpl_find_max_array, _tmpl_increment_ptr,
    _tmpl_dot_product, _tmpl_is_palindrome, _tmpl_clamp, _tmpl_linear_search,
    _tmpl_reverse_array,
]


class SyntheticCGenerator:
    """Generates synthetic C functions for training data."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = Random(seed)

    def _call_template(self, tmpl) -> str:
        """Call *tmpl* after pointing the module-level _rng at this instance's RNG."""
        global _rng
        _rng = self._rng
        return tmpl()

    def generate_function(self) -> str:
        """Generate one random C function string."""
        tmpl = self._rng.choice(_TEMPLATES)
        return self._call_template(tmpl)

    def generate_batch(self, n: int) -> List[str]:
        """Generate *n* random C function strings."""
        return [self.generate_function() for _ in range(n)]

    def generate_dataset(self, n: int, output_dir: str) -> None:
        """Generate *n* functions and save each as a separate .c file."""
        os.makedirs(output_dir, exist_ok=True)
        for idx in range(n):
            code = self.generate_function()
            path = os.path.join(output_dir, f"func_{idx:05d}.c")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code + "\n")

    def __repr__(self) -> str:
        return f"SyntheticCGenerator(templates={len(_TEMPLATES)})"


if __name__ == "__main__":
    gen = SyntheticCGenerator(seed=42)
    for fn in gen.generate_batch(5):
        print(fn)
        print()
