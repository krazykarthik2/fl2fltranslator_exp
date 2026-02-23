# fl2fltranslator_exp — Neural Compiler Loop v0

> 🔥 **Neural Compiler Loop v0** — A neural C-to-Rust compiler with feedback-driven self-improvement

## Architecture

```
C source code
    ↓
C Tokenizer
    ↓
Encoder (Transformer: 6L × 512d × 8h × 2048FFN ≈ 40M params)
    ↓
Latent Representation  ← auxiliary losses (ownership / mutability / lifetime / unsafe)
    ↓
IR Decoder (S-expression normalized C AST)
    ↓
Rust Decoder
```

The compiler is split into two learned stages separated by a symbolic IR:

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| 1 | `CToIRModel` | C tokens | S-expression IR |
| 2 | `IRToRustModel` | IR tokens | Rust tokens |

## Project Structure

```
fl2fltranslator_exp/
├── src/
│   ├── tokenizer/
│   │   └── c_tokenizer.py       # Regex-based C lexer + vocab builder
│   ├── ir/
│   │   ├── ir_types.py          # IRNode dataclass + S-expression serialization
│   │   ├── c_to_ir.py           # pycparser-based C → IR converter
│   │   └── ir_to_rust.py        # IR → Rust emitter
│   ├── model/
│   │   ├── transformer.py       # Full encoder-decoder transformer (~40M params)
│   │   ├── multitask_head.py    # Ownership/mutability/lifetime/unsafe heads
│   │   ├── c_to_ir_model.py     # Stage 1 model (C → IR + aux heads)
│   │   └── ir_to_rust_model.py  # Stage 2 model (IR → Rust)
│   ├── data/
│   │   ├── synthetic_gen.py     # Synthetic C function generator (23+ templates)
│   │   └── dataset.py           # TranslationDataset + DataCollator
│   ├── feedback/
│   │   ├── cargo_checker.py     # Runs `cargo check` on generated Rust
│   │   └── error_parser.py      # Parses cargo JSON output into CompileError
│   └── training/
│       ├── train_c_to_ir.py     # Stage 1 training loop
│       ├── train_ir_to_rust.py  # Stage 2 training loop
│       └── self_play.py         # Self-play refinement loop
├── dataset/
│   └── samples/
│       ├── c/                   # 51 example C functions
│       ├── ir/                  # Corresponding IR S-expressions
│       └── rust/                # Corresponding Rust functions
├── tests/
│   ├── test_tokenizer.py
│   ├── test_ir.py
│   ├── test_model.py
│   ├── test_data_gen.py
│   └── test_feedback.py
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
pip install -r requirements.txt

# Verify everything works
python -m pytest tests/ -v

# Generate more synthetic data
python -m src.data.synthetic_gen

# Train Stage 1 (C → IR)
python -m src.training.train_c_to_ir --data-dir dataset/samples --epochs 20

# Train Stage 2 (IR → Rust)
python -m src.training.train_ir_to_rust --data-dir dataset/samples --epochs 20
```

## IR Format

S-expression normalized C AST — no macros, no typedef, explicit pointer levels and mutability:

```
(fn (name add)
  (ret_type (type int))
  (params
    (param (type int) (ident a))
    (param (type int) (ident b)))
  (block
    (return (binop + (ident a) (ident b)))))
```

### Type Mappings

| C type | IR | Rust |
|--------|----|------|
| `int` | `(type int)` | `i32` |
| `long` | `(type long)` | `i64` |
| `int*` | `(ptr (mut) (type int))` | `*mut i32` |
| `const int*` | `(ptr (const) (type int))` | `*const i32` |
| `char*` | `(ptr (mut) (type char))` | `*mut i8` |
| `void` | `(type void)` | `()` |

## Multi-Task Auxiliary Losses

The encoder output drives four classification heads that encourage the model to learn Rust ownership semantics:

| Head | Classes | Purpose |
|------|---------|---------|
| `OwnershipClassifier` | owned, borrowed, borrowed_mut, raw_ptr | Infer Rust ownership |
| `MutabilityClassifier` | immutable, mutable | Track `mut` annotations |
| `LifetimeClassifier` | static, local, parameter, heap | Lifetime origin |
| `UnsafeClassifier` | safe, unsafe | Flag unsafe operations |

## Self-Play Refinement

```
C source
   │
   ▼
CToIRModel  ──►  IRToRustModel  ──►  cargo check
                                          │
                              ┌───────────┴────────────┐
                           success                   failure
                              │                         │
                    positive dataset             RustErrorParser
                                                         │
                                              correction prompt
                                                         │
                                              negative dataset
```

Run the self-play loop:

```python
from src.training.self_play import SelfPlayTrainer
from src.feedback.cargo_checker import CargoChecker

with CargoChecker() as checker:
    trainer = SelfPlayTrainer(c_to_ir_model, ir_to_rust_model, checker,
                               src_vocab, tgt_vocab)
    summary = trainer.run_loop(c_samples, n_iterations=200)
    print(summary)
```

## Dataset

`dataset/samples/` contains 51 hand-crafted C/IR/Rust triples covering:

- Simple arithmetic (`add`, `subtract`, `multiply`, `square`, `cube`)
- Comparisons (`max_val`, `min_val`, `abs_val`, `clamp`, `sign`)
- Pointer operations (`swap`, `deref_add`, `set_via_ptr`, `double_deref`)
- Array manipulation (`sum_array`, `bubble_sort`, `reverse_array`, `find_max`)
- Loops (`sum_range`, `count_down`, `fill_array`, `copy_array`)
- Bit operations (`toggle_bit`, `set_bit`, `clear_bit`, `check_bit`)
- Recursion (`factorial`, `fibonacci`, `gcd`)

## Transformer Parameters

For `vocab_size=8000, d_model=512, n_heads=8, n_layers=6, d_ff=2048`:

```
Encoder embedding:    8000 × 512  =  4,096,000
Per encoder layer:    4 × 512²   +  2 × 512 × 2048  ≈  3,145,728
6 encoder layers:                                     ≈ 18,874,368
6 decoder layers (×2 due to cross-attn):              ≈ 25,165,824
Output projection:    512 × 8000  =  4,096,000
─────────────────────────────────────────────────────
Total:                                                ≈ 43M params
```

## Training Configuration

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4       # Adam with warmup
    n_epochs: int = 20
    warmup_steps: int = 4000          # Transformer LR schedule
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    aux_loss_weight: float = 0.1      # Weight for auxiliary heads
    label_smoothing: float = 0.1
```

## Running Tests

```bash
pytest tests/ -v
# 60 tests: tokenizer, IR conversion, model shapes, data generation, feedback parsing
```
