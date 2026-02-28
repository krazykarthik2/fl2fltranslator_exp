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
Latent-Space IR  ← auxiliary losses (ownership / mutability / lifetime / unsafe)
    ↓
Rust Decoder
    ↓
Rust source code
```

The compiler uses a **single unified model** where the encoder's hidden states form a
**latent-space intermediate representation**.  Auxiliary classification heads encourage
this latent space to learn structured semantic features (ownership, mutability, lifetime,
unsafe boundaries) without ever producing explicit symbolic IR tokens.

| Model | Input | Latent Space | Output |
|-------|-------|--------------|--------|
| `CToRustModel` | C tokens | Encoder hidden states (latent-space IR) | Rust tokens |

## Project Structure

```
fl2fltranslator_exp/
├── src/
│   ├── tokenizer/
│   │   └── c_tokenizer.py          # Regex-based C lexer + vocab builder
│   ├── model/
│   │   ├── transformer.py          # Full encoder-decoder transformer (~40M params)
│   │   ├── multitask_head.py       # Ownership/mutability/lifetime/unsafe heads
│   │   └── c_to_rust_model.py      # Unified model: C → latent-space IR → Rust
│   ├── data/
│   │   ├── synthetic_gen.py        # Synthetic C function generator (23+ templates)
│   │   └── dataset.py              # TranslationDataset + DataCollator
│   ├── feedback/
│   │   ├── cargo_checker.py        # Runs `cargo check` on generated Rust
│   │   └── error_parser.py         # Parses cargo JSON output into CompileError
│   ├── tools/
│   │   └── run_inference.py        # CLI inference driver (c2rust)
│   └── training/
│       ├── train_c_to_rust.py      # Training loop (C → Rust via latent-space IR)
│       └── self_play.py            # Self-play refinement loop
├── dataset/
│   └── samples/
│       ├── c/                      # 51 example C functions
│       └── rust/                   # Corresponding Rust functions
├── ARCHITECTURE.md
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
pip install -r requirements.txt

# Train the unified C → Rust model
python -m src.training.train_c_to_rust --data-dir dataset/samples --epochs 20
```

## Latent-Space IR

Instead of materialising an explicit S-expression IR, the encoder's hidden states
form a continuous representation shaped by four auxiliary classification heads:

| Head | Classes | Purpose |
|------|---------|---------|
| `OwnershipClassifier` | owned, borrowed, borrowed\_mut, raw\_ptr | Infer Rust ownership |
| `MutabilityClassifier` | immutable, mutable | Track `mut` annotations |
| `LifetimeClassifier` | static, local, parameter, heap | Lifetime origin |
| `UnsafeClassifier` | safe, unsafe | Flag unsafe operations |

### Benefits over Explicit IR

- **No information bottleneck**: continuous vectors preserve more detail than discrete tokens.
- **End-to-end optimisation**: a single model is trained jointly, avoiding error compounding between stages.
- **Single forward pass**: inference is faster with one model instead of two sequential ones.

## Self-Play Refinement

```
C source
   │
   ▼
CToRustModel  ──►  cargo check
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
    trainer = SelfPlayTrainer(model, checker, src_vocab, tgt_vocab)
    summary = trainer.run_loop(c_samples, n_iterations=200)
    print(summary)
```

## Dataset

`dataset/samples/` contains 51 hand-crafted C/Rust pairs covering:

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
Auxiliary heads:      512 × (4+2+4+2)  =      6,144
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

## Batch Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `do.bat setup` | Create venv, install deps | Before first run |
| `do.bat quick` | Train model for 1 epoch | Verify pipeline works |
| `do.bat train` | Train C→Rust model | Full training |
| `do.bat test` | Run pytest | Validate code |
| `do.bat all` | Train + test | Full pipeline |
| `modelctl.bat input.c [checkpoint_dir]` | Run inference | Convert C→Rust |
| `convert.bat input.c [output.rs]` | Convert C file to Rust | End-to-end conversion |
