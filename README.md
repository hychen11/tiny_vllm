# nano-vLLM

`nano-vLLM` is a single-GPU, single-process inference engine for learning the core ideas behind vLLM. The project deliberately focuses on the execution path that matters most: request lifecycle, continuous batching, KV-cache allocation, paged memory, and testable text generation.

The immediate target is a compact core replica for a dense **Qwen3 decoder-only** checkpoint, loaded from a Hugging Face-compatible local directory. Qwen3 provides the important real inference features—RMSNorm, RoPE, grouped-query attention (GQA), Q/K normalization, and SwiGLU—without multimodal or MoE complexity.

The milestone-by-milestone plan is in [ROADMAP.md](ROADMAP.md).

## Scope

The first track will support:

- decoder-only Qwen-style autoregressive generation;
- continuous batching across several requests;
- paged KV-cache allocation, sharing, release, and copy-on-write;
- greedy, temperature, top-k, top-p, EOS, and `max_tokens` controls;
- deterministic CPU tests with a fake model runner;
- a single-GPU Qwen3 runner and an offline Python API.

It intentionally defers distributed execution, quantization, LoRA, multimodal models, mixture-of-experts, speculative decoding, HTTP serving, custom CUDA/Triton kernels, and production authentication. They belong to the later hardening track, not the initial core replica.

## Repository layout

```text
.
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── src/nano_vllm/
│   ├── config.py
│   ├── sampling_params.py
│   ├── sequence.py
│   ├── core/                 # scheduler, blocks, KV cache
│   ├── engine/               # LLMEngine and output events
│   ├── worker/               # model runners
│   ├── model_executor/       # Qwen layers and weight loader
│   └── entrypoints/          # CLI and HTTP API
└── tests/
    ├── unit/
    ├── integration/
    └── conftest.py
```

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once. From the repository root, create the project environment and install the development tools:

```bash
uv sync --extra dev
uv run pytest -q
```

`uv sync` creates `.venv` and installs the project in editable mode. `uv run` uses that environment even if your shell still shows Conda `(base)`, avoiding unrelated global pytest plugins.

Install the Qwen loading dependencies at M3. The web dependencies are only needed at M9:

```bash
uv sync --extra dev --extra model
uv sync --extra dev --extra model --extra api
```

`pyproject.toml` deliberately keeps `transformers`, FastAPI, and Uvicorn optional. The scheduler/cache unit tests must remain fast and independent of network access, model downloads, and CUDA.

## Qwen3 model boundary

Start with a small dense Qwen3 model, preferably `Qwen/Qwen3-0.6B`. Parse `config.json` into a local `QwenConfig`; do not make scheduler and cache code depend on Hugging Face configuration objects.

```python
@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    use_qk_norm: bool
```

The forward path is:

```text
token ids → embedding
  → repeated [RMSNorm → GQA attention with RoPE → residual
              RMSNorm → SwiGLU MLP → residual]
  → final RMSNorm → LM head → logits
```

Important implementation details:

- GQA means `num_key_value_heads` may be smaller than `num_attention_heads`.
- RoPE positions must be identical in prefill and decode paths.
- Qwen3 needs its configuration read exactly: Q/K normalization, projection bias, and RoPE settings must come from the selected checkpoint rather than from hard-coded assumptions.
- The initial implementation supports the checkpoint's normal RoPE settings. Long-context RoPE scaling is a later extension.
- Unit tests use a fake runner only. Real-model parity tests receive a local checkpoint through `NANO_VLLM_QWEN_MODEL=/absolute/path` and skip if absent.

## Component boundaries

| Component | Owns | Must not own |
| --- | --- | --- |
| `SamplingParams` | validation of generation settings | scheduling |
| `Sequence` | one request's tokens, state, output | physical blocks |
| `Scheduler` | waiting/running selection under budgets | model execution |
| `BlockSpaceManager` | block allocation, sharing, refcounts | request priority |
| `ModelRunner` | Qwen3 prefill/decode and logits | request lifecycle |
| `LLMEngine` | orchestration and output events | HTTP protocol |

Use the state machine below and allow only the scheduler/engine to transition states:

```text
WAITING → RUNNING → FINISHED_STOPPED
                  └→ FINISHED_LENGTH_CAPPED
                  └→ FINISHED_ABORTED
```

## Paged KV cache

For `block_size = 4`, ten tokens occupy three logical blocks:

```text
logical token positions:  0 1 2 3 | 4 5 6 7 | 8 9
logical block index:          0   |     1   |  2
block table:              [12, 7, 19]
physical block id:          12    7    19
```

`Sequence.block_table` maps logical blocks to physical blocks. Prompt-sharing sequences may reference the same blocks. If a sequence appends to a shared partial tail block, it first receives a copied block and changes only its own table: copy-on-write. This behavior needs focused tests; an end-to-end generation assertion is not enough.

Core invariants:

- A physical block contains exactly `block_size` token positions.
- `num_computed_tokens <= len(sequence.token_ids)`.
- A block returns to the free queue only when its reference count becomes zero.
- A scheduler step does not exceed `max_num_batched_tokens`.

## Test and commit contract

For every milestone:

1. Write a failing, focused test.
2. Implement the smallest correct behavior.
3. Run the milestone tests and then all tests.
4. Make one Git commit only after all tests pass.

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run pytest -q
```

Use a seeded `torch.Generator` whenever a test samples randomly. Unit tests must never need CUDA, an internet connection, or downloaded weights. Real Qwen parity tests have both `integration` and `model` markers; CUDA-only tests additionally have `cuda`.

## Delivery plan and estimated size

The roadmap deliberately starts with a compact Qwen3 core replica, then expands only after it works. It excludes model weights, dependencies, generated reports, and deployment infrastructure.

| Scope | Production | Tests | Total |
| --- | ---: | ---: | ---: |
| Track A M0–M5: eager Qwen3 core | 1,500–2,500 LOC | 600–1,200 LOC | 2,100–3,700 LOC |
| Track A M6: optional CUDA fast path | +300–800 LOC | +150–350 LOC | +450–1,150 LOC |
| Track B: hardening, API, metrics, scale-out | +1,500–3,000 LOC | +1,500–2,500 LOC | +3,000–5,500 LOC |

The immediate goal is Track A only. It is deliberately close to the compact reference-project scale, but still includes enough focused tests to make scheduler and block-manager bugs visible.
