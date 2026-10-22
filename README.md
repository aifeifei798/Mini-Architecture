# Mini-Architecture

[English](README.md) | [简体中文](README.zh-CN.md)

A miniature DeepSeek-style MoE LLM playground: study the minimal reproducible
DeepSeek V4.1 architecture with 20 training samples and ~45 seconds of
training. Text-only (no vision, no fp8 quantization).

The architecture lives in exactly one file (`model_v41.py`) — training and
chat scripts import it; there is no copy-pasted second implementation. Body:
3.56M parameters (36.7M total including embeddings + LM head, dominated by
the vocab-129280 input/output layers).

## Quick start

**v2 · 20-sample memorization baseline** (proves trainability):

```bash
.venv/bin/python train_v41.py    # train (20 samples, 300 epochs, ~45s, CUDA)
.venv/bin/python chat_v41.py     # interactive chat + expert/indexer diagnostic heatmap
```

**v3 · long context** (real corpus, S=512, requires `./minimind_dataset/*.jsonl`):

```bash
.venv/bin/python data_long.py    # jsonl → tokenized packed .bin (~1 min)
.venv/bin/python train_long.py   # 5000 pretrain steps + 3000 SFT steps, ~33 min
.venv/bin/python ablate_long.py  # 6-config equal-budget ablation, ~2 h (optional)
MODEL_DIR=./my_deepseek_v41_long .venv/bin/python chat_v41.py
```

You must `import model_v41` before loading checkpoints (registers the
`deepseek_v41_mini` architecture); `chat_v41.py` already does this. Model
artifacts (`./my_deepseek_v41_*`) are not committed.

## Architecture highlights (minimal V4.1)

- **mHC (Hyper-Connections)**: one module per sub-layer (attn/ffn); streams =
  4 embedding copies, Sinkhorn doubly-stochastic mixing of old streams,
  collapsed by a HyperHead at the end; no in-block residual
- **MiniMLA attention**: low-rank q + dual norms; shared single-head direct
  K=V projection (V4 dropped V3's kv_lora); partial interleaved RoPE rotates
  only the trailing rope dims, output conjugated (-sin); per-head learnable
  attention sinks; grouped low-rank output projection
- **Sparse attention**: indexer scores ∑_h w_h·ReLU(q·K)·d^-0.5; indexers are
  shared across layers (L2/L6 as sources)
- **MoE**: 8 routed experts top-2 + 1 shared expert; sqrtsoftplus scoring with
  the correction bias used for selection only (official noaux_tc semantics);
  SwiGLU clamped before activation. Expert dispatch is a **Triton grouped
  GEMM** (routed rows argsort+bincount into CSR segments, both forward and
  backward are kernels, no Python loop), with an automatic einsum reference
  fallback on CPU / without triton
- **Cross-layer KV compression reuse** [INFERRED], **MTP** [INFERRED]
  (fuse(h_t, emb(t+1)) predicts t+2, loss coefficient 0.3), **engram**
  [INFERRED] (n-gram hash-table lookup injection)
- `dspark_*` is a placeholder, unimplemented

Full official-alignment notes live in the `model_v41.py` module docstring;
engineering conventions and traps are documented in `AGENTS.md`.

## v3 long context (real corpus, sparse attention actually engages)

Data comes from the open [MiniMind](https://github.com/jingyaogong/minimind)
dataset (`pretrain_t2t_mini.jsonl` plain text / `sft_t2t_mini.jsonl`
multi-turn dialogues), reservoir-sampled and eos-packed into fixed S=512
windows (18.4M pretrain + 12.2M SFT tokens).

**Baselines** (5000 steps ≈ 41M tokens pretraining + 3000 SFT steps, ~33 min):

| Stage | best val | ppl |
|---|---|---|
| Pretraining (pretrain val) | 5.369 | 215 |
| SFT (sft val) | 5.009 | 150 |

At S=512 the indexer density is ~0.065 (0.4 at S=48) — each query "sees" only
about 33/512 positions; sparse attention performs real block selection.

**Architecture ablation** (equal 5000-step budget, fixed val batches,
`ablate_long.py`):

| Config | Change | best val | indexer density |
|---|---|---|---|
| dense | select all blocks (≈ full-attention upper bound) | **5.348** | 0.501 |
| idx32 | index_topk 8→32 | 5.545 | 0.076 |
| base | default (index_topk=8) | 5.648 | 0.042 |
| no_engram | engram off (saves 1.09M params) | 5.701 | 0.041 |
| idx4 | index_topk 8→4 | 5.733 | 0.037 |
| no_reuse | cross-layer KV reuse off | 5.936 | 0.043 |

(The ablation ran under pre-window-fix semantics; configurations remain
mutually comparable.)

Three takeaways:
1. **Monotone sparsity-quality trade-off**: density 4%→full costs 0.30 loss;
   top-8 buys 90% of the quality with 1/12 of the visible tokens
2. **Cross-layer KV reuse is the single biggest factor**: removing it costs
   0.29 loss — it is not just KV savings but a cross-layer representation
   sharing regularizer / depth gain
3. **engram pays off**: 3% of the parameters (1.09M) buys 0.05 loss

**v3.1 sliding-window fix**: `SparseIndexer`'s window difference was written
as `j-i`; after `& causal` the whole sliding_window=16 degenerated into
"self-attention only"; fixing it to `i-j` (look-back) semantics moved pretrain
val 5.65→5.37. Pre-fix checkpoints no longer match the new semantics and were
all retrained.

**KV-cache incremental decoding**: `prefill` caches each layer's indexer k
projection and each own-KV-layer raw KV; each token then costs one O(S) step
(the indexer scores only the new query row), bit-identical in semantics to the
full forward (greedy tokens match exactly). Note: at S≤512 on a strong GPU
both paths are dominated by kernel-launch counts, measuring only ~1.1× — the
asymptotic win (O(n·S²)→O(n·S)) shows up at longer sequences or larger batches.

## Tests

No pytest; three directly runnable verification scripts:

```bash
.venv/bin/python test/smoke_v41.py     # smoke: causality assertion (perturb a future
                                       # token → prefix logits diff < 1e-4) + S∈{1,3,7,48,49}
                                       # fwd/bwd finite + diagnostics API (CPU OK)
.venv/bin/python test/moe_eq_test.py   # MoE: Triton vs einsum reference equivalence
                                       # (bitwise fwd + all-param grads, fp32/bf16; CUDA)
.venv/bin/python test/kv_cache_test.py # KV cache: incremental vs full-forward greedy
                                       # token match + timing (CUDA + long checkpoint)
```

Run the smoke test before training after any model change; MoE changes also
need the equivalence test; attention/cache changes need the kv test.

## Acceptance baselines

- **v2 (20-sample memorization)**: train loss 0.0056 (44.5s), all 4 prompts
  greedily hit
- **v3 (long context)**: pretrain val 5.369 (ppl 215) → SFT val 5.009 (ppl 150)

## Layout

| Path | Description |
|---|---|
| `model_v41.py` | Single source of the V4.1 architecture (config + model + KV cache; registers `deepseek_v41_mini`) |
| `train_v41.py` | v2 training: 20-sample memorization baseline + save + greedy verification |
| `chat_v41.py` | Chat (incremental decoding) + expert/indexer diagnostics; `MODEL_DIR` env var selects the checkpoint |
| `data_v41.py` | 20 training samples (`问：… 答：…`, half-width space before 答) |
| `data_long.py` | v3 data pipeline: minimind jsonl → reservoir sampling → S=512 packed .bin |
| `train_long.py` | v3 training: random-window full-label LM, pretrain→SFT stages (each gated on its own val) |
| `ablate_long.py` | 6-config equal-budget ablation (KV reuse / sparsity / engram), resumable via results.json |
| `test/` | smoke + MoE Triton equivalence + KV cache consistency tests |
| `tokenizer/` | DeepSeek-family tokenizer (`PreTrainedTokenizerFast`, vocab 129280) |

A standalone spin-off library with the standard `(B,H,S,D)` KV-compression
contract lives at [flashcodec](https://github.com/aifeifei798/flashcodec).

## Environment

- Python 3.12 + torch 2.14 (CUDA 13) + transformers 5.20; venv built by uv
- triton ships with torch (used for MoE grouped-GEMM dispatch, nothing to install)
- No requirements.txt: `.venv` is the source of truth; install missing
  packages with uv

## License

[MIT](./LICENSE)
