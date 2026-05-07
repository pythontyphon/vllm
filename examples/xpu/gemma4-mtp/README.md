# Gemma 4 31B + MTP on Intel Arc Pro B70 (XPU)

A reference setup for running [`RedHatAI/gemma-4-31B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-Dynamic)
with Multi-Token Prediction (MTP) speculative decoding on **2× Intel Arc Pro B70 (32 GB)**
via Docker, in tensor-parallel.

The MTP support is the cherry-pick of upstream PR
[vllm-project/vllm#41745](https://github.com/vllm-project/vllm/pull/41745)
plus the `Gemma4Proposer` adapter at `vllm/v1/spec_decode/gemma4.py`. With
hybrid KV cache enabled and MTP γ=4, this setup serves a
**163 840 token (160 k) context** with concurrency headroom on dual
Battlemage G31 (32 GB each) cards.

> **Note on KV cache dtype.** An earlier version of this recipe used
> `--kv-cache-dtype fp8_e4m3` to fit the full 262 144-token native
> context. That config booted, but bench measurements showed steady-state
> decode TPS dropping ~30–46 % vs bf16 KV cache (the gap grew with
> context length, which is the signature of per-attention-step dequant
> overhead). The 160 k bf16 config below is the better trade-off for
> interactive use; bump `--max-model-len` lower for slightly more decode
> headroom, or re-enable `--kv-cache-dtype fp8_e4m3` and `--max-model-len
> 262144` if you need maximum context and accept the decode penalty.

## Hardware / software

- 2× Intel Arc Pro B70 32 GB (Battlemage G31, PCI ID `8086:e223`)
- Intel oneAPI 2025.3 (the upstream `Dockerfile.xpu` is pinned to this — do
  not let it upgrade to 2026.x)
- HuggingFace `transformers` from `main` (PR #45788 added the
  `gemma4_assistant` model type, merged 2026-05-05)
- `vllm-xpu-kernels >= 0.1.7` (carries decode-attention fixes from PRs
  #204, #257, #308, #318)
- vLLM XPU base image built from `Dockerfile.xpu`
- Linux kernel with i915 / xe driver supporting Battlemage

## Open upstream PRs included in this branch

This branch carries two open upstream vLLM PRs as cherry-picks beyond
the Gemma 4 MTP commit:

- [vllm-project/vllm#40327](https://github.com/vllm-project/vllm/pull/40327)
  — adds `USE_TD` constexpr to the unified Triton attention path,
  enabling HW 2D block reads on Intel Xe2/Xe3. Auto-enables on XPU,
  opt-out via `VLLM_TRITON_ATTN_USE_TD=0`. Helps the
  `--attention-backend TRITON_ATTN` path. (We bench FLASH_ATTN below
  because it's faster on this hardware regardless, but anyone using
  TRITON_ATTN gets the win.)
- [vllm-project/vllm#40356](https://github.com/vllm-project/vllm/pull/40356)
  — removes a forced contiguous-copy of Q/K/V per FLASH_ATTN call on
  XPU. Worth ~+9 % at 16 k context in our bench, neutral elsewhere.

## Build the image

The build is two layers:

1. Base XPU image (one-time, uses upstream `Dockerfile.xpu`):

   ```bash
   docker build -f docker/Dockerfile.xpu -t vllm-xpu-env:base .
   docker tag vllm-xpu-env:base vllm-xpu-env:mtp
   ```

2. MTP overlay — installs `transformers` from main and copies the in-tree
   `vllm/` (with the Gemma 4 MTP changes) over the image's site-packages:

   ```bash
   docker build -f examples/xpu/gemma4-mtp/Dockerfile.mtp -t vllm-xpu-env:mtp .
   ```

   Re-run step 2 every time you change `vllm/` Python sources locally.

## Launch

Two convenience scripts under `launchers/`:

- `gemma31` — reasoning mode on (Gemma 4 reasoning parser)
- `gemma31nr` — reasoning mode off (cleaner output for benchmarking)

Both launch the same container `vllm-gemma`, listening on `:8080`. Copy them
into your `$PATH` (e.g. `~/.local/bin/`) and run.

The serving args used by both launchers:

```text
--tensor-parallel-size 2
--enforce-eager
--attention-backend FLASH_ATTN
--max-model-len 163840
--gpu-memory-utilization 0.95
--no-disable-hybrid-kv-cache-manager
--enable-prefix-caching
--max-num-seqs 16
--max-num-batched-tokens 8192
--speculative-config '{"model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 4}'
```

## Why these flags matter on XPU

- `--attention-backend FLASH_ATTN` — Intel's sycl-tla FMHA. Decode TPS drops
  ~50 % across context lengths if you let it fall back to `TRITON_ATTN`.
- `--enforce-eager` — XPU graphs are disabled at TP > 1
  (`vllm/platforms/xpu.py:198`); this avoids the warmup hang.
- `--no-disable-hybrid-kv-cache-manager` — without this, Gemma 4's 51 sliding
  attention layers get charged full per-token KV memory regardless of the
  1024-token sliding window. With it, only the 9 full-attention layers scale
  with sequence length, dropping per-token KV cost from ~150 KiB to ~54 KiB
  at bf16. That's the only reason 160 k context fits with concurrency
  headroom on 64 GB total VRAM.
- `--max-model-len 163840` (160 k) is the sweet spot here at bf16 KV cache.
  vLLM's auto-estimate said ~172 544 was the ceiling at `--gpu-memory-utilization
  0.95`; 160 k is a clean number that leaves room for prefix caching and
  draft-model overhead without OOM under load.

## Boot sanity-check

After launch, look for these lines in `~/llama-server.log`:

```text
Available KV cache memory: 10.12 GiB
GPU KV cache size: 169,314 tokens
Maximum concurrency for 163,840 tokens per request: 1.03x
Gemma4 MTP: draft layer 0 (sliding_attention) -> language_model.model.layers.58.self_attn.attn
Gemma4 MTP: draft layer 1 (sliding_attention) -> language_model.model.layers.58.self_attn.attn
Gemma4 MTP: draft layer 2 (sliding_attention) -> language_model.model.layers.58.self_attn.attn
Gemma4 MTP: draft layer 3 (full_attention)    -> language_model.model.layers.59.self_attn.attn
```

The MTP draft layers wire to the last two backbone layers (58 + 59) via
cross-model KV sharing — that's the core of the speculative decoder.

## Caveats

- This branch is **not** in upstream vLLM. The Gemma 4 MTP commit
  (`[Spec Decode] Add Gemma4 MTP speculative decoding with centroids
  masking`) sits on top of vLLM main as of early May 2026 and tracks
  upstream PR #41745.
- Built and validated on Intel Arc Pro B70 32 GB. Smaller-VRAM Battlemage
  cards (B60 24 GB) likely need `--max-model-len` bumped down to ≤ 65536.

### Known long-context decode bottleneck

Empirical bench: decode TPS at γ=4 falls from ~58 TPS at ctx ≤ 4 k to
~15 TPS at ctx 65 k and ~9 TPS at ctx 131 k. The slope is ~7× steeper
than memory-bandwidth alone predicts (`9 full-attn layers × KV bytes / 912
GB/s aggregate`), which means the gap is software, not hardware.

This is tracked upstream by [vllm-xpu-kernels#271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271)
(another B70 user diagnosed the kernel-internal causes — `BLOCK_KV=4`
hardcoded, `num_warps=1` on decode stage 1 underutilizing BMG's 256
EUs, software dequant per attention step). [Issue #185](https://github.com/vllm-project/vllm-xpu-kernels/issues/185)
is the umbrella decode-perf tracker.

Things we tried that didn't help:
- `--attention-backend TRITON_ATTN` with USE_TD: 13–66 % slower than
  FLASH_ATTN on this model.
- `--kv-cache-dtype fp8_e4m3`: forces a chunked-prefill fallback (paged
  decode fast path requires `!is_fp8kv`); penalty grows with context.
- MTP γ=2 vs γ=4: γ=4 wins everywhere, so per-step MTP overhead isn't
  the dominant cost.

If you need predictable long-context decode TPS today, this is the wrong
stack. If you can keep most of your traffic ≤ 4 k context, MTP delivers
~2× over baseline at ~50 TPS sustained.
