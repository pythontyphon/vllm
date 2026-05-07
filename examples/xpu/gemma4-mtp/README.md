# Gemma 4 31B + MTP on Intel Arc Pro B70 (XPU)

A reference setup for running [`RedHatAI/gemma-4-31B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-Dynamic)
with Multi-Token Prediction (MTP) speculative decoding on **2× Intel Arc Pro B70 (32 GB)**
via Docker, in tensor-parallel.

The MTP support is the cherry-pick of upstream PR
[vllm-project/vllm#41745](https://github.com/vllm-project/vllm/pull/41745)
plus the `Gemma4Proposer` adapter at `vllm/v1/spec_decode/gemma4.py`. With
hybrid KV cache + fp8 KV cache + MTP γ=4, this setup serves the full
**262 144 token (256 k) native context** at ≥ 1× concurrency on dual
Battlemage G31 (32 GB each) cards.

## Hardware / software

- 2× Intel Arc Pro B70 32 GB (Battlemage G31, PCI ID `8086:e223`)
- Intel oneAPI 2025.3 (the upstream `Dockerfile.xpu` is pinned to this — do
  not let it upgrade to 2026.x)
- HuggingFace `transformers` from `main` (PR #45788 added the
  `gemma4_assistant` model type, merged 2026-05-05)
- vLLM XPU base image built from `Dockerfile.xpu`
- Linux kernel with i915 / xe driver supporting Battlemage

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
--max-model-len 262144
--gpu-memory-utilization 0.95
--no-disable-hybrid-kv-cache-manager
--kv-cache-dtype fp8_e4m3
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
  with sequence length, dropping per-token KV cost from ~150 KiB to ~26 KiB
  (with fp8 cache).
- `--kv-cache-dtype fp8_e4m3` — halves KV memory; required to fit 256 k
  context on 64 GB total VRAM with concurrency headroom. Verified that XPU
  FLASH_ATTN accepts this dtype at boot.

## Boot sanity-check

After launch, look for these lines in `~/llama-server.log`:

```text
Available KV cache memory: 10.12 GiB
GPU KV cache size: 391,845 tokens
Maximum concurrency for 262,144 tokens per request: 1.49x
Gemma4 MTP: draft layer 0 (sliding_attention) -> language_model.model.layers.58.self_attn.attn
Gemma4 MTP: draft layer 1 (sliding_attention) -> language_model.model.layers.58.self_attn.attn
Gemma4 MTP: draft layer 2 (sliding_attention) -> language_model.model.layers.58.self_attn.attn
Gemma4 MTP: draft layer 3 (full_attention)    -> language_model.model.layers.59.self_attn.attn
```

The MTP draft layers wire to the last two backbone layers (58 + 59) via
cross-model KV sharing — that's the core of the speculative decoder.

## Caveats

- vLLM warns about the fp8 KV cache's potential accuracy drop without proper
  scaling factors. If you care about Gemma 4 quality at fp8 KV, run an eval
  on your workload before relying on it.
- This branch is **not** in upstream vLLM. The Gemma 4 MTP commit
  (`[Spec Decode] Add Gemma4 MTP speculative decoding with centroids
  masking`) sits on top of vLLM main as of early May 2026 and tracks
  upstream PR #41745.
- Built and validated on Intel Arc Pro B70 32 GB. Smaller-VRAM Battlemage
  cards (B60 24 GB) likely need `--max-model-len` bumped down to ≤ 131072.
