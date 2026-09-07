# MAEMM — training a universal activation→text inverter

**MAEMM** (Max-Activating-Example Meta-Model) trains a small LoRA adapter that turns an *activation
direction* inside a language model back into **text** — text whose own activation, on a clean forward
pass, points the same way as the injected direction. In other words: given a direction in a model's
residual stream (an SAE feature, a probe direction, a raw activation, …), it generates a short span
that maximally *evokes* that direction. It's a learned, general-purpose "inverse" of the readout lens.

This repo is the training pipeline for that inverter on **Qwen/Qwen3.6-27B** (read layer 42, inject
layer 1), but the method is model-agnostic — change `mxf/config.py`.

## Method

1. **Inject** a unit direction `v` at `INJECT_LAYER` on a marker token (norm-matched: `h ← h + ‖h‖·v`).
2. **Generate** a short continuation.
3. **Score** it by re-reading the *clean* (adapter-off) activation at `READ_LAYER` and taking the
   max-over-token cosine with `v`. Good inversions produce text that genuinely drives the direction.

Training is two stages:

- **SFT (match-activation):** for each direction, the target text is the *real* corpus span whose
  activation produced it. This matches the objective by construction.
- **RL (Dr. GRPO):** reward = max-over-rollout cosine (× a scale), optimizing the inverter to *actively*
  evoke each direction. RL does most of the heavy lifting; SFT gives it a running start.

**Direction families** (the "data mix" — this is what matters most):

| family | what it is |
|---|---|
| `realact` | raw real activations `unit(h − μ)` at sampled corpus positions |
| `probe` / `cluster` | linear probe / cluster-centroid directions |
| `bsf` | block-sparse subspace-featurizer (SASA) projections |
| `sae` | SAE encoder columns |
| `jlens` | J-lens (pullback) vectors |

Empirically, a **balanced mix of real-activations + probes** generalizes best across held-out families;
adding **long-context activations to the RL mix** (contexts far longer than SFT ever sees) is a cheap way
to cover features that only fire at long range. See `bank/` for every family's builder.

## Layout

```
mxf/        core library: config, injection hooks, prompts, SAE loader, MFU
data/       activation collection + direction-bank building (+ their Modal launchers)
            collect_acts.py / collect_acts_longctx.py / collect_acts27b_worker.py   READ_LAYER residuals (fixed-len, long-ctx, 27B sharded)
            train_sasa.py                                                            the SASA/BSF subspace featurizer
            build_universal_bank.py / build_big_sft_bank.py / build_rl_bank.py        (direction, target-span) banks
            modal_acts27b.py, modal_bank_everything.py, modal_big_bank.py, modal_last5_bank.py, modal_pool_last5.py, modal_bsf_retrain.py
sft/        match-activation SFT: pretrain.py (LoRA; single- or multi-GPU via torchrun) + modal_sft.py
rl/         RL: rl.py (GRPO, actor + vLLM on every rank), rl_disagg.py (X vLLM rollout GPUs + Y trainer GPUs — the fast path),
            fast_lens_ext.py (vLLM steering hook), rl_ddp.sh, modal_rl_disagg.py (production launcher), modal_rl*.py (older launchers),
            patches/ (historical), test_rl_disagg_queue.py, test_rl_disagg_scalerl.py (CPU tests of the opt-in
            ScaleRL variant: --recipe scalerl = PipelineRL-8 lag, CISPO, prompt-level agg, batch adv norm, zero-var filter, NPR)
eval/       THE EVAL SUITE (see eval/README.md): eval_universal.py (held-out family cosine + mean_all, SAE norm_act / rank / cos /
            %unverbalized, random control, ctx-bucket + in-distribution families), snippet_locality.py, autointerp_detection.py,
            inline_extra_evals.py (locality + autointerp AUC + WildChat fire-prediction + adversarial confirmation, judge = Sonnet 5),
            eval_ckpt_daemon.py + modal_eval_ckpt.py (1-GPU per-checkpoint evaluator with its own vLLM engine),
            build_ctx_eval.py / build_indist_eval.py / wildchat_bank.py (eval-set builders), analysis/ (plots), modal_eval*.py (older daemons)
scripts/    vllm_smoke.py — vLLM + vllm_lens steering smoke test
```

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD            # so `import mxf...` resolves
export HF_TOKEN=...               # for model + corpus downloads
# optional: export WANDB_API_KEY=... ANTHROPIC_API_KEY=...   (logging / LLM-judge evals)

# 1) collect activations (needs a GPU with the model)
python data/collect_acts.py --n-seq 20000 --seq-len 512 --out-dir data/acts

# 2) build banks
python data/build_big_sft_bank.py --n-realact 500000 --n-probe 500000 --out data/pool_sft
python data/collect_acts_longctx.py --shard 0 --n-shards 5   # (one per GPU) -> data/acts_long
python data/build_rl_bank.py --n-each 250000 --out data/pool_rl_mix

# 3) SFT  (multi-GPU: use gloo — see note)
DDP_BACKEND=gloo TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD \
  torchrun --standalone --nproc_per_node=5 sft/pretrain.py \
    --data-dir data/pool_sft --lr 3e-5 --batch-size 16 --epochs 2 --save-dir ckpts/sft

# 4) RL from the SFT init (data-parallel; drop the launcher for a single-GPU run)
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash rl/rl_ddp.sh \
  --data-dir data/pool_rl_mix --bank-file vecs.f32 \
  --init-adapter ckpts/sft/final --lr 1e-5 --reward-metric cosine --reward-scale 1000 \
  --min-new-tokens 16 --max-new-tokens 96 --len-penalty-start 16 --len-penalty-per-tok 1.0 \
  --div-coef 2000 --kl-coef 0.03 --groups-per-step 32 --group-size 16 \
  --total-steps 400 --save-dir ckpts/rl

# 5) eval (held-out families + %unverbalized SAE — full suite docs in eval/README.md)
python eval/eval_universal.py --adapter ckpts/rl/final \
  --sae-path <ae.pt> --maxacts-path <max_acts.pt> --heldout-pool data/pool_heldout
```

> **Multi-GPU note:** use **`DDP_BACKEND=gloo`**, not NCCL. Only the (tiny) LoRA gradients are
> all-reduced, so gloo's CPU-socket comms are plenty — and NCCL can deadlock at DDP init on some
> single-node multi-GPU boxes. Also set `TOKENIZERS_PARALLELISM=false` (forked-rank tokenizer thrash).

## Config

`mxf/config.py` — `MODEL`, `D_MODEL`, `READ_LAYER`, `INJECT_LAYER`, `STEER_COEFF`, corpus, LoRA/RL hparams.
Defaults: rsLoRA r64/α16 all-linear, AdamW lr 3e-5 (SFT) / 1e-5 (RL).

## B200 pretraining with gradient accumulation

For the LoRA SFT path, `--prefix-cache --prefix-share-step` computes the common,
**pre-injection prefix once per optimizer step**, including its backward pass, across
both `--grad-accum` and `--prefix-accum` micro-batches. The injected marker and every
token after it are computed separately for each example; they are never shared.
The prefix cache is rebuilt after each optimizer update. Cache gradients accumulate in FP32 and propagate
through the prefix before the completed parameter gradients are averaged across GPUs.
The prefix still learns. The loss weighting, batch order, and optimizer schedule are
preserved; BF16 gradient summation order changes, so updates are not bit-identical.

Add `--prefix-share-step` to an existing prefix-cache run, for example:

```bash
python sft/pretrain.py --data-dir data/pool_sft \
  --prefix-cache --prefix-share-step --grad-ckpt 0 --autocast-bf16 \
  --batch-size 64 --grad-accum 8 --max-seq 160
```

This remains opt-in pending B200 throughput/parity measurements. It requires zero
dropout and the existing transformers prefix-cache fork
(`ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3`); full-FT/FSDP2
is not supported by this flag. Batch 64 is intended for short targets; longer targets
may need smaller micro-batches. The Modal launcher passes the flag through `extra_args`.

Pretraining also batch-tokenizes target text and constructs the shared chat prompt once
when loading a bank. That startup optimization is enabled for all training paths.

CPU loss/gradient, optimizer-step, and two-rank synchronization checks:

```bash
OMP_NUM_THREADS=1 python -m pytest -q sft/test_prefix_sharing.py sft/test_head_on_labels.py
```

The existing B200 harness can check BF16 gradient parity, then compare throughput on
identical synthetic batches without changing model weights or production checkpoints:

```bash
modal run sft/test_prefix_cache.py::share_step --batch 64 --grad-accum 8
modal run sft/test_prefix_cache.py::bench --steps 5 \
  --configs cached_gradaccum:64x8,cached_sharedstep:64x8 \
  --out sft/results/prefix_share_step_bench.json
```

## Scaling & infra

- **Data-parallel RL (gloo DDP) — `rl/rl_ddp.sh`.** `rl/rl.py` is DDP-aware: under torchrun,
  each rank rolls out / scores / backprops `groups_per_step / world` **whole** groups (Dr. GRPO
  advantages stay intra-rank), then the LoRA grads are all-reduced in one flat **CPU** buffer over
  **gloo**, token-weighted so the update is *exactly* the single-GPU gradient over the union batch
  (world=1 is byte-identical to the plain script; verified per-rank LoRA checksums match to 10
  decimals per step). gloo, not NCCL: only the tiny LoRA grads move, and NCCL deadlocks at the first
  collective on some single-node boxes. 3.8× step-time speedup at world=4 (223s → 59s/step on 27B).
- **Modal 8×B200 — `modal_rl.py`.** The same run as a detached Modal app (`modal run --detach
  modal_rl.py`): persistent `maemm-data` Volume for the bank / SFT-init adapter / HF cache /
  checkpoints, `prewarm` + 1-GPU `smoke` + 8-GPU `train` entrypoints, secrets via Modal
  (`maemm-hf`, `maemm-wandb`) — no keys in code. Includes the hard-won guards: SFT-init (not deep-RL
  warm-start, which collapsed), len-penalty 1.0/tok, gate-masked diversity bonus (asserts the fix is
  present), `expandable_segments` + offline HF loading for 8-rank stability.
- **vLLM rollout path — `scripts/vllm_smoke.py`.** Fast rollouts use vLLM 0.19 + `vllm_lens`
  per-request `SteeringVector(norm_match=True)` == the trainer's norm-matched inject. The gotcha:
  a vllm_lens install with **missing dist-info never registers its vLLM plugin entry point**, so
  steering *silently no-ops* — restore the dist-info (and use transformers ≥5) and it fires. The
  smoke test proves it numerically (Δresidual at the marker == ‖h_clean‖·unit(v), cos > 0.99) and
  behaviorally; needs `attention_backend=TRITON_ATTN` (FLASHINFER silently no-ops too).
- **Diversity bonus (`--div-coef`).** Within-group bonus for rollouts whose clean activations
  (mean-pooled over tokens up to the peak-activation token) are mutually orthogonal after projecting
  out the target direction — pushes a group to find *different* ways to evoke the same direction.
  **Must be masked by the fluency gate** (degenerate rollouts otherwise farm the bonus → collapse).
  Pareto-swept: `--div-coef 2000` ≈ +11% diversity for ~2% cosine cost. `patches/` holds the patch
  scripts that introduced this (already applied to `rl/rl.py`; kept for provenance).

## Notes

Research code — no warranty, APIs may change. No data, weights, checkpoints, or credentials are included;
supply your own model/corpus/keys via env vars.
