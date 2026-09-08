# Tensor-parallel full fine-tuning (`pretrain.py --full-ft --tp 8`)

Design note for `sft/tp.py`: one-node tensor parallelism (TP-8) for the full fine-tune of Qwen3.6-27B, as an alternative
to the FSDP2 path in `sft/fullft.py`. Same objective, same checkpoints, same prefix cache; different partition of the work.

## 1. Why

The prefix-cached SFT step is B suffixes of ~21 tokens on top of a shared 102-token prompt. Under FSDP2 each rank runs
its own B=64 rows (~1,350 tokens) through the full model and re-gathers all 54 GB of bf16 weights per micro-batch
(forward + backward). Memory pins B at 64 (140 GB: 54 GB fp32 masters+Adam, ~35 GB gathered params, MLP activations
19 GB, fp32 norm upcasts 25 GB, GDN states 18 GB, head 5 GB), so every matmul has M ~ 1,350 and the 64-layer step is
launch/communication bound: 80 ex/s/rank = 641 ex/s, 272-283 TFLOP/s per GPU (~12% of B200 bf16).

Under TP every rank runs ALL rows of the micro-batch through a 1/8 slice of every weight matrix: M grows 8x (B=512 ->
~11k tokens), no weight gathers, optimizer state only for the local shard, and the wide activations (MLP intermediate
17,408, GDN 48 value heads, attention 24 heads, 248k logits) are sharded 8x. Only the 5,120-wide residual stream is
replicated. Effective batch 4,096 = micro-batch 512 x grad-accum 1 (or 256 x 2, 128 x 4).

## 2. Sharding table (tp = 8, Qwen3.6-27B: d 5120, d_ff 17408, 48 GDN + 16 attention layers)

| module | parameter (full shape) | layout | per rank | comm |
|---|---|---|---|---|
| embed_tokens | weight [248320, 5120] | vocab rows chunk r | [31040, 5120] | out-of-range ids masked to 0, output **all-reduce** (fwd) |
| GDN in_proj_qkv | weight [10240, 5120] = [q 2048 \| k 2048 \| v 6144] | column, BY HEAD: rows q[256r:256r+256] + k[2048+256r:...] + v[4096+768r:4096+768r+768] | [1280, 5120] | input: identity fwd / **all-reduce** bwd (shared with z/b/a: one pre-hook on the module) |
| GDN conv1d | weight [10240, 1, 4] depthwise | channels = the same index as in_proj_qkv rows | [1280, 1, 4] | none |
| GDN in_proj_z | weight [6144, 5120] | column, value heads 6r..6r+5 | [768, 5120] | (with qkv) |
| GDN in_proj_b / in_proj_a | weight [48, 5120] | column, heads 6r..6r+5 | [6, 5120] | (with qkv) |
| GDN A_log / dt_bias | [48] | heads 6r..6r+5 | [6] | none |
| GDN norm (gated RMSNorm) | weight [128] (per head_v_dim, shared by all heads) | replicated | [128] | grad **all-reduce SUM** (each rank sees 6 heads) |
| GDN out_proj | weight [5120, 6144] | row, cols 768r..768r+767 | [5120, 768] | output **all-reduce** (fwd) |
| attn q_proj | weight [12288, 5120] = 24 heads x (q 256 + gate 256) | column, heads 3r..3r+2 (contiguous 1536 rows) | [1536, 5120] | input identity / **all-reduce** bwd |
| attn k_proj / v_proj | weight [1024, 5120] = 4 kv heads | kv head r // 2 (4 heads < 8 ranks -> each head replicated on 2 ranks) | [256, 5120] | grad **all-reduce SUM over the rank pair** {2h, 2h+1} |
| attn q_norm / k_norm | weight [256] | replicated | [256] | grad **all-reduce SUM** |
| attn o_proj | weight [5120, 6144] | row, cols 768r.. | [5120, 768] | output **all-reduce** (fwd) |
| MLP gate_proj / up_proj | weight [17408, 5120] | column rows 2176r.. | [2176, 5120] | input identity / **all-reduce** bwd |
| MLP down_proj | weight [5120, 17408] | row cols 2176r.. | [5120, 2176] | output **all-reduce** (fwd) |
| input / post-attention / final RMSNorm | weight [5120] | replicated | [5120] | none (gradient already complete and identical on every rank; averaged anyway as a drift guard) |
| lm_head | weight [248320, 5120] | vocab rows chunk r | [31040, 5120] | input identity / **all-reduce** bwd; loss via vocab-parallel CE: all-reduce MAX (detached), 2x all-reduce SUM of [n] |

Attention with `num_key_value_groups` 6 -> 3 per rank (3 q heads over 1 kv head); sdpa reads the module attribute.
GDN `repeat_interleave(num_v_heads // num_k_heads = 3)` is unchanged: local value head i uses local key head i // 3,
i.e. global value head 6r+i uses global key head 2r + i//3 = (6r+i)//3, exactly the full model's mapping.

Local parameters per rank: 3.37B of 27B (1/8 of every matrix + the replicated norms, ~0.7M). fla's chunk kernels see
[B, T, 6, 128] contiguous per-head tensors -- the same layout as the full model with fewer heads (no re-striding).

**Communication per layer per micro-batch**: 2 all-reduces of a [tokens, 5120] bf16 tensor in the forward (after out_proj
/ o_proj and after down_proj) and 2 in the backward (input gradients of the column-parallel projections). 64 layers ->
256 all-reduces of ~116 MB (11k tokens) per micro-batch, plus embed (1 fwd) and head (1 bwd + 3 tiny). All collectives
are `torch.autograd.Function`s (Megatron's f/g); there is no DTensor and no sequence parallelism (the residual is only
5,120 wide, so the replicated elementwise work is the norms -- handled by the fused kernel below -- and two adds).

## 3. Precision and optimizer

Identical to FSDP2's `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`: the module holds **bf16** params
(the model as loaded, sliced in place) and computes in bf16; every backward's bf16 parameter gradient is added into an
**fp32 accumulator** by a `register_post_accumulate_grad_hook` on the bf16 param (== FSDP2's fp32 reduce-scatter into the
sharded grad; grad accumulation across micro-batches / the shared-prefix backward happens in fp32); **fp32 master
weights** + AdamW moments exist for the local shard only (`TPModel.build_masters`, any `--optim` variant of
`fullft.make_optimizer`); after `opt.step()` the masters are copied into the bf16 params (`sync_params`).

Gradient completion after the step's backwards (`reduce_grads`): SUM over the TP group for the per-head norm weights,
SUM over the kv rank pair for k_proj/v_proj, average (a no-op when identical) for the replicated norms; everything
else is complete locally. Clipping (`clip_grad_norm`) forms the global norm from the local shards (kv-pair params
weighted 1/2, replicated norms counted once), same coefficient formula as `torch.nn.utils.clip_grad_norm_`.

## 4. Prefix cache under TP

`sft/prefix_cache.py` is unchanged. The prefix forward (B=1, 102 tokens) produces, on each rank, GDN conv states
[1, 1280, 4] and recurrent states [1, 6, 128, 128] and attention K/V [1, 1, 102, 256] for the LOCAL heads -- these ARE
the per-rank slices of the full model's states, because the module simply has fewer heads. `batch_repeat_interleave`
/ `expand_cache_copy` expand them to the micro-batch; the suffix forward consumes them; `PrefixGradientAccumulator`
sums the cache gradients and walks the prefix graph once. No FSDP hooks -> `keep_prefix_grad_path` is off, and
`--prefix-accum > 1` (retain_graph) is allowed again.

The injection hook (`mxf.inject`, layer 1 output, `add_clone`) acts on the replicated residual identically on every rank.

## 5. Fused RMSNorm (`--tp-fused-norm 1`, default)

The 129 residual-stream RMSNorms run on the replicated [tokens, 5120] stream, i.e. every rank does 8x FSDP2's per-rank
norm work. HF's eager norm saves ~4 fp32 copies of the input (x.float(), normalized, product) = ~0.9 GB/layer at 11k
tokens, ~60 GB per micro-batch, and costs ~1.3 ms per norm fwd+bwd. `install_fused_rmsnorm` swaps in fla 0.5.2's
Triton `rms_norm` (fp32 math, one rounding to bf16, saves x + rstd), with weight `1 + w` formed in fp32 so HF's
zero-centred parametrisation and its gradient are unchanged. The attention q/k norms and the GDN gated norm (sharded
tensors) stay eager.

## 6. Memory estimate per rank (bf16 activations unless noted; T = suffix tokens per micro-batch, ~22 per row)

| component | mb 256 (T = 5.6k) | mb 512 (T = 11.3k) |
|---|---|---|
| bf16 params (local 3.37B) | 6.8 GB | 6.8 GB |
| fp32 masters + fp32 grad accumulators + AdamW m, v | 54 GB | 54 GB |
| residual-stream activations: 2 norm inputs + 2 projection inputs per layer x 64 (bf16) | 15 GB | 30 GB |
| MLP (gate, up, silu, product; sharded 2176 wide) | 6.5 GB | 13 GB |
| GDN (fla saved q/k/v/g/beta + chunk h-states + expanded fp32 recurrent states, 6 heads) | 17 GB | 33 GB |
| attention (3 heads; K/V cache expanded) | 2.5 GB | 5 GB |
| head: local logits [T_lab, 31040] fp32 + exp | 1.4 GB | 2.8 GB |
| **total (estimate)** | **~105 GB** | **~145 GB** |

(FSDP2 at mb 64: 140 GB; B200 usable ~178 GB.) Without the fused norm add +25 / +50 GB. Measured numbers: section 9.

## 7. Checkpoint format and resume

`fullft.save_full_ckpt(..., gather=tpm.gather_full)`: for every parameter, in `named_parameters()` order on all ranks,
the local shards are all-gathered and placed by their index lists into the full tensor; rank 0 writes bf16 safetensors
in the base-repo layout (`model.language_model.*`, `lm_head.weight`, non-text tensors copied, SAVE_DONE last). Byte-for-
byte the same schema as the FSDP2 checkpoints, so `AutoModelForCausalLM.from_pretrained(ckpt)`, vLLM, the evaluator
(`--full-model`) and the RL `--policy-base` path are unaffected. `--init-adapter <ckpt> --skip-steps N` resume works
as before (the checkpoint is a plain HF model; it is re-sharded on load). Data: under TP there is no DP, so
`load_shard(rank=0, world=1)` -- every rank holds the whole bank (~100 B/row) and the deterministic `step_groups`
composition is identical on all ranks; `--save-examples` counts examples as `batch_size x grad_accum` per step.

## 8. Exactness

* `sft/test_tp.py` (CPU/gloo, TP-2, tiny hybrid 2 GDN + 1 attention, d 256, fp32): full-sequence loss identical,
  gathered gradients global cos 1.0000000, worst per-param rel 9e-5 (A_log fp32 summation noise, |d| 4e-7); prefix-cache
  path + PrefixGradientAccumulator: loss |d| 5e-7, cos 1.0; DP-emulating `loss_group_ids`: |d| 5e-7; clip norm rel 7e-6;
  AdamW step params rel 3e-5; kv-head replication (1 kv head, TP-2) all OK.
* `sft/tp_exactness.py` (8xB200, real 27B, same DP-dealt micro-batch window): see section 9.

## 9. Measured (8xB200, Modal, torch 2.10 / fla 0.5.2 / transformers fork e52940e; tiny bank replicated to 132k rows,
targets 8-57 tokens, legacy length-bucket composition, `--prefix-cache --prefix-share-step --pad-multiple 1`)

Throughput (`sft/results/tp_bench_eff512.json`, `tp_bench_eff4096.json`; bench = 6-8 logged steps after a 3-4 min step 0
that JIT-compiles / autotunes the fla + Triton kernels in BOTH paths; median of the steps >= 2):

| config | examples / step | s / step (median) | ex/s (node) | TFLOP/s per GPU | peak GB / rank |
|---|---|---|---|---|---|
| FSDP2 mb 64 x ga 8 (production, `--fsdp-prefetch 2`) | 4,096 | 6.46 | **629** (78.6 / rank) | 267 | 138 |
| TP-8 mb 512 x ga 8 | 4,096 | 5.77 (5.0-5.9 once warm) | **696** (711-814 warm) | 281-362 | **166.7** |
| TP-8 mb 256 x ga 16 | 4,096 | 6.49 | 607 | 251-291 | 112 |
| TP-8 mb 512 x ga 8, eager HF norm (`--tp-fused-norm 0`) | 4,096 | OOM | -- | -- | > 178 |
| TP-8 mb 512 x ga 8, fp32 all-reduces (`--tp-reduce-fp32 1`) | 4,096 | OOM at step 3 (6.7-7.3 s before) | -- | -- | > 178 |
| TP-8 mb 256 x ga 16, fp32 all-reduces | 4,096 | 6.66 | 613 | 254-287 | 112 |
| TP-8 mb 512 x ga 1 | 512 | 1.33 (0.9-2.3, tokens 7-17k) | 386 | 165-197 | 154 |
| TP-8 mb 256 x ga 2 | 512 | 2.35 | 223 | ~100 | 112 |
| TP-8 mb 128 x ga 4 | 512 | 2.49 | 218 | ~90 | 85 |

At effective batch 4,096 TP-8 with mb 512 is 1.1x (median) to 1.3x (warm steps) the FSDP2 baseline and mb 256 is 0.97x;
the eff-512 rows show the per-step fixed cost (prefix fwd+bwd at B=1, optimizer over 54 GB, ~0.4 s) dominating small
steps. Per-token marginal cost at mb 512 is ~0.13 s per 1k suffix tokens per GPU (~155-200 TFLOP/s).

Profile of one mb-512 micro-batch step (`--profile-step 3`, torch.profiler, self device time; `sft/results/tp_bench_eff4096.json`
notes): GPU busy ~4.5 s of a ~5.8 s unprofiled step (77%; the rest is CPU launch overhead -- fla's Triton launchers cost
1-6 ms CPU per call, 432 GDN backward calls + 1,159 norm calls per step). Of the GPU time: **GEMMs 29%** (aten::mm 1.31 s
for 1.83 PFLOP per GPU = ~1.4 PFLOP/s, i.e. the matmuls themselves run at ~60% of B200 peak), **NCCL all-reduce 17%**
(2,586 calls x 293 us for ~116 MB = ~400 GB/s algbw, not overlapped), **fla GDN chunk kernels 15%**, **depthwise conv1d
torch fallback 10%** (the `causal_conv1d` CUDA package is not installed: 432 x 0.3 ms fwd + 0.5 ms bwd), elementwise
glue (copy_/add_/mul/sum: 70k launches) ~22%, sdpa mem-efficient attention backward 3%, fused RMSNorm 3%.
So the TP step is NOT GEMM-bound: the same non-GEMM work that made FSDP2 launch-bound at 1,350 tokens/rank is still 70%
of the GPU time at 11k tokens/rank, and the CPU cannot feed the GPU faster than ~77%.

Memory at mb 512 (11-18k suffix tokens; `[mem]` attribution at the within-step peak, 166.7 GB): MLP activations 18 GB,
fused-norm saved inputs 13.6 GB, residual adds (the next norm's input) 21 GB, fla `chunk_gated_delta_rule_fwd_h` h-states
9 GB, expanded fp32 recurrent states (`batch_repeat_interleave`) 9 GB, GDN `apply_mask_to_padding_states` copies 8 GB,
GDN gated-norm fp32 upcasts ~10 GB, conv 2.4 GB, l2norm 2.4 GB, head ~3 GB; static (bf16 params 6.8 + fp32 masters,
grads, Adam 54) ~61 GB. The eager HF norm adds ~50 GB and OOMs. mb 512 has no headroom for the longest length buckets
(512 x 57 tokens = 29k tokens would need ~230 GB) -> production must use mb 256 (or `--length-bucket`/a token cap).

Exactness on the real 27B (`sft/tp_exactness.py`, `sft/results/tp_exactness_v*.json`; the same DP-dealt window, mb 32 per DP
rank = TP mb 256, DP-emulating loss normalisation; step 0 compared before the optimizer step, then 20 constant-lr AdamW steps):

| leg (vs one FSDP2 reference run) | step-0 loss (ref 5.6048) | grad norm (ref 76.7) | grad cos global / median / worst | params cos < 0.999 | 20-step trace mean / max abs loss diff |
|---|---|---|---|---|---|
| FSDP2 **re-run** (noise floor: fla atomics, NCCL order) | 5.6039 (-9e-4) | 76.4 | 0.99972 / 0.99972 / 0.992 | 23 | 0.14 / 1.24 |
| TP-8, fused norm (production default) | 5.6100 (+5.1e-3; v1 run 5.6099) | 79.9 (v1 77.8) | 0.9926 / 0.9939 / 0.86 | 819 | 0.065 / 0.20 (v1: 0.18 / 2.1) |
| TP-8, eager HF norm (`--tp-fused-norm 0`) | 5.6052 (+3.5e-4) | 78.7 | 0.9907 / 0.9938 / 0.69 | 816 | 0.05 / 0.29 |
| TP-8, fused norm, fp32 activation all-reduces (`--tp-reduce-fp32 1`) | 5.6080 (+4.1e-3) | 79.6 | 0.9943 / 0.9952 / 0.85 | 791 | 0.12 / 0.34 |
| TP-8, eager norm, fp32 all-reduces | 5.6047 (+7.9e-4) | 79.0 | 0.9950 / 0.9956 / 0.78 | 796 | 0.063 / 0.31 |
| FSDP2 re-run with each rank's micro-batch as 2 halves (`--split 2`: same objective and kernels, GEMM M halved) | 5.6039 (0.0) | 76.5 | 0.99997 / 0.99998 / 0.998 | 1 | 0.095 / 0.69 |

Reading: the bf16 FSDP2 path is itself only reproducible to ~1e-3 in loss and 0.9997 in gradient cosine, and its 20-step
trace wanders by up to 1.2 in loss on the same data (chaotic after a few steps) -- every TP trace sits inside that band.
The step-0 LOSS matches to within the FSDP2 re-run noise once HF's eager norm is used (+3.5e-4 / +7.9e-4); the fused fla
norm shifts it by +4-5e-3 (its bf16 output rounds differently), i.e. right at the 5e-3 tolerance. The step-0 GRADIENT is
~20x further from the reference than a re-run (median cos 0.994-0.996 vs 0.9997) in all four TP variants -- neither the
norm kernel nor the bf16 accumulation of the partial GEMM sums (fp32 all-reduces: 0.9943 -> 0.9952) explains it. What
remains is that TP computes every activation with different GEMM shapes (K split 8-way, M 8x larger, different tiles),
so the bf16 rounding STRUCTURE differs: every row-parallel output (out_proj / o_proj / down_proj, 128 per forward) is the
sum of 8 partial GEMMs each rounded to bf16 before the reduce, and every column-parallel input gradient (128 per backward)
likewise, where the full model rounds the complete K=6144 / 17408 sum once. The `--split 2` FSDP2 control (same code, GEMM
M halved: only the fp32 accumulation order inside each GEMM changes) stays at cos 0.99997, so tile-order noise is NOT the
explanation -- the 8-way partial rounding is the one structural difference left. Fix (not implemented here): compute the
row-parallel partials and the column-parallel backward dX partials with an fp32 GEMM output (`torch.mm(..., out_dtype=
torch.float32)`), all-reduce in fp32, round to bf16 once -- a custom autograd linear for the 3 row-parallel and 4
column-parallel projection groups (~80 lines), at +232 MB of transient fp32 per [11k, 5120] output. The CPU/gloo unit test (deterministic fp32 torch kernels) matches the single-process
reference to 1e-6 relative, so the sharding arithmetic itself is exact; on GPU the same test at fp32 gives global cos
1.0000000 with per-param deviations only on the fla g-path parameters (A_log / dt_bias / in_proj_a, rel 3e-3: fla's
chunk backward accumulates dg with atomics).

Wall-clock for a 104M-example run (eff batch 4,096, 25.4k steps, excluding the ~4 min kernel JIT per leg and checkpoint
saves): FSDP2 629 ex/s -> 45.9 h; TP-8 mb 512 x 8 at 696 ex/s -> 41.5 h (36-38 h at the 750-810 ex/s warm steps, but
166.7 GB peak: not safe for the longest length buckets); TP-8 mb 256 x 16 at 607-613 ex/s -> 47-48 h. Under TP every rank
also tokenizes the WHOLE bank at start-up (load_shard with world 1): ~8x the FSDP2 start-up tokenization (~15-20 min for
104M rows vs 2-3 min), an easy follow-up (tokenize the shard, all-gather the flat TokRows arrays).

## 10. Risks / not done

* GDN state layout: the per-rank recurrent/conv states are slices by construction, but any consumer that reads the cache
  tensors assuming full head counts (e.g. a generation loop on the TP model) would see local heads only. Training only.
* fla contiguity: q/k/v are reshaped from the conv output exactly as in the full model; `repeat_interleave` still
  materialises q/k copies (same as today).
* Replicated-norm drift: the norms' gradients are complete on every rank by construction; they are additionally
  averaged every step (2.6 MB) so bit-level drift cannot accumulate.
* Sequence parallelism is not implemented (would shard the norms/adds and the residual activations 8x more; comm
  volume unchanged). The fused norm removes most of the replicated-norm cost.
* `--suffix-ckpt` under TP: the layer wrapper re-runs the collectives in the recompute on every rank in lockstep;
  compatible in principle, not measured.
* Hybrid DP x TP (e.g. 2 x 4) is not implemented (`--tp` must equal the world size).
