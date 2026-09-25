"""Disaggregated GRPO for the MAEMM universal inverter: X vLLM ROLLOUT GPUs + Y HF TRAINER GPUs in ONE
container (N = X + Y processes, one GPU each), coupled through the container-local filesystem.

Why. rl/rl.py hosts BOTH the HF actor (61 GB resident, ~98 GB peak) AND a vLLM engine on EVERY GPU:
micro-batch is stuck at 3, vLLM gets a third of the GPU, old_logp is recomputed in HF, and the vllm_lens
hook is O(layers x reqs x keys) per decode step -> 145-250 s per 1024-rollout step on 4 GPUs. Here every
GPU does ONE job, the trainer never runs vLLM (-> no grad checkpointing, micro-batch 16-32), the sampler's
own per-token logprobs ARE old_logp (no HF recompute; the 1-2 step policy lag is importance-corrected by
the PPO clip + TIS cap from rl.py), and rollout GPUs run vLLM at 0.85 memory / 1024 seqs with a fast
steering hook (rl/fast_lens_ext.py) and optional CUDA graphs for decode.

Roles (this file, selected by --role; the launcher spawns the others):
  launch   parent: N children, CUDA_VISIBLE_DEVICES=<one gpu> each (GPUs [0,Y) trainers, [Y,N) rollouts),
           streams their stdout with [T<k>]/[R<r>] prefixes, tears everything down when the trainer
           finishes or anything dies.
  trainer  DDP group of Y ranks (gloo or nccl) over the HF actor + LoRA + AdamW (+ frozen 'ref' adapter):
           consume rollout blocks -> reward with rl.py's score() on the clean base -> compute_advantages
           -> clipped policy gradient with vLLM logprobs as old_lp -> capped-k3 KL to init -> exact
           token/seq-weighted grad all-reduce -> AdamW -> rank 0 publishes the adapter (+ ||h_marker||)
           for the rollout ranks every --publish-every steps, checkpoints every --save-every, wandb.
  rollout  one vLLM engine (TP=1, LoRA, full GPU); loop: pick up the newest published adapter, draw a
           block of directions from the bank, generate G samples each with per-request steering at the
           marker, write a rollout block file; block until the queue drains below --max-queue-blocks.

Filesystem protocol (all under --work-dir, default /tmp/disagg):
  lora/step_<k>/{adapter_model.safetensors, adapter_config.json, meta.json}   vLLM key layout
      (rl.py _save_adapter_for_vllm) + meta {"step", "hnorm": ||h_marker|| with the adapter ON}
  lora/latest            the integer k, written atomically (tmp + os.replace); rollouts reload on change
  queue/blk_<adapter_step>_<t_ns>_<rank>.pt   torch.save dict: dir_idx, dirs [B_r,d], gen_ids, lps
      (vLLM per-token logprobs; None where the engine dropped the stop token -> ratio 1 there),
      adapter_step, gen_s, n_tok, rank; written atomically. Trainer consumes FIFO (oldest adapter first)
      or --drop-stale (newest, older blocks discarded); consumed files are deleted.
  STOP                   trainer done -> rollout ranks exit.

Sampling rng: rollout rank r draws from numpy default_rng(seed*7919 + 1000 + r) -- per-rank streams,
NOT rl.py's every-rank-draws-the-same-B-then-slices scheme (there is no shared step any more). The first
--n-eval-dirs unique bank blocks are reserved exactly as in rl.py.

Metrics keep rl.py's names (reward/mean, policy/entropy, policy/kl_to_init, ratio/mean, ratio/clipfrac,
grad_norm, rollout/len_mean, time/step_s, var/*, rollouts/samples ...) plus
policy/offpolicy_lag_steps, policy/sampler_abs_dlogp, time/wait_rollouts_s, time/grad_sync_s,
rollout/queue_depth, rollout/gen_s, rollout/tok_per_s_per_replica, rollout/blocks_dropped.

ScaleRL variant (Khatri et al. 2025, "The Art of Scaling Reinforcement Learning Compute for LLMs", arXiv 2510.13786):
OPT-IN -- every flag below defaults to the legacy behaviour above (default run = bit-identical to before); --recipe scalerl
sets the whole bundle, flags given explicitly still win. Pure pieces are unit-tested on CPU in rl/test_rl_disagg_scalerl.py.
  --max-lag 8           PipelineRL-8: the rollout queue may hold 8 steps' worth of blocks, so a block can be up to ~8 policy
                        updates stale (legacy: 1). old_lp stays the SAMPLER's logprobs of the adapter that generated the
                        block, so the per-token IS weight below is the off-policy correction. Adapters swap between blocks
                        (a block = one <=96-token generate() call), not inside one, which is our grain of "in-flight" updates.
  --loss cispo          -sg(min(rho, eps_max)) * A * log pi   (truncated-IS REINFORCE, MiniMax-M1 / ScaleRL): no PPO clip of
                        the objective, every token keeps a gradient. --cispo-eps-max (paper ablates {4,5,8}: no difference).
  --loss-agg prompt     token-mean within each direction's G rollouts, then mean over directions (each prompt weighs 1).
  --adv-mode batch      (r - group_mean) / std of ALL surviving advantages in the global batch (rl.py's ScaleRL mode).
  --zero-var-filter     groups with identical rewards (std <= --zero-var-eps) leave the effective batch: loss weight 0 AND
                        out of every denominator (the paper's "effective batch"). Near-inert for a continuous cosine reward.
  --npr-threshold 0.9   No-Positive-Resampling, ADAPTED to the continuous reward: a rollout is a "positive" when its raw
                        cosine >= --npr-pass-cos; a direction whose cumulative pass rate over all its visits >= threshold is
                        dropped from all future sampling (trainer rank 0 publishes npr/dropped.json, rollout ranks exclude it).
  --autocast-bf16       trainer: policy forward under torch.autocast(bf16) with PEFT's LoRA input casting off -> bf16 LoRA
                        matmuls + bf16 saved activations (fp32 masters/AdamW, fp32 vocab math, inject hook, fp32 head unchanged)
  --fp32-head           trainer recomputes the frozen lm_head projection in fp32 (the log_softmax over the logits already
                        was fp32 -- see _chunked_logp; the vLLM sampler stays bf16-head / fp32-softmax, we cannot change it).
  --length-control      penalty (default, ALSO under --recipe scalerl: keeps the hinge LP) | interrupt (no LP; cap-hit
                        snippets are scored as generated = our analogue of the forced wrap-up; needs --trunc-reward unset).
  metrics under scalerl/*: is_weight_mean, is_trunc_frac, zero_var_dropped_frac, effective_groups, npr_* , lag_max,
  trunc_frac, step_skipped. Every pre-existing metric name is unchanged.

Trainer speed knobs (opt-in, defaults = the behaviour above; exact up to bf16 kernel noise, rl/test_rl_disagg_prefix.py):
  --prefix-cache        the ~102-token shared prompt prefix runs ONCE per pass (policy: 1 fwd + 1 bwd per step via fp32
                        cache-gradient accumulation; KL-ref: 1 no-grad fwd), only [marker]+response per rollout on the
                        batch-expanded cache. Needs the transformers fork (modal_rl_disagg.py DISAGG_TRANSFORMERS).
  --score-length-bucket the scoring pass runs on length-sorted rollouts (each --score-batch pads to its own longest response).
                        The update's micro-batches have always been length-sorted (update_disagg chunks()); trainer/pad_frac and
                        trainer/body_tokens_per_rollout log the residual padding / the tokens actually run per rollout.

--policy-base <dir|hub id>  (default MODEL = byte-identical to before): the POLICY lives on another base -- a FULL fine-tuned
                      checkpoint in sft/fullft.py layout (SAVE_DONE required) -- while the REWARD stays the ORIGINAL base.
                      Rollout engines serve <policy-base> (+ the LoRA slots), the trainer builds/continues the LoRA on
                      <policy-base> (--init-adapter may be omitted = fresh rsLoRA r64/a16; `--init-adapter none` unsets the
                      launcher default), and every trainer rank loads a SECOND, frozen copy of MODEL for rl.py score() /
                      the inline-eval scorers (with the gates off it keeps only layers [0, READ_LAYER]: read_resid stops
                      there; --scorer-layers). --kl-coef without any ref/init adapter anchors to <policy-base> itself (LoRA
                      disabled). Checkpoints are PEFT adapters as before + run_meta.json ({"policy_base": ...}) next to
                      them, which eval/eval_ckpt_daemon.py reads to rebuild policy = base + adapter, scorer = MODEL.

Launch inside the container (see modal_rl_disagg.py):
    python RL/rl_disagg.py --role launch --n-rollout 1 --n-trainer 3 --data-dir <pool> --init-adapter <sft> ...
"""
import argparse
import contextlib
import gc
import glob
import json
import math
import os
import re
import pickle
import shutil
import subprocess
import sys
import threading
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


# ----------------------------------------------------------------------------------------------
# args (superset of rl.py's flags so the v15 TRAIN_ARGS list can be reused verbatim)
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("launch", "bench", "trainer", "rollout", "bench-rollout", "bench-trainer"), default="launch",
                    help="launch/bench = parent (spawns children); the rest are the per-GPU child roles")
    ap.add_argument("--n-rollout", type=int, default=1, help="X: vLLM rollout GPUs")
    ap.add_argument("--n-trainer", type=int, default=3, help="Y: HF trainer GPUs (DDP)")
    ap.add_argument("--work-dir", default="/tmp/disagg")
    ap.add_argument("--backend", default="nccl", choices=("gloo", "nccl"),
                    help="trainer DDP backend (nccl: GPU flat-buffer grad all-reduce; gloo = rl.py's CPU path, ~4.5 s for 2 GB)")
    ap.add_argument("--master-port", type=int, default=29611)
    # data / init / resume (rl.py)
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--bank-file", default="vecs.f32")
    ap.add_argument("--direction-source", choices=("cluster", "random"), default="cluster")
    ap.add_argument("--n-eval-dirs", type=int, default=64)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--ref-adapter", default=None)
    ap.add_argument("--policy-base", default=None,
                    help="HF model the POLICY is built on (the vLLM rollout engines serve it, the trainer puts the LoRA on it): a FULL "
                         "fine-tuned checkpoint dir in sft/fullft.py layout (must carry SAVE_DONE) or a hub id. Default = mxf.config.MODEL "
                         "(the original base; byte-identical to before). The REWARD scorer always stays MODEL: with another policy base "
                         "every trainer rank loads a second, frozen copy of MODEL for scoring (--scorer-layers). 'none' = default.")
    ap.add_argument("--scorer-layers", type=int, default=0,
                    help="--policy-base != MODEL only: decoder layers kept in the frozen scorer copy of MODEL (the reward reads layer "
                         "READ_LAYER=42 and read_resid stops there, so layers 43+ and lm_head never run). 0 = auto: READ_LAYER+1 when the "
                         "fluency/distinct gates are off (~-18 GB per trainer rank), the full model otherwise; -1 = always the full model")
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--save-dir", default="checkpoints/rl_disagg")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-steps", default="", help="comma-separated extra checkpoint steps, e.g. 25,40,60,90,130,200,300,450,675,1000 (log-spaced)")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup over the first N global steps (stability)")
    ap.add_argument("--lr-decay", choices=("none", "linear", "cosine"), default="none",
                    help="LR decay from --lr (after warmup) to --lr-min-frac*lr at --total-steps. Default none = constant LR, which "
                         "blew up RL-C at step ~300 (entropy collapse -> grad-norm explosion, see memory 2026-09-05)")
    ap.add_argument("--lr-min-frac", type=float, default=0.0, help="final LR as a fraction of --lr for --lr-decay")
    ap.add_argument("--fresh-optim", action="store_true", help="do NOT load <init-adapter>/optim.pt on resume (fresh AdamW moments). Needed when the advantage "
                    "scale changes between the saved run and this one (e.g. batch-normalized -> raw), otherwise the stale second moment rescales the effective lr")
    ap.add_argument("--lr-decay-total-steps", type=int, default=None, help="decay horizon (global step at which lr reaches lr-min-frac*lr); default = --total-steps. "
                    "Lets a short resumed ablation follow the schedule a full 400-step run would have.")
    ap.add_argument("--run-name", default="mxf-rl-disagg")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # batch / sampling (rl.py)
    ap.add_argument("--groups-per-step", type=int, default=128, help="B: directions consumed per trainer step (global)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rollout-chunk", type=int, default=64, help="IGNORED (rl.py hf engine)")
    ap.add_argument("--logp-chunk", type=int, default=16, help="IGNORED (no HF old_logp recompute here)")
    ap.add_argument("--rollout-engine", default="vllm", help="IGNORED (always vllm)")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.85, help="rollout ranks own the GPU: 0.85-0.9")
    ap.add_argument("--vllm-logp-tol", type=float, default=0.10, help="IGNORED (see policy/sampler_abs_dlogp at step 0)")
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = auto: largest of --mb-candidates that fits at max length")
    ap.add_argument("--mb-candidates", default="64,48,40,32,24,16,12,8,6,4",
                    help="micro-batch probe candidates (with --prefix-cache the untouched default also tries 128,96)")
    ap.add_argument("--ref-micro-batch", type=int, default=32)
    ap.add_argument("--score-batch", type=int, default=128)
    ap.add_argument("--vocab-chunk", type=int, default=32, help="positions per fp32 log_softmax chunk over the 248k vocab")
    # reward (rl.py)
    ap.add_argument("--reward-metric", choices=("proj", "cosine"), default="proj")
    ap.add_argument("--reward-scale", type=float, default=1.0)
    ap.add_argument("--log-reward", action="store_true")
    ap.add_argument("--reward-window-last", type=int, default=0)
    ap.add_argument("--reward-topk", type=int, default=1)
    ap.add_argument("--reward-pos-penalty", type=float, default=0.0)
    ap.add_argument("--fluency-floor", type=float, default=-4.5)
    ap.add_argument("--distinct-floor", type=float, default=0.5)
    ap.add_argument("--gate-penalty", type=float, default=25.0)
    ap.add_argument("--len-penalty-start", type=int, default=64)
    ap.add_argument("--len-penalty-per-tok", type=float, default=0.5)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--no-len-penalty", action="store_true")
    # optimization (rl.py)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decoupled weight decay (ScaleRL paper: 0.01; default 0 = previous behaviour)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--tis-cap", type=float, default=2.0)
    ap.add_argument("--adv-mode", choices=["none", "group", "batch"], default=None)
    ap.add_argument("--loss-agg", choices=["token", "seq", "prompt"], default=None,
                    help="token (default) | seq | prompt (ScaleRL: token-mean within each direction's group, then mean over directions)")
    ap.add_argument("--trunc-reward", type=float, default=None)
    ap.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.999), metavar=("B1", "B2"))
    ap.add_argument("--grad-ckpt", action="store_true", help="IGNORED (never needed off-vLLM)")
    ap.add_argument("--std-norm", action="store_true")
    ap.add_argument("--batch-norm", action="store_true")
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--entropy-target", type=float, default=0.0,
                    help="adaptive entropy bonus: after each step coef *= exp(rate*(target - H)) so the policy's per-token "
                         "entropy is held near TARGET nats (0 = fixed --entropy-coef). Replaces the KL leash as the anti-collapse term.")
    ap.add_argument("--entropy-adapt-rate", type=float, default=0.05)
    ap.add_argument("--entropy-coef-max", type=float, default=0.05)
    ap.add_argument("--entropy-coef-min", type=float, default=1e-4)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--kl-cap", type=float, default=10.0)
    # inline eval flags accepted for launcher compatibility; see inline_eval_stub()
    ap.add_argument("--inline-eval-every", type=int, default=0)
    ap.add_argument("--eval-cache", default=os.environ.get("MAEMM_EVAL_CACHE", "/data/eval_universal_ho/eval_sets_heldout.pt"))
    ap.add_argument("--eval-sae", default="/data/sae/ae.pt")
    ap.add_argument("--eval-bo", type=int, default=4)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--eval-max-new", type=int, default=64)
    ap.add_argument("--eval-min-new", type=int, default=16)
    ap.add_argument("--no-extra-evals", action="store_true")
    ap.add_argument("--eval-n-per-family", type=int, default=0)
    ap.add_argument("--transcript-every", type=int, default=5)
    ap.add_argument("--transcript-groups", type=int, default=4)
    ap.add_argument("--transcript-samples", type=int, default=4)
    ap.add_argument("--div-coef", type=float, default=0.0)
    ap.add_argument("--firsttok-coef", type=float, default=0.0)
    # disaggregation
    ap.add_argument("--rollout-block-groups", type=int, default=0,
                    help="directions per rollout block (per generate call); 0 = groups_per_step / n_rollout")
    ap.add_argument("--max-num-seqs", type=int, default=0, help="vLLM max_num_seqs; 0 = block_groups*group_size")
    ap.add_argument("--gdn-prefill-backend", choices=("triton", "flashinfer", "auto"), default="triton",
                    help="vLLM GatedDeltaNet PREFILL kernel. vLLM 0.19's 'auto' picks flashinfer's gdn_prefill on sm90 (H100/H200) -- a "
                         "JIT-compiled CUDA extension that needs nvcc/CUDA_HOME, absent from our image (EngineDeadError on 4xH200) -- and the "
                         "vendored fla Triton kernel everywhere else (what every B200 measurement ran). 'triton' = the Triton kernel on all "
                         "archs (default); 'auto' = vLLM's choice; 'flashinfer' = force the JIT kernel (needs nvcc in the image).")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="vLLM FULL_DECODE_ONLY cudagraphs (compilation mode NONE) instead of the plugin-forced eager")
    ap.add_argument("--stock-lens-hook", action="store_true", help="use vllm_lens' stock O(reqs x keys x layers) hook (for A/B timing)")
    ap.add_argument("--max-queue-blocks", type=int, default=0,
                    help="rollout backpressure: max unconsumed blocks; 0 = one step's worth (lag stays 1 with full overlap)")
    ap.add_argument("--drop-stale", action="store_true", help="consume the NEWEST blocks and discard older ones (min lag, wastes rollouts)")
    ap.add_argument("--publish-every", type=int, default=1)
    ap.add_argument("--keep-loras", type=int, default=10,
                    help="published adapters kept: an eval request needs its adapter on disk until every rollout rank has generated its shard "
                         "(vLLM has 2 LoRA slots; the live policy advancing each step can evict the eval adapter -> reload from disk)")
    ap.add_argument("--publish-fp32", action="store_true",
                    help="publish the adapter in fp32 (rl.py behaviour). Default bf16: vLLM casts LoRA weights to the model dtype (bf16) on load anyway, so the served policy is identical and the write is half the size")
    ap.add_argument("--no-fla", action="store_true", help="trainer: block the fla (flash-linear-attention) GDN kernels -> HF torch fallback")
    # inline eval (disaggregated): rollout ranks generate the held-out eval texts as a side job, trainer ranks score
    ap.add_argument("--eval-chunk-seqs", type=int, default=512,
                    help="rollout ranks: eval sequences per generate() call when filling idle time (one chunk per idle slot)")
    ap.add_argument("--eval-max-delay-s", type=float, default=120.0,
                    help="rollout ranks: after this many seconds a pending eval request is worked on even if the rollout queue is not full")
    ap.add_argument("--eval-drop-after-s", type=float, default=2400.0,
                    help="trainer: an eval request whose shards have not all arrived after this long is dropped with an error")
    ap.add_argument("--bench-sizes", default="128,256,512,1024", help="bench-rollout: sequences per generate call")
    ap.add_argument("--bench-configs", default="eager:512,graphs:512,eager:1024,graphs:1024",
                    help="bench-rollout: <eager|graphs|stock>:<max_num_seqs> list, sharded over rollout ranks "
                         "(stock = eager with vllm_lens' stock hook, the rl.py baseline)")
    ap.add_argument("--bench-rollouts-per-rank", default="128,256,512,1024", help="bench-trainer: update() sizes to time")
    # ScaleRL variant (module docstring). default=None means "not given", so --recipe fills the bundle without clobbering
    # explicit flags; _resolve_recipe() turns the Nones into the legacy values otherwise.
    ap.add_argument("--recipe", choices=("", "scalerl"), default="", help="scalerl = the whole ScaleRL bundle (explicit flags win)")
    ap.add_argument("--loss", choices=("ppo", "cispo"), default=None,
                    help="ppo (default: rl.py's clipped surrogate on the TIS-capped ratio) | cispo (truncated-IS REINFORCE)")
    ap.add_argument("--cispo-eps-max", type=float, default=None, help="CISPO IS-weight truncation (paper ablates {4,5,8}; bundle 5)")
    ap.add_argument("--zero-var-filter", dest="zero_var_filter", action="store_true", default=None,
                    help="drop zero-variance groups from the effective batch (loss weight 0 and out of the denominators)")
    ap.add_argument("--no-zero-var-filter", dest="zero_var_filter", action="store_false")
    ap.add_argument("--zero-var-eps", type=float, default=1e-6, help="a group is zero-variance when its reward std <= this")
    ap.add_argument("--npr-threshold", type=float, default=None,
                    help="No-Positive-Resampling: drop a direction from future sampling once its cumulative pass rate >= this (0 = off)")
    ap.add_argument("--npr-pass-cos", type=float, default=0.7,
                    help="continuous-reward adaptation of 'correct': a rollout is a positive when its RAW cosine >= this")
    ap.add_argument("--max-lag", type=int, default=None,
                    help="max off-policyness in trainer steps: the rollout queue holds this many steps' worth of blocks "
                         "(an explicit --max-queue-blocks overrides); legacy 1, ScaleRL 8")
    ap.add_argument("--autocast-bf16", action="store_true",
                    help="trainer: run the POLICY forward under torch.autocast(bf16) with PEFT's LoRA input-dtype casting disabled, so the "
                         "LoRA matmuls and their saved activations are bf16 (the fp32 LoRA master weights + AdamW, the fp32 chunked "
                         "log-softmax, the inject hook and the fp32 head are unchanged). Off = byte-identical to the default path.")
    ap.add_argument("--fp32-head", dest="fp32_head", action="store_true", default=None,
                    help="trainer: recompute the frozen lm_head projection in fp32 (ScaleRL/MiniMax precision fix, trainer side)")
    ap.add_argument("--no-fp32-head", dest="fp32_head", action="store_false")
    ap.add_argument("--length-control", choices=("penalty", "interrupt"), default=None,
                    help="penalty = the --len-penalty-* hinge (default, also under --recipe scalerl) | interrupt = no length "
                         "penalty, cap-hit snippets scored as generated (our analogue of ScaleRL's forced interruption)")
    # trainer speed knobs (both opt-in, both exact up to bf16 kernel noise; defaults = the behaviour above)
    ap.add_argument("--prefix-cache", action="store_true",
                    help="trainer: run the shared prompt prefix (every token before the marker, identical for all rollouts) ONCE per "
                         "pass -- with grad for the policy pass (one prefix fwd + one prefix bwd per step; the micro-batches' cache "
                         "gradients are accumulated in fp32 and pushed through the prefix graph once) and without grad for the KL-ref "
                         "pass -- expand its cache (attention K/V + GDN conv/recurrent states) to each micro-batch and run only "
                         "[marker]+response per rollout (sft/prefix_cache.py machinery). Needs the transformers fork "
                         "ceselder/transformers@maemm-prefix-cache (modal_rl_disagg.py: DISAGG_TRANSFORMERS).")
    ap.add_argument("--score-length-bucket", action="store_true",
                    help="trainer: run the SCORING pass on the rollouts sorted by response length so every --score-batch pads to its "
                         "own longest response instead of the step's (reorder only: rl.py score() reads each response standalone on "
                         "the clean base, so the rewards are unchanged up to bf16 kernel noise). The policy/KL-ref update needs no "
                         "such flag: update_disagg has always length-sorted its micro-batches (chunks(); trainer/pad_frac logs the "
                         "residual padding).")
    # --full-param (rl/rl_fullparam.py): the POLICY is the whole model -- FSDP2 fp32 masters + AdamW over the Y trainer ranks
    # (sft/fullft.py), bf16 weights pushed to the vLLM engines every step, full-model checkpoints. Off = every path above
    # byte-identical (the LoRA path never touches these flags).
    ap.add_argument("--full-param", action="store_true",
                    help="train EVERY weight of the policy (FSDP2 over the trainer ranks, fp32 masters/AdamW, bf16 compute) instead of a "
                         "LoRA. Init = --policy-base (a full model dir with SAVE_DONE) or MODEL; no --init-adapter/--ref-adapter; the "
                         "reward scorer is a frozen FSDP2-sharded copy of MODEL; checkpoints are full HF model dirs (sft/fullft.py "
                         "layout) at --save-steps/--save-every + final; inline eval unsupported (eval_ckpt_daemon --full-model)")
    ap.add_argument("--publish-mode", choices=("nccl", "fs"), default="nccl",
                    help="--full-param weight publish: nccl = NCCL broadcast trainer rank 0 -> every vLLM worker at a block boundary "
                         "(default) | fs = bf16 shard files in --work-dir (RAM-backed; ~54 GB per kept step), engines load at their "
                         "next block boundary")
    ap.add_argument("--wu-port", type=int, default=0, help="--publish-mode nccl: TCP store port of the weight-update group (0 = --master-port + 111)")
    ap.add_argument("--fs-keep-steps", type=int, default=2, help="--publish-mode fs: shard sets kept in --work-dir (each ~54 GB)")
    ap.add_argument("--fsdp-prefetch", type=int, default=None, help="--full-param: explicit FSDP2 all-gather prefetch depth (sft/fullft.py; default 2 = the fast configuration; 0 = implicit one-ahead)")
    ap.add_argument("--no-scorer-shard", dest="scorer_shard", action="store_false", default=True,
                    help="--full-param: keep the frozen scorer copy of MODEL UNsharded on every rank (~35 GB/rank instead of ~7) -- debug only")
    ap.add_argument("--mb-target-frac", type=float, default=0.0,
                    help="micro-batch probe: largest candidate predicted under this fraction of GPU memory (0 = 0.85). --full-param: the "
                         "budget also reserves the not-yet-allocated AdamW moments (2 x the fp32 master shard) and the probe walks the "
                         "candidates upwards, measuring each (an OOM inside an FSDP2 forward is not recoverable)")
    ap.add_argument("--suffix-ckpt", dest="suffix_ckpt", action="store_true", default=None,
                    help="--full-param: exact per-layer activation checkpointing of the SUFFIX forward (sft/prefix_cache.SuffixCheckpointer; "
                         "needs --prefix-cache): recompute in the backward -> the micro-batch grows 3x (8 -> 24 on 5 B200 ranks), i.e. that many fewer "
                         "per-micro-batch FSDP2 all-gathers/reduce-scatters (the entire update cost). DEFAULT ON with --full-param + --prefix-cache "
                         "(68 -> 44 s/step measured); --no-suffix-ckpt = the first validated configuration")
    ap.add_argument("--no-suffix-ckpt", dest="suffix_ckpt", action="store_false")
    ap.add_argument("--chunked-head", dest="chunked_head", action="store_true", default=None,
                    help="--full-param: never materialize [tokens x 248k] logits -- lm_head + fp32 log-softmax + gather/entropy per --vocab-chunk "
                         "positions under torch.utils.checkpoint (rl_fullparam.ChunkedHead; same math as --fp32-head + _chunked_logp). DEFAULT ON "
                         "with --full-param; --no-chunked-head = the fp32-head hook")
    ap.add_argument("--no-chunked-head", dest="chunked_head", action="store_false")
    ap.add_argument("--save-optim", action="store_true", help="--full-param: also write the sharded AdamW state (torch.distributed.checkpoint, fp32, ~2x the model) next to each checkpoint")
    ap.add_argument("--load-optim", default=None, help="--full-param: <ckpt>/optim_dcp dir to restore the AdamW state from (same world size)")
    a = ap.parse_args(argv)
    for k in ("init_adapter", "ref_adapter", "policy_base", "load_optim"):   # launcher lists can only APPEND flags: `--init-adapter none` unsets an earlier one
        if getattr(a, k) in ("", "none", "None"):
            setattr(a, k, None)
    assert a.div_coef == 0 and a.firsttok_coef == 0
    assert not (a.std_norm and a.batch_norm)
    assert a.temperature == 1.0, "T must be 1.0: the sampler's logprobs are the behaviour policy"
    if a.no_gates:
        a.fluency_floor = a.distinct_floor = None
    if a.no_len_penalty:
        a.len_penalty_start = None
    _resolve_recipe(a)
    if a.wu_port <= 0:
        a.wu_port = a.master_port + 111
    if a.mb_target_frac <= 0:
        a.mb_target_frac = 0.85
    if a.full_param:
        assert a.init_adapter is None and a.ref_adapter is None, "--full-param: the policy init is --policy-base (a full model dir) or MODEL; no LoRA adapters (--init-adapter/--ref-adapter must be unset or 'none')"
        assert a.inline_eval_every == 0, "--full-param: inline eval is not supported (checkpoints are full-model dirs: eval/eval_ckpt_daemon.py --full-model)"
        assert a.backend == "nccl", "--full-param needs --backend nccl (FSDP2)"
        assert not a.publish_fp32, "--full-param publishes bf16 weights (the engines run bf16)"
        assert a.role in ("launch", "trainer", "rollout"), "--full-param has no bench roles"
        if a.role in ("launch", "trainer"):
            assert a.n_trainer >= 2, "--full-param needs >= 2 trainer ranks (fp32 masters + AdamW of the 27B do not fit one GPU)"
        if a.autocast_bf16:   # FSDP2 MixedPrecisionPolicy(param_dtype=bf16) already runs the compute in bf16 (sft/pretrain.py does the same)
            a.autocast_bf16 = False
        # the FAST configuration is the default (bench 2026-09-09: 68 -> 44 s/step, peak 151 -> 139 GB); explicit --no-* flags opt out
        if a.suffix_ckpt is None:
            a.suffix_ckpt = bool(a.prefix_cache)
        if a.chunked_head is None:
            a.chunked_head = True
        if a.fsdp_prefetch is None:
            a.fsdp_prefetch = 2
        assert not a.suffix_ckpt or a.prefix_cache, "--suffix-ckpt is a --prefix-cache knob"
    else:
        assert not (a.suffix_ckpt or a.chunked_head), "--suffix-ckpt / --chunked-head are --full-param knobs"
        a.suffix_ckpt, a.chunked_head = False, False
        a.fsdp_prefetch = a.fsdp_prefetch or 0
    if a.rollout_block_groups <= 0:
        a.rollout_block_groups = max(1, a.groups_per_step // max(a.n_rollout, 1))
    assert a.groups_per_step % a.rollout_block_groups == 0, "groups_per_step must be a multiple of rollout_block_groups"
    a.blocks_per_step = a.groups_per_step // a.rollout_block_groups
    if a.max_queue_blocks <= 0:
        a.max_queue_blocks = a.max_lag * a.blocks_per_step      # legacy max_lag 1 -> one step's worth (unchanged)
    if a.max_num_seqs <= 0:
        a.max_num_seqs = a.rollout_block_groups * a.group_size
    return a


# The two bundles _resolve_recipe() fills unspecified (None) variant flags from. LEGACY == the pre-ScaleRL defaults.
SCALERL_BUNDLE = {"loss": "cispo", "cispo_eps_max": 5.0, "loss_agg": "prompt", "zero_var_filter": True,
                  "npr_threshold": 0.9, "max_lag": 8, "fp32_head": True, "length_control": "penalty"}
LEGACY_BUNDLE = {"loss": "ppo", "cispo_eps_max": 5.0, "loss_agg": "token", "zero_var_filter": False,
                 "npr_threshold": 0.0, "max_lag": 1, "fp32_head": False, "length_control": "penalty"}


def _resolve_recipe(a):
    """Fill every ScaleRL-variant flag the user did not give (None) from the bundle --recipe selects; --recipe scalerl also
    picks --adv-mode batch unless an advantage mode was given. With --recipe '' and no variant flags nothing changes."""
    bundle = SCALERL_BUNDLE if a.recipe == "scalerl" else LEGACY_BUNDLE
    for k, v in bundle.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.recipe == "scalerl" and a.adv_mode is None and not (a.std_norm or a.batch_norm):
        a.adv_mode = "batch"
    if a.length_control == "interrupt":
        a.len_penalty_start = None
        assert a.trunc_reward is None, "--length-control interrupt scores cap-hit rollouts as generated: unset --trunc-reward"
    assert a.cispo_eps_max > 0 and a.max_lag >= 1 and 0.0 <= a.npr_threshold <= 1.0
    assert a.max_lag == 1 or not a.drop_stale, "--max-lag > 1 needs the FIFO queue (--drop-stale would discard the lagged blocks)"
    return a


# ----------------------------------------------------------------------------------------------
# small shared helpers (filesystem protocol)
# ----------------------------------------------------------------------------------------------
def _atomic_write_text(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_latest(work):
    p = f"{work}/lora/latest"
    try:
        with open(p) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _queue_files(work):
    return sorted(glob.glob(f"{work}/queue/blk_*.pt"))


def _stop_requested(work):
    return os.path.exists(f"{work}/STOP")


def _log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def _bank_open(a):
    """Direction bank memmap + rl.py's eval-row reservation. Returns (bank, n_vecs, eval_rows)."""
    import numpy as np
    from mxf.config import D_MODEL
    stats_p = f"{a.data_dir}/build_stats.json"
    n_vecs = (json.load(open(stats_p))["n_examples"] if os.path.exists(stats_p)
              else os.path.getsize(f"{a.data_dir}/{a.bank_file}") // (4 * D_MODEL))
    bank = np.memmap(f"{a.data_dir}/{a.bank_file}", dtype=np.float32, mode="r", shape=(n_vecs, D_MODEL))
    eval_rows = 0
    if a.n_eval_dirs > 0:
        blocks, i = 0, 0
        while i < n_vecs and blocks < a.n_eval_dirs:
            row = np.asarray(bank[i]); i += 1; blocks += 1
            while i < n_vecs and np.array_equal(np.asarray(bank[i]), row):
                i += 1
        eval_rows = i
    return bank, n_vecs, eval_rows


# ----------------------------------------------------------------------------------------------
# --policy-base: the policy on a different (full fine-tuned) base than the reward scorer. Pure parts unit-tested on CPU in
# rl/test_rl_disagg_policy_base.py.
# ----------------------------------------------------------------------------------------------
def policy_base_of(a):
    """The HF model the policy (rollout engines + trainer LoRA) is built on: MODEL unless --policy-base is given."""
    from mxf.config import MODEL
    return getattr(a, "policy_base", None) or MODEL


def policy_base_is_model(a):
    from mxf.config import MODEL
    return policy_base_of(a) == MODEL


def check_policy_base(a):
    """A local --policy-base must be a COMPLETE sft/fullft.py checkpoint (SAVE_DONE is written last, after the rename). Returns the path."""
    pb = policy_base_of(a)
    if os.path.isdir(pb):
        assert os.path.exists(f"{pb}/config.json"), f"--policy-base {pb}: no config.json"
        assert os.path.exists(f"{pb}/SAVE_DONE"), f"--policy-base {pb}: no SAVE_DONE -- incomplete full-FT checkpoint (fullft.py writes it last)"
    return pb


class BaseActor:
    """A plain HF model with the PEFT-actor surface the SCORING code touches (rl.py score(), eval_universal._reencode,
    inline_extra_evals._profiles, rl._marker_norm(adapter=False)): disable_adapter() is a no-op context (it IS the clean base),
    get_base_model() returns the model, forward/generate/eval pass through. Used wherever the clean ORIGINAL base is held next
    to a policy built on another base (--policy-base trainer scorer; eval_ckpt_daemon --full-model / --policy-base)."""

    def __init__(self, base):
        self.base = base

    def __call__(self, *args, **kw):
        return self.base(*args, **kw)

    def forward(self, *args, **kw):
        return self.base(*args, **kw)

    def generate(self, *args, **kw):
        return self.base.generate(*args, **kw)

    def get_base_model(self):
        return self.base

    def disable_adapter(self):
        return contextlib.nullcontext()

    def eval(self):
        self.base.eval(); return self

    def train(self, mode=True):
        self.base.train(mode); return self

    def parameters(self, *args, **kw):
        return self.base.parameters(*args, **kw)

    def named_modules(self, *args, **kw):
        return self.base.named_modules(*args, **kw)

    @property
    def training(self):
        return self.base.training

    @property
    def generation_config(self):
        return self.base.generation_config

    @property
    def config(self):
        return self.base.config


def _truncate_scorer(base, n_keep):
    """Keep decoder layers [0, n_keep) of a CausalLM and replace lm_head by a module that raises. Every scorer read path is
    mxf.inject.read_resid, whose forward hook on layer READ_LAYER raises _Stop BEFORE layer READ_LAYER+1 runs (the HF forward
    enumerates self.layers[:num_hidden_layers], so a shorter ModuleList is simply a shorter loop; layer_types[i] is still indexed
    by the kept i) -> layer-42 states are unchanged to the bit; only logits consumers (the gates) would notice, and they raise.
    Returns (n_layers_before, head_dropped)."""
    import torch
    n_layers = len(base.model.layers)
    if not (0 < n_keep < n_layers):
        return n_layers, False
    base.model.layers = base.model.layers[:n_keep]
    tied = base.lm_head.weight.data_ptr() == base.model.embed_tokens.weight.data_ptr()
    if not tied:
        class _NoHead(torch.nn.Module):
            def forward(self, *args, **kw):
                raise RuntimeError("truncated scorer: lm_head unavailable (only the fluency/distinct gates need logits; use --scorer-layers -1)")
        base.lm_head = _NoHead()
    return n_layers, not tied


def load_scorer(a, actor, device, tag, use_gates):
    """The REWARD model. --policy-base == MODEL (default): the actor itself -- rl.py score() disables the LoRA, i.e. the clean base,
    byte-identical to before. Otherwise the policy's own base is a fine-tuned model, so `actor.disable_adapter()` is NOT the
    original base any more: load a second, frozen bf16 copy of MODEL on this rank's GPU (BaseActor), truncated per --scorer-layers
    (auto = READ_LAYER+1 layers, lm_head dropped, when the gates are off; the gates need logits -> full model)."""
    if policy_base_is_model(a):
        return actor
    import torch
    from transformers import AutoModelForCausalLM
    from mxf.config import MODEL, READ_LAYER
    n_keep = a.scorer_layers
    if n_keep == 0:
        n_keep = -1 if use_gates else READ_LAYER + 1
    assert n_keep == -1 or n_keep > READ_LAYER, f"--scorer-layers {n_keep} would drop the read layer {READ_LAYER}"
    assert n_keep == -1 or not use_gates, "the fluency/distinct gates need the scorer's logits: use --scorer-layers -1"
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    n_layers, head_dropped = _truncate_scorer(base, n_keep)
    gc.collect(); torch.cuda.empty_cache()
    _log(tag, f"scorer = frozen ORIGINAL base {MODEL} ({len(base.model.layers)}/{n_layers} layers{', lm_head dropped' if head_dropped else ''}) "
              f"loaded in {time.time() - t0:.0f}s | resident now {torch.cuda.memory_allocated() / 2**30:.1f} GB | policy base = {policy_base_of(a)}")
    return BaseActor(base)


@contextlib.contextmanager
def _ref_policy(actor):
    """The KL reference policy for one pass: the frozen 'ref' adapter when one was loaded (--ref-adapter / --init-adapter; the
    set_adapter round trip exactly as before), else the policy base with the LoRA disabled (fresh LoRA on --policy-base: the
    reference IS the base the RL started from)."""
    if "ref" in (getattr(actor, "peft_config", None) or {}):
        actor.set_adapter("ref")
        try:
            yield
        finally:
            actor.set_adapter("default")
    else:
        with actor.disable_adapter():
            yield


def write_run_meta(a, path, micro_batch=None, step=None):
    """run_meta.json next to the checkpoints (the run dir and every step_*/final): how to REBUILD the policy that produced them
    (the adapter ON WHICH base) and what scored them. eval/eval_ckpt_daemon.py reads `policy_base` from <ckpt_dir>/run_meta.json
    to serve policy_base + adapter in vLLM while scoring with MODEL."""
    from mxf.config import INJECT_LAYER, MODEL, READ_LAYER, TrainConfig
    tr = TrainConfig()
    ref_src = (a.ref_adapter or a.init_adapter) if a.kl_coef > 0 else None
    full_param = bool(getattr(a, "full_param", False))
    meta = {"format": "rl_disagg_fullparam_v1" if full_param else "rl_disagg_lora_v1", "full_param": full_param,
            "policy_base": policy_base_of(a), "policy_base_is_model": policy_base_is_model(a),
            "scorer_base": MODEL, "tokenizer": MODEL, "init_adapter": a.init_adapter, "ref_adapter": ref_src,
            "kl_ref": "none" if a.kl_coef <= 0 else ("frozen_init_copy" if full_param else ("adapter" if ref_src else "policy_base_lora_off")),
            "lora": (None if full_param else "from init_adapter" if a.init_adapter else
                     {"r": tr.lora_r, "alpha": tr.lora_alpha, "rslora": True, "target_modules": "all-linear"}),
            "publish_mode": getattr(a, "publish_mode", None) if full_param else "lora_adapter",
            "checkpoint_layout": "full_model_bf16_base_layout (sft/fullft.py; SAVE_DONE)" if full_param else "peft_adapter",
            "inject_layer": INJECT_LAYER, "read_layer": READ_LAYER, "run_name": a.run_name, "save_dir": a.save_dir, "seed": a.seed,
            "micro_batch": micro_batch, "step": step, "argv": sys.argv[1:], "written_at": time.time()}
    os.makedirs(path, exist_ok=True)
    tmp = f"{path}/run_meta.json.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, f"{path}/run_meta.json")
    return meta


# ----------------------------------------------------------------------------------------------
# ScaleRL pieces: pure functions / plain state, unit-tested on CPU in rl/test_rl_disagg_scalerl.py
# ----------------------------------------------------------------------------------------------
def _sample_block_idx(rng, lo, hi, size, dropped=None):
    """`size` distinct bank rows in [lo, hi), sorted. Without a drop set this is EXACTLY the original rollout draw
    (same rng stream); with one (No-Positive-Resampling) the dropped rows are excluded by rejection -- the surviving
    prefix of a uniform without-replacement draw is a uniform without-replacement draw from the allowed rows."""
    import numpy as np
    if not dropped:
        return lo + np.sort(rng.choice(hi - lo, size=size, replace=False))
    n_drop = sum(1 for i in dropped if lo <= i < hi)
    assert hi - lo - n_drop >= size, f"only {hi - lo - n_drop} directions left after NPR dropped {n_drop}"
    k = size
    while True:
        k = min(hi - lo, 2 * k)
        cand = lo + rng.choice(hi - lo, size=k, replace=False)
        cand = cand[np.fromiter((int(c) not in dropped for c in cand), dtype=bool, count=len(cand))]
        if len(cand) >= size:
            return np.sort(cand[:size])


class NPRTracker:
    """No-Positive-Resampling (ScaleRL: "maintaining a history of pass rates and permanently removing any prompt with pass
    rate >= 0.9 from subsequent epochs"), ADAPTED to a continuous reward: a rollout is a 'positive' when its raw cosine
    >= pass_cos; a direction's pass rate is positives / rollouts over ALL its visits so far (the "history"); once that rate
    >= threshold the direction is dropped from every future block (G=8, 0.9: 8/8 on one visit, 15/16 over two, ...).
    Owned by trainer rank 0 (which sees every rank's rewards); publish() writes the drop list _NPRDropList reads."""

    def __init__(self, threshold, pass_cos):
        self.threshold, self.pass_cos = float(threshold), float(pass_cos)
        self.hist = {}            # dir_idx -> [positives, rollouts]
        self.dropped = set()
        self._dirty = False

    def update(self, dir_idx, raw_cos, group_size):
        """dir_idx [B] bank rows (-1 = random direction, ignored); raw_cos [B*G] group-major RAW cosines. -> step stats."""
        import numpy as np
        idx = np.asarray(dir_idx).reshape(-1)
        pos = np.asarray(raw_cos, dtype=np.float64).reshape(len(idx), int(group_size)) >= self.pass_cos
        n_new, n_flag = 0, 0
        for i, row in zip(idx.tolist(), pos):
            if i < 0:
                continue
            h = self.hist.setdefault(i, [0, 0])
            h[0] += int(row.sum()); h[1] += int(row.size)
            if h[0] / h[1] >= self.threshold:
                n_flag += 1
                if i not in self.dropped:
                    self.dropped.add(i); n_new += 1; self._dirty = True
        return {"scalerl/npr_pass_frac": float(pos.mean()) if pos.size else 0.0,
                "scalerl/npr_batch_flagged_frac": n_flag / max(len(idx), 1),
                "scalerl/npr_new_dropped": float(n_new), "scalerl/npr_dropped_total": float(len(self.dropped)),
                "scalerl/npr_directions_seen": float(len(self.hist))}

    def publish(self, path):
        """Atomic json list of the dropped rows, rewritten only when the set changed. Returns True when written."""
        if not self._dirty:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_write_text(path, json.dumps(sorted(self.dropped)))
        self._dirty = False
        return True


class _NPRDropList:
    """Rollout-side reader of NPRTracker.publish(): re-reads the file only when its mtime changes."""

    def __init__(self, path):
        self.path, self.mtime, self.dropped = path, None, set()

    def refresh(self):
        try:
            mt = os.path.getmtime(self.path)
        except FileNotFoundError:
            return self.dropped
        if mt != self.mtime:
            try:
                with open(self.path) as f:
                    self.dropped = set(json.load(f))
                self.mtime = mt
            except (ValueError, OSError):     # mid-replace / half-written: keep the previous list, retry next block
                pass
        return self.dropped


def compute_advantages_disagg(r, n_groups, group_size, mode, zero_var_eps=1e-6, zero_var_filter=False):
    """rl.py compute_advantages (none = Dr. GRPO centering; group = / per-group std; batch = ScaleRL: zero-variance groups
    zeroed, / ONE std of all surviving advantages of the GLOBAL batch, all_reduce'd over DDP) with the zero-variance
    threshold exposed, plus the ScaleRL effective-batch mask: keep [n_groups*group_size] bool, None when the filter is
    off (-> the loss weights are the original ones bit for bit). Zero-variance = reward std <= zero_var_eps."""
    import torch
    import torch.distributed as dist
    rg = r.view(n_groups, group_size)
    adv = rg - rg.mean(1, keepdim=True)
    nz_g = rg.std(1) > zero_var_eps
    if mode == "group":
        adv = adv / (rg.std(1, keepdim=True) + 1e-6)
    elif mode == "batch":
        nz = nz_g[:, None].expand(-1, group_size)
        adv = adv * nz
        stats = torch.tensor([adv[nz].double().pow(2).sum(), adv[nz].double().sum(), nz.sum()], dtype=torch.float64)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(stats)                                        # CPU tensor -> gloo
        n = stats[2].item()
        std = math.sqrt(max(stats[0].item() / n - (stats[1].item() / n) ** 2, 0.0)) if n > 1 else 1.0
        adv = adv / (std + 1e-6)
    elif mode != "none":
        raise ValueError(mode)
    keep = nz_g.repeat_interleave(group_size) if zero_var_filter else None
    return adv.flatten().detach(), keep


def loss_weights(gen_mask, loss_agg, group_size=1, keep=None):
    """Per-token loss weights w_all [n,T] (sum 1 over the local batch) + the weight _sync_grads uses so the DDP
    all-reduce equals the single-GPU gradient over the union batch:
      token  : every completion token weighs 1/total_tok                              (sync_w = total_tok)  rl.py / DAPO
      seq    : every rollout weighs 1/n, its tokens 1/|y_i| within                    (sync_w = n)          GRPO sample-mean
      prompt : every GROUP (direction) weighs 1/n_groups, its tokens 1/sum_g |y_g| within (sync_w = n_groups) ScaleRL
    keep [n] bool (zero-variance filter): dropped rollouts weigh 0 and leave every denominator (effective batch).
    keep=None reproduces the original rl_disagg expressions bit for bit."""
    import torch
    n = gen_mask.shape[0]
    gm = gen_mask.float()
    if keep is None:
        if loss_agg == "seq":
            return gm / gen_mask.sum(1, keepdim=True).clamp(min=1).float() / n, float(n)
        if loss_agg == "token":
            total_tok = max(int(gen_mask.sum()), 1)
            return gm / total_tok, float(total_tok)
        keep = torch.ones(n, dtype=torch.bool)
    gm = gm * keep.float()[:, None]
    if loss_agg == "token":
        tot = max(int(gm.sum()), 1)
        return gm / tot, float(tot)
    if loss_agg == "seq":
        n_eff = max(int(keep.sum()), 1)
        return gm / gm.sum(1, keepdim=True).clamp(min=1) / n_eff, float(n_eff)
    if loss_agg == "prompt":
        G = int(group_size)
        assert n % G == 0, f"{n} rollouts is not a multiple of the group size {G}"
        m3 = gm.view(n // G, G, -1)
        tok_g = m3.sum((1, 2))                                            # completion tokens per group (0 when dropped)
        n_eff = max(int((tok_g > 0).sum()), 1)
        return (m3 / tok_g.clamp(min=1)[:, None, None] / n_eff).view(n, -1), float(n_eff)
    raise ValueError(loss_agg)


def pg_token_loss(new_lp, old_lp, A, loss, clip_eps, tis_cap, cispo_eps_max):
    """Per-token policy-gradient loss BEFORE the aggregation weights, for A broadcast over tokens ([n,1]):
      ppo   : rl.py's clipped surrogate on ratio = min(exp(new-old), tis_cap): -min(ratio*A, clip(ratio, 1-+eps)*A)
      cispo : ScaleRL / MiniMax-M1 truncated-IS REINFORCE: -sg(min(exp(new-old), eps_max)) * A * new_lp -- no clip of the
              objective, every token keeps a gradient; the truncation IS the off-policy correction (lower bound 0).
    Returns (loss_tok [n,T], ratio [n,T] = the IS weight the gradient sees (detached for cispo), rho [n,T] raw ratio, no grad)."""
    import torch
    if loss == "cispo":
        rho = torch.exp(new_lp.detach() - old_lp)
        is_w = rho.clamp(max=cispo_eps_max)
        return -(is_w * A * new_lp), is_w, rho
    ratio = torch.exp(new_lp - old_lp).clamp(max=tis_cap)
    with torch.no_grad():
        rho = torch.exp(new_lp - old_lp)
    return -torch.minimum(ratio * A, ratio.clamp(1 - clip_eps, 1 + clip_eps) * A), ratio, rho


def install_fp32_head(actor):
    """ScaleRL / MiniMax-M1 "FP32 at the LM head", trainer side: a forward hook replaces the lm_head output with
    F.linear(x.float(), W_fp32) so the logits the loss sees are not bf16-rounded (their log_softmax already was fp32, see
    _chunked_logp). W_fp32 is ONE persistent detached copy (2x the head's bf16 bytes; ~5 GB for a 248k x 5120 head) so no
    per-call cast is kept alive for backward. lm_head must be a frozen nn.Linear (PEFT 'all-linear' never wraps it).
    Returns the hook handle."""
    import torch
    import torch.nn.functional as F
    heads = [(n, m) for n, m in actor.named_modules() if n.endswith("lm_head")]
    assert len(heads) == 1, f"expected exactly one lm_head module, found {[n for n, _ in heads]}"
    name, head = heads[0]
    assert isinstance(head, torch.nn.Linear) and not any(p.requires_grad for p in head.parameters()), \
        f"{name} must be a frozen nn.Linear (got {type(head).__name__})"
    w32 = head.weight.detach().float()
    b32 = None if head.bias is None else head.bias.detach().float()

    def _fp32_out(mod, inp, out):
        with torch.autocast(inp[0].device.type, enabled=False):   # composable with --autocast-bf16: the head stays fp32 (no-op otherwise)
            return F.linear(inp[0].float(), w32, b32)
    return head.register_forward_hook(_fp32_out)


@contextlib.contextmanager
def _policy_precision(actor, enabled):
    """--autocast-bf16 region for the POLICY forward. Off: a pure no-op (default path byte-identical). On: (1) PEFT's LoRA
    input-dtype casting is disabled (peft.helpers.disable_input_dtype_casting, else the same `cast_input_dtype_enabled`
    toggle by hand) -- otherwise every LoRA layer still materialises an fp32 copy of its input; (2) torch.autocast(bf16) on the
    actor's device, so F.linear(x_bf16, W_lora_fp32) runs as a bf16 matmul and autograd saves bf16 activations. The LoRA
    master weights stay fp32 (grads arrive in fp32 through autocast's cast nodes), AdamW is untouched, rsLoRA `scaling` is a
    Python float applied after the matmul, lora_dropout is nn.Identity at p=0. The bf16 base layers are unaffected (their
    inputs are bf16 already; RMSNorm's explicit fp32 upcasts are explicit casts, which autocast does not override)."""
    if not enabled:
        yield
        return
    import torch
    dev = next(actor.parameters()).device.type
    try:
        from peft.helpers import disable_input_dtype_casting
        cm = disable_input_dtype_casting(actor)
    except ImportError:
        cm = _disable_input_dtype_casting_manual(actor)
    with cm, torch.autocast(dev, dtype=torch.bfloat16):
        yield


@contextlib.contextmanager
def _disable_input_dtype_casting_manual(model):
    """Fallback == peft.helpers.disable_input_dtype_casting: flip `cast_input_dtype_enabled` off on every tuner layer, restore after."""
    saved = {}
    for name, m in model.named_modules():
        if hasattr(m, "cast_input_dtype_enabled"):
            saved[name] = m.cast_input_dtype_enabled
            m.cast_input_dtype_enabled = False
    try:
        yield
    finally:
        for name, m in model.named_modules():
            if name in saved:
                m.cast_input_dtype_enabled = saved[name]


def _hook_outside_autocast(hook, enabled):
    """Wrap a forward hook so it runs with autocast DISABLED (the inject hook's ||h|| and add stay bf16-exact as in rl.py)."""
    if not enabled:
        return hook
    import torch

    def wrapped(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        with torch.autocast(h.device.type, enabled=False):
            return hook(mod, inp, out)
    return wrapped


# ----------------------------------------------------------------------------------------------
# --prefix-cache: the shared prompt prefix once per pass (sft/prefix_cache.py machinery, RL-update flavour).
# Every rollout sequence = the same ~102-token prompt (chat template + marker) + its response; everything strictly before
# the marker is identical across rollouts and, by causality, untouched by the injection -> its forward (and the backward
# through it) is shared. The suffix forward gets the prefix's cache expanded to the micro-batch; the injection hook then
# fires at SUFFIX index 0 (the marker). Exact incl. gradients up to bf16 kernel noise (measured in rl/test_rl_disagg_prefix.py).
# ----------------------------------------------------------------------------------------------
def _prefix_cache_module():
    try:
        from sft import prefix_cache as pc
    except ImportError:                 # modal_rl_disagg.py mounts sft/prefix_cache.py next to mxf/ (PYTHONPATH /pmx/helpers)
        import prefix_cache as pc
    return pc


class _PrefixGradAccumulator:
    """Share one differentiable prefix forward AND backward across every micro-batch of an optimizer step (mirror of
    sft/prefix_cache.PrefixGradientAccumulator). ``cache`` holds DETACHED leaves of the prefix cache tensors: the suffix
    backwards stop there; ``accumulate`` moves the leaves' grads into fp32 sums; ``backward`` pushes the summed cache
    gradients through the original prefix graph ONCE (chain rule -- not a frozen prefix). Summation order differs from the
    naive full-sequence path, so bf16 gradients agree to kernel noise, not bitwise. Build a new one after every update."""

    def __init__(self, cache, extra_outputs=None):
        """extra_outputs (--full-param): prefix-graph tensors that must receive a ZERO gradient in the final backward -- the
        prefix logits: the loss depends on the prefix only through the caches, so the LAST layer's / the root's output never
        gets a gradient and FSDP2's pre-backward unshard hook (hung on module outputs) would not fire for them
        (sft/prefix_cache.PrefixGradientAccumulator extra_outputs; no-op for the LoRA path)."""
        import copy
        import torch
        self._pairs, self._sums, self._finished = [], [], False
        self._extra = [t for t in (extra_outputs or []) if t is not None and t.requires_grad]
        seen = {}

        def detach(v):
            if isinstance(v, torch.Tensor):
                if id(v) not in seen:
                    leaf = v.detach().requires_grad_(v.requires_grad)
                    seen[id(v)] = leaf
                    if v.requires_grad:
                        self._pairs.append((v, leaf)); self._sums.append(None)
                return seen[id(v)]
            if isinstance(v, dict):
                return {k: detach(x) for k, x in v.items()}
            if isinstance(v, list):
                return [detach(x) for x in v]
            if isinstance(v, tuple):
                return tuple(detach(x) for x in v)
            return v
        self.cache = copy.copy(cache)
        self.cache.layers = []
        for layer in cache.layers:
            new = copy.copy(layer)
            for name, v in vars(layer).items():
                setattr(new, name, detach(v))
            self.cache.layers.append(new)

    def accumulate(self):
        """After each suffix backward (no GPU sync)."""
        import torch
        if self._finished:
            raise RuntimeError("prefix gradients already consumed; build a new accumulator")
        for i, (_, leaf) in enumerate(self._pairs):
            if leaf.grad is not None:
                if self._sums[i] is None:   # bf16/fp16 grads accumulate in fp32 (fp64 stays fp64 for the CPU equivalence test)
                    self._sums[i] = leaf.grad.detach().to(torch.float64 if leaf.dtype == torch.float64 else torch.float32)
                else:
                    self._sums[i].add_(leaf.grad.detach())
                leaf.grad = None

    def backward(self):
        """Once per step: the accumulated cache gradients through the prefix graph."""
        import torch
        self.accumulate()
        outs, grads = [], []
        for (orig, _), g in zip(self._pairs, self._sums):
            if g is not None:
                outs.append(orig); grads.append(g.to(orig.dtype))
        for t in self._extra:                       # zero-weight grad path (FSDP2 unshard hooks on the last layer's / root's output)
            outs.append(t); grads.append(torch.zeros_like(t))
        if outs:
            torch.autograd.backward(outs, grads)
        self._finished = True
        self.cache = None
        self._pairs.clear(); self._sums.clear(); self._extra = []


class PrefixRunner:
    """--prefix-cache for one actor: ``run_prefix`` = the batch-1 forward over prompt_ids[:marker] (use_cache=True; grad iff
    enabled by the caller) -> cache; ``suffix_logits`` = the micro-batch forward of [marker]+response (+ right pad) on a
    batch-expanded COPY of that cache -> logits [B, S, V] for all S suffix positions (position s predicts response token s;
    the last one is discarded by the caller exactly like the full-sequence path's logits_to_keep=Tc+1 [:, :-1])."""

    def __init__(self, actor, prompt_ids, marker, device):
        import torch
        pc = _prefix_cache_module()
        pc.check_transformers()          # stock transformers: in-place cache writes break autograd, no GDN batch expansion
        self._expand = pc.expand_cache_copy
        assert 0 < marker == len(prompt_ids) - 1, "the marker must be the LAST prompt token (suffix prompt == [marker])"
        self.actor, self.device, self.P = actor, device, int(marker)
        self._prefix = torch.tensor(list(prompt_ids[:marker]), dtype=torch.long, device=device)[None]

    def run_prefix(self, autocast_cm=None, return_logits=False):
        with (autocast_cm if autocast_cm is not None else contextlib.nullcontext()):
            out = self.actor(input_ids=self._prefix, use_cache=True, logits_to_keep=1)
        assert out.past_key_values is not None, "prefix forward returned no cache"
        if return_logits:   # --full-param: the [1, 1, V] prefix logits = the zero-grad path FSDP2 needs (see _PrefixGradAccumulator)
            return out.past_key_values, out.logits
        return out.past_key_values

    def suffix_logits(self, cache, ids_suf, attn_suf):
        """ids_suf / attn_suf: [B, S] on device (S = 1 + Tc: marker, response, right pad). Caller holds the autocast +
        injection-hook contexts. The expanded cache copy dies here (its post-suffix states are useless for training)."""
        import torch
        B, S = ids_suf.shape
        cache_b = self._expand(cache, B)
        full_mask = torch.cat([torch.ones((B, self.P), dtype=attn_suf.dtype, device=self.device), attn_suf], 1)
        pos = torch.arange(self.P, self.P + S, device=self.device)[None].expand(B, -1)
        out = self.actor(input_ids=ids_suf, attention_mask=full_mask, position_ids=pos, past_key_values=cache_b, use_cache=True)
        out.past_key_values = None
        del cache_b
        return out.logits


def score_bucketed(R, texts, dirs_rep, actor, tok, device, a, lens, with_fluency=False):
    """--score-length-bucket: rl.py score() cuts --score-batch batches in arrival order and pads each to its
    longest text; calling it on the rollouts sorted by response length makes every batch nearly pad-free. Pure reorder --
    every row is scored standalone (sink + response on the clean base) and SCORE_STATS only holds order-invariant
    aggregates -- so the rewards equal the unsorted call up to bf16 kernel noise."""
    import torch
    order = torch.argsort(torch.as_tensor(lens, dtype=torch.long), stable=True)
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    out = R.score([texts[i] for i in order.tolist()], dirs_rep[order.to(dirs_rep.device)], actor, tok, device, a,
                  with_fluency=with_fluency)
    return tuple(t[inv] for t in out) if with_fluency else out[inv]


# ----------------------------------------------------------------------------------------------
# inline-eval plumbing (pure functions; unit-tested on CPU in train/test_rl_disagg_queue.py)
#   eval_req/req_<k>.pt   trainer rank 0 -> everything the rollout side needs to generate ckpt k's eval texts
#   eval_gen/<k>_r<r>.pt  rollout rank r -> its row shard (rows i with i % X == r) of every eval set
# ----------------------------------------------------------------------------------------------
def _eval_req_path(work, k):
    return f"{work}/eval_req/req_{k:07d}.pt"


def _eval_shard_path(work, k, rank):
    return f"{work}/eval_gen/{k:07d}_r{rank}.pt"


def _eval_requests(work):
    """ckpt steps with a pending request, oldest first."""
    ks = []
    for f in glob.glob(f"{work}/eval_req/req_*.pt"):
        try:
            ks.append(int(os.path.basename(f)[4:-3]))
        except ValueError:
            pass
    return sorted(ks)


def _eval_shards_ready(work, k, n_rollout):
    return all(os.path.exists(_eval_shard_path(work, k, r)) for r in range(n_rollout))


def _eval_plan(req, rank, n_rollout, chunk_seqs):
    """Split ckpt k's eval generation into this rank's chunks (rows i % n_rollout == rank), each <= chunk_seqs
    sequences. Returns a list of dicts {set, kind, rows, n, temp, min_new, max_new, seeds} in a fixed order.
    Row -> seed: held-out families use eval_universal GEN_SEED*1000+i (== rl.py inline_eval), the extra-eval
    testbed uses snippet_locality GEN_SEED*1000+i mod 2^31-1 (== inline_extra_evals.run_extra_evals_gpu)."""
    chunks = []
    for st in req["sets"]:
        rows = [i for i in range(st["n_rows"]) if i % n_rollout == rank]
        per = max(1, chunk_seqs // max(int(st["n"]), 1))
        for c0 in range(0, len(rows), per):
            rr = rows[c0 : c0 + per]
            chunks.append({"set": st["name"], "kind": st["kind"], "rows": rr, "n": int(st["n"]), "temp": float(st["temp"]),
                           "min_new": int(st["min_new"]), "max_new": int(st["max_new"]),
                           "seeds": [int(st["seed_base"] + i) % 2147483647 for i in rr]})
    return chunks


def _eval_merge_shards(shards):
    """[{set: {row: [texts]}}, ...] -> {set: {row: [texts]}} (rows from all rollout ranks)."""
    out = {}
    for sh in shards:
        for name, rows in sh["texts"].items():
            out.setdefault(name, {}).update({int(i): v for i, v in rows.items()})
    return out


def _eval_sets_from_assets(EV, EX, a):
    """The eval 'sets' list for a request: one entry per held-out cosine family + the SAE family (+ the
    extra-eval testbed features). dirs are unit fp32 [n_rows, d]."""
    import torch.nn.functional as F
    EU, es = EV["EU"], EV["es"]
    sets = []
    for fam in EV["fams"]:
        du = F.normalize(es[f"{fam}_dirs"].float(), dim=-1)
        sets.append({"name": fam, "kind": "cos", "n_rows": int(du.shape[0]), "dirs": du, "n": a.eval_bo, "temp": a.eval_temp,
                     "min_new": a.eval_min_new, "max_new": a.eval_max_new, "seed_base": EU.GEN_SEED * 1000})
    du = F.normalize(es["sae_dirs"].float(), dim=-1)
    sets.append({"name": "sae", "kind": "sae", "n_rows": int(du.shape[0]), "dirs": du, "n": a.eval_bo, "temp": a.eval_temp,
                 "min_new": a.eval_min_new, "max_new": a.eval_max_new, "seed_base": EU.GEN_SEED * 1000, "feats": list(EV["feats"])})
    if EX is not None:
        import snippet_locality as SL
        cfg = EX["tb_config"]
        max_new = min(int(cfg.get("max_new", 64)), int(a.max_new_tokens))
        sets.append({"name": "extra", "kind": "extra", "n_rows": len(EX["feats"]), "dirs": F.normalize(EX["dirs"].float(), dim=-1),
                     "n": int(EX["n_rollouts"]), "temp": float(cfg.get("temp", 1.0)), "min_new": min(int(cfg.get("min_new", 16)), max_new),
                     "max_new": max_new, "seed_base": SL.GEN_SEED * 1000, "feats": list(EX["feats"])})
    return sets


class _GenCfgStub:
    """rl.py's _eos_ids(tok, actor) only reads actor.generation_config -- rollout ranks have no actor."""
    def __init__(self, gen_cfg):
        self.generation_config = gen_cfg



# ----------------------------------------------------------------------------------------------
# --full-param (rl/rl_fullparam.py): trainer-side glue. Everything here is only reached when a.full_param is set.
# ----------------------------------------------------------------------------------------------
class FullParamCtx:
    """What update_disagg / find_micro_batch / the publish + save paths need to know about the FSDP2 policy."""

    def __init__(self, FP, FT, world, rank):
        self.FP, self.FT, self.world, self.rank = FP, FT, world, rank
        self.inject_mode = "add_clone"        # FSDP2 hangs backward hooks on the layer output: never modify it in place
        self.ref_model = self.ref_submodule = self.ref_pfx = None
        self.manifest = None
        self.comm = None                      # rank 0: TrainerWeightComm (publish-mode nccl), created at the first publish
        self.nontext_shard = None
        self.last_pub = {}
        self.pub_hist = []
        self.mem = {}
        self.ckpt = None                      # --suffix-ckpt: rl_fullparam.suffix_checkpointer(actor)
        self.head = None                      # --chunked-head: rl_fullparam.ChunkedHead(actor)

    @contextlib.contextmanager
    def suffix_ctx(self):
        """Around every suffix forward/backward of the policy: checkpointing on, chunked head capturing."""
        if self.ckpt is not None:
            self.ckpt.enabled = True
        if self.head is not None:
            self.head.active = True
        try:
            yield
        finally:
            if self.ckpt is not None:
                self.ckpt.enabled = False
            if self.head is not None:
                self.head.active, self.head.hidden = False, None

    def logp_from(self, logits, tgt, vocab_chunk, need_entropy_grad, fp32_head):
        """new_lp / entropy for a micro-batch: from the stashed head input (--chunked-head) or from the logits (_chunked_logp)."""
        if self.head is not None:
            return self.head.logp(self.head.hidden[:, :-1], tgt, vocab_chunk, need_entropy_grad, fp32=fp32_head)
        return _chunked_logp(logits, tgt, vocab_chunk, need_entropy_grad)


def load_scorer_fullparam(a, device, world, tag, use_gates, need_logits=False):
    """--full-param REWARD model: a frozen bf16 copy of the ORIGINAL base MODEL on every trainer rank, truncated to layers
    [0, READ_LAYER] with a TinyHead when nothing needs logits (gates off, no KL-on-MODEL), FSDP2-sharded over the trainer
    ranks (--scorer-shard, default: ~7 GB/rank instead of ~35). Read through rl_fullparam.read_resid_noraise (installed as
    rl_hf.read_resid): the forward runs to its end so FSDP2's hooks all fire."""
    import torch
    import rl_fullparam as FP
    from transformers import AutoModelForCausalLM
    from mxf.config import MODEL, READ_LAYER
    n_keep = a.scorer_layers
    if n_keep == 0:
        n_keep = -1 if (use_gates or need_logits) else READ_LAYER + 1
    assert n_keep == -1 or n_keep > READ_LAYER, f"--scorer-layers {n_keep} would drop the read layer {READ_LAYER}"
    assert n_keep == -1 or not (use_gates or need_logits), "the gates / KL-on-MODEL need the scorer's logits: use --scorer-layers -1"
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    n_layers, head_dropped = FP.truncate_scorer_fsdp_safe(base, n_keep)
    if a.scorer_shard:
        FP.shard_frozen_bf16(base, world)
    gc.collect(); torch.cuda.empty_cache()
    _log(tag, f"scorer = frozen ORIGINAL base {MODEL} ({len(base.model.layers)}/{n_layers} layers{', lm_head -> TinyHead' if head_dropped else ''}, "
              f"{'FSDP2-sharded over ' + str(world) + ' ranks' if a.scorer_shard else 'UNsharded'}) loaded in {time.time() - t0:.0f}s | "
              f"resident now {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    return BaseActor(base)


def _publish_fullparam(fp, actor, submodule, prompt, marker, device, work, step, tag, a, is_main, world, initial=False):
    """COLLECTIVE over the trainer ranks: ||h_marker|| of the policy (FSDP forward) + the bf16 weights to the engines
    (rl_fullparam.publish_nccl / write_shard_file), then rank 0 writes lora/step_<k>/meta.json and flips `latest`.
    initial=True (step_offset): the engines loaded exactly these weights from disk -- meta only."""
    import torch
    import torch.distributed as dist
    import rl_hf as R
    FP = fp.FP
    t0 = time.time()
    with torch.no_grad():
        hnorm = R._marker_norm(actor, submodule, prompt, marker, device, adapter=True)
    FP.reshard_root(actor)   # the no-grad forward left embed/norm/lm_head gathered: back to the fp32 DTensor shards before iterating params
    t_norm = time.time() - t0
    d = f"{work}/lora/step_{step}"
    if is_main:
        os.makedirs(d, exist_ok=True)
    tm, checks, mode = {}, {}, "initial"
    if not initial and a.publish_mode == "nccl":
        mode = "nccl"

        def comm_getter():   # rank 0, after every engine wrote its ready file: the workers join the group in wu_init right after
            if fp.comm is None:
                fp.comm = FP.TrainerWeightComm("127.0.0.1", a.wu_port, a.n_rollout, device)
                _log(tag, f"weight-update NCCL group up ({fp.comm.world} members: trainer rank 0 + {a.n_rollout} vLLM workers) in {fp.comm.init_s:.1f}s")
            return fp.comm
        tm, checks = FP.publish_nccl(actor, fp.manifest, comm_getter, is_main, work, step, a.n_rollout, ready_timeout_s=1800,
                                     log=lambda m: _log(tag, m), stop=lambda: _stop_requested(work))
        if tm.get("aborted"):
            return hnorm, time.time() - t0
    elif not initial:
        mode = "fs"
        tm_r = FP.write_shard_file(actor, fp.manifest, f"{d}/shard_{fp.rank}.safetensors", fp.rank)
        loc = FP.local_check_abs_sums(actor, fp.manifest)
        checks = FP.all_reduce_dict_sum(loc, device)
        if world > 1:
            dist.barrier()
        tm = {"t_transfer": time.time() - t0 - t_norm, "gb": tm_r["gb"] * world, "to_cpu_s": tm_r["to_cpu_s"], "write_s": tm_r["write_s"]}
        if is_main:
            FP.prune_fs_steps(work, a.fs_keep_steps)
    if is_main:
        meta = {"step": step, "hnorm": hnorm, "mode": mode, "checks": checks, "t": time.time(), "timing": tm, "n_tensors": len(fp.manifest["names"])}
        _atomic_write_text(f"{d}/meta.json", json.dumps(meta))
        _atomic_write_text(f"{work}/lora/latest", str(step))
        for old in sorted(glob.glob(f"{work}/lora/step_*"), key=lambda q: int(q.rsplit("_", 1)[-1]))[:-a.keep_loras]:
            shutil.rmtree(old, ignore_errors=True)
    tot = time.time() - t0
    fp.last_pub = {**{k: v for k, v in tm.items() if isinstance(v, (int, float))}, "hnorm_s": t_norm, "total_s": tot}
    fp.pub_hist.append({"step": step, "mode": mode, **fp.last_pub})
    if is_main and (step <= a.step_offset + 2 or step % 10 == 0):
        _log(tag, f"publish step {step} [{mode}]: hnorm {t_norm:.2f}s | " + " ".join(f"{k} {v:.2f}" for k, v in tm.items() if isinstance(v, (int, float))) + f" | total {tot:.2f}s")
    return hnorm, tot


def _save_fullparam_ckpt(fp, actor, path, tok, is_main, world, a, mb, step, opt, tag, final=False):
    """COLLECTIVE: full HF model dir (sft/fullft.py layout, SAVE_DONE last) + run_meta.json (+ optim_dcp/ with --save-optim)."""
    import torch.distributed as dist
    from mxf.config import MODEL
    t0 = time.time()
    fp.FP.save_full_checkpoint(actor, path, tok, MODEL, is_main, world, fp.nontext_shard, log=lambda m: _log(tag, m),
                               extra_meta={"step": step, "optimizer_updates": (step if final else step + 1), "run_name": a.run_name, "lr": a.lr,
                                           "rl_fullparam": True, "policy_base": policy_base_of(a), "publish_mode": a.publish_mode})
    if is_main:
        write_run_meta(a, path, mb, step=step)
    if a.save_optim:
        fp.FP.save_optim_dcp(actor, opt, f"{path}/optim_dcp", log=lambda m: _log(tag, m))
    if world > 1:
        dist.barrier()
    if is_main:
        _log(tag, f"full-model checkpoint {path} written in {time.time() - t0:.0f}s (step {step}{', final' if final else ''})")


class _FullParamSync:
    """Rollout-side --full-param weight refresh (one per rollout rank). nccl: when the trainer flags lora/pending = k, write
    lora/ready_<r>_<k>, join the weight-update group (first time), receive step k into the engine (worker.wu_recv), wait for
    lora/latest == k and verify the checksums. fs: load lora/step_<k>/shard_*.safetensors when `latest` moves. Only ever
    called between generate() calls (a block boundary)."""

    def __init__(self, llm, a, rank, work, tag):
        self.llm, self.a, self.rank, self.work, self.tag = llm, a, rank, work, tag
        self.manifest_path = f"{work}/lora/manifest.json"
        self.inited = False
        self.cur_step = None
        self.hnorm = None
        self.hist = []

    def _meta(self, k, timeout_s=1800):
        p = f"{self.work}/lora/step_{k}/meta.json"
        t0 = time.time()
        while True:
            try:
                with open(p) as f:
                    return json.load(f)
            except (FileNotFoundError, ValueError):
                if time.time() - t0 > timeout_s:
                    raise
                time.sleep(0.02)

    def initial(self):
        k = _read_latest(self.work)
        m = self._meta(k)
        self.cur_step, self.hnorm = k, float(m["hnorm"])
        return k

    def pending(self):
        return self.a.publish_mode == "nccl" and os.path.exists(f"{self.work}/lora/pending")

    def _verify(self, k, meta, res):
        import rl_fullparam as FP
        bad = FP.verify_checks(meta.get("checks", {}), res.get("checks", {}))
        if bad:
            raise RuntimeError(f"weight publish step {k}: engine checksums differ from the trainer's: {bad}")

    def poll(self):
        """-> True when the engine now serves a newer step."""
        a, work = self.a, self.work
        if a.publish_mode == "nccl":
            pend = f"{work}/lora/pending"
            try:
                with open(pend) as f:
                    k = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                return False
            if k == self.cur_step:
                return False
            t0 = time.time()
            open(f"{work}/lora/ready_{self.rank}_{k}", "w").close()
            if not self.inited:
                import rl_fullparam as FP
                FP.wait_for_files([self.manifest_path], 1800)
                info = self.llm.collective_rpc("wu_init", args=("127.0.0.1", a.wu_port, 1 + self.rank, 1 + a.n_rollout, self.manifest_path))[0]
                self.inited = True
                _log(self.tag, f"weight-update group joined as rank {info['rank']}/{info['world']} in {info['init_s']:.1f}s ({info['n_tensors']} tensors, {info['gb']:.1f} GB bf16 per publish)")
            res = self.llm.collective_rpc("wu_recv", args=(k,))[0]
            t_recv = time.time() - t0
            while _read_latest(work) != k:          # rank 0 writes meta + latest right after its last broadcast
                if _stop_requested(work):
                    return False
                time.sleep(0.02)
            meta = self._meta(k)
            self._verify(k, meta, res)
            self.cur_step, self.hnorm = k, float(meta["hnorm"])
            rec = {"step": k, "mode": "nccl", "wall_s": time.time() - t0, "recv_s": t_recv, "engine_load_s": res["load_s"], "gb": res["gb"], "gbps": res["gbps"],
                   "n_vllm_params": res["n_vllm_params_loaded"]}
            self.hist.append(rec)
            if k <= a.step_offset + 2 or k % 10 == 0:
                _log(self.tag, f"weights -> step {k} [nccl]: {res['gb']:.1f} GB in {res['load_s']:.2f}s ({res['gbps']:.0f} GB/s), {res['n_vllm_params_loaded']} vLLM params, "
                               f"checksums OK, engine stalled {rec['wall_s']:.2f}s")
            return True
        # fs
        k = _read_latest(work)
        if k is None or k == self.cur_step:
            return False
        meta = self._meta(k)
        if meta.get("mode") == "initial":
            self.cur_step, self.hnorm = k, float(meta["hnorm"])
            return True
        t0 = time.time()
        res = self.llm.collective_rpc("wu_load_fs", args=(f"{work}/lora/step_{k}", self.manifest_path))[0]
        self._verify(k, meta, res)
        self.cur_step, self.hnorm = k, float(meta["hnorm"])
        rec = {"step": k, "mode": "fs", "wall_s": time.time() - t0, "engine_load_s": res["load_s"], "gb": res["gb"], "gbps": res["gbps"],
               "n_vllm_params": res["n_vllm_params_loaded"]}
        self.hist.append(rec)
        if k <= a.step_offset + 2 or k % 10 == 0:
            _log(self.tag, f"weights -> step {k} [fs]: {res['gb']:.1f} GB in {res['load_s']:.2f}s ({res['gbps']:.0f} GB/s), checksums OK, engine stalled {rec['wall_s']:.2f}s")
        return True

    def dump(self):
        try:
            json.dump(self.hist, open(f"{self.work}/fullparam_sync_r{self.rank}.json", "w"), indent=1)
        except Exception:  # noqa
            pass


# ==============================================================================================
# ROLLOUT rank
# ==============================================================================================
def _multimodal():
    """mxf.config.MULTIMODAL (model profile): True for Qwen3.6-27B (Qwen3_5ForConditionalGeneration), False for plain CausalLMs."""
    from mxf.config import MULTIMODAL
    return bool(MULTIMODAL)


def _build_engine(a, rank, p_len, max_seqs, use_graphs, tag):
    """vLLM engine on this rank's (only visible) GPU. vllm_lens' plugin is loaded first so its
    LLM.generate/steering patches are live; its EngineArgs patch (which FORCES enforce_eager and
    installs the slow stock hook) is replaced by ours: fast_lens_ext + our own eager/graph choice."""
    hidden = {k: os.environ.pop(k) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                                             "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT") if k in os.environ}
    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        import vllm_lens._activations_plugin as P
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        orig = P._original_create_engine_config
        assert orig is not None, "vllm_lens plugin did not register (dist-info missing?)"
        ext_cls = None if a.stock_lens_hook else "fast_lens_ext.FastSteerExtension"

        def _cfg(self, *args, **kw):
            if ext_cls and not self.worker_extension_cls:
                self.worker_extension_cls = ext_cls
            if not self.worker_extension_cls:
                self.worker_extension_cls = "vllm_lens._worker_ext.HiddenStatesExtension"
            return orig(self, *args, **kw)
        EngineArgs.create_engine_config = _cfg
        from mxf.config import MODEL
        max_len = p_len + a.max_new_tokens + 8
        # engine_model / engine_lora (eval_ckpt_daemon --full-model): serve a FULL fine-tuned checkpoint dir instead of
        # base+LoRA; --policy-base: that base with the LoRA slots. Defaults = MODEL with 2 LoRA slots (live policy + eval
        # ckpt), byte-identical to before.
        engine_model = getattr(a, "engine_model", None) or getattr(a, "policy_base", None) or MODEL
        engine_lora = bool(getattr(a, "engine_lora", True))
        kw = dict(model=engine_model, tensor_parallel_size=1, gpu_memory_utilization=a.vllm_gpu_mem, max_model_len=max_len,
                  attention_backend="TRITON_ATTN", enable_prefix_caching=False,
                  **({"language_model_only": True} if _multimodal() else {}),   # Qwen3.6 = multimodal wrapper; Qwen3-8B = plain LM
                  enable_lora=engine_lora, max_num_seqs=int(max_seqs),
                  **({"max_loras": 2, "max_lora_rank": 64} if engine_lora else {}),
                  # never chunk a prompt (the marker must be prefilled in a hooked, eager pass): budget = every seq's full prompt+gen
                  # unless overridden (the eval daemon uses a smaller budget so the profiling run leaves KV memory for concurrency)
                  max_num_batched_tokens=int(getattr(a, "max_num_batched_tokens", 0) or 0) or max(8192, int(max_seqs) * max_len),
                  seed=a.seed * 1000 + 500 + rank, dtype="bfloat16")
        gdn = str(getattr(a, "gdn_prefill_backend", "triton") or "triton").lower()
        if gdn != "auto":   # EngineArgs.gdn_prefill_backend -> additional_config["gdn_prefill_backend"] -> ChunkGatedDeltaRule (sm90: flashinfer JIT unless 'triton')
            kw["gdn_prefill_backend"] = gdn
        if use_graphs:
            kw["enforce_eager"] = False
            kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                        "max_cudagraph_capture_size": int(max_seqs)}
        else:
            kw["enforce_eager"] = True
        t0 = time.time()
        llm = LLM(**kw)
        llm.collective_rpc("install_hooks")
        _log(tag, f"engine up in {time.time() - t0:.0f}s | graphs={use_graphs} max_num_seqs={max_seqs} "
                  f"mem={a.vllm_gpu_mem} ext={'stock' if a.stock_lens_hook else 'fast'} gdn_prefill={gdn} "
                  f"model={engine_model} lora={engine_lora}")
        return llm
    finally:
        os.environ.update(hidden)


def _verify_injection(llm, prompt_ids, marker, hnorm, tag, seed=0):
    """verify_vllm_injection without an HF actor: the injected vector is ABSOLUTE (norm_match=False), so
    the captured marker-row delta must equal hnorm*STEER_COEFF*unit(v): cos>0.99, ratio in [0.95,1.05];
    pre-marker rows untouched. Runs on the BASE weights (no LoRA request), greedy, 1 token."""
    import torch
    import torch.nn.functional as F
    import rl_hf as R
    from vllm import SamplingParams
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    g = torch.Generator().manual_seed(seed)
    v = F.normalize(torch.randn(D_MODEL, generator=g), dim=0)

    def run(steer):
        extra = {"output_residual_stream": [INJECT_LAYER]}
        if steer:
            extra["apply_steering_vectors"] = [R._steer_vec(v, hnorm, marker)]
        out = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                           [SamplingParams(temperature=0.0, max_tokens=1, extra_args=extra)], use_tqdm=False)[0]
        act = getattr(out, "activations", None)
        assert act is not None and "residual_stream" in act, "capture returned nothing -- hooks not live?"
        return act["residual_stream"][0].float()
    h_clean, h_steer = run(False), run(True)
    delta = h_steer[marker] - h_clean[marker]
    cos = F.cosine_similarity(delta, v, dim=0).item()
    ratio = (delta.norm() / (STEER_COEFF * hnorm)).item()
    other = (h_steer[:marker] - h_clean[:marker]).norm(dim=-1).max().item() if marker > 0 else 0.0
    chk = {"cos": cos, "norm_ratio": ratio, "hnorm_published": hnorm, "hnorm_vllm_base": h_clean[marker].norm().item(),
           "max_other_row_delta": other, "ok": cos > 0.99 and 0.95 < ratio < 1.05}
    _log(tag, f"injection check: cos={cos:.4f} ratio={ratio:.3f} ||h||_vllm_base={chk['hnorm_vllm_base']:.1f} "
              f"(published adapter-on {hnorm:.1f}) pre-marker max|d|={other:.2e} -> {'OK' if chk['ok'] else 'FAIL'}")
    return chk


def _generate_block(llm, a, tok, prompt_ids, marker, dirs, hnorm, lora_req, eos_ids, key_prefix):
    """ONE generate() call: one request per direction, n=G, steering keyed by _steering_id (one RPC for
    the whole block). Returns gen_ids (group-major, stop token kept via _trim_at_stop), per-token vLLM
    logprobs (None where the engine dropped the stop token and we re-appended it), appended count, gen_s."""
    import rl_hf as R
    from vllm import SamplingParams
    G = a.group_size
    keys = [f"{key_prefix}_{i}" for i in range(len(dirs))]
    payload = {k: [R._steer_vec(v, hnorm, marker)] for k, v in zip(keys, dirs)}
    if a.stock_lens_hook:   # stock plugin protocol: per-request apply_steering_vectors (it does the RPCs itself)
        params = [SamplingParams(n=G, temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                                 max_tokens=a.max_new_tokens, min_tokens=a.min_new_tokens, stop_token_ids=sorted(eos_ids),
                                 logprobs=0, extra_args={"apply_steering_vectors": payload[k]}) for k in keys]
    else:
        llm.collective_rpc("set_steering_data_many", args=(pickle.dumps(payload),))
        params = [SamplingParams(n=G, temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                                 max_tokens=a.max_new_tokens, min_tokens=a.min_new_tokens, stop_token_ids=sorted(eos_ids),
                                 logprobs=0, extra_args={"_steering_id": k}) for k in keys]
    reqs = [{"prompt_token_ids": list(prompt_ids)} for _ in keys]
    t1 = time.time()
    try:
        outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
    finally:
        if not a.stock_lens_hook:
            llm.collective_rpc("clear_steering_data_many", args=(keys,))
    gen_s = time.time() - t1
    gen_ids, lps, appended = [], [], 0
    for out in outs:
        assert len(out.outputs) == G, f"expected {G} samples, got {len(out.outputs)}"
        for o in out.outputs:
            g = list(o.token_ids)
            lp = [None] * len(g)
            if o.logprobs:
                lp = [(d[t].logprob if (d is not None and t in d) else None) for d, t in zip(o.logprobs, g)]
            if o.finish_reason == "stop" and (not g or g[-1] not in eos_ids):
                g.append(int(o.stop_reason) if isinstance(o.stop_reason, int) else int(tok.eos_token_id))
                lp.append(None); appended += 1
            g2 = R._trim_at_stop(g, eos_ids)
            gen_ids.append(g2); lps.append(lp[: len(g2)])
    return gen_ids, lps, appended, gen_s


def _generate_eval_chunk(llm, a, tok, prompt_ids, marker, ch, dirs, hnorm, lora_req, eos_ids, key_prefix):
    """One generate() for an eval chunk: n samples per row at the set's temperature / token limits with the
    per-row seeds (deterministic like rl.py inline_eval), steering via _steering_id (one RPC per chunk).
    Returns {row: [n texts]} (stop token trimmed, decoded, empty -> ' ' as in inline_extra_evals)."""
    import rl_hf as R
    from vllm import SamplingParams
    keys = [f"{key_prefix}_{i}" for i in ch["rows"]]
    payload = {k: [R._steer_vec(dirs[i], hnorm, marker)] for k, i in zip(keys, ch["rows"])}
    llm.collective_rpc("set_steering_data_many", args=(pickle.dumps(payload),))
    params = [SamplingParams(n=ch["n"], temperature=ch["temp"], top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                             max_tokens=ch["max_new"], min_tokens=ch["min_new"], stop_token_ids=sorted(eos_ids), seed=sd,
                             extra_args={"_steering_id": k}) for k, sd in zip(keys, ch["seeds"])]
    reqs = [{"prompt_token_ids": list(prompt_ids)} for _ in keys]
    try:
        outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
    finally:
        llm.collective_rpc("clear_steering_data_many", args=(keys,))
    res = {}
    for i, out in zip(ch["rows"], outs):
        assert len(out.outputs) == ch["n"], f"expected {ch['n']} samples, got {len(out.outputs)}"
        res[i] = [(tok.decode(R._trim_at_stop(list(o.token_ids), eos_ids), skip_special_tokens=True).strip() or " ")
                  for o in out.outputs]
    return res


class _EvalJob:
    """Rollout-side state of one eval request: the chunks still to generate + the texts generated so far."""

    def __init__(self, work, k, rank, n_rollout, a, tag):
        import torch
        from vllm.lora.request import LoRARequest
        self.k, self.rank, self.tag, self.work = k, rank, tag, work
        self.req = torch.load(_eval_req_path(work, k), weights_only=False)
        self.adapter_step = int(self.req["adapter_step"])
        d = f"{work}/lora/step_{self.adapter_step}"
        self.error = None if os.path.isdir(d) else f"adapter step {self.adapter_step} no longer published"
        self.hnorm = float(json.load(open(f"{d}/meta.json"))["hnorm"]) if self.error is None else None
        self.lora_req = LoRARequest(lora_name=f"step{self.adapter_step}", lora_int_id=self.adapter_step + 1, lora_path=d)
        self.dirs = {st["name"]: st["dirs"] for st in self.req["sets"]}
        self.chunks = _eval_plan(self.req, rank, n_rollout, a.eval_chunk_seqs)
        self.texts = {st["name"]: {} for st in self.req["sets"]}
        self.t_gen, self.t0, self.n_seq = 0.0, time.time(), 0
        _log(tag, f"eval request ckpt {k}: {len(self.chunks)} chunks / {sum(len(c['rows']) * c['n'] for c in self.chunks)} seqs for this rank"
                  + (f" | ERROR {self.error}" if self.error else ""))

    @property
    def age(self):
        return time.time() - float(self.req.get("t", self.t0))

    def done(self):
        return self.error is not None or not self.chunks

    def step(self, llm, a, tok, prompt_ids, marker, eos_ids):
        ch = self.chunks.pop(0)
        t1 = time.time()
        try:
            res = _generate_eval_chunk(llm, a, tok, prompt_ids, marker, ch, self.dirs[ch["set"]], self.hnorm, self.lora_req, eos_ids,
                                       key_prefix=f"ev{self.k}r{self.rank}{ch['set']}")
            self.texts[ch["set"]].update(res)
            self.n_seq += len(ch["rows"]) * ch["n"]
        except Exception as e:  # noqa
            self.error = f"rank{self.rank}: {type(e).__name__}: {str(e)[:300]}"
            self.chunks = []
        self.t_gen += time.time() - t1

    def write(self):
        import torch
        fn = _eval_shard_path(self.work, self.k, self.rank)
        torch.save({"ckpt_step": self.k, "adapter_step": self.adapter_step, "rank": self.rank, "texts": self.texts,
                    "t_gen": self.t_gen, "n_seq": self.n_seq, "error": self.error, "t_done": time.time()}, fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        _log(self.tag, f"eval shard ckpt {self.k}: {self.n_seq} seqs in {self.t_gen:.1f}s gen, {self.age:.0f}s after the request"
                       + (f" | ERROR {self.error}" if self.error else ""))


def run_rollout(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, GenerationConfig
    import rl_hf as R
    from mxf.config import D_MODEL, MODEL
    from mxf.prompts import build_prompt_ids
    from vllm.lora.request import LoRARequest

    rank = int(os.environ["DISAGG_RANK"]); tag = f"R{rank}"
    work = a.work_dir
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    eos_ids = R._eos_ids(tok, _GenCfgStub(GenerationConfig.from_pretrained(MODEL)))
    rng = np.random.default_rng(a.seed * 7919 + 1000 + rank)
    bank = n_vecs = None; eval_rows = 0
    if a.direction_source == "cluster":
        bank, n_vecs, eval_rows = _bank_open(a)
        assert n_vecs - eval_rows >= a.rollout_block_groups
    Bb = a.rollout_block_groups

    # the engine can load while the trainer is still loading the actor; the first block waits for step 0
    if not policy_base_is_model(a):
        _log(tag, f"policy base = {check_policy_base(a)} (full-FT checkpoint served by vLLM; tokenizer/prompt/eos from {MODEL})")
    if a.full_param:   # the engine serves the policy weights themselves (no LoRA slots); they are refreshed IN PLACE every step
        a.engine_lora = False
        _log(tag, f"FULL-PARAMETER policy: engine serves {policy_base_of(a)} without LoRA; weights refreshed via {a.publish_mode} at block boundaries")
    llm = _build_engine(a, rank, p_len, a.max_num_seqs, a.cuda_graphs, tag)
    t_wait = time.time()
    while _read_latest(work) is None:
        if _stop_requested(work):
            return
        time.sleep(1.0)
    _log(tag, f"first adapter published after {time.time() - t_wait:.0f}s of waiting")
    cur_step, lora_req, hnorm = None, None, None
    fsync = _FullParamSync(llm, a, rank, work, tag) if a.full_param else None

    def refresh():
        nonlocal cur_step, lora_req, hnorm
        if fsync is not None:
            sw = fsync.poll() if fsync.cur_step is not None else bool(fsync.initial())
            cur_step, hnorm = fsync.cur_step, fsync.hnorm
            return sw
        k = _read_latest(work)
        if k is None or k == cur_step:
            return False
        d = f"{work}/lora/step_{k}"
        meta = json.load(open(f"{d}/meta.json"))
        hnorm = float(meta["hnorm"])
        lora_req = LoRARequest(lora_name=f"step{k}", lora_int_id=k + 1, lora_path=d)
        cur_step = k
        return True
    refresh()
    chk = _verify_injection(llm, prompt_ids, marker, hnorm, tag, seed=a.seed)
    json.dump(chk, open(f"{work}/verify_r{rank}.json", "w"))
    if not chk["ok"]:
        raise RuntimeError(f"vLLM steering does NOT match the HF inject hook: {chk}")

    blk = 0
    inflight = f"{work}/queue/.inflight_{rank}"
    n_rollout = int(os.environ["DISAGG_WORLD"])
    ev_job, ev_done = None, set()
    npr_drop = _NPRDropList(f"{work}/npr/dropped.json") if a.npr_threshold > 0 else None   # No-Positive-Resampling

    def depth():   # complete blocks + blocks other ranks are generating right now (so N producers cannot all overshoot the cap)
        return len(_queue_files(work)) + len([f for f in glob.glob(f"{work}/queue/.inflight_*") if f != inflight])

    def eval_job():
        """The oldest pending eval request this rank has not finished (None if there is none)."""
        nonlocal ev_job
        if ev_job is None:
            for k in _eval_requests(work):
                if k not in ev_done and not os.path.exists(_eval_shard_path(work, k, rank)):
                    try:
                        ev_job = _EvalJob(work, k, rank, n_rollout, a, tag)
                    except Exception as e:  # noqa — request file half-written / adapter vanished: retry next slot
                        _log(tag, f"eval request ckpt {k} not loadable yet ({type(e).__name__}: {e})")
                    break
        return ev_job
    while not _stop_requested(work):
        # ---- eval generation fills the slack: whenever the rollout queue is full (the trainer is the bottleneck) do ONE
        # chunk of pending eval work instead of sleeping; a request older than --eval-max-delay-s is worked on regardless ----
        job = eval_job()
        if job is not None and (depth() >= a.max_queue_blocks or job.age > a.eval_max_delay_s):
            if not job.done():
                job.step(llm, a, tok, prompt_ids, marker, eos_ids)
            if job.done():
                job.write(); ev_done.add(job.k); ev_job = None
            continue
        while depth() >= a.max_queue_blocks:                           # backpressure: bounded lag, no wasted rollouts
            if _stop_requested(work):
                return
            if fsync is not None and fsync.pending():                  # --full-param nccl: the trainer waits for THIS rank at a block boundary
                refresh()
            if eval_job() is not None:
                break                                                  # -> top of the loop: eval chunk instead of idling
            time.sleep(0.25 + 0.05 * rank)
        if depth() >= a.max_queue_blocks:
            continue
        open(inflight, "w").close()
        t0 = time.time()
        swapped = refresh()
        if a.direction_source == "random":
            idx = np.full(Bb, -1, dtype=np.int64)
            dirs = F.normalize(torch.randn(Bb, D_MODEL, dtype=torch.float32), dim=-1)
        else:
            idx = _sample_block_idx(rng, eval_rows, n_vecs, Bb, npr_drop.refresh() if npr_drop is not None else None)
            dirs = F.normalize(torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
        gen_ids, lps, appended, gen_s = _generate_block(llm, a, tok, prompt_ids, marker, dirs, hnorm, lora_req, eos_ids,
                                                        key_prefix=f"r{rank}b{blk}")
        n_tok = sum(len(g) for g in gen_ids)
        rec = {"block": blk, "rank": rank, "adapter_step": cur_step, "dir_idx": idx, "dirs": dirs, "gen_ids": gen_ids,
               "lps": lps, "appended": appended, "gen_s": gen_s, "n_tok": n_tok, "t_done": time.time(),
               "lora_swapped": swapped}
        fn = f"{work}/queue/blk_{cur_step:07d}_{time.time_ns()}_{rank}.pt"
        torch.save(rec, fn + ".tmp"); os.replace(fn + ".tmp", fn)
        try:
            os.remove(inflight)
        except FileNotFoundError:
            pass
        _log(tag, f"block {blk} | adapter step {cur_step}{' (swapped)' if swapped else ''} | {len(gen_ids)} seqs "
                  f"{n_tok} tok in gen {gen_s:.1f}s ({n_tok / gen_s:.0f} tok/s, {len(gen_ids) / gen_s:.1f} seq/s) "
                  f"| appended_stop {appended} | wall {time.time() - t0:.1f}s | queue {len(_queue_files(work))}")
        blk += 1
    if fsync is not None:
        fsync.dump()
    _log(tag, "STOP seen, exiting")


def run_bench_rollout(a):
    """Rollout throughput table: for each <eager|graphs>:<max_num_seqs> config assigned to this rank
    (configs[rank::n_rollout]) build an engine, verify injection, then time generate() for every size in
    --bench-sizes (fresh directions, the SFT adapter converted to vLLM layout, real sampling params).
    Writes <work>/bench_rollout_r<rank>.json."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, GenerationConfig
    import rl_hf as R
    from mxf.config import MODEL
    from mxf.prompts import build_prompt_ids
    from vllm.lora.request import LoRARequest

    rank = int(os.environ["DISAGG_RANK"]); world = int(os.environ["DISAGG_WORLD"]); tag = f"B{rank}"
    work = a.work_dir
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    eos_ids = R._eos_ids(tok, _GenCfgStub(GenerationConfig.from_pretrained(MODEL)))
    # SFT adapter -> vLLM key layout (no actor needed: pure key rename, cf. rl.py _save_adapter_for_vllm)
    lora_dir = f"{work}/bench_lora_r{rank}"
    os.makedirs(lora_dir, exist_ok=True)
    sd = load_file(f"{a.init_adapter}/adapter_model.safetensors")
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.contiguous()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(f"{a.init_adapter}/adapter_config.json", f"{lora_dir}/adapter_config.json")
    lora_req = LoRARequest(lora_name="bench", lora_int_id=1, lora_path=lora_dir)
    bank, n_vecs, eval_rows = _bank_open(a)
    rng = np.random.default_rng(a.seed + rank)
    configs = [c for c in a.bench_configs.split(",") if c][rank::max(world, 1)]
    sizes = [int(s) for s in a.bench_sizes.split(",")]
    results = []
    _log(tag, f"configs for this rank: {configs} | sizes {sizes}")
    for cfg in configs:
        mode, mns = cfg.split(":"); mns = int(mns)
        a.stock_lens_hook = (mode == "stock")
        llm = _build_engine(a, rank, p_len, mns, mode == "graphs", tag)
        # base-weights clean marker norm from a capture is a fine stand-in for the trainer's published hnorm here
        from vllm import SamplingParams
        from mxf.config import INJECT_LAYER
        o = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                         [SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [INJECT_LAYER]})],
                         use_tqdm=False)[0]
        hnorm = o.activations["residual_stream"][0].float()[marker].norm().item()
        chk = _verify_injection(llm, prompt_ids, marker, hnorm, tag, seed=a.seed)
        for n_seqs in sizes:
            if n_seqs > mns * 4:      # more than 4 waves of the engine's capacity is not a config we would run
                continue
            Bb = max(1, n_seqs // a.group_size)
            idx = eval_rows + np.sort(rng.choice(n_vecs - eval_rows, size=Bb, replace=False))
            dirs = F.normalize(torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
            # warm-up (LoRA load + graph warm) then the timed call
            _generate_block(llm, a, tok, prompt_ids, marker, dirs[: max(1, Bb // 4)], hnorm, lora_req, eos_ids, f"warm{n_seqs}")
            gen_ids, lps, appended, gen_s = _generate_block(llm, a, tok, prompt_ids, marker, dirs, hnorm, lora_req, eos_ids, f"bench{n_seqs}")
            n_tok = sum(len(g) for g in gen_ids)
            stats = None
            try:
                stats = llm.collective_rpc("fast_lens_stats")[0] if not a.stock_lens_hook else None
            except Exception:  # noqa
                pass
            row = {"mode": mode, "max_num_seqs": mns, "n_seqs": len(gen_ids), "gen_s": gen_s, "n_tok": n_tok,
                   "tok_per_s": n_tok / gen_s, "seq_per_s": len(gen_ids) / gen_s, "len_mean": n_tok / len(gen_ids),
                   "appended_stop": appended, "verify": chk, "lens_stats": stats,
                   "sample": tok.decode(gen_ids[0], skip_special_tokens=True)[:100]}
            results.append(row)
            _log(tag, f"{mode} mns={mns} n={len(gen_ids)}: {gen_s:.1f}s -> {row['tok_per_s']:.0f} tok/s, {row['seq_per_s']:.1f} seq/s, "
                      f"len {row['len_mean']:.1f} | appended {appended} | lens {stats}")
            json.dump(results, open(f"{work}/bench_rollout_r{rank}.json", "w"), indent=1)
        del llm
        gc.collect(); torch.cuda.empty_cache()
        time.sleep(3)
    _log(tag, "bench done")


# ==============================================================================================
# TRAINER rank
# ==============================================================================================
def _sync_grads(params, weight, backend, device):
    """Exact weighted all-reduce of the LoRA grads (one flat buffer): global grad = sum_r w_r g_r / sum_r w_r.
    weight = local rollout count (seq-mean loss) or local completion tokens (token-mean loss) -> equals the
    single-GPU gradient over the union batch even with UNEVEN shards. GPU buffer under nccl, CPU under gloo."""
    import torch
    import torch.distributed as dist
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    dev = device if backend == "nccl" else "cpu"
    flat = torch.cat([g.detach().reshape(-1).float() for g in grads] + [torch.ones(1, device=grads[0].device)]).to(dev)
    flat.mul_(float(weight))
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    tot = flat[-1].item()
    if tot <= 0:                  # every rank's effective batch is empty (zero-variance filter): nothing to average
        for g in grads:
            g.zero_()
        return 0.0
    flat.div_(tot)
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off : off + n].view_as(g))
        off += n
    return tot


def _chunked_logp(logits, targets, vocab_chunk, need_entropy_grad):
    """log_softmax over the 248k vocab in fp32, `vocab_chunk` positions at a time (bounds the transient to
    mb*chunk*V*4 bytes). Returns new_lp [mb,T] (with grad) and per-token entropy [mb,T] (no grad unless asked).
    Math identical to rl.py: log_softmax(logits.float()).gather(targets)."""
    import torch
    T = logits.shape[1]
    lp_chunks, ent_chunks = [], []
    for c0 in range(0, T, vocab_chunk):
        c1 = min(c0 + vocab_chunk, T)
        lpf = torch.log_softmax(logits[:, c0:c1].float(), -1)
        lp_chunks.append(lpf.gather(-1, targets[:, c0:c1, None]).squeeze(-1))
        if need_entropy_grad:
            ent_chunks.append(-(lpf.exp() * lpf).sum(-1))
        else:
            with torch.no_grad():
                ent_chunks.append(-(lpf.exp() * lpf).sum(-1))
        del lpf
    return torch.cat(lp_chunks, 1), torch.cat(ent_chunks, 1)


def update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb, keep=None, pfx=None, fp=None):
    """rl.py update() with: vLLM sampler logprobs as old_lp (ratio := 1 where the sampler logp is unknown, i.e.
    the re-appended stop token), logits only for the completion positions (logits_to_keep), fp32 vocab math
    in chunks, use_cache off, exact weighted grad sync. Returns rl.py's stats + sampler_abs_dlogp.
    ScaleRL variant: a.loss (ppo | cispo), a.loss_agg (token | seq | prompt) and keep (effective-batch mask from the
    zero-variance filter, None = everything) go through pg_token_loss() / loss_weights(); the legacy flags reproduce the
    original arithmetic bit for bit.
    pfx (--prefix-cache, a PrefixRunner): the shared prompt prefix runs once per pass and only [marker]+response per rollout
    (suffix logits [:, :-1] == the full path's logits_to_keep=Tc+1 [:, :-1]); loss weights, denominators, IS weights and the
    micro-batch composition are untouched -- only how the logits are computed changes.
    fp (--full-param, a FullParamCtx): FSDP2 policy -- inject hook on a CLONE (FSDP2 hangs backward hooks on the layer output),
    the KL reference is fp.ref (a separate frozen model + its own inject layer / prefix runner), no _sync_grads (FSDP2 reduce-
    scatters every backward and AVERAGES over ranks, so the local loss is scaled by sync_w * world / sum(sync_w) up front =
    the same weighted mean), DTensor-aware grad clipping, prefix logits as the zero-grad path. fp=None = the LoRA path, unchanged."""
    import torch
    import torch.distributed as dist
    import rl_hf as R
    from mxf.config import STEER_COEFF
    n, L = ids.shape
    T = L - p_len
    gen_mask = attn[:, p_len:].bool()
    total_tok = max(int(gen_mask.sum()), 1)
    w_all, sync_w = loss_weights(gen_mask, a.loss_agg, a.group_size, keep)
    lo, hi = 1 - a.clip_eps, 1 + a.clip_eps
    trunc_cap = a.cispo_eps_max if a.loss == "cispo" else a.tis_cap
    inj_pos = 0 if pfx is not None else marker            # prefix-cache: the marker is suffix index 0
    inj_mode = fp.inject_mode if fp is not None else "add"
    scale = 1.0
    tot_w_fp = None
    n_mb_max = n_ref_max = 0
    if fp is not None:   # weighted-mean grad over uneven shards, FSDP2 style (rl_fullparam.loss_scale; sum(sync_w) == 0 -> skipped step)
        tot_w_fp = fp.FP.all_reduce_scalar(sync_w, device)
        scale = fp.FP.loss_scale(sync_w, tot_w_fp, fp.world)
        # every FSDP2 forward/backward is a collective: with uneven shards (512 groups over 5 ranks = 103/103/102/102/102) the ranks
        # would otherwise run different numbers of micro-batches and deadlock -> ranks with fewer run zero-weight dummies (below)
        n_mb_max = int(fp.FP.all_reduce_max(-(-n // mb), device))
        n_ref_max = int(fp.FP.all_reduce_max(-(-n // a.ref_micro_batch), device)) if a.kl_coef > 0 else 0
    t_ref = time.time()
    # micro-batches of LENGTH-SORTED rollouts, each padded only to ITS longest sequence: the loss weights are
    # per-sequence and independent of batching, so this is exactly the same gradient as global padding while
    # most micro-batches run at ~p_len+30 instead of p_len+96 tokens
    lens = gen_mask.sum(1)
    order = torch.argsort(lens)

    def chunks(size):
        for s in range(0, n, size):
            ix = order[s : s + size]
            # pad to a multiple of 16 completion tokens (right padding is masked -> numerically inert; fewer distinct
            # shapes -> far fewer fla Triton autotune stalls in the first steps)
            yield ix, p_len + min(T, -(-int(lens[ix].max()) // 16) * 16)

    def model_logits(model, pf, ix, Lc, cache=None):
        """[len(ix), Tc, V]: logits predicting completion tokens 0..Tc-1 (Tc = Lc - p_len), full-sequence or prefix-cached."""
        if pf is not None:
            return pf.suffix_logits(cache, ids[ix, marker:Lc].to(device), attn[ix, marker:Lc].to(device))[:, :-1]
        return model(input_ids=ids[ix, :Lc].to(device), attention_mask=attn[ix, :Lc].to(device), use_cache=False,
                     logits_to_keep=Lc - p_len + 1).logits[:, :-1]

    def policy_logits(ix, Lc, cache=None):
        return model_logits(actor, pfx, ix, Lc, cache)
    ref_lp_all = None
    if a.kl_coef > 0:
        ref_lp_all = torch.zeros_like(old_lp)
        # LoRA: the 'ref' adapter (as before) or, without one, the policy base with the LoRA off. --full-param: fp.ref = a frozen
        # copy of the init (its own inject layer and prefix runner), run exactly like the policy pass below.
        ref_model, ref_sub, ref_pfx = (fp.ref_model, fp.ref_submodule, fp.ref_pfx) if fp is not None else (actor, submodule, pfx)
        with (contextlib.nullcontext() if fp is not None else _ref_policy(actor)), torch.no_grad():
            cache_ref = ref_pfx.run_prefix() if ref_pfx is not None else None      # ref policy's prefix, no grad
            for ix, Lc in chunks(a.ref_micro_batch):
                Tc = Lc - p_len
                tgt = ids[ix, p_len:Lc].to(device)
                hook = R.make_inject_hook([dirs_rep[i : i + 1] for i in ix.tolist()], [[inj_pos]] * len(ix),
                                          STEER_COEFF, device, torch.bfloat16, mode=inj_mode)
                with R.hooked(ref_sub, hook):
                    lg = model_logits(ref_model, ref_pfx, ix, Lc, cache_ref)
                for c0 in range(0, Tc, a.vocab_chunk):
                    c1 = min(c0 + a.vocab_chunk, Tc)
                    ref_lp_all[ix, c0:c1] = torch.log_softmax(lg[:, c0:c1].float(), -1).gather(
                        -1, tgt[:, c0:c1, None]).squeeze(-1).cpu()
                del lg
            for _ in range(n_ref_max - (-(-n // a.ref_micro_batch))):      # FSDP2: equal forward counts on every rank
                ix1, Lc1 = order[:1], p_len + min(T, 16)
                hook = R.make_inject_hook([dirs_rep[i : i + 1] for i in ix1.tolist()], [[inj_pos]], STEER_COEFF, device, torch.bfloat16, mode=inj_mode)
                with R.hooked(ref_sub, hook):
                    lg = model_logits(ref_model, ref_pfx, ix1, Lc1, cache_ref)
                del lg
            del cache_ref
    t_ref = time.time() - t_ref
    opt.zero_grad(set_to_none=True)
    loss_sum, clipped_tok, ent_sum, kl_sum, ratio_sum, dlp_sum, dlp_n = 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0
    isw_sum, trunc_tok = 0.0, 0
    t_fb = time.time()
    acc = None
    body_tok = 0                 # tokens actually run through the transformer body this pass (prompt/prefix + response + pad)
    if pfx is not None:   # ONE differentiable prefix forward for the whole step; its backward runs once after the micro-batches
        with _policy_precision(actor, a.autocast_bf16):
            if fp is not None:   # FSDP2: the prefix logits carry the zero-weight grad path (see _PrefixGradAccumulator)
                cache0, lg0 = pfx.run_prefix(return_logits=True)
                acc = _PrefixGradAccumulator(cache0, extra_outputs=[lg0])
                del cache0, lg0
            else:
                acc = _PrefixGradAccumulator(pfx.run_prefix())
        body_tok += pfx.P
    for ix, Lc in chunks(mb):
        Tc = Lc - p_len
        body_tok += len(ix) * ((Lc - marker) if pfx is not None else Lc)
        tgt = ids[ix, p_len:Lc].to(device)
        m = gen_mask[ix, :Tc].to(device); w = w_all[ix, :Tc].to(device); A = adv[ix, None].to(device)
        olp = old_lp[ix, :Tc].to(device); kn = known[ix, :Tc].to(device)
        hook = _hook_outside_autocast(R.make_inject_hook([dirs_rep[i : i + 1] for i in ix.tolist()], [[inj_pos]] * len(ix), STEER_COEFF, device, torch.bfloat16, mode=inj_mode),
                                      a.autocast_bf16)
        with R.hooked(submodule, hook), (fp.suffix_ctx() if fp is not None else contextlib.nullcontext()):
            with _policy_precision(actor, a.autocast_bf16):   # --autocast-bf16: bf16 LoRA matmuls/activations; fp32 vocab math below is outside
                logits = policy_logits(ix, Lc, acc.cache if acc is not None else None)
            if fp is not None:
                new_lp, ent = fp.logp_from(logits, tgt, a.vocab_chunk, a.entropy_coef > 0, a.fp32_head)
            else:
                new_lp, ent = _chunked_logp(logits, tgt, a.vocab_chunk, a.entropy_coef > 0)
            del logits
            olp_eff = torch.where(kn, olp, new_lp.detach())
            loss_tok, ratio, rho = pg_token_loss(new_lp, olp_eff, A, a.loss, a.clip_eps, a.tis_cap, a.cispo_eps_max)
            loss = (loss_tok * w).sum()
            ent_sum += float((ent.detach() * m).sum())
            if a.entropy_coef > 0:
                loss = loss - a.entropy_coef * (ent * w).sum()
            if a.kl_coef > 0:
                ref_lp = ref_lp_all[ix, :Tc].to(device)
                delta = ref_lp - new_lp
                kl = (torch.exp(delta) - delta - 1).clamp(0.0, a.kl_cap)
                loss = loss + a.kl_coef * (kl * w).sum()
                kl_sum += float((kl.detach() * m).sum())
            if fp is not None:
                loss = loss * scale       # FSDP2 mean over ranks -> sync_w-weighted mean (loss/loss stats stay the unscaled local value)
            loss.backward()
            if fp is not None and scale > 0:
                loss = loss / scale
        if acc is not None:
            acc.accumulate()
        loss_sum += loss.item()
        clipped_tok += int((((ratio < lo) | (ratio > hi)) & m).sum())
        ratio_sum += float((ratio.detach() * m).sum())
        with torch.no_grad():   # the IS weight the gradient actually sees: min(rho, eps_max) (cispo) / ratio where the PPO clip is inactive
            eff = ratio.detach() if a.loss == "cispo" else ratio.detach() * ~(((ratio > hi) & (A > 0)) | ((ratio < lo) & (A < 0)))
            isw_sum += float((eff * m).sum()); trunc_tok += int(((rho > trunc_cap) & m).sum())
        mk = m & kn
        dlp_sum += float(((new_lp.detach() - olp).abs() * mk).sum()); dlp_n += int(mk.sum())
        del new_lp, ent, ratio, loss, olp_eff, loss_tok, rho, eff
    for _ in range(n_mb_max - (-(-n // mb))):   # FSDP2: zero-weight dummy micro-batches so every rank issues the same collectives
        ix1, Lc1 = order[:1], p_len + min(T, 16)
        hook = R.make_inject_hook([dirs_rep[i : i + 1] for i in ix1.tolist()], [[inj_pos]], STEER_COEFF, device, torch.bfloat16, mode=inj_mode)
        with R.hooked(submodule, hook), fp.suffix_ctx():
            lg = policy_logits(ix1, Lc1, acc.cache if acc is not None else None)
            # the dummy loss must reach EVERY parameter the real micro-batches reach: FSDP2 reduce-scatters only the params that
            # have a grad, as ONE flat collective per group -> a rank whose lm_head.grad is None (chunked head bypassed) would
            # issue a shorter reduce-scatter than the others and hang. Go through the (chunked) head like a real micro-batch.
            lp1, _ = fp.logp_from(lg, ids[ix1, p_len:Lc1].to(device), a.vocab_chunk, False, a.fp32_head)
            (lp1.float().sum() * 0.0).backward()
        del lg, lp1
        if acc is not None:
            acc.accumulate()
    if acc is not None:
        acc.backward()               # the summed cache gradients through the shared prefix forward, once
    t_fb = time.time() - t_fb
    params = [p for p in actor.parameters() if p.requires_grad]
    t_sync = time.time()
    tot_w = sync_w
    if fp is not None:
        tot_w = tot_w_fp             # FSDP2 already reduce-scattered (and averaged) every micro-batch's grads; the loss scaling did the weighting
    elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        tot_w = _sync_grads(params, sync_w, a.backend, device)
    t_sync = time.time() - t_sync
    skipped = 0
    if tot_w <= 0:               # zero-variance filter emptied the global effective batch (never with keep=None)
        opt.zero_grad(set_to_none=True); gn = 0.0; skipped = 1
        print("[update] empty effective batch (every group zero-variance) -- skipping step", flush=True)
    else:
        gn = fp.FT.clip_grad_norm(params, a.max_grad_norm) if fp is not None else float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
        if math.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad(set_to_none=True); skipped = 1
            print(f"[update] non-finite grad norm ({gn}) -- skipping step", flush=True)
    # padding accounting: completion positions the policy micro-batches ran vs real completion tokens (the pad-to-16 rule +
    # length spread inside a micro-batch), and body tokens per rollout (what the transformer actually processed)
    comp_pos = sum(len(ix) * (Lc - p_len) for ix, Lc in chunks(mb))
    return {"loss": loss_sum, "grad_norm": gn, "clipfrac": clipped_tok / total_tok, "entropy": ent_sum / total_tok,
            "kl": kl_sum / total_tok, "ratio_mean": ratio_sum / total_tok, "sampler_abs_dlogp": dlp_sum / max(dlp_n, 1),
            "t_ref": t_ref, "t_fb": t_fb, "t_sync": t_sync, "n_unknown_lp": int((gen_mask & ~known).sum()),
            "is_weight_mean": isw_sum / total_tok, "is_trunc_frac": trunc_tok / total_tok, "sync_w": sync_w, "skipped": skipped,
            "pad_frac": 1.0 - int(gen_mask.sum()) / max(comp_pos, 1), "body_tok_per_rollout": body_tok / max(n, 1),
            "real_tok_per_rollout": (int(gen_mask.sum()) + (1 if pfx is not None else p_len) * n) / max(n, 1)}


def _make_prefix_runner(actor, prompt_ids, marker, device, a, tag):
    """--prefix-cache -> PrefixRunner (after the actor + adapters exist), else None. Refuses stock transformers loudly."""
    if not getattr(a, "prefix_cache", False):
        return None
    import transformers
    pfx = PrefixRunner(actor, prompt_ids, marker, device)
    _log(tag, f"prefix cache ON: shared prefix {pfx.P} tokens run once per pass, [marker]+response per rollout "
              f"(transformers {transformers.__version__} fork at {os.path.dirname(transformers.__file__)})")
    return pfx


_MB_CANDIDATES_DEFAULT = "64,48,40,32,24,16,12,8,6,4"


def _mb_candidates(a):
    """--mb-candidates; with --prefix-cache and the untouched default the probe also tries 128/96 (the suffix-only
    activations are ~3x smaller per rollout, so the old 64 cap is far below what fits)."""
    cands = [int(x) for x in a.mb_candidates.split(",")]
    if getattr(a, "prefix_cache", False) and a.mb_candidates == _MB_CANDIDATES_DEFAULT:
        cands = [128, 96] + cands
    return cands


def find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, cands, tag, pfx=None, fp=None):
    """Largest micro-batch whose forward+backward at MAX length (prompt + max_new_tokens) fits with <90% of the GPU
    allocated. Synthetic tokens; same hook, same chunked vocab math as update_disagg. pfx (--prefix-cache): the probe
    runs the prefix-cached suffix path (prefix fwd + expanded cache + [marker]+max_new_tokens suffix + prefix bwd), i.e.
    exactly the per-micro-batch memory shape of the real update.

    Strategy (Sep 3): measure mb=1 and mb=2 (always fit), fit peak ~= fixed + mb * per_seq, predict the largest candidate
    under 85% of the GPU and VERIFY it (<90%); only on a failed verification step down. The old descending scan started at
    mb=64 and OOM'd on purpose -- on the H200:4 run a CUDA OOM mid-forward left the partial graph resident (every later
    candidate saw ~138 GB 'allocated by PyTorch' on a 140 GB card, with only 56.6 GB resident before the scan), so this
    probe never OOMs by design. Each attempt runs in its own function so no local of a failed attempt can outlive it.
    fp (--full-param): the FSDP2 policy -- clone-mode inject hook, prefix logits as the zero-grad path, and a lower target
    fraction (--mb-target-frac 0.80): an OOM inside an FSDP2 forward/backward leaves its state machine inconsistent, so a
    failed verification is FATAL here (pass --micro-batch explicitly) instead of stepping down."""
    import torch
    import torch.nn.functional as F
    import rl_hf as R
    from mxf.config import D_MODEL, STEER_COEFF
    inj_mode = fp.inject_mode if fp is not None else "add"
    frac = float(getattr(a, "mb_target_frac", 0.85) or 0.85)
    L = len(prompt_ids) + a.max_new_tokens
    p_len = len(prompt_ids)
    total = torch.cuda.get_device_properties(0).total_memory
    GB = 2**30
    gc.collect(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    free0, _ = torch.cuda.mem_get_info()
    _log(tag, f"micro-batch probe @ L={L} (prompt {p_len} + {a.max_new_tokens} new): resident {base / GB:.1f} GB, "
              f"free {free0 / GB:.1f} / {total / GB:.0f} GB (device {torch.cuda.current_device()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    def attempt(mb):
        """-> (ok, peak_bytes | None, err | None). All tensors are locals here and die with the frame."""
        ok, peak, err = False, None, None
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 100000, (mb, L), device=device)
            ids[:, :p_len] = torch.tensor(prompt_ids, device=device)
            attn = torch.ones_like(ids)
            dirs = F.normalize(torch.randn(mb, D_MODEL, device=device), dim=-1)
            ac = getattr(a, "autocast_bf16", False)
            hook = _hook_outside_autocast(R.make_inject_hook([dirs[i : i + 1] for i in range(mb)], [[0 if pfx is not None else marker]] * mb,
                                                             STEER_COEFF, device, torch.bfloat16, mode=inj_mode), ac)
            acc = None
            if pfx is not None:
                with _policy_precision(actor, ac):
                    if fp is not None:
                        c0, l0 = pfx.run_prefix(return_logits=True)
                        acc = _PrefixGradAccumulator(c0, extra_outputs=[l0]); del c0, l0
                    else:
                        acc = _PrefixGradAccumulator(pfx.run_prefix())
            with R.hooked(submodule, hook):
                with _policy_precision(actor, ac):
                    if pfx is not None:
                        logits = pfx.suffix_logits(acc.cache, ids[:, marker:], attn[:, marker:])[:, :-1]
                    else:
                        logits = actor(input_ids=ids, attention_mask=attn, use_cache=False, logits_to_keep=L - p_len + 1).logits[:, :-1]
                new_lp, ent = _chunked_logp(logits, ids[:, p_len:], a.vocab_chunk, False)
                del logits
                loss = new_lp.mean() * 0.0
                loss.backward()
            if acc is not None:
                acc.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            ok = peak < 0.90 * total
        except torch.cuda.OutOfMemoryError as e:
            ok = False
            err = re.sub(r"\s+", " ", str(e))[:420]
        finally:
            opt.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
        return ok, peak, err

    res, chosen = {}, None

    def record(mb, ok, peak, err, note=""):
        res[mb] = {"ok": ok, "peak_gb": (peak / GB) if peak else None}
        after = torch.cuda.memory_allocated()
        leak = f" | LEAK: {(after - base) / GB:.1f} GB still allocated after the attempt" if after > base + GB else ""
        _log(tag, f"mb {mb}: {'OK' if ok else 'OOM/too tight'}" + (f" peak {peak / GB:.1f} GB / {total / GB:.0f}" if peak else "")
                  + (f" {note}" if note else "") + (f" | {err}" if err else "") + leak)
        return after > base + GB

    ok1, p1, e1 = attempt(1); record(1, ok1, p1, e1, "(calibration)")
    ok2, p2, e2 = attempt(2); leaked = record(2, ok2, p2, e2, "(calibration)")
    if ok1 and ok2 and not leaked:
        per = max(p2 - p1, 64 * 2**20)
        fixed = max(p1 - base - per, 0)
        pred = int((frac * total - base - fixed) // per)
        if fp is not None and fp.world > 1:   # FSDP2: every rank must run the SAME attempts (each backward is a collective) -> agree on the prediction
            import torch.distributed as dist
            t = torch.tensor([pred], dtype=torch.int64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            pred = int(t.item())
        _log(tag, f"probe fit: {per / GB:.2f} GB/seq + {fixed / GB:.1f} GB fixed on {base / GB:.1f} GB resident -> "
                  f"predicted max mb {pred} at {frac:.0%} of {total / GB:.0f} GB" + (" (min over ranks)" if fp is not None else ""))
        for mb in sorted({c for c in cands if 2 < c <= pred}, reverse=True):
            ok, peak, err = attempt(mb); leaked = record(mb, ok, peak, err, "(verify)")
            if ok:
                chosen = mb
                break
            if fp is not None:
                raise RuntimeError(f"--full-param micro-batch verification at mb={mb} failed ({err}); FSDP2 state is not trustworthy after an "
                                   f"OOM -- relaunch with --micro-batch <= {max(2, mb // 2)} or a lower --mb-target-frac (probe fit: {per / GB:.2f} GB/seq, "
                                   f"{fixed / GB:.1f} GB fixed, {base / GB:.1f} GB resident)")
            if leaked:
                _log(tag, "verification OOM leaked GPU memory; not probing further")
                break
        if chosen is None and ok2:
            chosen = 2 if pred >= 2 else 1
    else:
        # calibration itself failed (should not happen with >30 GB free) -- fall back to the old descending scan
        _log(tag, f"calibration failed (ok1={ok1} ok2={ok2} leaked={leaked}); falling back to the descending scan over {cands}")
        for mb in cands:
            ok, peak, err = attempt(mb); record(mb, ok, peak, err)
            if ok:
                chosen = mb
                break
    return chosen, res



def plan_probe_step(measured, cands, budget_bytes, max_growth=2.0, per_seq_floor=0.0):
    """--full-param micro-batch probe planning (pure; unit-tested). measured: {mb: peak_bytes} of the attempts so far (all fit),
    cands: the candidate list. Returns the next mb to try or None when done. Rule: walk the candidates upwards; the next one must
    be <= max_growth x the largest measured mb and its peak, extrapolated linearly from the two largest measured points (slope
    floored at per_seq_floor), must stay under the budget. The first two attempts (the two smallest candidates >= 4) are free:
    at tiny micro-batches the peak is the mb-independent fp32-grad allocation, so a slope from mb 1/2 says nothing."""
    ok = sorted(measured)
    todo = [c for c in sorted(set(cands)) if c not in measured and (not ok or c > ok[-1])]
    if not todo:
        return None
    if len(ok) < 2:
        return todo[0]
    a, b = ok[-2], ok[-1]
    per = max((measured[b] - measured[a]) / max(b - a, 1), per_seq_floor)
    for c in todo:
        if c > max_growth * b:
            return None
        pred = measured[b] + per * (c - b)
        return c if pred <= budget_bytes else None
    return None


def find_micro_batch_fullparam(actor, opt, submodule, prompt_ids, marker, a, device, cands, tag, pfx, fp):
    """--full-param: the largest micro-batch whose MEASURED forward+backward peak at max length, plus the AdamW moments that
    torch.optim.AdamW allocates at the first step (2 x the fp32 master shard; not present during the probe), stays under
    --mb-target-frac of the GPU. Walks the candidates upwards (plan_probe_step) so no attempt can OOM by more than one
    doubling's worth of extrapolation error; every rank runs the same attempts (each backward is an FSDP2 collective) and
    uses the MAX peak over ranks. An OOM is fatal (FSDP2 cannot resume after an exception inside a forward)."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import rl_hf as R
    from mxf.config import D_MODEL, STEER_COEFF
    L = len(prompt_ids) + a.max_new_tokens
    p_len = len(prompt_ids)
    total = torch.cuda.get_device_properties(0).total_memory
    GB = 2**30
    frac = float(a.mb_target_frac)
    fp.FP.reshard_root(actor)   # after the initial publish's no-grad forward the root params are still gathered (5 GB, plain tensors)
    gc.collect(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    reserve = 0 if len(opt.state) else 2 * sum((p.to_local().numel() if fp.FP.is_dtensor(p) else p.numel()) * 4 for p in actor.parameters())
    world = fp.world
    reserve = fp.FP.all_reduce_max(reserve, device)      # shard padding differs per rank: ONE budget everywhere (identical probe decisions)
    budget = frac * total - reserve
    _log(tag, f"micro-batch probe (full-param) @ L={L}: resident {base / GB:.1f} GB, AdamW reserve {reserve / GB:.1f} GB, budget for the measured "
              f"fwd/bwd peak {budget / GB:.1f} GB ({frac:.0%} of {total / GB:.0f} GB minus the reserve)")

    def attempt(mb):
        peak, err = None, None
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 100000, (mb, L), device=device)
            ids[:, :p_len] = torch.tensor(prompt_ids, device=device)
            attn = torch.ones_like(ids)
            dirs = F.normalize(torch.randn(mb, D_MODEL, device=device), dim=-1)
            hook = R.make_inject_hook([dirs[i : i + 1] for i in range(mb)], [[0 if pfx is not None else marker]] * mb,
                                      STEER_COEFF, device, torch.bfloat16, mode=fp.inject_mode)
            acc = None
            if pfx is not None:
                c0, l0 = pfx.run_prefix(return_logits=True)
                acc = _PrefixGradAccumulator(c0, extra_outputs=[l0]); del c0, l0
            with R.hooked(submodule, hook), fp.suffix_ctx():
                if pfx is not None:
                    logits = pfx.suffix_logits(acc.cache, ids[:, marker:], attn[:, marker:])[:, :-1]
                else:
                    logits = actor(input_ids=ids, attention_mask=attn, use_cache=False, logits_to_keep=L - p_len + 1).logits[:, :-1]
                new_lp, ent = fp.logp_from(logits, ids[:, p_len:], a.vocab_chunk, False, a.fp32_head)
                del logits
                loss = new_lp.mean() * 0.0
                loss.backward()
            if acc is not None:
                acc.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError as e:
            err = re.sub(r"\s+", " ", str(e))[:300]
        finally:
            opt.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
        return peak, err

    def sync_max(x):
        t = torch.tensor([float(x)], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    measured, res = {}, {}
    cands = sorted({c for c in cands if c >= 4})
    while True:
        mb = plan_probe_step(measured, cands, budget)
        if mb is None:
            break
        peak, err = attempt(mb)
        peak_all = sync_max(peak if peak is not None else float("inf"))
        if peak is None or peak_all == float("inf"):
            raise RuntimeError(f"--full-param micro-batch probe: mb={mb} ran out of memory ({err}); measured so far "
                               f"{ {k: round(v / GB, 1) for k, v in measured.items()} } GB; relaunch with --micro-batch <= {max(measured) if measured else 4}")
        res[mb] = {"ok": peak_all <= budget, "peak_gb": peak_all / GB, "peak_plus_reserve_gb": (peak_all + reserve) / GB}
        _log(tag, f"mb {mb}: peak {peak_all / GB:.1f} GB (max over ranks; + AdamW reserve = {(peak_all + reserve) / GB:.1f} of {total / GB:.0f} GB) -> "
                  f"{'fits' if peak_all <= budget else 'over budget'}")
        if peak_all > budget:
            break
        measured[mb] = peak_all
    chosen = max(measured) if measured else None
    return chosen, res


def _save_adapter_for_vllm(actor, lora_dir, dtype):
    """rl.py's _save_adapter_for_vllm (module names renamed to the Qwen3_5ForConditionalGeneration layout vLLM
    serves) with a configurable dtype. bf16 halves the write; vLLM casts LoRA weights to the model dtype on
    load, so the served policy is bit-identical either way. Returns (n_tensors, timings)."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    os.makedirs(lora_dir, exist_ok=True)
    t0 = time.time()
    sd = get_peft_model_state_dict(actor, adapter_name="default")
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.detach().to(dtype).to("cpu", copy=True).contiguous()
    torch.cuda.synchronize()
    t1 = time.time()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    actor.peft_config["default"].save_pretrained(lora_dir)
    t2 = time.time()
    return len(out), {"state_dict_s": t1 - t0, "write_s": t2 - t1, "bytes": sum(v.numel() * v.element_size() for v in out.values())}


def _write_eval_request(work, k, adapter_step, EV, EX, a, tag):
    """rank 0: everything the rollout ranks need to generate ckpt k's eval texts (~30 MB of unit directions)."""
    import torch
    os.makedirs(f"{work}/eval_req", exist_ok=True)
    req = {"ckpt_step": k, "adapter_step": adapter_step, "t": time.time(), "sets": _eval_sets_from_assets(EV, EX, a)}
    p = _eval_req_path(work, k)
    torch.save(req, p + ".tmp"); os.replace(p + ".tmp", p)
    _log(tag, f"eval request written for ckpt {k} (adapter step {adapter_step}): "
              + ", ".join(f"{st['name']}x{st['n_rows']}x{st['n']}" for st in req["sets"]))


def _score_eval_block(k, shard_paths, EV, EX, IX, actor, tok, device, rank, world, a):
    """ALL trainer ranks: load the rollout ranks' shards, score rows i % world == rank of every set on the CLEAN base
    (exactly rl.py inline_eval's scoring + inline_extra_evals.run_extra_evals_gpu's scoring half), all_gather, and
    on rank 0 reduce to rl.py's eval/* keys (+ extra/locality + adversarial of the previous ckpt's judge texts).
    Returns (ev, ex) dicts on rank 0 ({} elsewhere); errors travel as data so no rank can deadlock."""
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    t0 = time.time()
    EU, es, sae = EV["EU"], EV["es"], EV["sae"]
    pending = IX._broadcast_pending(EX, rank, world) if (EX is not None and IX is not None) else []
    local = {}
    try:
        shards = [torch.load(p, weights_only=False) for p in shard_paths]
        errs = [sh["error"] for sh in shards if sh.get("error")]
        if errs:
            raise RuntimeError("rollout shard errors: " + " | ".join(errs))
        texts = _eval_merge_shards(shards)
        bo = a.eval_bo
        for fam in EV["fams"]:
            du = es[f"{fam}_dirs"]
            rows = [i for i in sorted(texts.get(fam, {})) if i % world == rank]
            if rows:
                flat = [t for i in rows for t in texts[fam][i]]
                rd = F.normalize(torch.stack([du[i] for i in rows for _ in range(bo)]).float(), dim=-1)
                cos = EU.score_probe_cos(flat, rd, actor, tok, device).view(len(rows), bo).max(1).values
                local[fam] = {int(i): float(c) for i, c in zip(rows, cos.tolist())}
            else:
                local[fam] = {}
        feats = EV["feats"]
        rows = [i for i in sorted(texts.get("sae", {})) if i % world == rank]
        if rows:
            flat = [t for i in rows for t in texts["sae"][i]]
            fl = [feats[i] for i in rows for _ in range(bo)]
            acts, peaks = EU.score_sae_peaks(flat, fl, sae, actor, tok, device)
            acts = acts.view(len(rows), bo)
            best, arg = acts.max(1)
            pk = peaks.view(len(rows), bo, -1)[torch.arange(len(rows)), arg]
            local["sae"] = {int(i): float(v) for i, v in zip(rows, best.tolist())}
            local["sae_peak"] = {int(i): pk[j].half().numpy().tobytes() for j, i in enumerate(rows)}
        else:
            local["sae"], local["sae_peak"] = {}, {}
        local["extra"] = None
        if EX is not None and IX is not None:
            xf, subsae, fidx = EX["feats"], EX["subsae"], EX["fidx"]
            n_roll = EX["n_rollouts"]
            xrows = [i for i in sorted(texts.get("extra", {})) if i % world == rank]
            flat_t = [t for i in xrows for t in texts["extra"][i]]
            flat_f = [xf[i] for i in xrows for _ in range(n_roll)]
            loc_rows = IX.locality_rows_from_profiles(IX._profiles(flat_t, flat_f, actor, tok, device, subsae) if flat_t else [], EX["fire"])
            loc = {xf[i]: loc_rows[j * n_roll:(j + 1) * n_roll] for j, i in enumerate(xrows)}
            adv = []
            for item in pending:
                res = {"src": item["src_ckpt_step"], "true": {}, "naive": {}}
                at, af, tags = [], [], []
                for arm in ("true", "naive"):
                    for f, txts in item[arm].items():
                        f = int(f)
                        if f in fidx and fidx[f] % world == rank:
                            for t in txts:
                                at.append(t); af.append(f); tags.append((arm, f))
                profs = IX._profiles(at, af, actor, tok, device, subsae) if at else []
                for (arm, f), pr in zip(tags, profs):
                    res[arm].setdefault(f, []).append(float(pr.max()) if len(pr) else 0.0)
                adv.append(res)
            local["extra"] = {"texts": {xf[i]: texts["extra"][i] for i in xrows}, "loc": loc, "adv": adv}
        local["t_gen"] = float(max(sh.get("t_gen", 0.0) for sh in shards))
    except Exception as e:  # noqa
        local = {"error": f"rank{rank}: {type(e).__name__}: {str(e)[:300]}"}
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return {}, {}
    errs = [g["error"] for g in gathered if "error" in g]
    if errs:
        return {"error": " | ".join(errs)}, {}
    merged = {}
    for g in gathered:
        for key, d in g.items():
            if key in ("extra", "t_gen"):
                continue
            merged.setdefault(key, {}).update(d)
    out = {}
    for fam in EV["fams"]:
        vals = np.array([merged[fam][i] for i in sorted(merged[fam])], dtype=np.float64)
        out[f"eval/{fam}/cos"] = float(vals.mean())
    idx = sorted(merged["sae"])
    best = np.array([merged["sae"][i] for i in idx], dtype=np.float64)
    cp = EV["cp"][idx]
    na = best / np.maximum(cp, 1e-6)
    out["eval/sae/norm_act"] = float(na.mean())
    if merged.get("sae_peak"):
        peak_h = torch.from_numpy(np.stack([np.frombuffer(merged["sae_peak"][i], dtype=np.float16) for i in idx]).astype(np.float32))
        ranks = EU.sae_rank_at_peaks(sae, peak_h, [EV["feats"][i] for i in idx]).astype(np.float64)
        out["eval/sae/rank1_frac"] = float(np.mean(ranks == 1))
        out["eval/sae/rank_le5"] = float(np.mean(ranks <= 5))
        out["eval/sae/mean_rank"] = float(ranks.mean())
        out["eval/sae/mrr"] = float(np.mean(1.0 / ranks))
    out["eval/sae/fired"] = float(np.mean(best > EU.SAE_FIRE))
    out["eval/sae/beat_corpus"] = float(np.mean(best > cp))
    out["eval/sae/unverbalized_frac"] = float(np.mean(best <= EU.SAE_FIRE))
    out["eval/sae/unverbalized_p10"] = float(np.mean(na < 0.10))
    out["eval/sae/gate"] = float(EU.SAE_FIRE)
    out["eval/sae/fired_1p0"] = float(np.mean(best > EU.SAE_FIRE_LEGACY))
    out["eval/sae/unverbalized_1p0"] = float(np.mean(best <= EU.SAE_FIRE_LEGACY))
    cos_keys = [kk for kk in out if kk.startswith("eval/") and kk.endswith("/cos") and kk.split("/")[1] not in EU.CONTROL_FAMS and kk.split("/")[1] != "sae"]   # sae/cos is a diagnostic, not a mean_all family
    out["eval/mean_all"] = float(np.mean([out[kk] for kk in cos_keys]))
    for fam in EV["fams"]:
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
    out["eval/all/sae_norm_act"] = out["eval/sae/norm_act"]
    out["eval/all/sae_unverbalized"] = out["eval/sae/unverbalized_frac"]
    out["time/inline_eval_s"] = time.time() - t0
    out["time/inline_eval_gen_s"] = float(max(g.get("t_gen", 0.0) for g in gathered))
    ex = {}
    if EX is not None and IX is not None and all(g.get("extra") is not None for g in gathered):
        texts_x, loc = {}, {}
        adv_true, adv_naive = {}, {}
        for g in gathered:
            texts_x.update(g["extra"]["texts"]); loc.update(g["extra"]["loc"])
            for res in g["extra"]["adv"]:
                for arm, store in (("true", adv_true), ("naive", adv_naive)):
                    for f, acts in res[arm].items():
                        store.setdefault(res["src"], {}).setdefault(int(f), []).extend(acts)
        ex = IX.aggregate_locality(loc, EX["corpus_peak"])
        for src in sorted(set(adv_true) | set(adv_naive)):
            m = IX.adversarial_metrics(adv_true.get(src, {}), adv_naive.get(src, {}), EX["corpus_peak"], fire=EX["fire"])
            IX._RESULTS_Q.put((src, m))
        ex["extra/adversarial/n_pending_scored"] = float(len(pending))
        ex["time/extra_eval_gpu_s"] = time.time() - t0
        EX["last_rollouts"] = {"ckpt_step": k, "rollouts": texts_x}
        try:
            os.makedirs(EX["out_dir"], exist_ok=True)
            IX._save_json_atomic({"ckpt_step": k, "rollouts": {str(f): v for f, v in texts_x.items()},
                                  "locality": {str(f): v for f, v in loc.items()}}, os.path.join(EX["out_dir"], f"rollouts_ckpt{k}.json"))
        except Exception as e:  # noqa
            _log("T0", f"could not write extra-eval rollouts artifact: {e}")
    return out, ex


def _publish_adapter(actor, submodule, prompt, marker, device, work, step, keep, tag, fp32=False):
    """rank 0: adapter in vLLM key layout + ||h_marker|| (adapter ON) -> lora/step_<k>, then flip `latest`."""
    import torch
    import rl_hf as R
    t0 = time.time()
    hnorm = R._marker_norm(actor, submodule, prompt, marker, device, adapter=True)
    t_norm = time.time() - t0
    d = f"{work}/lora/step_{step}"
    n, tm = _save_adapter_for_vllm(actor, d, torch.float32 if fp32 else torch.bfloat16)
    json.dump({"step": step, "hnorm": hnorm, "n_tensors": n, "t": time.time()}, open(f"{d}/meta.json", "w"))
    _atomic_write_text(f"{work}/lora/latest", str(step))
    for old in sorted(glob.glob(f"{work}/lora/step_*"), key=lambda p: int(p.rsplit("_", 1)[-1]))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    tot = time.time() - t0
    if step <= 1:
        _log(tag, f"publish step {step}: hnorm {t_norm:.2f}s | state_dict->cpu {tm['state_dict_s']:.2f}s | write {tm['bytes'] / 2**30:.2f} GB in {tm['write_s']:.2f}s -> {d}")
    return hnorm, tot


def _pick_blocks(work, n_needed, drop_stale):
    files = _queue_files(work)
    if len(files) < n_needed:
        return None, []
    if drop_stale:
        take, stale = files[-n_needed:], files[:-n_needed]
    else:
        take, stale = files[:n_needed], []
    return take, stale


def run_trainer(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    if a.no_fla:
        sys.modules["fla"] = None      # transformers' use_kernel_func_from_hub_with_fallback then keeps the torch GDN path
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb
    import rl_hf as R
    from mxf.config import INJECT_LAYER, MODEL, TrainConfig
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids

    rank = int(os.environ["DISAGG_RANK"]); world = int(os.environ["DISAGG_WORLD"]); is_main = rank == 0
    tag = f"T{rank}"
    work = a.work_dir
    torch.manual_seed(a.seed + rank)
    torch.cuda.set_device(0)
    device = "cuda:0"
    if world > 1:   # nccl for the GPU flat-buffer grad all-reduce, gloo for the CPU-tensor collectives of the eval code
        dist.init_process_group("cpu:gloo,cuda:nccl" if a.backend == "nccl" else "gloo",
                                init_method=f"tcp://127.0.0.1:{a.master_port}", rank=rank, world_size=world)
    try:
        import fla  # noqa
        fla_v = getattr(fla, "__version__", "?")
    except Exception:  # noqa
        fla_v = None
    _log(tag, f"world {world} backend {a.backend} | fla {'v' + str(fla_v) if fla_v else 'ABSENT (torch GDN fallback)'} "
              f"| torch {torch.__version__} | gpu {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    B, G = a.groups_per_step, a.group_size
    tr = TrainConfig()

    t0 = time.time()
    pb = check_policy_base(a)   # MODEL unless --policy-base (a full-FT checkpoint dir); tokenizer, prompt and eos ids stay MODEL's
    fp = None
    if a.full_param:   # rl/rl_fullparam.py: FSDP2 whole-model policy (sft/fullft.py sharding), frozen sharded scorer, bf16 weight publish
        import rl_fullparam as FP
        FT = FP.fullft_module()
        R.read_resid = FP.read_resid_noraise      # rl.py score() reads layer 42 without the _Stop exception (FSDP2-safe); full-param mode only
        fp = FullParamCtx(FP, FT, world, rank)
        actor = AutoModelForCausalLM.from_pretrained(pb, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
        _log(tag, f"bf16 policy base loaded in {time.time() - t0:.0f}s | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB -> FSDP2 sharding over {world} ranks")
        actor = FT.shard_full_model(actor, world, device, log=(lambda m: _log(tag, m)) if is_main else (lambda *x, **k: None),
                                    prefetch=a.fsdp_prefetch)
        actor.train()
        fp.mem["after_shard_gb"] = torch.cuda.memory_allocated() / 2**30
        if a.chunked_head:   # lm_head + fp32 log-softmax per position chunk, recomputed in the backward: no [tokens x 248k] logits ever
            fp.head = FP.ChunkedHead(actor)
            _log(tag, f"lm_head chunked + recomputed ({'fp32' if a.fp32_head else 'bf16'} logits, {a.vocab_chunk} positions per chunk; rl_fullparam.ChunkedHead)")
        elif a.fp32_head:   # trainable head: fp32 logits + grad path through the gathered bf16 weight (rl_fullparam)
            FP.install_fp32_head_trainable(actor)
            _log(tag, "lm_head recomputed in fp32 (trainable FSDP2 head: F.linear(x.float(), W_bf16.float()) per micro-batch)")
        if a.suffix_ckpt:
            fp.ckpt = FP.suffix_checkpointer(actor)
            _log(tag, "suffix activation checkpointing ON (per decoder layer, exact with the prefix cache; sft/prefix_cache.SuffixCheckpointer)")
        opt = torch.optim.AdamW(list(actor.parameters()), lr=a.lr, weight_decay=a.weight_decay, eps=a.adam_eps, betas=tuple(a.adam_betas))
        if a.load_optim:
            FP.load_optim_dcp(actor, opt, a.load_optim, log=lambda m: _log(tag, m))
        if is_main and _multimodal():             # vision/MTP tensors exist only in the Qwen3.6 base repo
            fp.nontext_shard = "/tmp/base_nontext.safetensors"
            FT.prepare_nontext_shard(MODEL, fp.nontext_shard, log=lambda m: _log(tag, m))
        submodule = get_layer(actor, INJECT_LAYER)
        if is_main:
            _log(tag, f"kernel backends: {FT.kernel_backends(actor)}")
    else:
        actor = AutoModelForCausalLM.from_pretrained(pb, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
        if a.init_adapter:
            actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
        else:
            actor = get_peft_model(actor, LoraConfig(r=tr.lora_r, lora_alpha=tr.lora_alpha, lora_dropout=0.0, use_rslora=True,
                                                     target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
            _log(tag, f"fresh rsLoRA r{tr.lora_r}/a{tr.lora_alpha} all-linear on {pb}")
        actor.train()
        if a.fp32_head:   # before the micro-batch search so its +memory is part of the OOM probe
            install_fp32_head(actor)
            _log(tag, "lm_head recomputed in fp32 (ScaleRL precision fix, trainer side; the vLLM sampler stays bf16-head/fp32-softmax)")
        opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay,
                                eps=a.adam_eps, betas=tuple(a.adam_betas))
        optim_p = os.path.join(a.init_adapter or "", "optim.pt")
        if a.init_adapter and os.path.exists(optim_p) and not a.fresh_optim:
            opt.load_state_dict(torch.load(optim_p, map_location="cpu"))
            for _g in opt.param_groups:   # a loaded optimizer state carries the OLD run's lr/betas/eps — re-apply this run's flags (ablation arms rely on it)
                _g["lr"], _g["betas"], _g["eps"] = a.lr, tuple(a.adam_betas), a.adam_eps
            _log(tag, f"optimizer state loaded from {optim_p}; hyperparams re-applied: lr {a.lr} betas {tuple(a.adam_betas)} eps {a.adam_eps}")
            _log(tag, f"AdamW state restored from {optim_p}")
        submodule = get_layer(actor, INJECT_LAYER)
        if a.kl_coef > 0:
            ref_src = a.ref_adapter or a.init_adapter
            if ref_src:
                actor.load_adapter(ref_src, adapter_name="ref")
                actor.set_adapter("default")
            else:   # fresh LoRA (typically on a full-FT --policy-base): the KL anchor is the policy base itself = the LoRA disabled
                _log(tag, f"KL reference = {pb} with the LoRA disabled (no --init-adapter / --ref-adapter)")
    n_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    _log(tag, f"actor ready in {time.time() - t0:.0f}s | policy base {pb} | trainable {n_train / 1e6:.0f}M | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    use_gates = a.fluency_floor is not None or a.distinct_floor is not None
    if fp is not None:
        # the REWARD model: a frozen, FSDP2-sharded copy of the ORIGINAL base (the policy's weights move, so "LoRA off" no longer exists)
        scorer = load_scorer_fullparam(a, device, world, tag, use_gates, need_logits=(a.kl_coef > 0 and policy_base_is_model(a)))
        fp.mem["after_scorer_gb"] = torch.cuda.memory_allocated() / 2**30
        if a.kl_coef > 0:   # KL anchor = a frozen copy of the init: the scorer itself when the policy started from MODEL, else a second sharded copy of the policy base
            if policy_base_is_model(a):
                fp.ref_model = scorer.get_base_model()
                _log(tag, "KL reference = the frozen scorer copy of MODEL (policy base == MODEL; scorer kept full-depth for its logits)")
            else:
                t_ref0 = time.time()
                ref = AutoModelForCausalLM.from_pretrained(pb, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
                fp.ref_model = FP.shard_frozen_bf16(ref, world)
                gc.collect(); torch.cuda.empty_cache()
                _log(tag, f"KL reference = frozen FSDP2-sharded copy of {pb} in {time.time() - t_ref0:.0f}s | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB "
                          f"(+~{54 / world:.0f} GB/rank; reuse the scorer by starting from MODEL to avoid it)")
            fp.ref_submodule = get_layer(fp.ref_model, INJECT_LAYER)
    else:
        scorer = load_scorer(a, actor, device, tag, use_gates)   # the REWARD model: the actor with its LoRA off (default) or a frozen copy of MODEL
    pfx = _make_prefix_runner(actor, prompt_ids, marker, device, a, tag)
    if fp is not None and fp.ref_model is not None and pfx is not None:
        fp.ref_pfx = PrefixRunner(fp.ref_model, prompt_ids, marker, device)

    # publish the init policy FIRST so the rollout ranks start generating while we tune the micro-batch
    if is_main:
        for d in ("lora", "queue"):
            os.makedirs(f"{work}/{d}", exist_ok=True)
    if fp is not None:   # COLLECTIVE: manifest (all ranks' shard geometry) + marker norm (FSDP forward); the engines already hold these weights
        fp.manifest = FP.build_manifest(actor, world)
        if is_main:
            FP.write_manifest(f"{work}/lora/manifest.json", fp.manifest)
        hnorm0, t_pub = _publish_fullparam(fp, actor, submodule, prompt, marker, device, work, a.step_offset, tag, a, is_main, world, initial=True)
        if is_main:
            _log(tag, f"published init policy as step {a.step_offset} (||h_marker|| = {hnorm0:.2f}; {len(fp.manifest['names'])} tensors, "
                      f"{fp.manifest['bytes_bf16'] / 2**30:.1f} GB bf16 per publish, mode {a.publish_mode}) in {t_pub:.1f}s")
    elif is_main:
        hnorm0, t_pub = _publish_adapter(actor, submodule, prompt, marker, device, work, a.step_offset, a.keep_loras, tag, a.publish_fp32)
        _log(tag, f"published init adapter as step {a.step_offset} (||h_marker|| = {hnorm0:.2f}) in {t_pub:.1f}s")

    mb = a.micro_batch
    mb_res = {}
    if mb <= 0:
        cands = _mb_candidates(a)
        if fp is not None:
            mb, mb_res = find_micro_batch_fullparam(actor, opt, submodule, prompt_ids, marker, a, device, cands, tag, pfx, fp)
        else:
            mb, mb_res = find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, cands, tag, pfx=pfx, fp=fp)
        assert mb is not None, f"no micro-batch candidate fits: {mb_res}"
        if world > 1:
            t = torch.tensor([mb], dtype=torch.int64, device=device if a.backend == "nccl" else "cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            mb = int(t.item())
    _peak = (mb_res.get(mb) or {}).get("peak_gb")
    _log(tag, f"micro-batch = {mb} (no gradient checkpointing; autocast bf16 policy forward {'ON' if a.autocast_bf16 else 'off'}; "
              f"prefix cache {'ON' if pfx is not None else 'off'}; score length-bucket {'ON' if a.score_length_bucket else 'off'}"
              + (f"; probe peak {_peak:.1f} GB" if _peak else "") + ")")
    # ---- inline eval assets (held-out sets + SAE on every trainer rank; extra-eval testbed/judge) — after the
    # micro-batch search so the SAE's 2.7 GB is not part of the OOM probe ----
    EV = R.load_eval_assets(a, device, is_main) if a.inline_eval_every > 0 else None
    EX, IX = None, None
    if EV is not None and not a.no_extra_evals:
        try:
            import inline_extra_evals as IX
            EX = IX.prepare_extra_eval_assets(a, device, rank, world, is_main, sae=EV["sae"])
        except Exception as e:  # noqa
            EX, IX = None, None
            if is_main:
                _log(tag, f"extra-eval DISABLED ({type(e).__name__}: {e})")
    if world > 1:   # every rank must agree on whether EV exists (a failed load on one rank would otherwise desync the collectives)
        t = torch.tensor([0 if EV is None else 1], dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        if int(t.item()) == 0:
            EV, EX = None, None
    if is_main and EV is not None:
        _log(tag, f"inline eval every {a.inline_eval_every} steps: rollout ranks generate {len(EV['fams'])} families x {len(EV['es'][EV['fams'][0] + '_dirs'])} "
                  f"dirs x Bo{a.eval_bo} + sae + {'extra testbed' if EX is not None else 'NO extra evals'}; trainer ranks score sharded")

    adv_mode = a.adv_mode or ("batch" if a.batch_norm else ("group" if a.std_norm else "none"))
    tgt_map = {}
    if is_main and a.transcript_every > 0 and a.direction_source == "cluster" and os.path.exists(f"{a.data_dir}/records.jsonl"):
        with open(f"{a.data_dir}/records.jsonl") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except Exception:  # noqa
                    continue
                vi = rec.get("vec_idx", i)
                if vi not in tgt_map and rec.get("target_text") is not None:
                    tgt_map[vi] = (rec.get("family"), rec["target_text"][:240])
    npr, npr_path, n_bank_avail = None, f"{work}/npr/dropped.json", 0
    if a.npr_threshold > 0 and a.direction_source == "cluster":
        npr = NPRTracker(a.npr_threshold, a.npr_pass_cos)          # rank 0 drives it; every rank sees the same gathered rewards
        _, _nv, _er = _bank_open(a)
        n_bank_avail = int(_nv - _er)
    if is_main:
        _log(tag, f"B={B} groups x G={G} per step from {a.blocks_per_step} block(s) of {a.rollout_block_groups} | adv {adv_mode} "
                  f"| gates {use_gates} | mb {mb} | ref-mb {a.ref_micro_batch} | queue cap {a.max_queue_blocks} | {'drop-stale' if a.drop_stale else 'FIFO'}")
        _log(tag, f"recipe {a.recipe or 'legacy'} | loss {a.loss} " + (f"eps_max {a.cispo_eps_max}" if a.loss == "cispo" else f"clip {a.clip_eps} tis {a.tis_cap}")
                  + f" | loss-agg {a.loss_agg} | zero-var filter {a.zero_var_filter} (eps {a.zero_var_eps}) | NPR {a.npr_threshold}"
                  + (f" (pass cos {a.npr_pass_cos}, {n_bank_avail} directions)" if npr is not None else " (off)")
                  + f" | max lag {a.max_lag} step(s) | fp32 head {a.fp32_head} | length control {a.length_control}")
        _log(tag, f"policy base {pb} | scorer {'the actor with its LoRA off (= MODEL)' if scorer is actor else ('a frozen FSDP2-sharded copy of MODEL' if fp is not None else 'a frozen copy of MODEL')}"
                  + (f" | FULL-PARAMETER policy: {sum(p.numel() for p in actor.parameters()) / 1e9:.2f}B trainable, publish {a.publish_mode}, "
                     f"resident {torch.cuda.memory_allocated() / 2**30:.1f} GB/rank before step 0" if fp is not None else ""))
        if not a.no_wandb:
            wandb.init(project="maxact-fast", name=a.run_name, config={**vars(a), "micro_batch_used": mb, "mb_search": mb_res,
                                                                        "fla": fla_v, "disagg": True, "policy_base_resolved": pb},
                       id=a.wandb_id or None, resume="must" if a.wandb_id else None)
            wandb.define_metric("ckpt_step")
            wandb.define_metric("eval/*", step_metric="ckpt_step")
            wandb.define_metric("extra/*", step_metric="ckpt_step")
        os.makedirs(a.save_dir, exist_ok=True)
        write_run_meta(a, a.save_dir, mb)
        json.dump({"micro_batch": mb, "search": mb_res}, open(f"{work}/trainer_mb.json", "w"))
    eos_set = set(R._eos_ids(tok, actor))
    step_hist = []

    for step in range(a.step_offset, a.total_steps):
        t0 = time.time()
        # ---- rank 0 picks the blocks; everyone loads them (shared /tmp) and takes its whole-group shard ----
        pick = [None, None, None]
        if is_main:
            ev_ready = None
            if EV is not None:   # a complete eval block (all rollout shards present) is scored at the START of the step
                for kk in _eval_requests(work):
                    if _eval_shards_ready(work, kk, a.n_rollout):
                        ev_ready = (kk, [_eval_shard_path(work, kk, r) for r in range(a.n_rollout)])
                        break
                    if time.time() - os.path.getmtime(_eval_req_path(work, kk)) > a.eval_drop_after_s:
                        _log(tag, f"inline-eval ckpt {kk}: shards never completed after {a.eval_drop_after_s:.0f}s — dropped")
                        os.remove(_eval_req_path(work, kk))
            while True:
                take, stale = _pick_blocks(work, a.blocks_per_step, a.drop_stale)
                if take is not None:
                    break
                time.sleep(0.2)
            pick = [take, stale, ev_ready]
        if world > 1:
            dist.broadcast_object_list(pick, src=0)
        take, stale, ev_ready = pick
        if ev_ready is not None:
            kk, paths = ev_ready
            ev, ex = _score_eval_block(kk, paths, EV, EX, IX, scorer, tok, device, rank, world, a)
            if is_main:
                for pth in paths + [_eval_req_path(work, kk)]:
                    try:
                        os.remove(pth)
                    except FileNotFoundError:
                        pass
                if "error" in ev:
                    _log(tag, f"inline-eval FAILED for ckpt {kk}: {ev['error']}")
                else:
                    print(f"  [inline-eval] ckpt {kk}: mean_all {ev['eval/mean_all']:.4f} | sae norm_act {ev['eval/sae/norm_act']:.4f} "
                          f"unverb {ev['eval/sae/unverbalized_frac']:.3f} rank1 {ev.get('eval/sae/rank1_frac', float('nan')):.3f} | realact "
                          f"{ev.get('eval/realact/cos', float('nan')):.4f} | scored in {ev['time/inline_eval_s']:.0f}s (rollout gen {ev['time/inline_eval_gen_s']:.0f}s)"
                          + (f" | locality win5 {ex.get('extra/locality/win5_share', float('nan')):.3f} fire {ex.get('extra/locality/fire_frac', float('nan')):.3f}" if ex else ""),
                          flush=True)
                    if not a.no_wandb:
                        wandb.log({**ev, **ex, "ckpt_step": kk})
                    if ex and IX is not None:
                        try:
                            IX.launch_judge_stage(None, kk, EX, a)
                        except Exception as e:  # noqa
                            _log(tag, f"judge launch failed: {type(e).__name__}: {e}")
        t_wait = time.time() - t0
        blocks = [torch.load(f, weights_only=False) for f in take]
        if world > 1:
            dist.barrier()
        if is_main:
            for f in take + stale:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
        dirs_all = torch.cat([b["dirs"] for b in blocks])                       # [B, d]
        idx_all = np.concatenate([np.asarray(b["dir_idx"]) for b in blocks])
        gen_all = [g for b in blocks for g in b["gen_ids"]]
        lps_all = [l for b in blocks for l in b["lps"]]
        assert dirs_all.shape[0] == B and len(gen_all) == B * G, f"batch shape {dirs_all.shape[0]} groups / {len(gen_all)} seqs"
        lag = float(np.mean([step - b["adapter_step"] for b in blocks]))
        lag_max = float(max(step - b["adapter_step"] for b in blocks))
        gen_s = float(np.mean([b["gen_s"] for b in blocks]))
        blk_tok = float(np.sum([b["n_tok"] for b in blocks]))
        my_groups = np.array_split(np.arange(B), world)[rank]
        g0, g1 = int(my_groups[0]), int(my_groups[-1]) + 1
        Bl = g1 - g0
        dirs = dirs_all[g0:g1]
        idx = idx_all[g0:g1]
        gen_ids = gen_all[g0 * G : g1 * G]
        lps = lps_all[g0 * G : g1 * G]
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)

        # ---- reward + shaping (identical to rl.py main; --score-length-bucket = the same score() on length-sorted rows) ----
        t_sc = time.time()
        if a.score_length_bucket:
            _lens = [len(g) for g in gen_ids]
            if use_gates:
                r, flu, dis = score_bucketed(R, texts, dirs_rep, scorer, tok, device, a, _lens, with_fluency=True)
            else:
                r = score_bucketed(R, texts, dirs_rep, scorer, tok, device, a, _lens)
        elif use_gates:
            r, flu, dis = R.score(texts, dirs_rep, scorer, tok, device, a, with_fluency=True)
        else:
            r = R.score(texts, dirs_rep, scorer, tok, device, a)
        if fp is not None:   # FSDP2 scorer: score() ran ceil(non-empty texts / score_batch) collective forwards on THIS rank -> equalize
            n_sc = -(-sum(1 for t in texts if t.strip()) // a.score_batch)
            for _ in range(int(FP.all_reduce_max(n_sc, device)) - n_sc):
                FP.dummy_scorer_forward(scorer, tok, device)
        r = r * a.reward_scale
        raw_r, gate_frac = r.clone(), 1.0                      # raw_r = the TRUE cosine (logged/transcripts), before any shaping (rl.py fd2d144)
        trunc = torch.tensor([len(g) >= a.max_new_tokens and (not g or g[-1] not in eos_set) for g in gen_ids])
        trunc_frac = trunc.float().mean().item()
        if a.trunc_reward is not None and bool(trunc.any()):   # EasyNLA: cap-hit rollouts score a fixed failure reward and still train
            r[trunc] = a.trunc_reward
        gate = torch.ones(Bl * G, dtype=torch.bool)
        if use_gates:
            if a.fluency_floor is not None:
                gate &= flu >= a.fluency_floor
            if a.distinct_floor is not None:
                gate &= dis >= a.distinct_floor
            r = r - a.gate_penalty * (~gate).float()
            gate_frac = gate.float().mean().item()
        if a.len_penalty_start is not None:
            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids], dtype=torch.float32)
            r = r - a.len_penalty_per_tok * over * gate.float()
        adv, keep = compute_advantages_disagg(r, Bl, G, adv_mode, a.zero_var_eps, a.zero_var_filter)
        t_sc = time.time() - t_sc

        _table = None
        if is_main and a.transcript_every > 0 and step % a.transcript_every == 0:
            rows_t = []
            for g in range(min(a.transcript_groups, Bl)):
                vi = int(idx[g])
                fam, tgt = tgt_map.get(vi, (None, None))
                for j in range(min(a.transcript_samples, G)):
                    i = g * G + j
                    rows_t.append({"step": step, "group": g, "vec_idx": vi, "family": fam, "target": tgt, "text": texts[i],
                                   "cos": float(raw_r[i]) / a.reward_scale, "reward": float(r[i]), "adv": float(adv[i]), "n_tok": len(gen_ids[i])})
            with open(f"{a.save_dir}/transcripts.jsonl", "a") as f:
                for row in rows_t:
                    f.write(json.dumps(row) + "\n")
            if not a.no_wandb:
                _table = wandb.Table(columns=list(rows_t[0].keys()), data=[list(x.values()) for x in rows_t])

        # ---- pad + update (old_lp = the SAMPLER's logprobs) ----
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len))
        known = torch.zeros((Bl * G, L - p_len), dtype=torch.bool)
        pt = torch.tensor(prompt_ids, dtype=torch.long)
        for i, (g, lp) in enumerate(zip(gen_ids, lps)):
            ids[i, :p_len] = pt
            ids[i, p_len : p_len + len(g)] = torch.tensor(g)
            attn[i, : p_len + len(g)] = 1
            for j, v in enumerate(lp):
                if v is not None:
                    old_lp[i, j] = float(v); known[i, j] = True
        t_up = time.time()
        lr_now = a.lr * (min(1.0, (step + 1) / a.warmup_steps) if a.warmup_steps > 0 else 1.0)   # linear warmup
        if a.lr_decay != "none":   # decay over [warmup_steps, total_steps) to lr_min_frac * lr; step is the GLOBAL step (resume-safe)
            _T = a.lr_decay_total_steps or a.total_steps
            frac = min(1.0, max(0.0, (step - a.warmup_steps) / max(1, _T - a.warmup_steps)))
            shape = (1.0 - frac) if a.lr_decay == "linear" else 0.5 * (1.0 + math.cos(math.pi * frac))
            lr_now = lr_now * (a.lr_min_frac + (1.0 - a.lr_min_frac) * shape)
        if a.warmup_steps > 0 or a.lr_decay != "none":
            for _g in opt.param_groups:
                _g["lr"] = lr_now

        stats = update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb, keep=keep, pfx=pfx, fp=fp)
        if a.entropy_target > 0:   # SAC-style temperature adaptation on the measured per-token entropy of this step
            a.entropy_coef = float(min(a.entropy_coef_max, max(a.entropy_coef_min,
                                   a.entropy_coef * math.exp(a.entropy_adapt_rate * (a.entropy_target - stats["entropy"])))))
        t_up = time.time() - t_up

        # ---- publish the new policy for the rollout ranks ----
        t_pub, hnorm = 0.0, float("nan")
        if fp is not None and a.publish_every > 0 and (step + 1) % a.publish_every == 0:   # COLLECTIVE (every trainer rank gathers)
            hnorm, t_pub = _publish_fullparam(fp, actor, submodule, prompt, marker, device, work, step + 1, tag, a, is_main, world)
        elif is_main and a.publish_every > 0 and (step + 1) % a.publish_every == 0:
            hnorm, t_pub = _publish_adapter(actor, submodule, prompt, marker, device, work, step + 1, a.keep_loras, tag, a.publish_fp32)
            # rl.py evaluates ckpt `step` (the weights after this update) with the adapter published right here
            if EV is not None and step % a.inline_eval_every == 0:
                _write_eval_request(work, step, step + 1, EV, EX, a, tag)

        # ---- logging (reward stats over ALL trainer ranks; uneven shards -> weighted) ----
        secs = time.time() - t0
        mem_alloc = torch.cuda.memory_allocated(device) / 2**30
        mem_peak = torch.cuda.max_memory_allocated(device) / 2**30
        mem_peak_max = mem_peak
        if fp is not None:   # COLLECTIVE: the worst rank's peak this step
            _, mem_peak_max = FP.peak_gb_all_ranks(device)
        torch.cuda.reset_peak_memory_stats(device)
        n_gen = float(sum(len(g) for g in gen_ids))
        _rg = raw_r.view(Bl, G)
        n_dropped_g = 0.0 if keep is None else float((~keep.view(Bl, G)[:, 0]).sum())
        loc = torch.tensor([n_gen, gate_frac * Bl, r.view(Bl, G).std(1).sum().item(), Bl,
                            _rg.std(1).sum().item(), (_rg.std(1) < 1e-6).float().sum().item(),
                            adv.pow(2).sum().item(), adv.abs().sum().item(), float(Bl * G),
                            stats["n_unknown_lp"], n_dropped_g], dtype=torch.float64)
        gmin, gmax = torch.tensor([_rg.std(1).min().item()]), torch.tensor([_rg.std(1).max().item()])
        if world > 1:
            dev = device if a.backend == "nccl" else "cpu"
            raw_list = [None] * world; r_list = [None] * world; gm_list = [None] * world
            dist.all_gather_object(raw_list, raw_r); dist.all_gather_object(r_list, r); dist.all_gather_object(gm_list, _rg.mean(1))
            raw_r_all, r_all, gmeans = torch.cat(raw_list), torch.cat(r_list), torch.cat(gm_list)
            loc = loc.to(dev); dist.all_reduce(loc); loc = loc.cpu()
            gmin, gmax = gmin.to(dev), gmax.to(dev)
            dist.all_reduce(gmin, op=dist.ReduceOp.MIN); dist.all_reduce(gmax, op=dist.ReduceOp.MAX)
            gmin, gmax = gmin.cpu(), gmax.cpu()
        else:
            raw_r_all, r_all, gmeans = raw_r, r, _rg.mean(1)
        n_gen_all, n_groups_all, n_seq_all = float(loc[0]), float(loc[3]), float(loc[8])
        log = {"reward/mean": raw_r_all.mean().item(), "reward/std": raw_r_all.std().item(), "reward/max": raw_r_all.max().item(),
               "reward/shaped_mean": r_all.mean().item(), "reward/within_group_std": float(loc[2] / n_groups_all),
               "reward/gate_frac": float(loc[1] / n_groups_all), "reward/trunc_frac": trunc_frac,
               "ratio/clipfrac": stats["clipfrac"], "ratio/mean": stats["ratio_mean"],
               "policy/entropy": stats["entropy"], "policy/kl_to_init": stats["kl"], "policy/entropy_coef": a.entropy_coef,
               "policy/sampler_abs_dlogp": stats["sampler_abs_dlogp"], "policy/offpolicy_lag_steps": lag,
               "loss": stats["loss"], "grad_norm": stats["grad_norm"], "grad_norm_did_clip": float(stats["grad_norm"] > a.max_grad_norm), "lr": float(opt.param_groups[0]["lr"]),
               "rollout/mean_logp": float(old_lp[known].mean()) if bool(known.any()) else float("nan"),
               "rollout/len_mean": n_gen_all / (B * G), "tokens_per_sec": n_gen_all / secs,
               "rollout/gen_s": gen_s, "rollout/tok_per_s_per_replica": blk_tok / max(gen_s * len(blocks), 1e-6),
               "rollout/appended_stop": float(sum(b["appended"] for b in blocks)), "rollout/marker_hnorm": hnorm,
               "rollout/queue_depth": float(len(_queue_files(work))), "rollout/blocks_dropped": float(len(stale)),
               "rollout/unknown_lp_tokens": float(loc[9]),
               "time/step_s": secs, "time/wait_rollouts_s": t_wait, "time/score_s": t_sc, "time/update_s": t_up,
               "time/ref_pass_s": stats["t_ref"], "time/fwd_bwd_s": stats["t_fb"], "time/grad_sync_s": stats["t_sync"],
               "time/publish_s": t_pub, "time/rollout_s": gen_s,
               "mem/hf_alloc_gb": mem_alloc, "mem/hf_peak_gb": mem_peak, "mem/hf_peak_gb_max_rank": mem_peak_max, "micro_batch": mb,
               "trainer/pad_frac": stats["pad_frac"], "trainer/body_tokens_per_rollout": stats["body_tok_per_rollout"],
               "trainer/real_tokens_per_rollout": stats["real_tok_per_rollout"],
               # ScaleRL diagnostics (present in every run; zero/inert when the variant flags are off)
               "scalerl/is_weight_mean": stats["is_weight_mean"], "scalerl/is_trunc_frac": stats["is_trunc_frac"],
               "scalerl/zero_var_dropped_frac": float(loc[10] / n_groups_all), "scalerl/effective_groups": float(n_groups_all - loc[10]),
               "scalerl/lag_max": lag_max, "scalerl/trunc_frac": trunc_frac, "scalerl/step_skipped": float(stats["skipped"])}
        if fp is not None:    # publish breakdown (rl_fullparam: block-boundary wait, transfer, GB, GB/s; engine-side load time arrives via meta on the rollout side)
            log.update({f"publish/{k}": float(v) for k, v in fp.last_pub.items() if isinstance(v, (int, float))})
        if npr is not None:   # No-Positive-Resampling bookkeeping on the whole batch (every rank has the gathered rewards; rank 0 publishes)
            log.update(npr.update(idx_all, (raw_r_all / a.reward_scale).numpy(), G))
            log["scalerl/npr_dropped_frac_of_bank"] = len(npr.dropped) / max(n_bank_avail, 1)
            if is_main:
                npr.publish(npr_path)
        if R.SCORE_STATS.get("peak_dist"):
            _pd = torch.cat(R.SCORE_STATS["peak_dist"]); log["reward/peak_dist_mean"] = _pd.mean().item()
            log["reward/peak_in_last5_frac"] = (_pd <= 4).float().mean().item()
            log["reward/peak_last_frac"] = (_pd == 0).float().mean().item()      # reward argmax exactly at the last token
        w_std = float(loc[4] / n_groups_all)
        b_std = float(gmeans.std().item()) if len(gmeans) > 1 else 0.0
        log.update({"var/within_group_std_raw": w_std, "var/between_group_std_raw": b_std,
                    "var/zero_var_group_frac": float(loc[5] / n_groups_all),
                    "var/adv_std": float(math.sqrt(max(loc[6] / n_seq_all, 0.0))),   # advantages are zero-mean per group -> std = rms
                    "var/adv_abs_mean": float(loc[7] / n_seq_all),
                    "var/group_std_min": float(gmin), "var/group_std_max": float(gmax), "var/signal_ratio": w_std / (b_std + 1e-9)})
        if _table is not None:
            log["rollouts/samples"] = _table
        step_hist.append({"step": step, "step_s": secs, "wait_s": t_wait, "update_s": t_up, "score_s": t_sc, "lag": lag,
                          "gen_s": gen_s, "reward": log["reward/mean"], "entropy": log["policy/entropy"], "ratio": log["ratio/mean"],
                          "clipfrac": log["ratio/clipfrac"], "dlogp": log["policy/sampler_abs_dlogp"], "len": log["rollout/len_mean"],
                          "peak_gb": mem_peak, "peak_gb_max_rank": mem_peak_max, "t_ref": stats["t_ref"], "t_fb": stats["t_fb"], "t_sync": stats["t_sync"], "t_pub": t_pub,
                          "n_local": Bl * G, "gnorm": log["grad_norm"], "publish": dict(fp.last_pub) if fp is not None else None})
        if is_main:
            print(f"step {step:05d} | r {log['reward/mean']:.3f} (max {log['reward/max']:.2f}) | ent {log['policy/entropy']:.2f} "
                  f"| ratio {log['ratio/mean']:.3f} clip {log['ratio/clipfrac']:.2%} |dlogp| {log['policy/sampler_abs_dlogp']:.4f} lag {lag:.1f} "
                  f"| kl {log['policy/kl_to_init']:.4f} | len {log['rollout/len_mean']:.0f} | gnorm {log['grad_norm']:.3f} "
                  f"| {secs:.0f}s (wait {t_wait:.0f} score {t_sc:.0f} update {t_up:.0f} [ref {stats['t_ref']:.0f} fb {stats['t_fb']:.0f} sync {stats['t_sync']:.1f}] pub {t_pub:.0f}) "
                  f"| gen {gen_s:.0f}s | peak {mem_peak:.0f}G | queue {int(log['rollout/queue_depth'])}", flush=True)
            if step % 10 == 0:
                print(f"  sample r={raw_r[0]:.2f}: {texts[0][:110]!r}", flush=True)
            if not a.no_wandb:
                wandb.log(log, step=step)
            if IX is not None and EX is not None:
                for cs, m in IX.poll_judge_results():          # judge results arrive minutes later; x-axis = ckpt_step
                    print(f"  [extra-eval] judge results for ckpt {cs}: " + " ".join(
                        f"{kk.split('/')[-1]}={v:.3f}" for kk, v in m.items() if kk.startswith("extra/") and "auc" in kk), flush=True)
                    if not a.no_wandb:
                        wandb.log({**m, "ckpt_step": cs})
            json.dump(step_hist, open(f"{work}/trainer_steps.json", "w"))
        _save_steps = {int(x) for x in a.save_steps.split(",") if x.strip()}
        do_save = bool((a.save_every and step and step % a.save_every == 0) or (step in _save_steps))
        if fp is not None and do_save:   # COLLECTIVE full-model checkpoint (every rank all-gathers, rank 0 writes)
            _save_fullparam_ckpt(fp, actor, f"{a.save_dir}/step_{step}", tok, is_main, world, a, mb, step, opt, tag)
        if is_main and fp is None:
            if do_save:
                actor.save_pretrained(f"{a.save_dir}/step_{step}")
                write_run_meta(a, f"{a.save_dir}/step_{step}", mb, step=step)
                torch.save(opt.state_dict(), f"{a.save_dir}/step_{step}/optim.pt")
                if a.save_every:   # rolling optim.pt cleanup only for the periodic schedule (log-spaced ckpts keep theirs)
                    stale_o = os.path.join(a.save_dir, f"step_{step - 2 * a.save_every}", "optim.pt")
                    if os.path.exists(stale_o) and (step - 2 * a.save_every) not in _save_steps:
                        os.remove(stale_o)
    if fp is not None:   # COLLECTIVE full-model final checkpoint
        _save_fullparam_ckpt(fp, actor, f"{a.save_dir}/final", tok, is_main, world, a, mb, a.total_steps, opt, tag, final=True)
        if is_main:
            ok = os.path.exists(f"{a.save_dir}/final/SAVE_DONE") and os.path.exists(f"{a.save_dir}/final/model.safetensors.index.json")
            _log(tag, f"final full-model checkpoint {a.save_dir}/final: {'complete (SAVE_DONE + index)' if ok else 'INCOMPLETE'} | "
                      f"memory: {json.dumps({k: round(v, 1) for k, v in fp.mem.items()})}")
            json.dump({"mem": fp.mem, "publish_hist": fp.pub_hist, "manifest_summary": {k: fp.manifest[k] for k in ('n_trainer', 'n_params', 'bytes_bf16', 'checks')},
                       "micro_batch": mb}, open(f"{work}/fullparam_summary.json", "w"), indent=1)
    if is_main and fp is None:
        actor.save_pretrained(f"{a.save_dir}/final")
        write_run_meta(a, f"{a.save_dir}/final", mb, step=a.total_steps)
        if a.save_every:
            torch.save(opt.state_dict(), f"{a.save_dir}/final/optim.pt")
        # rl.py-format round trip: the checkpoint must load as a PEFT adapter next to the live one
        try:
            actor.load_adapter(f"{a.save_dir}/final", adapter_name="chk_final")
            actor.set_adapter("default")
            _log(tag, f"checkpoint {a.save_dir}/final loads via PeftModel.load_adapter: OK")
        except Exception as e:  # noqa
            _log(tag, f"checkpoint load-back FAILED: {type(e).__name__}: {e}")
    if is_main:
        if IX is not None and EX is not None:
            IX.wait_for_judge_stages(900)
            for cs, m in IX.poll_judge_results():
                if not a.no_wandb:
                    wandb.log({**m, "ckpt_step": cs})
        _atomic_write_text(f"{work}/STOP", "done")
        print("RL_DONE", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def run_bench_trainer(a):
    """Trainer-side cost model: micro-batch search, then score()+update_disagg() wall time for several
    rollouts-per-rank sizes on SYNTHETIC rollouts at realistic lengths (uniform 8..96 gen tokens, mean ~40),
    on world = n_trainer ranks (so grad sync is included). Writes <work>/bench_trainer_r<rank>.json."""
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    if a.no_fla:
        sys.modules["fla"] = None
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import rl_hf as R
    from mxf.config import D_MODEL, INJECT_LAYER, MODEL, TrainConfig
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    rank = int(os.environ["DISAGG_RANK"]); world = int(os.environ["DISAGG_WORLD"]); tag = f"BT{rank}"
    torch.cuda.set_device(0); device = "cuda:0"
    if world > 1:
        dist.init_process_group(a.backend, init_method=f"tcp://127.0.0.1:{a.master_port}", rank=rank, world_size=world)
    try:
        import fla  # noqa
        fla_v = getattr(fla, "__version__", "?")
    except Exception:  # noqa
        fla_v = None
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    t0 = time.time()
    actor = AutoModelForCausalLM.from_pretrained(check_policy_base(a), dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    if a.init_adapter:
        actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
    else:
        tr = TrainConfig()
        actor = get_peft_model(actor, LoraConfig(r=tr.lora_r, lora_alpha=tr.lora_alpha, lora_dropout=0.0, use_rslora=True,
                                                 target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    actor.train()
    if a.fp32_head:
        install_fp32_head(actor)
    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay, eps=a.adam_eps, betas=tuple(a.adam_betas))
    submodule = get_layer(actor, INJECT_LAYER)
    if a.kl_coef > 0 and (a.ref_adapter or a.init_adapter):
        actor.load_adapter(a.ref_adapter or a.init_adapter, adapter_name="ref"); actor.set_adapter("default")
    scorer = load_scorer(a, actor, device, tag, use_gates=False)
    resident = torch.cuda.memory_allocated() / 2**30
    _log(tag, f"actor in {time.time() - t0:.0f}s | resident {resident:.1f} GB | fla {fla_v} | world {world} {a.backend}")
    pfx = _make_prefix_runner(actor, prompt_ids, marker, device, a, tag)
    out = {"fla": fla_v, "resident_gb": resident, "gpu": torch.cuda.get_device_name(0), "world": world, "backend": a.backend,
           "prefix_cache": pfx is not None, "score_length_bucket": bool(a.score_length_bucket)}
    mb = a.micro_batch
    if mb <= 0:
        mb, res = find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, _mb_candidates(a), tag, pfx=pfx)
        out["mb_search"] = res
        if world > 1:
            t = torch.tensor([mb], dtype=torch.int64, device=device if a.backend == "nccl" else "cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MIN); mb = int(t.item())
    out["micro_batch"] = mb
    t_pub = time.time()
    hn = R._marker_norm(actor, submodule, prompt, marker, device, adapter=True)
    _, tm32 = _save_adapter_for_vllm(actor, f"{a.work_dir}/bench_pub32", torch.float32)
    _, tm16 = _save_adapter_for_vllm(actor, f"{a.work_dir}/bench_pub16", torch.bfloat16)
    out["publish_s"] = time.time() - t_pub; out["hnorm"] = hn; out["publish_fp32"] = tm32; out["publish_bf16"] = tm16
    _log(tag, f"mb {mb} | publish fp32 {tm32} | bf16 {tm16}")
    rng = np.random.default_rng(0)
    G = a.group_size
    rows = []
    for n_roll in [int(x) for x in a.bench_rollouts_per_rank.split(",")]:
        Bl = max(1, n_roll // G)
        lens = rng.integers(a.min_new_tokens, a.max_new_tokens + 1, size=Bl * G)
        gen_ids = [list(rng.integers(1000, 100000, size=int(l))) + [tok.eos_token_id] for l in lens]
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
        dirs = F.normalize(torch.randn(Bl, D_MODEL), dim=-1)
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)
        torch.cuda.synchronize(); t_s = time.time()
        if a.score_length_bucket:
            r = score_bucketed(R, texts, dirs_rep, scorer, tok, device, a, [len(g) for g in gen_ids])
        else:
            r = R.score(texts, dirs_rep, scorer, tok, device, a)
        torch.cuda.synchronize(); t_s = time.time() - t_s
        adv = R.compute_advantages(r, Bl, G, "group")
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long); attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len)); known = torch.zeros((Bl * G, L - p_len), dtype=torch.bool)
        for i, g in enumerate(gen_ids):
            ids[i, :p_len] = prompt.cpu(); ids[i, p_len : p_len + len(g)] = torch.tensor(g); attn[i, : p_len + len(g)] = 1
            old_lp[i, : len(g)] = -2.5; known[i, : len(g)] = True
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t_u = time.time()
        st = update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb, pfx=pfx)
        torch.cuda.synchronize(); t_u = time.time() - t_u
        row = {"rollouts_per_rank": Bl * G, "score_s": t_s, "update_s": t_u, "ref_s": st["t_ref"], "fwd_bwd_s": st["t_fb"], "sync_s": st["t_sync"],
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "len_mean": float(lens.mean()), "L": L}
        rows.append(row)
        _log(tag, f"{Bl * G} rollouts/rank (L={L}): score {t_s:.1f}s | update {t_u:.1f}s (ref {st['t_ref']:.1f} fb {st['t_fb']:.1f} sync {st['t_sync']:.1f}) "
                  f"| peak {row['peak_gb']:.0f} GB")
        out["rows"] = rows
        json.dump(out, open(f"{a.work_dir}/bench_trainer_r{rank}.json", "w"), indent=1)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    _log(tag, "bench done")


# ==============================================================================================
# LAUNCHER
# ==============================================================================================
def _n_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:  # noqa
        return 0


def run_launch(a, argv):
    n = _n_gpus()
    X, Y = a.n_rollout, a.n_trainer
    bench = a.role == "bench"
    assert X + Y == n, f"n_rollout + n_trainer = {X + Y} but the container has {n} GPUs"
    work = a.work_dir
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    for d in ("lora", "queue"):
        os.makedirs(f"{work}/{d}", exist_ok=True)
    base_env = os.environ.copy()
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        base_env.pop(k, None)
    procs = []

    def spawn(role, gpu, rank, world):
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu), DISAGG_RANK=str(rank), DISAGG_WORLD=str(world))
        if role in ("trainer", "bench-trainer") and base_env.get("DISAGG_TRAINER_PYTHONPATH"):
            # Hopper: fla 0.5.2 refuses its gated chunk_bwd_dqkwg on Triton 3.4-3.7.0 (fla #640, wrong results). The HF
            # trainer children (no vLLM) get a newer Triton from this dir; the vLLM rollout children keep torch's pin.
            env["PYTHONPATH"] = base_env["DISAGG_TRAINER_PYTHONPATH"] + os.pathsep + base_env.get("PYTHONPATH", "")
        child_argv = [x for x in argv]
        child_argv[child_argv.index("--role") + 1] = role
        tagp = {"trainer": "T", "rollout": "R", "bench-rollout": "B", "bench-trainer": "BT"}[role] + str(rank)
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + child_argv, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append((tagp, role, p))

        def pump():
            for line in p.stdout:
                sys.stdout.write(f"[{tagp}] {line}"); sys.stdout.flush()
        threading.Thread(target=pump, daemon=True).start()
        return p

    if bench:   # trainer bench on GPUs [0,Y) and rollout bench on [Y,N), concurrently, independent
        for k in range(Y):
            spawn("bench-trainer", k, k, Y)
        for r in range(X):
            spawn("bench-rollout", Y + r, r, X)
    else:
        for k in range(Y):
            spawn("trainer", k, k, Y)
        for r in range(X):
            spawn("rollout", Y + r, r, X)
    print(f"[launch] {len(procs)} processes on {n} GPUs: {[(t, role) for t, role, _ in procs]}", flush=True)
    rc = 0
    try:
        while True:
            alive = [(t, role, p) for t, role, p in procs if p.poll() is None]
            dead = [(t, role, p) for t, role, p in procs if p.poll() is not None]
            for t, role, p in dead:
                if p.returncode != 0 and (role != "rollout" or not _stop_requested(work)):
                    if bench:   # benches are independent: let the others finish, report the failure at the end
                        if getattr(p, "_reported", False) is False:
                            print(f"[launch] {t} ({role}) exited rc={p.returncode}", flush=True); p._reported = True
                        rc = rc or p.returncode
                    else:
                        print(f"[launch] {t} ({role}) exited rc={p.returncode} -> aborting", flush=True); rc = p.returncode
            if rc and not bench:
                break
            if bench:
                if not alive:
                    break
            else:
                if not [x for x in alive if x[1] == "trainer"]:
                    _atomic_write_text(f"{work}/STOP", "trainers exited")
                    t_end = time.time()
                    while any(p.poll() is None for _, _, p in procs) and time.time() - t_end < 60:
                        time.sleep(1)
                    break
            time.sleep(2)
    finally:
        for t, role, p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(5)
        for t, role, p in procs:
            if p.poll() is None:
                p.kill()
    print(f"[launch] done rc={rc}", flush=True)
    return rc


def main():
    argv = sys.argv[1:]
    a = parse_args(argv)
    if a.role in ("launch", "bench"):
        sys.exit(run_launch(a, argv))
    if a.role == "trainer":
        run_trainer(a)
    elif a.role == "rollout":
        run_rollout(a)
    elif a.role == "bench-rollout":
        run_bench_rollout(a)
    elif a.role == "bench-trainer":
        run_bench_trainer(a)


if __name__ == "__main__":
    main()
