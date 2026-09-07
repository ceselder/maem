"""Stage 4: pretrain the generator on the (direction, target_text) firehose.

Conditioning vector for each record is row `vec_idx` of the memmap vec bank (probe directions in
READ_LAYER residual space; vecs.f32 or vecs.f16, N x D_MODEL); injected at INJECT_LAYER at the
marker. Teacher-force the target.

    torchrun --standalone --nproc_per_node=8 scripts/pretrain.py --data-dir data/pretrain --epochs 1

Speed knobs (all exact -- none changes the optimization):
    --compile [--compile-mode default|max-autotune|reduce-overhead]   torch.compile the forward
    --grad-ckpt 0 --autocast-bf16                                     no recompute, bf16 LoRA matmuls
    --head-on-labels                                                  lm_head + CE only at label positions
    --grad-accum N                                                    N micro-batches per optimizer step
    --prefix-cache --prefix-share-step                               one prefix fwd/bwd per optimizer step
    --pad-multiple 1                                                  --prefix-cache: no suffix-length rounding (default 8 = ~15% pad)
Composition knob (NOT exact -- changes which examples share an optimizer step, not the objective):
    --length-bucket   i.i.d. example windows per optimizer step, length-sorted into micro-batches (see step_groups)
"""
import argparse
import contextlib
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from mxf.config import D_MODEL, INJECT_LAYER, MODEL, STEER_COEFF, TrainConfig
from mxf.inject import FixedPositionInjector, get_layer, hooked, make_inject_hook, make_packed_inject_hook
from mxf.mfu import mfu
from mxf.prompts import build_prompt_ids, build_sft_ids


@contextlib.contextmanager
def autocast_region(model, enabled):
    """--autocast-bf16: PEFT input-dtype casting off + torch.autocast(bf16) around the policy forward (mirrors
    rl_disagg._policy_precision). Off = no-op, byte-identical to the legacy path."""
    if not enabled:
        yield
        return
    try:
        from peft.helpers import disable_input_dtype_casting
        cm = disable_input_dtype_casting(model)
    except ImportError:
        cm = contextlib.nullcontext()
    with cm, torch.autocast("cuda", dtype=torch.bfloat16):
        yield


# ---- vector bank: vecs.f32 (legacy) or vecs.f16 (half the bytes), same N x D_MODEL row layout ----
VEC_BANK_FILES = (("vecs.f32", np.float32), ("vecs.f16", np.float16))


def open_vec_bank(data_dir, n_vecs):
    """Read-only memmap over whichever of vecs.f32 / vecs.f16 exists (f32 preferred if both)."""
    for fname, dt in VEC_BANK_FILES:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return np.memmap(path, dtype=dt, mode="r", shape=(n_vecs, D_MODEL)), fname
    raise FileNotFoundError(f"no vecs.f32 / vecs.f16 in {data_dir}")


def gather_rows(vecs, idx):
    """Bank rows -> float32 CPU tensor (f16 banks are upcast here; f32 rows are a plain copy)."""
    return torch.from_numpy(np.array(vecs[idx], dtype=np.float32))   # np.array = always a writable copy


def tokenize_records(records, tok, max_seq, chunk_size=4096):
    """Tokenize targets in batches and build the constant chat prompt only once.

    Same IDs/labels/truncation as build_sft_ids per record, without millions of repeated
    chat-template calls. Bounded tokenizer batches keep startup memory independent of
    the bank size beyond the tokenized rows the trainer already retains.
    """
    prompt, positions = build_prompt_ids(tok)
    prompt_labels = [-100] * len(prompt)
    rows = []
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        targets = tok([r["target_text"] for r in chunk], add_special_tokens=False,
                      padding=False, truncation=False)["input_ids"]
        for record, ids in zip(chunk, targets):
            target = list(ids) + [tok.eos_token_id]
            rows.append(((prompt + target)[:max_seq], (prompt_labels + target)[:max_seq],
                         positions, record["vec_idx"]))
    return rows


class LabelHeadLM(torch.nn.Module):
    """--head-on-labels: the transformer body, then lm_head + cross-entropy on ONLY the positions whose
    next-token label is not -100. Mean over label tokens = exactly HF ForCausalLM's loss (which computes
    full-vocab logits at every position, upcasts them to fp32 and masks ~85% of them). Wraps the PEFT
    model so DDP / no_sync / save_pretrained keep working on the same parameters; the frozen lm_head is
    applied under whatever autocast is active (bf16 matmul), then CE in fp32 -- as HF does."""

    def __init__(self, peft_model, ce_chunk=0):
        super().__init__()
        self.peft_model = peft_model
        self.ce_chunk = ce_chunk

    @staticmethod
    def _head_ce_sum(lm_head, h_rows, tgt_rows):
        return F.cross_entropy(lm_head(h_rows).float(), tgt_rows, reduction="sum")

    def hidden(self, input_ids, attention_mask, **body_kw):
        """Final-norm hidden states [B, L, d] from the transformer body (compiled if --compile)."""
        body_kw.setdefault("use_cache", False)
        return self.peft_model.get_base_model().model(
            input_ids=input_ids, attention_mask=attention_mask, **body_kw).last_hidden_state

    def loss_from_hidden(self, h, labels):
        lm_head = self.peft_model.get_base_model().lm_head
        tgt = labels[:, 1:]                                                # logits at t predict token t+1
        keep = tgt != -100
        h_sel, tgt_sel = h[:, :-1][keep], tgt[keep]                        # [n, d], [n]
        n = tgt_sel.numel()
        if self.ce_chunk and n > self.ce_chunk:
            # recompute each chunk's head+CE in the backward so peak memory is one chunk of fp32 logits
            total = sum(checkpoint(self._head_ce_sum, lm_head, h_sel[s : s + self.ce_chunk],
                                   tgt_sel[s : s + self.ce_chunk], use_reentrant=False)
                        for s in range(0, n, self.ce_chunk))
        else:
            total = self._head_ce_sum(lm_head, h_sel, tgt_sel)
        return total / n

    def forward(self, input_ids, attention_mask, labels, **body_kw):
        return self.loss_from_hidden(self.hidden(input_ids, attention_mask, **body_kw), labels)


@contextlib.contextmanager
def eager_forwards(compiled):
    """Temporarily undo `mod.forward = torch.compile(mod.forward)` for each (mod, original_forward) pair."""
    saved = [(m, m.forward) for m, _ in compiled]
    for m, f in compiled:
        m.forward = f
    try:
        yield
    finally:
        for m, f in saved:
            m.forward = f


@torch.no_grad()
def parity_check(peft_model, label_head, kw, compiled=(), tol=1e-3):
    """--parity-check on one batch (same hooks / autocast / weights). Three numbers:
      (1) eager HF ForCausalLM loss vs eager head-on-labels loss      -> asserted < tol
      (2) HF's own ForCausalLMLoss on the full logits vs the gathered CE, both from the SAME hidden states
          of the actual (compiled) training body                      -> asserted < tol (pure loss-math check)
      (3) the compiled head-on-labels training path vs (1)'s eager HF -> informational: torch.compile's bf16
          numerics, nothing to do with the loss formulation
    Eager-vs-eager matters: two different Dynamo graphs of a 27B bf16 body differ by ~1e-2 in loss."""
    from transformers.loss.loss_utils import ForCausalLMLoss

    with eager_forwards(compiled):
        loss_hf = peft_model(**kw).loss.float()
        loss_lh = label_head(**kw).float()
    d1 = (loss_hf - loss_lh).abs().item()
    base = peft_model.get_base_model()
    h = label_head.hidden(**{k: v for k, v in kw.items() if k != "labels"})
    loss_full = ForCausalLMLoss(base.lm_head(h), kw["labels"], base.config.vocab_size).float()
    loss_gath = label_head.loss_from_hidden(h, kw["labels"]).float()
    d2 = (loss_full - loss_gath).abs().item()
    d3 = (loss_gath - loss_hf).abs().item()
    ok = d1 < tol and d2 < tol
    print(f"[parity] (1) eager: hf {loss_hf.item():.6f} head_on_labels {loss_lh.item():.6f} |diff| {d1:.3e} | "
          f"(2) same hidden states: hf-formula {loss_full.item():.6f} gathered {loss_gath.item():.6f} |diff| {d2:.3e} | "
          f"(3) compiled-vs-eager |diff| {d3:.3e} (info) -> {'OK' if ok else 'FAIL'} @ tol {tol:g}", flush=True)
    assert ok, f"head-on-labels parity FAILED: eager |diff| {d1:.3e}, same-hidden |diff| {d2:.3e} (tol {tol:g})"
    return d1, d2, d3


def pack_examples(toks, pack_len, seed=0):
    """Greedy-pack (ids, labels, marker_pos, vec_idx) tuples end-to-end into blocks of <= pack_len
    tokens (example order shuffled once; an example that would overflow starts the next block, so
    examples are never split). Each block: ids/labels concat + per-example seg_lens, absolute
    marker positions, vec idxs."""
    order = np.random.default_rng(seed).permutation(len(toks))
    blocks, cur = [], None
    for i in order:
        ids, labs, pos, vidx = toks[i]
        if len(ids) > pack_len:
            continue
        if cur is None or len(cur["ids"]) + len(ids) > pack_len:
            if cur is not None:
                blocks.append(cur)
            cur = {"ids": [], "labels": [], "seg_lens": [], "markers": [], "vec_idxs": []}
        cur["markers"].append(len(cur["ids"]) + pos[0])
        cur["ids"] += ids
        cur["labels"] += labs
        cur["seg_lens"].append(len(ids))
        cur["vec_idxs"].append(vidx)
    if cur is not None and cur["seg_lens"]:
        blocks.append(cur)
    return blocks


def pack_batch(bblocks, pack_len, pad_id):
    """CPU tensors for a batch of packed blocks. seg = example index per token; tail pads get a
    unique seg id each (self-attention only, never attended by real tokens, labels -100).
    position_ids restart at 0 for every example. Returns per-marker (rows, cols) for injection."""
    B = len(bblocks)
    input_ids = torch.full((B, pack_len), pad_id, dtype=torch.long)
    labels = torch.full((B, pack_len), -100, dtype=torch.long)
    pos_ids = torch.zeros((B, pack_len), dtype=torch.long)
    seg = torch.arange(pack_len, dtype=torch.long).repeat(B, 1) + 1_000_000  # pads: isolated
    rows, cols, n_real = [], [], 0
    for b, blk in enumerate(bblocks):
        n = len(blk["ids"])
        n_real += n
        input_ids[b, :n] = torch.tensor(blk["ids"])
        labels[b, :n] = torch.tensor(blk["labels"])
        s = 0
        for j, sl in enumerate(blk["seg_lens"]):
            seg[b, s : s + sl] = j
            pos_ids[b, s : s + sl] = torch.arange(sl)
            s += sl
        rows += [b] * len(blk["markers"])
        cols += blk["markers"]
    return input_ids, labels, pos_ids, seg, torch.tensor(rows), torch.tensor(cols), n_real


def packed_attn_mask(seg, causal, dtype):
    """Additive [B,1,L,L] mask: 0 where (same example ∧ causal), finfo.min elsewhere. transformers
    returns already-4D masks as-is, so this reaches sdpa untouched."""
    allowed = (seg[:, None, :, None] == seg[:, None, None, :]) & causal
    zero = torch.zeros((), dtype=dtype, device=seg.device)
    neg = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=seg.device)
    return torch.where(allowed, zero, neg)


def step_groups(lengths, batch_size, grad_accum, epoch, length_bucket=False):
    """Micro-batch composition for one epoch of the per-example path: a list of optimizer-step groups, each a list
    of index arrays (micro-batches) into the per-rank example list. A pure function of (lengths, epoch, flags) with
    no hidden state, so --skip-steps fast-forwarding lands on exactly the micro-batches an uninterrupted run used,
    and every DDP rank (equal shards) gets the same group/micro-batch shapes.

    default (legacy, byte-identical order to the old inline code): stable-sort ALL examples by length, cut into
        batch_size micro-batches -- each one nearly length-homogeneous, so intra-micro-batch padding is already ~0
        (measured 0.3-1.2% before --pad-multiple rounding) -- shuffle the micro-batches, take grad_accum consecutive
        ones per step. Consequence: an optimizer step is grad_accum whole length-buckets, i.e. only grad_accum
        distinct target lengths per rank per step (one at grad_accum=1), and per-step token counts swing with them.
    length_bucket: shuffle EXAMPLES (seeded by epoch), cut the order into windows of grad_accum*batch_size -- one
        window = one optimizer step = an i.i.d. sample of the shard, so every step is a representative mix of
        lengths -- then stable-sort each window by length and cut it into grad_accum micro-batches. The multiset of
        examples per step is exactly the window (the un-sorted random order would give the same per-step set; only
        the grouping into micro-batches changes), and intra-micro-batch padding stays small because each micro-batch
        spans ~1/grad_accum of the window's length range (measured on 8-32-token targets: 2.5% at grad_accum 32,
        7% at 8, 12% at 4 with --pad-multiple 1; a randomly composed micro-batch pads 35-43%). Loss normalisation is
        untouched in both modes: each micro-batch's HF mean-over-its-target-tokens loss is scaled by 1/grad_accum.
    """
    lengths = np.asarray(lengths)
    n = len(lengths)
    if not length_bucket:
        order = np.argsort(lengths, kind="stable")                       # == list.sort(key=len): stable
        micro = [order[s : s + batch_size] for s in range(0, n, batch_size)]
        np.random.default_rng(epoch).shuffle(micro)                      # same draw as shuffling the old list of lists
        return [micro[s : s + grad_accum] for s in range(0, len(micro), grad_accum)]
    perm = np.random.default_rng(epoch).permutation(n)
    window = batch_size * grad_accum
    groups = []
    for s in range(0, n, window):
        w = perm[s : s + window]
        w = w[np.argsort(lengths[w], kind="stable")]                     # ties broken by position -> deterministic
        groups.append([w[k : k + batch_size] for k in range(0, len(w), batch_size)])
    return groups


def main():
    cfg = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--init-adapter", default=cfg.init_adapter)
    ap.add_argument("--save-dir", default=cfg.save_dir)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (effective batch = batch-size x world x N; each "
                         "micro-loss is scaled by 1/N; scheduler/ckpt/skip-steps count optimizer steps)")
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--max-seq", type=int, default=cfg.max_seq)
    ap.add_argument("--pack-len", type=int, default=0,
                    help="0 = per-example padded batches + compile (validated 57%% MFU, the default). "
                         ">0 packs into fixed blocks but REGRESSES on Blackwell (no flash-attn → dense "
                         "attn mask wastes off-block compute); only use with a block-sparse attn backend.")
    ap.add_argument("--pack-blocks", type=int, default=8,
                    help="packed blocks per device micro-batch (tokens/step = pack-blocks * pack-len)")
    ap.add_argument("--run-name", default=cfg.run_name)
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy (test injection still fires)")
    ap.add_argument("--compile-mode", default="default", choices=["default", "max-autotune", "reduce-overhead"],
                    help="torch.compile mode. reduce-overhead = CUDA graphs (sequences are rounded to multiples "
                         "of 64, so <=3 static shapes get recorded)")
    ap.add_argument("--grad-ckpt", type=int, default=1,
                    help="1 = gradient checkpointing (legacy default; +33%% recompute). 0 = off -- a 178 GB B200 holds batch 16 x "
                         "192 tokens without it (the RL trainer runs mb 12-16 at 295 tokens with no checkpointing).")
    ap.add_argument("--autocast-bf16", action="store_true",
                    help="bf16 LoRA matmuls/activations under torch.autocast with PEFT's fp32 input-dtype casting disabled "
                         "(fp32 LoRA masters unchanged; HF's loss still upcasts logits to fp32). Same region as rl_disagg.")
    ap.add_argument("--head-on-labels", action="store_true",
                    help="lm_head + fp32 CE only at label positions (HF computes 248k-vocab logits for every position, "
                         "~85%% of them masked). Identical loss; with --compile only the transformer body is compiled.")
    ap.add_argument("--ce-chunk", type=int, default=0,
                    help="--head-on-labels: rows of fp32 logits per recomputed chunk (0 = one chunk; ~1 MB/row)")
    ap.add_argument("--parity-check", action="store_true",
                    help="on the first batch, assert |HF loss - head-on-labels loss| < 1e-3 and print both")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="compute the shared prompt prefix (tokens before the marker) ONCE per micro-batch and run only "
                         "[marker]+target per example on top of the expanded cache (sft/prefix_cache.py). Exact incl. "
                         "gradients; needs the transformers fork ceselder/transformers@maemm-prefix-cache, --grad-ckpt 0 "
                         "and the per-example path (no --pack-len). Lets --batch-size go to 64-128 on one B200.")
    ap.add_argument("--prefix-accum", type=int, default=1,
                    help="with --prefix-cache: split each --batch-size batch into N micro-batches that SHARE one prefix "
                         "forward (token-weighted losses => identical to one big-batch mean-loss step). Amortizes the "
                         "B=1 prefix fwd+bwd (~225 ms/step on B200) and lowers peak memory: e.g. --batch-size 128 "
                         "--prefix-accum 2 fits one B200 where a single 128 micro-batch OOMs.")
    ap.add_argument("--prefix-share-step", action="store_true",
                    help="LoRA + --prefix-cache: share one prefix forward AND backward across all --grad-accum / "
                         "--prefix-accum micro-batches of an optimizer step. Accumulates cache gradients in fp32, "
                         "then reduces completed parameter gradients once. Same objective; bf16 reduction order differs.")
    ap.add_argument("--pad-multiple", type=int, default=8,
                    help="--prefix-cache: round each micro-batch's padded suffix length up to a multiple of N. The legacy 8 "
                         "pads ~15%% of suffix tokens on 8-32-token targets (micro-batches are already length-sorted, so "
                         "rounding is nearly ALL the padding); 1 = pad only to the longest suffix. Exact (pads carry no loss).")
    ap.add_argument("--length-bucket", action="store_true",
                    help="per-example path: compose each optimizer step from an i.i.d. window of grad-accum x batch-size "
                         "examples (seeded shuffle), length-sorted into its micro-batches (see step_groups). Default = the "
                         "legacy order: length-sorted micro-batches shuffled whole, so a step is only grad-accum distinct "
                         "lengths. Changes which examples share a step, not the loss formula; deterministic, so "
                         "--skip-steps resume stays exact. Not for --pack-len.")
    ap.add_argument("--fp8-base", action="store_true",
                    help="EXPERIMENTAL: run the frozen base linears in torchao float8 (fwd + grad_input GEMMs; LoRA A/B, "
                         "lm_head, embeddings and the 5120->48 GDN gates stay bf16). Recipe via MAEMM_FP8_RECIPE "
                         "(rowwise default | tensorwise). See sft/fp8.py.")
    ap.add_argument("--log-steps", type=int, default=20,
                    help="synchronize and report MFU every N optimizer steps (1 for trustworthy microbenchmarks)")
    ap.add_argument("--save-examples", default="",
                    help="comma-separated global example counts for scaling-curve checkpoints")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="0 = the whole bank; N = train on a seeded random subset of N records (equal per-rank shards). "
                         "Fixed data budget for lr / scaling sweeps; the OneCycle schedule spans the subset.")
    ap.add_argument("--max-examples-seed", type=int, default=0, help="seed of the --max-examples subset")
    ap.add_argument("--full-ft", action="store_true",
                    help="FULL fine-tuning (no LoRA): every weight trainable, FSDP2-sharded across the ranks with fp32 sharded "
                         "masters + AdamW states and bf16 compute (sft/fullft.py). --init-adapter then names a FULL model dir "
                         "(the base by default); checkpoints are full HF models in the base repo layout (bf16, ~54 GB each).")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--n-ckpts", type=int, default=0,
                    help=">0: save this many evenly-spaced checkpoints (every 100/N %% of training); else every 2000 steps")
    ap.add_argument("--skip-steps", type=int, default=0,
                    help="crash-resume: fast-forward N optimizer steps (scheduler + step counter advance, no "
                         "compute). Batch order is deterministic (seeded shuffle over identical data/world), so "
                         "pairing with --init-adapter <save-dir>/step_{N-1} resumes exactly; AdamW moments reset.")
    ap.add_argument("--wandb-id", default="", help="crash-resume: continue this wandb run id (resume='allow')")
    a = ap.parse_args()
    assert a.grad_accum >= 1, "--grad-accum must be >= 1"
    assert not a.prefix_share_step or (a.prefix_cache and not a.full_ft), \
        "--prefix-share-step requires --prefix-cache and LoRA (FSDP2/full-ft is not supported)"
    assert a.pad_multiple >= 1, "--pad-multiple must be >= 1"
    assert not (a.length_bucket and a.pack_len), "--length-bucket is for the per-example path (no --pack-len)"

    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0)); is_main = rank == 0
    if world > 1:
        dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl")); torch.cuda.set_device(local)
    device = f"cuda:{local}"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # stream + shard while reading: 20M-record banks would otherwise put 8 full copies of the record list in host RAM
    n_lines = sum(1 for _ in open(f"{a.data_dir}/records.jsonl", "rb"))
    n_use = min(n_lines, a.max_examples) if a.max_examples else n_lines
    keep = n_use // world                                     # equal shards (see below)
    if n_use < n_lines:   # --max-examples: seeded random subset of the bank, dealt round-robin into equal per-rank shards
        chosen = np.sort(np.random.default_rng(a.max_examples_seed).permutation(n_lines)[:n_use])
        mine = set(chosen[rank::world][:keep].tolist())
        take = mine.__contains__
        del chosen
    else:
        take = lambda i: i % world == rank
    records = []
    with open(f"{a.data_dir}/records.jsonl") as f:
        for i, l in enumerate(f):
            if take(i) and len(records) < keep:
                records.append(json.loads(l))
    vec_path, vec_bytes = next((os.path.join(a.data_dir, fn), np.dtype(dt).itemsize) for fn, dt in VEC_BANK_FILES
                               if os.path.exists(os.path.join(a.data_dir, fn)))
    n_vecs = os.path.getsize(vec_path) // (D_MODEL * vec_bytes)
    assert all(r["vec_idx"] < n_vecs for r in records), "records reference rows beyond the vector bank"
    vecs, vec_file = open_vec_bank(a.data_dir, n_vecs)
    if is_main:
        print(f"{len(records)*world} records, {n_vecs} vectors ({vec_file}), world={world}", flush=True)

    FT = None
    if a.full_ft:
        try:
            from sft import fullft as FT
        except ImportError:  # mounted next to this file (modal_sft.py puts both under /pmx/SL/)
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import fullft as FT
        assert world > 1, "--full-ft needs >= 2 ranks (fp32 masters + AdamW of a 27B do not fit one GPU)"
        assert not a.pack_len and not a.head_on_labels and not a.compile and not a.fp8_base, \
            "--full-ft supports the per-example path (optionally --prefix-cache) without --compile/--head-on-labels/--fp8-base"
        if a.autocast_bf16 and is_main:
            print("[fullft] --autocast-bf16 ignored: FSDP2 MixedPrecisionPolicy(param_dtype=bf16) already runs the compute in bf16", flush=True)
        a.autocast_bf16 = False
        src = a.init_adapter or MODEL   # a full model dir (our own checkpoints load like the base repo) or the base
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map={"": device})
        model = FT.shard_full_model(model, world, device, log=print if is_main else (lambda *x, **k: None))
        nontext_shard = "/tmp/base_nontext.safetensors"
        if is_main:
            FT.prepare_nontext_shard(MODEL, nontext_shard)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa",  # flash-attn has no sm_103 build
                                                     device_map={"": device})
        model.enable_input_require_grads()
        if a.init_adapter:
            model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, use_rslora=True,
                target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    if a.fp8_base:  # after PEFT (only frozen base_layers convert), before grad-ckpt / compile / DDP
        from fp8 import convert_frozen_base_to_fp8
        convert_frozen_base_to_fp8(model, verbose=is_main)
    # 27B/64-layer OOMs on the 178GB B200 without activation checkpointing (~176GB resident at any
    # batch). enable_input_require_grads() above is the prerequisite; recompute activations in the
    # backward to fit. use_reentrant=False is required for LoRA/frozen-base + checkpointing.
    # use_reentrant=True: the layer-1 injection forward-hook modifies activations, which breaks
    # non-reentrant checkpointing's forward-vs-recompute tensor-count determinism check. Reentrant
    # mode re-runs the forward without that check and tolerates the hook.
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()
    try:
        import fla  # noqa
        _fla = "v" + str(getattr(fla, "__version__", "?"))
    except Exception:  # noqa
        _fla = "ABSENT (torch GDN fallback -- slow)"
    if is_main:
        print(f"[pretrain] grad_ckpt={a.grad_ckpt} autocast_bf16={a.autocast_bf16} compile={a.compile} "
              f"compile_mode={a.compile_mode} head_on_labels={a.head_on_labels} grad_accum={a.grad_accum} "
              f"fla={_fla} gpu={torch.cuda.get_device_name(0)}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())  # full model incl. lm_head; LoRA adds <0.2%, fine for MFU
    submodule = get_layer(model, INJECT_LAYER)
    persistent_injector = persistent_handle = None
    if a.compile and not a.pack_len:
        # Install once before Dynamo's first trace. The buffer address remains stable while values
        # are copied in per batch, so injection stays in-graph without recompiling on every vector.
        _, _, fixed_positions = build_sft_ids(tok, "compile marker probe")
        assert len(fixed_positions) == 1
        persistent_injector = FixedPositionInjector(
            # prefix-cache path: the marker is index 0 of the suffix forward, not its absolute prompt index
            a.batch_size, D_MODEL, 0 if a.prefix_cache else fixed_positions[0], STEER_COEFF, device, torch.bfloat16
        )
        persistent_handle = submodule.register_forward_hook(persistent_injector.hook)
    # --head-on-labels: the gather has a data-dependent size, so only the transformer body is compiled
    # (one static graph per padded length); gather + lm_head + CE run eagerly on a few hundred rows.
    label_head = LabelHeadLM(model, a.ce_chunk)
    compiled = []   # (module, eager forward) pairs, so --parity-check can run the eager reference
    if a.compile:
        target = model.get_base_model().model if a.head_on_labels else model
        compiled.append((target, target.forward))
        # --prefix-cache: cache inputs vary per step, so the graph must be dynamic (measured: recompile-prone, not recommended)
        target.forward = torch.compile(target.forward, mode=a.compile_mode, dynamic=True if a.prefix_cache else None)
    train_mod = label_head if a.head_on_labels else model   # what DDP wraps / the loop calls
    if a.full_ft:
        ddp = model                                          # FSDP2 IS the parallelism (fully_shard in place; no wrapper)
    else:
        ddp = DDP(train_mod, device_ids=[local]) if world > 1 else train_mod
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    prefix_cache = None
    if a.prefix_cache:
        try:
            from sft.prefix_cache import PrefixCache, PrefixGradientAccumulator, sync_accumulated_gradients
        except ImportError:  # mounted next to this file (modal_sft.py puts both under /pmx/SL/)
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from prefix_cache import PrefixCache, PrefixGradientAccumulator, sync_accumulated_gradients
        assert not a.pack_len, "--prefix-cache is the per-example path (no --pack-len)"
        assert not a.head_on_labels, "--prefix-cache uses HF's loss on the short suffix (its head cost is already ~all labels); not combinable with --head-on-labels"
        assert not a.grad_ckpt, "--prefix-cache needs --grad-ckpt 0 (GradientCheckpointingLayer drops past_key_values)"
        prompt_ids, mpos = build_prompt_ids(tok)
        # suffix forward through `ddp` (primes DDP's reducer once per step), prefix forward through the bare model
        prefix_cache = PrefixCache(ddp, prompt_ids, mpos[0], tok.pad_token_id, submodule, STEER_COEFF, device,
                                   prefix_model=model, persistent_injector=persistent_injector, pad_multiple=a.pad_multiple,
                                   # FSDP2 registers backward hooks on the layer output: inject on a clone, never in place;
                                   # and its pre-backward unshard hangs on module outputs -> the prefix output needs a grad path
                                   inject_mode="add_clone" if a.full_ft else "add", keep_prefix_grad_path=a.full_ft)
        assert a.prefix_accum >= 1 and a.batch_size % a.prefix_accum == 0, "--prefix-accum must divide --batch-size"
        if a.prefix_share_step:
            # Sharing stochastic prefix computations changes the objective. This trainer creates
            # zero-dropout LoRAs, but reject a resumed adapter/base with active dropout as well.
            assert not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()), \
                "--prefix-share-step requires zero dropout"
        if is_main:
            print(f"[pretrain] prefix-cache ON: shared prefix {prefix_cache.prefix_len} tokens, suffix = "
                  f"{len(prefix_cache.suffix_prompt)} prompt token(s) + target, prefix shared by {a.prefix_accum} "
                  f"micro-batch(es) of {a.batch_size // a.prefix_accum}, suffix padded to a multiple of {a.pad_multiple}",
                  flush=True)
            if a.prefix_share_step:
                print("[pretrain] prefix-share-step ON: one prefix fwd/bwd per optimizer step; "
                      "fp32 cache-gradient accumulation, one parameter-gradient reduction", flush=True)

    # pre-tokenize once. Packed path: shuffle-once greedy packing into fixed pack-len blocks (zero
    # intra-block padding, one static shape for compile). Legacy path: length-bucketed padded batches.
    t_tokenize = time.time()
    toks_cache = tokenize_records(records, tok, a.max_seq)
    if is_main:
        print(f"[pretrain] tokenized {len(toks_cache)} local records in {time.time() - t_tokenize:.1f}s", flush=True)
    if a.pack_len:
        blocks = pack_examples(toks_cache, a.pack_len, seed=0)
        if world > 1:  # equalize block count across ranks (packing yields ±1 per rank → DDP deadlock)
            t = torch.tensor([len(blocks)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            blocks = blocks[: int(t.item())]
        bper = len(blocks) // a.pack_blocks  # drop remainder batch: keeps a single static shape
        micro_per_epoch = bper
        causal = torch.tril(torch.ones(a.pack_len, a.pack_len, dtype=torch.bool, device=device))
        if is_main:
            fill = sum(len(b["ids"]) for b in blocks) / (len(blocks) * a.pack_len)
            print(f"packed: {len(blocks)} blocks of {a.pack_len} (fill {fill:.1%}), "
                  f"{bper} micro-batches/epoch x {a.pack_blocks} blocks", flush=True)
    else:
        # micro-batch composition per epoch: step_groups (default = legacy length-sorted buckets; --length-bucket =
        # i.i.d. windows sorted within the step). Lengths are of the full prompt+target row (prompt is constant).
        ex_lengths = np.fromiter((len(t[0]) for t in toks_cache), dtype=np.int64, count=len(toks_cache))
        micro_per_epoch = math.ceil(len(toks_cache) / a.batch_size)
        if is_main:
            print(f"[pretrain] micro-batch composition: {'length-bucket (i.i.d. step windows, length-sorted micro-batches)' if a.length_bucket else 'legacy (length-sorted micro-batches, shuffled whole)'}", flush=True)
    # one optimizer step per --grad-accum micro-batches (the last group of an epoch may be shorter)
    steps_total = math.ceil(micro_per_epoch / a.grad_accum) * a.epochs
    # warmup >= 2 optimizer steps: OneCycleLR divides by (pct_start*total_steps - 1), which is 0 for runs of
    # < 100 steps at the 2% default (smoke tests / --grad-accum shrinking the step count); production unchanged.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps_total,
                                                pct_start=max(cfg.warmup_frac, 2.0 / steps_total),
                                                anneal_strategy="linear")
    save_every = max(1, steps_total // a.n_ckpts) if a.n_ckpts else 2000
    if is_main:
        print(f"steps_total {steps_total} ({micro_per_epoch} micro-batches/epoch, grad_accum {a.grad_accum}), "
              f"checkpoint every {save_every} steps", flush=True)
    if is_main and not a.no_wandb:
        wandb.init(project="maxact-fast", name=a.run_name, config=vars(a),
                   id=a.wandb_id or None, resume="allow" if a.wandb_id else None)
    os.makedirs(a.save_dir, exist_ok=True)
    save_examples = sorted({int(x) for x in a.save_examples.split(",") if x.strip()})
    saved_examples = set()

    def save_ckpt(path, at_step):
        """LoRA: rank 0 writes the adapter. --full-ft: COLLECTIVE full-model save (every rank all-gathers), rank 0 writes."""
        if a.full_ft:
            FT.save_full_ckpt(model, path, tok, MODEL, is_main, world, nontext_shard=nontext_shard, log=print,
                              extra_meta={"step": at_step, "run_name": a.run_name, "lr": a.lr, "full_ft": True})
        elif is_main:
            model.save_pretrained(path)

    step = 0
    parity_done = not a.parity_check
    t_train0 = time.time()
    for ep in range(a.epochs):
        if a.pack_len:
            order = np.random.default_rng(ep).permutation(len(blocks))
            micro = [[blocks[i] for i in order[s : s + a.pack_blocks]]
                     for s in range(0, bper * a.pack_blocks, a.pack_blocks)]
            groups = [micro[s : s + a.grad_accum] for s in range(0, len(micro), a.grad_accum)]
        else:
            groups = step_groups(ex_lengths, a.batch_size, a.grad_accum, ep, a.length_bucket)
        for group in groups:
            if step < a.skip_steps:  # crash-resume fast-forward (see --skip-steps help)
                sched.step(); step += 1
                continue
            log_now = is_main and step % a.log_steps == 0
            if log_now:
                torch.cuda.synchronize()  # drain queued work so the timed step is only this step
            t0 = time.time()
            n_real_step = n_ex_step = 0
            n_pad_real = n_pad_slots = 0     # padding accounting: real tokens vs padded slots in the padded tensors (prefix path: suffix only)
            loss_step = torch.zeros((), device=device)
            probe_ctx = (FT.injection_probe(model, INJECT_LAYER, log=print) if (a.full_ft and step == a.skip_steps and is_main)
                         else contextlib.nullcontext())
            probe_ctx.__enter__()
            shared_prefix = None
            if a.prefix_share_step:
                ac = lambda: autocast_region(model, a.autocast_bf16)
                shared_prefix = PrefixGradientAccumulator(prefix_cache.run_prefix(ac))
                n_real_step += prefix_cache.prefix_len
            for mi, batch in enumerate(group):
                last_in_group = mi == len(group) - 1
                if not a.pack_len:
                    batch = [toks_cache[i] for i in batch]      # index array -> (ids, labels, pos, vec_idx) rows
                if prefix_cache is not None:
                    # ---- --prefix-cache micro-batch: shared prompt prefix once, [marker]+target per example on the
                    # expanded cache (sft/prefix_cache.py; exact incl. gradients). toks_cache rows are prompt+target ids
                    # (truncated to max_seq), so the target is everything after the prompt. n_real = tokens actually run.
                    n_prompt = len(prompt_ids)
                    targets = [t[0][n_prompt:] for t in batch]
                    vmat = gather_rows(vecs, [t[3] for t in batch])
                    ac = lambda: autocast_region(model, a.autocast_bf16)  # noqa: E731
                    if shared_prefix is not None:
                        # All suffix backwards stop at the cache leaves. The prefix is traversed
                        # once below, before clipping/Adam. DDP must not reduce partial gradients.
                        n_tot = sum(len(t) for t in targets)
                        if n_tot == 0:
                            raise ValueError("prefix-cache batch has no target tokens; increase --max-seq")
                        mb = a.batch_size // a.prefix_accum
                        loss = torch.zeros((), device=device)
                        n_real = 0
                        for start in range(0, len(batch), mb):
                            sl = slice(start, start + mb)
                            ctx = ddp.no_sync() if world > 1 else contextlib.nullcontext()
                            with ctx:
                                out = prefix_cache.forward(vmat[sl], targets[sl], autocast=ac,
                                                           prefix_cache=shared_prefix.cache)
                                weight = out.n_target_tokens / n_tot
                                (out.loss * weight / len(group)).backward()
                            shared_prefix.accumulate()
                            loss += out.loss.detach() * weight
                            n_suf = int(out.suffix_mask.sum())
                            n_real += n_suf
                            n_pad_real += n_suf; n_pad_slots += out.suffix_len * out.suffix_mask.shape[0]
                            del out
                    elif a.prefix_accum == 1:
                        sync_ctx = ddp.no_sync() if (world > 1 and not last_in_group) else contextlib.nullcontext()
                        with sync_ctx:
                            out = prefix_cache.forward(vmat, targets, autocast=ac)
                            loss = out.loss
                            (loss / len(group)).backward()
                        n_suf = int(out.suffix_mask.sum())
                        n_pad_real += n_suf; n_pad_slots += out.suffix_len * len(batch)
                        n_real = prefix_cache.prefix_len + n_suf
                    else:
                        # one prefix forward, N micro-batches on copy-expanded caches; loss_k * (n_k / N_tokens) summed
                        # == the single mean-over-target-tokens loss, so the update is identical to one big micro-batch
                        mb = len(batch) // a.prefix_accum
                        n_tot = sum(len(t) for t in targets)
                        cache0 = prefix_cache.run_prefix(ac)
                        loss_val, n_real = 0.0, prefix_cache.prefix_len
                        for k in range(a.prefix_accum):
                            last = k == a.prefix_accum - 1
                            sl = slice(k * mb, len(batch) if last else (k + 1) * mb)
                            # DDP: all-reduce only on the very last backward of the optimizer step
                            ctx = ddp.no_sync() if (world > 1 and not (last and last_in_group)) else contextlib.nullcontext()
                            with ctx:
                                o = prefix_cache.forward(vmat[sl], targets[sl], autocast=ac, prefix_cache=cache0)
                                (o.loss * (o.n_target_tokens / n_tot) / len(group)).backward(retain_graph=not last)
                            loss_val += o.loss.item() * o.n_target_tokens / n_tot
                            n_suf = int(o.suffix_mask.sum())
                            n_real += n_suf
                            n_pad_real += n_suf; n_pad_slots += o.suffix_len * o.suffix_mask.shape[0]
                            del o
                        del cache0
                        loss = torch.tensor(loss_val, device=device)
                    n_ex_step += len(batch)
                    n_real_step += n_real
                    loss_step += loss.detach() / len(group)
                    continue
                # ---- one micro-batch: CPU tensors + the injection hook for its vectors ----
                if a.pack_len:
                    input_ids, labels, pos_ids, seg, rows, cols, n_real = pack_batch(
                        batch, a.pack_len, tok.pad_token_id)
                    mask4 = packed_attn_mask(seg.to(device), causal, torch.bfloat16)
                    vmat = gather_rows(vecs, [v for blk in batch for v in blk["vec_idxs"]])
                    hook_ctx = hooked(submodule, make_packed_inject_hook(
                        vmat, rows, cols, STEER_COEFF, device, torch.bfloat16))
                    kw = dict(input_ids=input_ids.to(device), attention_mask=mask4,
                              position_ids=pos_ids.to(device), labels=labels.to(device), use_cache=False)
                    n_ex_step += sum(len(blk["seg_lens"]) for blk in batch)
                    n_pad_real += n_real; n_pad_slots += len(batch) * a.pack_len
                else:
                    L = max(len(t[0]) for t in batch)
                    L = min(((L + 63) // 64) * 64, a.max_seq)  # round to mult-of-64 → ≤3 static shapes for compile
                    input_ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
                    labels = torch.full((len(batch), L), -100, dtype=torch.long)
                    attn = torch.zeros((len(batch), L), dtype=torch.bool)
                    pos = batch[0][2]
                    for i, (ii, ll, _, _) in enumerate(batch):
                        input_ids[i, : len(ii)] = torch.tensor(ii)
                        labels[i, : len(ll)] = torch.tensor(ll)
                        attn[i, : len(ii)] = True
                    n_real = int(attn.sum())
                    n_pad_real += n_real; n_pad_slots += len(batch) * L      # includes the multiple-of-64 rounding
                    if persistent_injector is not None:
                        # One memmap gather + H2D copy, versus one tiny transfer per row in the legacy
                        # hook. The registered hook reads this stable buffer inside the compiled graph.
                        persistent_injector.set_vectors(gather_rows(vecs, [t[3] for t in batch]).to(device))
                        hook_ctx = contextlib.nullcontext()
                    else:
                        vlist = [gather_rows(vecs, t[3]).unsqueeze(0) for t in batch]
                        hook_ctx = hooked(submodule, make_inject_hook(
                            vlist, [pos] * len(batch), STEER_COEFF, device, torch.bfloat16,
                            mode="add_clone" if a.full_ft else "add"))
                    # use_cache=False explicitly: HF's default (None -> config True) builds a DynamicCache every
                    # training forward and routes the GDN conv through the cache pad+slice path.
                    kw = dict(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                              labels=labels.to(device), use_cache=False)
                    n_ex_step += len(batch)
                n_real_step += n_real
                # ---- forward/backward. DDP all-reduces grads only on the group's last micro-batch. ----
                sync_ctx = ddp.no_sync() if (world > 1 and mi < len(group) - 1) else contextlib.nullcontext()
                with hook_ctx, autocast_region(model, a.autocast_bf16), sync_ctx:
                    if not parity_done:
                        parity_check(model, label_head, kw, compiled); parity_done = True
                    out = ddp(**kw)
                    loss = out if a.head_on_labels else out.loss
                    (loss / len(group)).backward()
                loss_step += loss.detach() / len(group)
            if shared_prefix is not None:
                ctx = ddp.no_sync() if world > 1 else contextlib.nullcontext()
                with ctx:
                    shared_prefix.backward()
                sync_accumulated_gradients(model.parameters())
            probe_ctx.__exit__(None, None, None)
            if a.full_ft:
                FT.clip_grad_norm([p for p in model.parameters() if p.requires_grad], 1.0)
            else:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if log_now:
                torch.cuda.synchronize()
                dt = time.time() - t0
                # MFU = 6·N·(real tokens this rank, all micro-batches) / step wall time. N is the full model
                # (lm_head included), so with --head-on-labels the skipped head FLOPs (~2-3% of 6ND) are
                # still credited -- the number is throughput in "nominal 6ND" units, not hardware FLOPs.
                tfl, m = mfu(n_real_step, dt, n_params, fwd_bwd=True)
                peak_gb = torch.cuda.max_memory_allocated() / 2**30
                pad_frac = 1.0 - n_pad_real / max(n_pad_slots, 1)   # fraction of padded-tensor slots that were padding
                print(f"ep{ep} step {step}/{steps_total} loss {loss_step.item():.4f} | "
                      f"{tfl:.0f} TFLOP/s MFU {m:.0%} | {n_ex_step / dt:.2f} ex/s {n_real_step / dt:.0f} tok/s "
                      f"pad {pad_frac:.1%} ({dt:.3f} s/step, {len(group)} micro) | peak {peak_gb:.1f} GB", flush=True)
                if not a.no_wandb:
                    wandb.log({"loss": loss_step.item(), "lr": sched.get_last_lr()[0], "mfu": m, "tflops": tfl,
                               "ex_per_s": n_ex_step / dt, "tok_per_s": n_real_step / dt, "pad_frac": pad_frac,
                               "peak_mem_gb": peak_gb}, step=step)
            if step % save_every == 0 and step:
                save_ckpt(f"{a.save_dir}/step_{step}", step)
            global_seen = min((step + 1) * a.batch_size * a.grad_accum * world, len(records) * world)
            for requested in save_examples:
                if requested <= global_seen and requested not in saved_examples:
                    path = f"{a.save_dir}/examples_{requested}"
                    save_ckpt(path, step + 1)
                    if is_main:
                        json.dump({"requested_examples": requested, "actual_examples": global_seen,
                                   "step": step + 1, "world_size": world,
                                   "batch_size_per_rank": a.batch_size, "grad_accum": a.grad_accum},
                                  open(f"{path}/training_progress.json", "w"), indent=2)
                        print(f"SAVED_SCALING_POINT requested={requested} actual={global_seen} "
                              f"step={step + 1}", flush=True)
                    saved_examples.add(requested)
            step += 1
    save_ckpt(f"{a.save_dir}/final", step)
    if is_main:
        print(f"[pretrain] {step} optimizer steps in {time.time() - t_train0:.0f} s | peak GPU memory: "
              f"allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
              f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)
        print("PRETRAIN_DONE", flush=True)
    if persistent_handle is not None:
        persistent_handle.remove()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
