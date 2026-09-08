"""Tensor parallelism (TP) for the --full-ft path of sft/pretrain.py: Qwen3.5/3.6 hybrid (gated-delta-net + GQA) models
sharded over the GPUs of ONE node, every rank running ALL tokens of the micro-batch through a 1/tp slice of every matmul.

Why (vs FSDP2, sft/fullft.py): the prefix-cached SFT micro-batch is B suffixes x ~21 tokens. Under FSDP2 each rank
runs its own B=64 rows (~1,350 tokens) through the FULL model, re-gathering 54 GB of bf16 weights per micro-batch, and
memory forbids larger B. Under TP every rank runs the whole micro-batch (B=256-512, ~5-11k tokens) through 1/tp of the
weights: matmul M grows tp-fold, no weight gathers, optimizer state covers the local shard only, and the wide activations
(MLP intermediate, attention / GDN heads, logits) are sharded tp-fold. Only the d_model-wide residual stream is replicated.

Sharding (per decoder layer; tp must divide every head count / width; kv heads may be FEWER than tp -> replicated in
groups of tp // n_kv, see docs/fullft_tp.md for the full table):
    MLP        gate_proj, up_proj column-parallel (rows of W)      down_proj row-parallel (cols) + all-reduce
    GDN        in_proj_qkv column-parallel BY HEAD (q/k/v row blocks sliced consistently), conv1d channels likewise,
               in_proj_z / in_proj_b / in_proj_a / A_log / dt_bias by value head, out_proj row-parallel + all-reduce;
               the gated RMSNorm weight (per head_v_dim, shared by all heads) is replicated with a summed gradient.
               The recurrent state [B, Hv/tp, dk, dv] and conv state [B, conv_dim/tp, K] a rank produces ARE the
               per-rank slices of the full states -> the prefix cache (sft/prefix_cache.py) works unchanged.
    attention  q_proj (q + gate, per head) column-parallel, k_proj / v_proj by kv head (or replicated per kv group when
               n_kv < tp, gradient summed over the group), o_proj row-parallel + all-reduce; q_norm / k_norm weights
               replicated with a summed gradient.
    embed      vocab-parallel rows, output all-reduced.      lm_head  vocab-parallel + vocab-parallel cross-entropy
    norms      input / post-attention / final RMSNorm replicated (their gradient is already complete on every rank).
Communication per layer per micro-batch: 2 all-reduces of [tokens, d_model] bf16 in the forward (after the mixer's
out projection and after down_proj) and 2 in the backward (the input of the column-parallel projections), plus the
same pair for embed / lm_head. All collectives are autograd functions (Megatron's f / g operators), no DTensor.

Precision (== FSDP2's MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)): the module holds bf16 params and runs
bf16 compute; each backward's bf16 parameter gradient is added into an fp32 accumulator by a post-accumulate-grad hook
(so grad accumulation across micro-batches is fp32); fp32 master weights + AdamW states cover the local shard only;
after the step the masters are copied back into the bf16 params.

Checkpoints: `gather_full` all-gathers each shard back into the full tensor, so fullft.save_full_ckpt writes the same
base-repo-layout HF directory the FSDP2 path writes (loads in vanilla transformers / vLLM / the evaluator).
"""
import contextlib
import math
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------------------------------------------
# collectives as autograd functions (Megatron's f / g)
# ---------------------------------------------------------------------------------------------------------------
class _CopyToTP(torch.autograd.Function):
    """Identity in the forward; all-reduce (sum) of the gradient in the backward. Marks the input of a column-parallel
    region: every rank consumes the replicated activation, each produces a partial input-gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.group)
        return g, None


class _ReduceFromTP(torch.autograd.Function):
    """All-reduce (sum) in the forward; identity in the backward. Closes a row-parallel region: every rank holds a
    partial output, the sum is the full one; the gradient of a sum is the same on every rank."""

    @staticmethod
    def forward(ctx, x, group):
        y = x.contiguous().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def backward(ctx, g):
        return g, None


def copy_to_tp(x, group):
    return _CopyToTP.apply(x, group) if dist.get_world_size(group) > 1 else x


def reduce_from_tp(x, group):
    return _ReduceFromTP.apply(x, group) if dist.get_world_size(group) > 1 else x


# ---------------------------------------------------------------------------------------------------------------
# shard bookkeeping
# ---------------------------------------------------------------------------------------------------------------
class Shard:
    """Layout of ONE parameter across the tp ranks.

    dim: sharded dimension of the FULL tensor (None = replicated). index[r]: LongTensor of full-tensor indices along
    `dim` that rank r holds (all ranks hold the same count; ranks may hold identical indices = group replication).
    grad_group / grad_op: collective to complete this rank's gradient after the backward (None = already complete).
    clip_weight: this rank's local grad^2 enters the global clip norm with this weight (1 for a true shard, 1/rep for a
    param replicated across `rep` ranks whose gradients were summed, 0 for fully replicated params -- counted once via
    `replicated_full`).
    """

    def __init__(self, full_shape, dim, index, grad_group=None, grad_op=None, clip_weight=1.0, replicated_full=False):
        self.full_shape = tuple(full_shape)
        self.dim = dim
        self.index = index
        self.grad_group = grad_group
        self.grad_op = grad_op
        self.clip_weight = clip_weight
        self.replicated_full = replicated_full

    @property
    def replicated(self):
        return self.dim is None


def _chunk_index(n, tp, r):
    assert n % tp == 0, f"{n} not divisible by tp={tp}"
    c = n // tp
    return torch.arange(r * c, (r + 1) * c)


class TPModel:
    """Handle returned by shard_model_tp: process groups, per-parameter Shard specs, fp32 masters + accumulators."""

    def __init__(self, model, tp, group, rank):
        self.model, self.tp, self.group, self.rank = model, tp, group, rank
        self.specs = {}           # param name -> Shard
        self.masters = []         # list of (name, bf16 param, fp32 master, Shard)
        self._hooks = []
        self.vocab_start = None
        self.vocab_local = None

    # ---- gradient plumbing ----
    def build_masters(self):
        """fp32 master copy of every (local) parameter; bf16 grads are accumulated into master.grad (fp32) by a
        post-accumulate-grad hook on the bf16 param, exactly like FSDP2's fp32 reduce into the sharded grads."""
        self.masters = []
        for name, p in self.model.named_parameters():
            spec = self.specs[name]
            m = p.detach().float().clone().requires_grad_(True)
            m.grad = None

            def make_hook(master):
                def hook(param):
                    if param.grad is None:
                        return
                    if master.grad is None:
                        master.grad = param.grad.detach().float()
                    else:
                        master.grad.add_(param.grad.detach())
                    param.grad = None
                return hook

            self._hooks.append(p.register_post_accumulate_grad_hook(make_hook(m)))
            self.masters.append((name, p, m, spec))
        return [m for _, _, m, _ in self.masters]

    def master_params(self):
        return [m for _, _, m, _ in self.masters]

    @torch.no_grad()
    def reduce_grads(self):
        """Complete the gradients that are partial on this rank: group-replicated params (kv heads shared by tp // n_kv
        ranks; per-head norm weights shared by all heads) are summed over their group. Fully replicated params
        (input / post-attention / final norms) already carry the complete, identical gradient on every rank; they are
        averaged anyway (an exact no-op when identical) so no rank can drift."""
        for name, p, m, spec in self.masters:
            if m.grad is None:
                continue
            if spec.grad_group is not None:
                dist.all_reduce(m.grad, op=spec.grad_op, group=spec.grad_group)
            elif spec.replicated_full:   # SUM then divide: ReduceOp.AVG is NCCL-only (the tests run on gloo)
                dist.all_reduce(m.grad, op=dist.ReduceOp.SUM, group=self.group)
                m.grad.div_(self.tp)

    @torch.no_grad()
    def clip_grad_norm(self, max_norm):
        """Global L2 norm over the FULL parameter set from the local shards (torch.nn.utils.clip_grad_norm_ semantics,
        coefficient max_norm / (norm + 1e-6) clamped to 1). Returns the norm as a python float."""
        dev = self.masters[0][2].device
        local_sq = torch.zeros((), dtype=torch.float32, device=dev)
        rep_sq = torch.zeros((), dtype=torch.float32, device=dev)
        for name, p, m, spec in self.masters:
            if m.grad is None:
                continue
            s = m.grad.pow(2).sum()
            if spec.replicated_full:
                rep_sq += s
            else:
                local_sq += s * spec.clip_weight
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=self.group)
        total = (local_sq + rep_sq).sqrt()
        coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
        for name, p, m, spec in self.masters:
            if m.grad is not None:
                m.grad.mul_(coef)
        return float(total)

    @torch.no_grad()
    def sync_params(self):
        """masters (fp32, updated by the optimizer) -> bf16 module params."""
        for name, p, m, spec in self.masters:
            p.copy_(m)

    @torch.no_grad()
    def zero_grad(self):
        for name, p, m, spec in self.masters:
            m.grad = None
            p.grad = None

    # ---- checkpoint gathering ----
    @torch.no_grad()
    def gather_full(self, name, p):
        """COLLECTIVE: the full tensor of parameter `name` (every rank returns it). Replicated params are local."""
        spec = self.specs[name]
        loc = p.detach().contiguous()
        if spec.replicated:
            return loc
        shards = [torch.empty_like(loc) for _ in range(self.tp)]
        dist.all_gather(shards, loc, group=self.group)
        full = torch.empty(spec.full_shape, dtype=loc.dtype, device=loc.device)
        for r in range(self.tp):
            full.index_copy_(spec.dim, spec.index[r].to(loc.device), shards[r])
        return full

    def local_param_count(self):
        return sum(p.numel() for p in self.model.parameters())


# ---------------------------------------------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------------------------------------------
def _shard_param(tpm, module, attr, prefix, dim, index_fn, **kw):
    """Replace module.<attr> by this rank's slice (index_fn(r) -> indices along `dim` held by rank r)."""
    p = getattr(module, attr)
    index = [index_fn(r) for r in range(tpm.tp)]
    n = len(index[0])
    assert all(len(ix) == n for ix in index), f"{prefix}.{attr}: unequal shard sizes"
    local = p.detach().index_select(dim, index[tpm.rank].to(p.device)).contiguous()
    new = nn.Parameter(local, requires_grad=True)
    setattr(module, attr, new)
    tpm.specs[f"{prefix}.{attr}"] = Shard(p.shape, dim, index, **kw)
    del p


def _replicate_param(tpm, module, attr, prefix, grad_group=None, grad_op=None, replicated_full=False):
    p = getattr(module, attr)
    p.requires_grad_(True)
    tpm.specs[f"{prefix}.{attr}"] = Shard(p.shape, None, None, grad_group=grad_group, grad_op=grad_op,
                                          clip_weight=0.0, replicated_full=replicated_full)


def _install_copy_pre_hook(module, group):
    def pre(mod, args, kwargs):
        if args:
            return (copy_to_tp(args[0], group),) + tuple(args[1:]), kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = copy_to_tp(kwargs["hidden_states"], group)
        return args, kwargs
    return module.register_forward_pre_hook(pre, with_kwargs=True)


def _install_reduce_hook(module, group):
    return module.register_forward_hook(lambda mod, inp, out: reduce_from_tp(out, group))


def _fused_rmsnorm_available():
    try:
        from fla.modules.layernorm import rms_norm  # noqa: F401
        return True
    except Exception:  # noqa
        return False


def install_fused_rmsnorm(norm_modules):
    """Replace the eager Qwen3_5RMSNorm forward (x.float() * rsqrt(mean x^2 + eps) * (1 + w.float()) -> x.dtype; autograd
    saves ~4 fp32 copies of [tokens, d]) by fla's Triton RMSNorm (same fp32 math, one rounding to x.dtype, saves x + rstd).
    (1 + w) is formed in fp32 so the weight keeps HF's zero-centred parametrisation; its gradient flows to w unchanged.
    Returns an undo callable. Under TP the residual stream is replicated, so every rank runs the norm over ALL tokens:
    the eager norm would cost ~8x FSDP2's per-rank norm time and memory -- this brings both back down."""
    from fla.modules.layernorm import rms_norm

    saved = []
    for mod in norm_modules:
        eps = mod.eps

        def forward(x, _m=mod, _eps=eps):
            return rms_norm(x, 1.0 + _m.weight.float(), None, eps=_eps)

        saved.append((mod, mod.__dict__.get("forward")))
        mod.forward = forward

    def undo():
        for mod, f in saved:
            if f is None:
                mod.__dict__.pop("forward", None)
            else:
                mod.forward = f
    return undo


def shard_model_tp(model, tp, device, log=print, fused_norm=True, group=None):
    """In-place: shard a loaded (bf16) Qwen3_5ForCausalLM over `tp` ranks (== dist world size unless `group` is given),
    install the TP collectives, the vocab-parallel embedding / LM head + parallel cross-entropy, and (GPU + fla) fused
    RMSNorms. Returns a TPModel whose `build_masters()` gives the fp32 optimizer params."""
    if group is None:
        group = dist.group.WORLD
    world = dist.get_world_size(group)
    assert world == tp, f"--tp {tp} needs exactly tp ranks in the TP group (got {world}); DP x TP is not implemented"
    rank = dist.get_rank(group)
    tpm = TPModel(model, tp, group, rank)
    cfg = model.config
    base = model.model
    layers = base.layers
    t0 = time.time()
    for p in model.parameters():
        p.requires_grad_(True)

    # ---- kv-head groups (attention with n_kv < tp): ranks r with the same r // rep share kv head r // rep ----
    H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    kv_groups = None
    if Hkv % tp != 0:
        assert tp % Hkv == 0, f"tp={tp} must divide or be a multiple of num_key_value_heads={Hkv}"
        rep = tp // Hkv
        kv_groups = [dist.new_group(list(range(h * rep, (h + 1) * rep))) if world > 1 else group for h in range(Hkv)]

    # ---- embeddings (vocab-parallel) ----
    V, d = base.embed_tokens.weight.shape
    _shard_param(tpm, base.embed_tokens, "weight", "model.embed_tokens", 0, lambda r: _chunk_index(V, tp, r))
    tpm.vocab_local = V // tp
    tpm.vocab_start = rank * tpm.vocab_local
    emb = base.embed_tokens
    pad_idx = emb.padding_idx
    v0, vl = tpm.vocab_start, tpm.vocab_local

    def embed_forward(input_ids):
        mask = (input_ids >= v0) & (input_ids < v0 + vl)
        local = (input_ids - v0).masked_fill(~mask, 0)
        lp = (pad_idx - v0) if (pad_idx is not None and v0 <= pad_idx < v0 + vl) else None
        e = F.embedding(local, emb.weight, padding_idx=lp)
        e = e.masked_fill(~mask[..., None], 0)
        return reduce_from_tp(e, group)

    emb.forward = embed_forward
    emb.num_embeddings = vl

    # ---- decoder layers ----
    n_gdn = n_attn = 0
    norm_modules = [base.norm]
    for i, layer in enumerate(layers):
        pre = f"model.layers.{i}"
        norm_modules += [layer.input_layernorm, layer.post_attention_layernorm]
        _replicate_param(tpm, layer.input_layernorm, "weight", f"{pre}.input_layernorm", replicated_full=True)
        _replicate_param(tpm, layer.post_attention_layernorm, "weight", f"{pre}.post_attention_layernorm", replicated_full=True)
        # MLP
        mlp = layer.mlp
        I = mlp.gate_proj.weight.shape[0]
        _shard_param(tpm, mlp.gate_proj, "weight", f"{pre}.mlp.gate_proj", 0, lambda r: _chunk_index(I, tp, r))
        _shard_param(tpm, mlp.up_proj, "weight", f"{pre}.mlp.up_proj", 0, lambda r: _chunk_index(I, tp, r))
        _shard_param(tpm, mlp.down_proj, "weight", f"{pre}.mlp.down_proj", 1, lambda r: _chunk_index(I, tp, r))
        for lin in (mlp.gate_proj, mlp.up_proj):
            lin.out_features = I // tp
        mlp.down_proj.in_features = I // tp
        mlp.intermediate_size = I // tp
        tpm._hooks.append(_install_copy_pre_hook(mlp, group))
        tpm._hooks.append(_install_reduce_hook(mlp.down_proj, group))
        if hasattr(layer, "linear_attn"):
            n_gdn += 1
            m = layer.linear_attn
            K, Vd, Hk, Hv = m.key_dim, m.value_dim, m.num_k_heads, m.num_v_heads
            assert K % tp == 0 and Vd % tp == 0 and Hk % tp == 0 and Hv % tp == 0, \
                f"GDN dims not divisible by tp={tp}: key_dim {K} value_dim {Vd} k_heads {Hk} v_heads {Hv}"
            Kl, Vl = K // tp, Vd // tp

            def qkv_index(r, K=K, Kl=Kl, Vl=Vl):
                return torch.cat([torch.arange(r * Kl, (r + 1) * Kl), K + torch.arange(r * Kl, (r + 1) * Kl),
                                  2 * K + torch.arange(r * Vl, (r + 1) * Vl)])

            mp = f"{pre}.linear_attn"
            _shard_param(tpm, m.in_proj_qkv, "weight", f"{mp}.in_proj_qkv", 0, qkv_index)
            _shard_param(tpm, m.conv1d, "weight", f"{mp}.conv1d", 0, qkv_index)
            if m.conv1d.bias is not None:
                _shard_param(tpm, m.conv1d, "bias", f"{mp}.conv1d", 0, qkv_index)
            _shard_param(tpm, m.in_proj_z, "weight", f"{mp}.in_proj_z", 0, lambda r: _chunk_index(Vd, tp, r))
            _shard_param(tpm, m.in_proj_b, "weight", f"{mp}.in_proj_b", 0, lambda r: _chunk_index(Hv, tp, r))
            _shard_param(tpm, m.in_proj_a, "weight", f"{mp}.in_proj_a", 0, lambda r: _chunk_index(Hv, tp, r))
            _shard_param(tpm, m, "dt_bias", mp, 0, lambda r: _chunk_index(Hv, tp, r))
            _shard_param(tpm, m, "A_log", mp, 0, lambda r: _chunk_index(Hv, tp, r))
            _shard_param(tpm, m.out_proj, "weight", f"{mp}.out_proj", 1, lambda r: _chunk_index(Vd, tp, r))
            # gated RMSNorm weight [head_v_dim] is shared by every value head: partial grad on each rank -> sum
            _replicate_param(tpm, m.norm, "weight", f"{mp}.norm", grad_group=group, grad_op=dist.ReduceOp.SUM)
            m.num_v_heads, m.num_k_heads = Hv // tp, Hk // tp
            m.key_dim, m.value_dim = Kl, Vl
            m.conv_dim = 2 * Kl + Vl
            m.conv1d.in_channels = m.conv1d.out_channels = m.conv1d.groups = m.conv_dim
            m.in_proj_qkv.out_features = m.conv_dim
            m.in_proj_z.out_features = Vl
            m.in_proj_b.out_features = m.in_proj_a.out_features = Hv // tp
            m.out_proj.in_features = Vl
            tpm._hooks.append(_install_copy_pre_hook(m, group))
            tpm._hooks.append(_install_reduce_hook(m.out_proj, group))
        elif hasattr(layer, "self_attn"):
            n_attn += 1
            m = layer.self_attn
            hd = m.head_dim
            assert H % tp == 0, f"num_attention_heads {H} not divisible by tp={tp}"
            mp = f"{pre}.self_attn"
            _shard_param(tpm, m.q_proj, "weight", f"{mp}.q_proj", 0, lambda r: _chunk_index(H * hd * 2, tp, r))
            if m.q_proj.bias is not None:
                _shard_param(tpm, m.q_proj, "bias", f"{mp}.q_proj", 0, lambda r: _chunk_index(H * hd * 2, tp, r))
            _shard_param(tpm, m.o_proj, "weight", f"{mp}.o_proj", 1, lambda r: _chunk_index(H * hd, tp, r))
            if Hkv % tp == 0:
                kv_index = lambda r: _chunk_index(Hkv * hd, tp, r)   # noqa: E731
                kv_kw = {}
                m.num_key_value_groups = (H // tp) // (Hkv // tp)
            else:
                rep = tp // Hkv

                def kv_index(r, hd=hd, rep=rep):
                    h = r // rep
                    return torch.arange(h * hd, (h + 1) * hd)

                kv_kw = dict(grad_group=kv_groups[rank // rep], grad_op=dist.ReduceOp.SUM, clip_weight=1.0 / rep)
                m.num_key_value_groups = H // tp
            for name in ("k_proj", "v_proj"):
                lin = getattr(m, name)
                _shard_param(tpm, lin, "weight", f"{mp}.{name}", 0, kv_index, **kv_kw)
                if lin.bias is not None:
                    _shard_param(tpm, lin, "bias", f"{mp}.{name}", 0, kv_index, **kv_kw)
                lin.out_features = len(kv_index(rank))
            m.q_proj.out_features = H * hd * 2 // tp
            m.o_proj.in_features = H * hd // tp
            _replicate_param(tpm, m.q_norm, "weight", f"{mp}.q_norm", grad_group=group, grad_op=dist.ReduceOp.SUM)
            _replicate_param(tpm, m.k_norm, "weight", f"{mp}.k_norm", grad_group=group, grad_op=dist.ReduceOp.SUM)
            tpm._hooks.append(_install_copy_pre_hook(m, group))
            tpm._hooks.append(_install_reduce_hook(m.o_proj, group))
        else:
            raise RuntimeError(f"layer {i}: neither linear_attn nor self_attn")

    # ---- final norm + LM head (vocab-parallel) + parallel cross-entropy ----
    _replicate_param(tpm, base.norm, "weight", "model.norm", replicated_full=True)
    _shard_param(tpm, model.lm_head, "weight", "lm_head", 0, lambda r: _chunk_index(V, tp, r))
    model.lm_head.out_features = V // tp
    install_tp_causal_lm_forward(model, tpm)

    missing = [n for n, _ in model.named_parameters() if n not in tpm.specs]
    assert not missing, f"parameters without a TP shard spec: {missing[:5]}"

    fused = False
    if fused_norm and str(device).startswith("cuda") and _fused_rmsnorm_available():
        install_fused_rmsnorm(norm_modules)
        fused = True
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    n_local = tpm.local_param_count()
    n_full = sum(math.prod(s.full_shape) for s in tpm.specs.values())
    mem = (f"resident {torch.cuda.memory_allocated() / 2**30:.1f} GB (peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)"
           if str(device).startswith("cuda") else "")
    log(f"[tp] sharded {len(layers)} layers ({n_gdn} GDN + {n_attn} attention) + embed/head over tp={tp} in {time.time() - t0:.0f}s | "
        f"{n_full / 1e9:.2f}B params, {n_local / 1e9:.3f}B local per rank | kv heads {Hkv} "
        f"{'sharded' if Hkv % tp == 0 else f'replicated x{tp // Hkv}'} | fused RMSNorm {'on' if fused else 'off'} | {mem}")
    tpm.fused_norm = fused
    return tpm


# ---------------------------------------------------------------------------------------------------------------
# vocab-parallel LM head + cross-entropy
# ---------------------------------------------------------------------------------------------------------------
def vocab_parallel_cross_entropy(h, w_local, targets, vocab_start, group, weights=None):
    """Sum_i w_i * CE_i where the logits of every token are split across the ranks along the vocab (rank holds
    rows [vocab_start, vocab_start + V/tp) of the head). h: [n, d] (already inside the TP region), targets [n] global
    token ids, weights [n] (None = 1/n each, i.e. HF ForCausalLM's mean over label tokens). fp32 like HF (bf16 logits
    upcast). The row max is a detached constant (it cancels in log-softmax), the two sums are autograd all-reduces."""
    logits = F.linear(h, w_local).float()                                  # [n, V/tp]
    vl = logits.shape[-1]
    m = logits.max(dim=-1, keepdim=True).values.detach()
    if dist.get_world_size(group) > 1:
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
    z = logits - m
    sum_exp = reduce_from_tp(z.exp().sum(dim=-1, keepdim=True), group)     # [n, 1]
    lse = sum_exp.log().squeeze(-1)
    mask = (targets >= vocab_start) & (targets < vocab_start + vl)
    local_t = (targets - vocab_start).masked_fill(~mask, 0)
    tl = z.gather(-1, local_t[:, None]).squeeze(-1) * mask.to(z.dtype)
    tl = reduce_from_tp(tl, group)
    per_token = lse - tl
    if weights is None:
        return per_token.sum() / per_token.numel()
    return (per_token * weights).sum()


def install_tp_causal_lm_forward(model, tpm):
    """Replace the CausalLM forward: labels given -> body, lm_head + parallel CE ONLY at the label positions (mean over
    label tokens == HF ForCausalLM's loss; `loss_group_ids` [B] optionally averages per-group token means instead, to
    emulate the DP loss normalisation of the FSDP2 path for exactness checks); no labels (the prefix forward) -> the
    LOCAL vocab slice of the requested logits (only the cache is used by the caller). Returns an undo callable."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    base = model
    orig_forward = base.forward
    lm_head = base.lm_head
    group = tpm.group

    def forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, logits_to_keep=0, loss_group_ids=None, **kwargs):
        out = base.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        h = out.last_hidden_state
        if labels is None:
            sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = F.linear(copy_to_tp(h[:, sl, :], group), lm_head.weight)
            return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=out.past_key_values)
        tgt = labels[:, 1:]                                   # logits at t predict token t+1
        keep = tgt != -100
        h_sel, tgt_sel = h[:, :-1][keep], tgt[keep]
        n = tgt_sel.numel()
        if n == 0:
            raise ValueError("no label tokens in this batch")
        weights = None
        if loss_group_ids is not None:
            gid = loss_group_ids.to(labels.device)[:, None].expand_as(tgt)[keep]
            counts = torch.bincount(gid, minlength=int(loss_group_ids.max()) + 1)
            n_groups = int((counts > 0).sum())
            weights = 1.0 / (counts.clamp_min(1)[gid].float() * n_groups)
        loss = vocab_parallel_cross_entropy(copy_to_tp(h_sel, group), lm_head.weight, tgt_sel, tpm.vocab_start, group, weights)
        return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=out.past_key_values)

    base.forward = forward

    def undo():
        base.forward = orig_forward
    return undo


# ---------------------------------------------------------------------------------------------------------------
# reference helpers (tests): full-model gradient of a replicated single-process model, gathered TP gradients
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def gathered_grads_one(tpm, name, master, spec):
    """Full fp32 gradient of ONE parameter from the masters' (complete) local grads. COLLECTIVE (every rank returns it)."""
    if spec.replicated:
        return master.grad.detach().clone()
    loc = master.grad.detach().contiguous()
    shards = [torch.empty_like(loc) for _ in range(tpm.tp)]
    dist.all_gather(shards, loc, group=tpm.group)
    full = torch.empty(spec.full_shape, dtype=loc.dtype, device=loc.device)
    for r in range(tpm.tp):
        full.index_copy_(spec.dim, spec.index[r].to(loc.device), shards[r])
    return full


@torch.no_grad()
def gathered_grads(tpm):
    """{name: full fp32 gradient} for every parameter with a gradient. COLLECTIVE."""
    return {name: gathered_grads_one(tpm, name, m, spec) for name, p, m, spec in tpm.masters if m.grad is not None}


def compare_grads(a, b):
    """Per-name (cosine, max|d|, rel Frobenius) + aggregates over two {name: tensor} dicts (same keys)."""
    rows, num, den_a, den_b = {}, 0.0, 0.0, 0.0
    worst_cos, worst_name, max_abs = 1.0, "", 0.0
    for n in a:
        if n not in b:
            continue
        x, y = a[n].double().flatten(), b[n].double().flatten()
        dot, nx, ny = float((x * y).sum()), float(x.norm()), float(y.norm())
        cos = dot / max(nx * ny, 1e-30)
        d = float((x - y).abs().max())
        rel = float((x - y).norm()) / max(ny, 1e-30)
        rows[n] = (cos, d, rel)
        num += dot; den_a += nx * nx; den_b += ny * ny
        max_abs = max(max_abs, d)
        if cos < worst_cos:
            worst_cos, worst_name = cos, n
    return SimpleNamespace(rows=rows, global_cos=num / max(math.sqrt(den_a * den_b), 1e-30), worst_cos=worst_cos,
                           worst_name=worst_name, max_abs=max_abs, n=len(rows))
