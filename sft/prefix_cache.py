"""Prefix-cached SFT forward: run the shared prompt prefix ONCE per micro-batch, expand its cache to the batch,
and run only ``[marker] + prompt tail + target`` per example -- exact (gradients included) w.r.t. the naive
full-sequence forward, modulo bf16 kernel noise.

Why it is exact. Every SFT example is the same prompt (``mxf.prompts.build_prompt_ids``) followed by its target;
the direction is injected at the marker token's residual at ``INJECT_LAYER``. Everything strictly BEFORE the marker
is identical across examples and -- causality -- unaffected by the injection, so its forward (and the backward
through it) can be shared. For Qwen3.5/3.6's hybrid stack that means: the attention layers' K/V for the prefix,
and for the gated-delta-net layers the last ``conv_kernel_size`` conv inputs plus the recurrent state at the end
of the prefix. transformers' ``DynamicCache`` already carries exactly those (see ``LinearAttentionLayer``), and the
GDN layer already honours them for a multi-token continuation (conv gets ``cat(conv_state, new)`` and the chunk
kernel gets ``initial_state``). What stock transformers (5.15.0) does NOT support is doing this under autograd:
the cache writes are in-place ``copy_`` into buffers that FLA's chunk kernel has saved for backward (version-counter
error), and the linear-attention layers have no batch expansion. Both are fixed in the fork this module requires:

    pip install git+https://github.com/ceselder/transformers@maemm-prefix-cache

(``PrefixCache.check_transformers`` refuses to run on stock transformers.)

Usage (mirrors sft/pretrain.py's per-example path)::

    pc = PrefixCache(model, prompt_ids, marker, tok.pad_token_id, get_layer(model, INJECT_LAYER), STEER_COEFF, device)
    out = pc.forward(vecs, targets, autocast=lambda: autocast_region(model, a.autocast_bf16))
    out.loss.backward()

``vecs``: [B, d] direction rows (any float dtype, CPU or GPU); ``targets``: list of B token-id lists (already
including EOS if you want it, exactly what ``build_sft_ids`` appends). ``out.loss`` is HF's ForCausalLM loss --
mean over target tokens (labels -100 on marker/tail/pad) -- so it matches the naive ``model(..., labels=...).loss``.
"""
import contextlib
import copy
from types import SimpleNamespace

import torch

from mxf.inject import hooked, make_inject_hook

try:
    from sft.loss import cross_entropy_sum
except ImportError:  # Modal sibling mount.
    from loss import cross_entropy_sum


def unwrap_base(model):
    """HF base model through DDP + PEFT wrappers."""
    m = model.module if hasattr(model, "module") else model
    return m.get_base_model() if hasattr(m, "get_base_model") else m


_LAYER_DICT_ATTRS = ("conv_states", "recurrent_states", "is_conv_states_initialized",
                     "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size")


def compile_mlp_blocks(model, dynamic=True):
    """Regional torch.compile: only each decoder layer's MLP (3 LoRA linears + SiLU-mul) -- no cache objects cross the
    compiled boundary, so Dynamo does not recompile per step (whole-forward compile does). Returns a zero-arg undo
    callable. MEASURED (B200, mb64): no speed gain (62.5 vs 62.4 ex/s), -5 GB peak -- the launch-bound part of the step
    is not in the MLPs. Kept for the bench (test_prefix_cache.py cached_mlpcompile); not wired into pretrain.py.
    Note: a compiled backward + retain_graph=True (shared-prefix accumulation) fails with 'donated buffers' unless
    torch._functorch.config.donated_buffer=False."""
    base = unwrap_base(model)
    layers = base.model.layers
    originals = [layer.mlp for layer in layers]
    for layer in layers:
        layer.mlp = torch.compile(layer.mlp, dynamic=dynamic)

    def undo():
        for layer, mlp in zip(layers, originals):
            layer.mlp = mlp
    return undo


def expand_cache_copy(cache, repeats):
    """A NEW cache whose per-layer tensors are ``cache``'s repeated ``repeats`` times along the batch dim; ``cache``
    itself is left untouched (so one prefix cache can feed several micro-batches -- grad accumulation). Shallow copies
    of the Cache and its layer objects (+ their state dicts); tensors are shared until ``batch_repeat_interleave``
    replaces them out-of-place, so autograd still points back at the single prefix forward."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        l2 = copy.copy(layer)
        for attr in _LAYER_DICT_ATTRS:
            if hasattr(l2, attr):
                setattr(l2, attr, dict(getattr(l2, attr)))
        new.layers.append(l2)
    new.batch_repeat_interleave(repeats)
    return new


class PrefixGradientAccumulator:
    """Share a differentiable prefix across an entire optimizer step.

    Suffix backward passes end at detached cache leaves. ``accumulate`` saves their
    gradients in fp32; ``backward`` applies the summed cache gradients to the original
    prefix graph ONCE. This is the chain rule, not a frozen/detached-prefix objective.
    It avoids both repeated prefix forwards and retain_graph/repeated prefix backwards.
    Rebuild after every optimizer update. Floating-point summation order differs from
    the repeated-prefix path, so bf16 gradients are not expected to be bit-identical.

    The supported cache is the fork's DynamicCache (attention K/V and GDN conv/recurrent
    states). Only expand COPIES of ``cache`` with ``expand_cache_copy`` for suffix calls.
    DDP must have automatic reduction disabled for all suffix/prefix backwards, followed
    by one explicit reduction of the completed parameter gradients (see below).
    """

    def __init__(self, cache, extra_outputs=None):
        """extra_outputs: tensors of the prefix graph that must receive a (zero) gradient in the final ``backward`` --
        the prefix logits under FSDP2 (see ``PrefixCache.keep_prefix_grad_path``). They are traversed ONCE, at the end,
        never in the suffix backwards: an FSDP2 pre-backward unshard hook fires only once per forward, so the prefix
        graph must be walked by exactly one backward call."""
        self._pairs = []
        self._sums = []
        self._finished = False
        self._extra = [t for t in (extra_outputs or []) if t is not None and t.requires_grad]
        tensors = {}

        def detach(value):
            if isinstance(value, torch.Tensor):
                if id(value) not in tensors:
                    leaf = value.detach().requires_grad_(value.requires_grad)
                    tensors[id(value)] = leaf
                    if value.requires_grad:
                        self._pairs.append((value, leaf))
                        self._sums.append(None)
                return tensors[id(value)]
            if isinstance(value, dict):
                return {k: detach(v) for k, v in value.items()}
            if isinstance(value, list):
                return [detach(v) for v in value]
            if isinstance(value, tuple):
                return tuple(detach(v) for v in value)
            return value

        self.cache = copy.copy(cache)
        self.cache.layers = []
        for layer in cache.layers:
            new = copy.copy(layer)
            for name, value in vars(layer).items():
                setattr(new, name, detach(value))
            self.cache.layers.append(new)

    def accumulate(self):
        """Call after each suffix backward; no GPU-to-CPU synchronization."""
        if self._finished:
            raise RuntimeError("prefix gradients have already been consumed; build a new prefix")
        for i, (_, leaf) in enumerate(self._pairs):
            if leaf.grad is not None:
                if self._sums[i] is None:
                    # Keep double precision in CPU equivalence tests; accumulate bf16/fp16 in fp32.
                    dtype = torch.float64 if leaf.dtype == torch.float64 else torch.float32
                    self._sums[i] = leaf.grad.detach().to(dtype)
                else:
                    self._sums[i].add_(leaf.grad.detach())
                leaf.grad = None

    def backward(self):
        """Propagate the accumulated cache gradients through the original prefix once."""
        self.accumulate()
        outputs, grads = [], []
        for (original, _), grad in zip(self._pairs, self._sums):
            if grad is not None:
                outputs.append(original)
                grads.append(grad.to(original.dtype))
        for t in self._extra:                       # zero-weight grad path (FSDP2 unshard hook on the last layer's output)
            outputs.append(t)
            grads.append(torch.zeros_like(t))
        if outputs:
            torch.autograd.backward(outputs, grads)
        self._finished = True
        self.cache = None
        self._pairs.clear()
        self._sums.clear()
        self._extra = []


def sync_accumulated_gradients(parameters, bucket_bytes=32 * 1024**2):
    """Average completed local gradients once, after suffix AND prefix backwards.

    Bounded-size buckets avoid allocating another full LoRA gradient vector. A global
    presence mask handles locally unused parameters without dropping another rank's
    contribution; globally unused parameters retain grad=None, as in DDP. Every rank
    must pass the same ordered parameter list. Call before gradient clipping/Adam.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return
    comm_device = params[0].device if dist.get_backend() == "nccl" else torch.device("cpu")
    present = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=comm_device)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    active = [p for p, used in zip(params, present.tolist()) if used]
    world = dist.get_world_size()

    def reduce(bucket):
        if not bucket:
            return
        for p in bucket:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = torch.cat([p.grad.reshape(-1) for p in bucket]).to(comm_device)
        dist.all_reduce(flat)
        flat.div_(world)
        flat = flat.to(bucket[0].device)
        offset = 0
        for p in bucket:
            p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()

    bucket, size = [], 0
    for p in active:
        nbytes = p.numel() * p.element_size()
        if bucket and (size + nbytes > bucket_bytes or p.dtype != bucket[0].dtype or p.device != bucket[0].device):
            reduce(bucket)
            bucket, size = [], 0
        bucket.append(p)
        size += nbytes
    reduce(bucket)


def check_transformers():
    """Refuse stock transformers: it lacks the autograd-safe linear-attention cache writes + batch expansion."""
    import transformers
    from transformers import cache_utils

    layer_cls = getattr(cache_utils, "LinearAttentionLayer", None)
    ok = (
        layer_cls is not None
        and hasattr(layer_cls, "batch_repeat_interleave")
        and hasattr(cache_utils, "_write_cached_state")
    )
    if not ok:
        raise RuntimeError(
            f"transformers {transformers.__version__} at {transformers.__file__} is not the prefix-cache fork: "
            "install `pip install git+https://github.com/ceselder/transformers@maemm-prefix-cache` "
            "(autograd-safe LinearAttentionLayer state writes + batch_repeat_interleave)."
        )


# ---------------------------------------------------------------------------------------------------------------
# Exact per-layer activation checkpointing WITH a prefix cache, and a gathered/chunked LM head for the suffix
# ---------------------------------------------------------------------------------------------------------------
_TENSOR = 0
_VALUE = 1


def _snapshot_cache_layer(cl):
    """Flatten a cache layer object's state: (structure, tensors). Tensors are returned separately so they can be passed
    as explicit checkpoint inputs; ``_restore_cache_layer`` puts the exact same objects back (attributes and dict items)."""
    struct, tensors = [], []

    def enc(v):
        if isinstance(v, torch.Tensor):
            tensors.append(v)
            return (_TENSOR, len(tensors) - 1)
        return (_VALUE, v)

    for name, val in vars(cl).items():
        if isinstance(val, dict):
            struct.append((name, {k: enc(v) for k, v in val.items()}))
        else:
            struct.append((name, enc(val)))
    return struct, tensors


def _restore_cache_layer(cl, struct, tensors):
    def dec(e):
        return tensors[e[1]] if e[0] == _TENSOR else e[1]

    for name, e in struct:
        if isinstance(e, dict):
            setattr(cl, name, {k: dec(v) for k, v in e.items()})
        else:
            setattr(cl, name, dec(e))


class SuffixCheckpointer:
    """Per-decoder-layer activation checkpointing that is EXACT with a prefix cache.

    HF's GradientCheckpointingLayer drops ``past_key_values`` under checkpointing (the recompute would otherwise re-read
    cache slots the layer's own forward has already overwritten with its final states: the fork ASSIGNS the new
    conv/recurrent states and the attention layers ``cat`` onto the cached K/V). Here each layer's forward is wrapped
    (INSIDE the FSDP2 boundary, so FSDP's pre/post hooks run once, in the real forward): the layer's cache slots are
    snapshotted before the call, passed to ``torch.utils.checkpoint`` as inputs (grads flow back into the expanded
    prefix cache), restored right before the layer runs -- identically in the forward and in the recompute -- and
    restored again after it, which also discards the layer's suffix final states (never used in training; 150 MB per
    example for the 27B). Deterministic layer, no dropout => recompute == forward.

    Only active while ``enabled`` (PrefixCache.forward switches it on for the suffix forward; the batch-1 prefix
    forward stays un-checkpointed: it is launch-bound, recomputing it would double that cost for ~nothing).
    """

    def __init__(self, model):
        from torch.utils.checkpoint import checkpoint

        self.enabled = False
        self._checkpoint = checkpoint
        base = unwrap_base(model)
        self._patched = []
        for idx, layer in enumerate(base.model.layers):
            orig = layer.forward
            layer.forward = self._wrap(layer, idx, orig)
            self._patched.append((layer, orig))

    def undo(self):
        for layer, orig in self._patched:
            layer.forward = orig
        self._patched.clear()

    def _wrap(self, layer, idx, orig):
        ckpt = self._checkpoint

        def forward(hidden_states, *args, past_key_values=None, **kwargs):
            if not (self.enabled and past_key_values is not None and torch.is_grad_enabled()):
                return orig(hidden_states, *args, past_key_values=past_key_values, **kwargs)
            cl = past_key_values.layers[idx]
            struct, tensors = _snapshot_cache_layer(cl)

            def run(hidden_states, *tensors, **kw):
                _restore_cache_layer(cl, struct, tensors)
                out = orig(hidden_states, *args, past_key_values=past_key_values, **kw)
                _restore_cache_layer(cl, struct, tensors)   # drop this layer's suffix final states (unused; frees them)
                return out

            return ckpt(run, hidden_states, *tensors, use_reentrant=False, preserve_rng_state=False, **kwargs)

        return forward


def install_prefix_label_head(model, ce_chunk=0, fused_ce=False):
    """Replace the CausalLM forward (when ``labels`` are given) by: transformer body -> lm_head + fp32 cross-entropy
    ONLY at the label positions, optionally in recomputed chunks of ``ce_chunk`` rows (peak = one chunk of fp32 logits
    instead of [B, L, 248k] bf16 + fp32 + softmax). Mean over label tokens == HF ForCausalLM's loss (same math as
    pretrain.py's LabelHeadLM, which the --parity-check asserted to < 1e-3). Calls without labels (the prefix forward,
    logits_to_keep=1) go through the original forward. Installed on the module instance so FSDP2's root hooks (module
    pre/post forward hooks) still wrap it. Returns an undo callable."""
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_outputs import CausalLMOutputWithPast

    base = unwrap_base(model)
    orig_forward = base.forward
    lm_head = base.lm_head

    def head_ce_sum(h_rows, tgt_rows):
        return cross_entropy_sum(lm_head(h_rows), tgt_rows, fused=fused_ce)

    def forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if labels is None:
            return orig_forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                                logits_to_keep=logits_to_keep, **kwargs)
        out = base.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        h = out.last_hidden_state
        tgt = labels[:, 1:]                                   # logits at t predict token t+1
        keep = tgt != -100
        h_sel, tgt_sel = h[:, :-1][keep], tgt[keep]
        n = tgt_sel.numel()
        if n == 0:
            raise ValueError("no label tokens in this batch")
        if ce_chunk and n > ce_chunk:
            total = sum(checkpoint(head_ce_sum, h_sel[s : s + ce_chunk], tgt_sel[s : s + ce_chunk],
                                   use_reentrant=False, preserve_rng_state=False) for s in range(0, n, ce_chunk))
        else:
            total = head_ce_sum(h_sel, tgt_sel)
        return CausalLMOutputWithPast(loss=total / n, logits=None, past_key_values=out.past_key_values)

    base.forward = forward

    def undo():
        base.forward = orig_forward
    return undo


class PrefixCache:
    """Shared-prefix forward for a fixed prompt. Construct once (prompt is fixed), call ``forward`` per micro-batch."""

    def __init__(self, model, prompt_ids, marker, pad_id, inject_module, coeff, device,
                 inject_dtype=torch.bfloat16, inject_mode="add", pad_multiple=8, prefix_model=None,
                 persistent_injector=None, compile_prefix=None, keep_prefix_grad_path=False):
        """
        model: the model to run (PEFT-wrapped, possibly DDP-wrapped) -- used for the SUFFIX forward.
        prefix_model: module used for the PREFIX forward. Under DDP pass the unwrapped ``ddp.module`` so DDP's reducer
            is primed by exactly one forward (the suffix) per backward; defaults to ``model``.
        prompt_ids / marker: from ``build_prompt_ids(tok)`` -> (prompt_ids, mpos); marker = mpos[0].
        inject_module: ``get_layer(model, INJECT_LAYER)``; coeff: STEER_COEFF.
        persistent_injector: optional ``mxf.inject.FixedPositionInjector(position=0, ...)`` whose ``.hook`` the
            caller registered on ``inject_module`` (torch.compile path: no per-step Python hook). It is switched
            ``active=False`` for the prefix forward and ``True`` for the suffix forward.
        compile_prefix: None | "default" | "reduce-overhead" -- torch.compile ONLY the prefix call (fully static
            shape [1, prefix_len]; the eager 64-layer PEFT forward+backward at batch 1 is launch/Python-bound).
            The suffix forward stays eager (its cache inputs make Dynamo recompile endlessly).
        keep_prefix_grad_path: add ``0 * prefix_logits.sum()`` to the loss. The loss depends on the prefix forward ONLY
            through the caches, so the LAST layer's prefix output never receives a gradient. FSDP2 (fully_shard) hangs its
            pre-backward "all-gather the params" hook on module OUTPUTS, so without this term the last layer's params are
            still sharded when the cache path's gradient reaches them (setStorage ... storage of size 0). Zero-weight,
            so the optimization is unchanged; needed for --full-ft, a no-op for DDP/LoRA.
        """
        check_transformers()
        if not (0 < marker < len(prompt_ids)):
            raise ValueError(f"marker {marker} must be inside the prompt (len {len(prompt_ids)}) and not at 0")
        if persistent_injector is not None and persistent_injector.position != 0:
            raise ValueError("persistent_injector.position must be 0 (the marker is suffix index 0)")
        self.model = model
        self.prefix_model = prefix_model if prefix_model is not None else model
        self.prefix_ids = list(prompt_ids[:marker])          # shared, injection-free
        self.suffix_prompt = list(prompt_ids[marker:])       # [marker] + tail (tail is empty for the current prompt)
        self.pad_id = pad_id
        self.inject_module = inject_module
        self.coeff = coeff
        self.device = device
        self.inject_dtype = inject_dtype
        self.inject_mode = inject_mode
        self.pad_multiple = pad_multiple
        self.persistent_injector = persistent_injector
        self._prefix_tensor = torch.tensor(self.prefix_ids, dtype=torch.long, device=device)[None]
        self.keep_prefix_grad_path = keep_prefix_grad_path
        self._prefix_logits = None
        self.compile_prefix = compile_prefix
        if compile_prefix:
            fwd = self.prefix_model.forward if hasattr(self.prefix_model, "forward") else self.prefix_model
            self._prefix_fn = torch.compile(fwd, mode=None if compile_prefix == "default" else compile_prefix, dynamic=False)
        else:
            self._prefix_fn = self.prefix_model

        self.suffix_ckpt = None   # optional SuffixCheckpointer: per-layer activation checkpointing of the SUFFIX forward only

    @property
    def prefix_len(self):
        return len(self.prefix_ids)

    def pop_prefix_logits(self):
        """Take the pending ``keep_prefix_grad_path`` logits (None when off / already consumed). A caller that shares one
        prefix across an optimizer step (PrefixGradientAccumulator) MUST pop them before the first suffix forward and hand
        them to the accumulator, else the first suffix backward would walk the whole prefix graph (zero grads) and the
        final prefix backward would hit FSDP2's already-consumed unshard hooks."""
        logits, self._prefix_logits = self._prefix_logits, None
        return logits

    def build_suffix(self, targets, max_len=None):
        """Right-padded suffix batch. Returns (input_ids [B,L], labels [B,L], suffix_mask [B,L] bool, L)."""
        rows = [self.suffix_prompt + list(t) for t in targets]
        if max_len is not None:
            rows = [r[:max_len] for r in rows]
        L = max(len(r) for r in rows)
        L = ((L + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        B = len(rows)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.bool)
        n_prompt = len(self.suffix_prompt)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = True
            if len(r) > n_prompt:
                labels[i, n_prompt : len(r)] = torch.tensor(r[n_prompt:])
        return ids, labels, mask, L

    def run_prefix(self, autocast=contextlib.nullcontext):
        """One forward over the shared prefix with grad enabled; returns the (batch-1) cache."""
        inj = self.persistent_injector
        if inj is not None:
            inj.active = False
        try:
            with autocast():
                out = self._prefix_fn(input_ids=self._prefix_tensor, use_cache=True, logits_to_keep=1)
        finally:
            if inj is not None:
                inj.active = True
        cache = out.past_key_values
        if cache is None:
            raise RuntimeError("prefix forward returned no cache (use_cache ignored?)")
        self._prefix_logits = out.logits if self.keep_prefix_grad_path else None   # [1, 1, V] (logits_to_keep=1)
        return cache

    def forward(self, vecs, targets, autocast=contextlib.nullcontext, max_len=None, prefix_cache=None, timings=None):
        """Full prefix-cached step. ``autocast``: zero-arg callable returning a context manager (e.g.
        ``lambda: autocast_region(model, True)``), applied to BOTH forwards. ``timings``: optional dict that receives
        CUDA-synchronized phase durations (prefix_fwd_s, expand_s, suffix_fwd_s) -- profiling only, it syncs.
        Returns SimpleNamespace(loss, logits, labels, suffix_mask, n_target_tokens, suffix_len, prefix_len)."""
        import time

        def _tick():
            if timings is None:
                return None
            torch.cuda.synchronize()
            return time.time()

        B = len(targets)
        vecs = torch.as_tensor(vecs)
        if vecs.ndim != 2 or vecs.shape[0] != B:
            raise ValueError(f"vecs must be [B={B}, d], got {tuple(vecs.shape)}")
        t0 = _tick()
        if prefix_cache is not None:
            # caller-owned prefix (shared across micro-batches of one optimizer step): expand a COPY, keep theirs intact
            t1 = _tick()
            cache = expand_cache_copy(prefix_cache, B)
        else:
            cache = self.run_prefix(autocast)
            t1 = _tick()
            cache.batch_repeat_interleave(B)   # out-of-place in the fork: grads flow back into the prefix graph
        t2 = _tick()
        P = self.prefix_len

        ids, labels, smask, L = self.build_suffix(targets, max_len)
        full_mask = torch.cat([torch.ones((B, P), dtype=torch.bool), smask], dim=1).to(self.device)
        position_ids = torch.arange(P, P + L, device=self.device)[None].expand(B, -1)
        if self.persistent_injector is not None:   # compile path: registered once, reads a stable buffer
            self.persistent_injector.set_vectors(vecs.to(self.device, self.persistent_injector.vectors.dtype))
            self.persistent_injector.active = True
            hook_cm = contextlib.nullcontext()
        else:                                      # legacy path: one Python hook per micro-batch
            hook = make_inject_hook([v[None] for v in vecs], [[0]] * B, self.coeff, self.device,
                                    self.inject_dtype, mode=self.inject_mode)   # marker == suffix index 0
            hook_cm = hooked(self.inject_module, hook)
        if self.suffix_ckpt is not None:
            self.suffix_ckpt.enabled = True
        try:
            with hook_cm, autocast():
                out = self.model(input_ids=ids.to(self.device), attention_mask=full_mask, position_ids=position_ids,
                                 past_key_values=cache, use_cache=True, labels=labels.to(self.device))
        finally:
            if self.suffix_ckpt is not None:
                self.suffix_ckpt.enabled = False
        # The suffix's final GDN states / KV are useless for training: drop them before backward (fp32 recurrent
        # states are 150 MB per example for the 27B).
        del cache
        out.past_key_values = None
        loss = out.loss
        if self.keep_prefix_grad_path and self._prefix_logits is not None:   # see __init__: FSDP2 needs a grad path through the prefix output
            loss = loss + 0.0 * self._prefix_logits.float().sum()
            self._prefix_logits = None                                        # once per prefix forward (prefix_accum > 1 shares it)
        t3 = _tick()
        if timings is not None:
            timings.update(prefix_fwd_s=t1 - t0, expand_s=t2 - t1, suffix_fwd_s=t3 - t2)
        return SimpleNamespace(loss=loss, logits=out.logits, labels=labels, suffix_mask=smask,
                               n_target_tokens=int((labels != -100).sum()), suffix_len=L, prefix_len=P)
