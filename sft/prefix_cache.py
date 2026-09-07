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

    def __init__(self, cache):
        self._pairs = []
        self._sums = []
        self._finished = False
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
        if outputs:
            torch.autograd.backward(outputs, grads)
        self._finished = True
        self.cache = None
        self._pairs.clear()
        self._sums.clear()


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

    @property
    def prefix_len(self):
        return len(self.prefix_ids)

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
        with hook_cm, autocast():
            out = self.model(input_ids=ids.to(self.device), attention_mask=full_mask, position_ids=position_ids,
                             past_key_values=cache, use_cache=True, labels=labels.to(self.device))
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
