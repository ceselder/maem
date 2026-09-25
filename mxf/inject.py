"""Norm-matched activation injection (activation-oracle formula) + residual read hook.

    resid[b, pos] += unit(v[b]) * ‖resid[b, pos]‖ * coeff        (v detached; grad flows via resid)
"""
import contextlib

import torch


def get_layer(model, layer: int):
    """The decoder block at `layer`, unwrapping DDP + PEFT."""
    m = model.module if hasattr(model, "module") else model
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.model.layers[layer]


def get_input_embeddings(model):
    """Input-token embedding module, unwrapping DDP + PEFT."""
    m = model.module if hasattr(model, "module") else model
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.get_input_embeddings()


def make_inject_hook(vecs, positions, coeff, device, dtype, mode="add"):
    """vecs: list of [1, d] unit-ish directions (one per batch row). positions: list[list[int]]."""
    if len(vecs) != len(positions):
        raise ValueError(f"{len(vecs)} vector rows != {len(positions)} position rows")
    counts = [len(p) for p in positions]
    if any(v.shape[0] != n for v, n in zip(vecs, counts)):
        raise ValueError("each vector row must have one vector per marker position")
    normed = torch.nn.functional.normalize(torch.cat(vecs).to(device, dtype), dim=-1)
    rows = torch.repeat_interleave(torch.arange(len(vecs), device=device),
                                  torch.tensor(counts, device=device), output_size=sum(counts))
    cols = torch.tensor([p for row in positions for p in row], device=device)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:  # decode step (KV-cache): marker already injected at prefill
            return out
        if h.shape[0] != len(vecs):
            raise RuntimeError(f"inject batch {h.shape[0]} != {len(vecs)} vector rows")
        base = h[rows, cols]
        if mode in ("add", "add_clone"):
            if mode == "add_clone":   # same arithmetic on a copy: FSDP2 hangs backward hooks on the layer output tensor
                h = h.clone()
            scale = base.norm(dim=-1, keepdim=True) * coeff
            h[rows, cols] = base + (normed * scale).to(h.dtype).detach()
        elif mode == "replace":
            # Input-embedding outputs are leaf tensors after enable_input_require_grads(); clone
            # before assignment so eager autograd and Dynamo both accept literal replacement.
            h = h.clone()
            h[rows, cols] = (normed * coeff).to(h.dtype).detach()
        else:
            raise ValueError(f"unknown injection mode: {mode}")
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return hook


def make_packed_inject_hook(vecs, rows, cols, coeff, device, dtype):
    """Packed-block variant: vecs [K, d]; direction j injected at (rows[j], cols[j]) — several
    markers per batch row, one direction per marker. Norm-matched formula identical to
    make_inject_hook. Training-forward only (seq_len == pack_len > 1), so the decode-step guard
    below never triggers; kept for symmetry."""
    normed = torch.nn.functional.normalize(vecs.to(device, dtype), dim=-1)   # [K, d]
    rows, cols = rows.to(device), cols.to(device)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:  # decode step (KV-cache): marker already injected at prefill
            return out
        base = h[rows, cols]                                 # [K, d]
        scale = base.norm(dim=-1, keepdim=True) * coeff
        h[rows, cols] = base + (normed * scale).to(h.dtype).detach()
        return out

    return hook


class FixedPositionInjector(torch.nn.Module):
    """Persistent, tensorized injection hook for fixed-position padded training batches.

    Register ``hook`` once before ``torch.compile`` and update the non-persistent vector buffer in
    place before each forward. Unlike creating/removing a Python hook every step, Dynamo can keep
    this hook inside the compiled graph; unlike ``make_inject_hook``, it launches one batched GPU
    operation rather than one operation per row.
    """

    def __init__(self, max_batch, d_model, position, coeff, device, dtype, mode="add"):
        super().__init__()
        self.position = position
        self.coeff = coeff
        if mode not in {"add", "replace"}:
            raise ValueError(f"unknown injection mode: {mode}")
        self.mode = mode
        # ``active=False`` turns the hook into a no-op. Prefix-cache path (sft/prefix_cache.py): the hook is
        # registered permanently for torch.compile but must not fire on the shared-prefix forward, only on the
        # suffix forward whose index 0 is the marker. Plain Python bool -> a single Dynamo guard.
        self.active = True
        self.register_buffer("vectors", torch.empty(max_batch, d_model, device=device, dtype=dtype),
                             persistent=False)

    @torch.no_grad()
    def set_vectors(self, vectors):
        """Copy a [batch, d_model] tensor into stable storage without changing graph inputs."""
        if vectors.ndim != 2 or vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"expected [batch, {self.vectors.shape[1]}], got {tuple(vectors.shape)}")
        if len(vectors) > len(self.vectors):
            raise ValueError(f"batch {len(vectors)} exceeds injector capacity {len(self.vectors)}")
        self.vectors[: len(vectors)].copy_(vectors)

    def hook(self, _module, _inputs, output):
        if not self.active:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] <= 1:
            return output
        base = h[:, self.position]
        directions = torch.nn.functional.normalize(self.vectors[: h.shape[0]], dim=-1)
        if self.mode == "add":
            scale = base.norm(dim=-1, keepdim=True) * self.coeff
            h[:, self.position] = base + (directions * scale).to(h.dtype).detach()
        else:
            h = h.clone()
            h[:, self.position] = (directions * self.coeff).to(h.dtype).detach()
        return (h, *output[1:]) if isinstance(output, tuple) else h


@contextlib.contextmanager
def hooked(module, hook):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class _Stop(Exception):
    pass


@torch.no_grad()
def read_resid(model, layer, batch, pool="mean"):
    """Layer-`layer` residual for a tokenized batch. pool: 'mean'|'last'|'all'. No injection, base model."""
    captured = {}

    def cap(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).float()
        raise _Stop

    h = None
    handle = get_layer(model, layer).register_forward_hook(cap)
    try:
        model(**batch)
    except _Stop:
        h = captured["h"]
    finally:
        handle.remove()
    mask = batch["attention_mask"].bool()
    if pool == "all":
        return h, mask
    if pool == "last":
        idx = mask.sum(1) - 1
        return h[torch.arange(h.shape[0]), idx]
    summed = (h * mask.unsqueeze(-1)).sum(1)
    return summed / mask.sum(1, keepdim=True).clamp(min=1)
