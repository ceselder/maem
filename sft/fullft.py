"""--full-ft for sft/pretrain.py: every weight of Qwen3.6-27B trainable, sharded with torch FSDP2 (fully_shard).

Layout (per rank, 8xB200): fp32 sharded master params (13.5 GB) + fp32 sharded grads (13.5 GB) + fp32 AdamW moments
(27 GB) = ~54 GB of optimizer state, plus one decoder layer's bf16 all-gather buffer at a time. Compute happens in bf16
(MixedPrecisionPolicy param_dtype=bf16, reduce_dtype=fp32), exactly the precision the LoRA path ran the frozen base in.

Loading: each rank loads the bf16 HF model onto its own GPU (54 GB, the same path pretrain.py always used), then every
decoder layer is upcast to fp32 and sharded IMMEDIATELY (peak = the bf16 model + one fp32 layer), then the root
(embed_tokens / final norm / lm_head).

Checkpoints are FULL HF models in the base repo's on-disk layout (Qwen3_5ForConditionalGeneration config, weights named
model.language_model.*, lm_head.weight, bf16 safetensors shards + index, tokenizer files, and the base's untouched
vision-tower / MTP tensors copied in so the directory is byte-for-byte the same schema as Qwen/Qwen3.6-27B). Both
transformers' AutoModelForCausalLM and vLLM (language_model_only) load it exactly like the base. The directory is
written as <path>.tmp and renamed at the end; SAVE_DONE (json) is written last -- consumers must require it.
"""
import contextlib
import json
import os
import shutil
import time

import torch
import torch.distributed as dist

_LM_PREFIX = "model.language_model."


class FSDPNoSync:
    """`ddp.no_sync()` stand-in: FSDP2 reduce-scatters every micro-batch into the fp32 sharded grads (correct
    accumulation, no 54 GB unsharded-grad buffer), so gradient accumulation needs NO communication skipping."""

    def __call__(self):
        return contextlib.nullcontext()


def shard_full_model(model, world, device, log=print, keep_unsharded_layers=0, prefetch=0, param_dtype=torch.bfloat16,
                     device_type="cuda"):
    """In-place: all params trainable, fp32 masters, FSDP2-sharded per decoder layer + root. Returns the model.

    keep_unsharded_layers: 0 = every layer re-all-gathers its bf16 params for every forward AND backward (FSDP2 default,
        reshard_after_forward=True). N > 0 = the root group (embed/norm/lm_head) and the first N decoder layers keep their
        gathered bf16 params resident for a whole optimizer step (reshard_after_forward=False at construction +
        set_reshard_after_backward(False) between the step's backwards, see set_persistent_unshard): ONE all-gather per
        step instead of 2 per micro-batch, at +0.84 GB/rank per layer (+5 GB root). -1 = all layers (+54 GB/rank). Grads
        are still reduce-scattered in fp32 after every micro-batch, so the update is unchanged (exact).
    prefetch: 0 = FSDP2's implicit one-module-ahead all-gather prefetch; N > 0 = explicit forward/backward prefetch of
        the next N layers (set_modules_to_{forward,backward}_prefetch; +0.84 GB/rank per extra module in flight).
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(device_type, (world,))
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(True)
    t0 = time.time()
    layers = model.model.layers
    n_keep = len(layers) if keep_unsharded_layers < 0 else min(keep_unsharded_layers, len(layers))
    for i, layer in enumerate(layers):
        layer.float()                                   # fp32 master for THIS layer only ...
        fully_shard(layer, mesh=mesh, mp_policy=mp,     # ... then shard it before touching the next one
                    reshard_after_forward=(i >= n_keep))
    for m in (model.model.embed_tokens, model.model.norm, model.lm_head):
        m.float()
    fully_shard(model, mesh=mesh, mp_policy=mp)         # root group: embed_tokens + norm + lm_head (never resharded after fwd)
    if prefetch > 0:
        L = list(layers)
        for i, layer in enumerate(L):
            layer.set_modules_to_forward_prefetch(L[i + 1 : i + 1 + prefetch])
            layer.set_modules_to_backward_prefetch(L[max(0, i - prefetch) : i][::-1])
    # groups whose gathered params persist across the step's micro-batches (toggled by set_persistent_unshard)
    model._fullft_persistent = ([model] + list(layers[:n_keep])) if n_keep else []
    if device_type == "cuda":
        torch.cuda.synchronize()
    n = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem = (f"resident {torch.cuda.memory_allocated() / 2**30:.1f} GB (peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)"
           if device_type == "cuda" else "")
    log(f"[fullft] FSDP2 sharded {len(layers)} layers + root over {world} ranks in {time.time() - t0:.0f}s | "
        f"{n / 1e9:.2f}B params, {n_tr / 1e9:.2f}B trainable | compute {str(param_dtype).split('.')[-1]} | "
        f"persistent-unshard: {'root + ' + str(n_keep) + ' layers' if n_keep else 'off'} | explicit prefetch {prefetch} | {mem}")
    model.no_sync = FSDPNoSync()   # pretrain.py calls ddp.no_sync() for grad accumulation
    return model


def set_persistent_unshard(model, on):
    """keep_unsharded_layers > 0: call with on=True at the start of an optimizer step (the persistent groups skip the
    post-backward reshard, so the next micro-batch's forward finds them gathered) and on=False right BEFORE the step's
    LAST backward, so that backward reshards them and the optimizer updates the sharded fp32 masters with nothing
    stale left gathered (the next step's first forward all-gathers the fresh weights). Exact: only comm changes."""
    for m in getattr(model, "_fullft_persistent", []):
        m.set_reshard_after_backward(not on, recurse=False)


# ---------------------------------------------------------------------------------------------------------------
# optimizers over FSDP2 DTensor params
# ---------------------------------------------------------------------------------------------------------------
def _local(t):
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(t, DTensor):
            return t.to_local()
    except ImportError:
        pass
    return t


class LowPrecisionStateAdamW(torch.optim.Optimizer):
    """AdamW with the two moment buffers stored in lower precision (fp32 masters + fp32 grads untouched). Elementwise, so
    it runs on the LOCAL shard of every FSDP2 DTensor (padding rows are zero in param and grad and stay zero). Update
    math == torch.optim.AdamW (lerp / mul+addcmul / sqrt-div-add / addcdiv, fp32); with fp32 moments it reproduces it.
    m_dtype/v_dtype = bfloat16 halves the moment memory (27 -> 13.5 GB/rank for the 27B). CAUTION on a bf16 exp_avg_sq:
    beta2=0.999 means each step changes v by ~0.1%%, below bf16's 8-bit mantissa (0.4%%) -- v only moves when g^2 differs
    from v by > ~0.4%%; the update is no longer exact AdamW (torchao's block-quantized 8-bit states re-scale per block and
    are the tested alternative). 'adamw-mbf16' (bf16 m, fp32 v) avoids that caveat at 3/4 of the memory."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 m_dtype=torch.bfloat16, v_dtype=torch.float32):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.m_dtype, self.v_dtype = m_dtype, v_dtype

    @torch.no_grad()
    def step(self, closure=None):
        import math
        for group in self.param_groups:
            lr = float(group["lr"]); b1, b2 = group["betas"]; eps = group["eps"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                pl, gl = _local(p), _local(p.grad)
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(pl, dtype=self.m_dtype)
                    st["exp_avg_sq"] = torch.zeros_like(pl, dtype=self.v_dtype)
                st["step"] += 1
                t = st["step"]
                g = gl.float()
                m = st["exp_avg"].float().lerp_(g, 1 - b1)                       # fp32 temporaries (alias the state if fp32)
                v = st["exp_avg_sq"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                if st["exp_avg"].dtype != torch.float32:
                    st["exp_avg"].copy_(m)
                if st["exp_avg_sq"].dtype != torch.float32:
                    st["exp_avg_sq"].copy_(v)
                if wd:
                    pl.mul_(1 - lr * wd)
                denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)             # out-of-place sqrt: v may alias the state
                pl.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
        return None


OPTIMS = ("adamw", "adamw8bit", "adamw4bit", "adamwfp8", "adamw-bf16", "adamw-mbf16", "adamw-fp32states")


def make_optimizer(name, params, lr):
    """--optim: adamw = torch.optim.AdamW (fp32 moments, 27 GB/rank for the 27B); adamw8bit / adamw4bit / adamwfp8 =
    torchao.optim block-quantized moments (FSDP2 DTensor-aware; ~7 / ~3.5 / ~7 GB); adamw-bf16 / adamw-mbf16 =
    LowPrecisionStateAdamW (13.5 / 20 GB); adamw-fp32states = LowPrecisionStateAdamW with fp32 moments (== adamw, sanity)."""
    params = [p for p in params if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name in ("adamw8bit", "adamw4bit", "adamwfp8"):
        import torchao.optim as tao
        cls = {"adamw8bit": tao.AdamW8bit, "adamw4bit": tao.AdamW4bit, "adamwfp8": tao.AdamWFp8}[name]
        return cls(params, lr=lr, weight_decay=0.0)
    if name == "adamw-bf16":
        return LowPrecisionStateAdamW(params, lr=lr, m_dtype=torch.bfloat16, v_dtype=torch.bfloat16)
    if name == "adamw-mbf16":
        return LowPrecisionStateAdamW(params, lr=lr, m_dtype=torch.bfloat16, v_dtype=torch.float32)
    if name == "adamw-fp32states":
        return LowPrecisionStateAdamW(params, lr=lr, m_dtype=torch.float32, v_dtype=torch.float32)
    raise ValueError(f"unknown --optim {name!r}; choose from {OPTIMS}")


def optimizer_state_gb(opt):
    """Bytes of every tensor in the optimizer state on this rank (tensor subclasses report their own storage)."""
    total = 0
    seen = set()
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor) and id(v) not in seen:
                seen.add(id(v))
                v = _local(v)
                try:
                    inner = getattr(v, "codes", None)   # torchao OptimState{8bit,4bit,Fp8}: codes (+ scale, + qmap)
                    if inner is not None:
                        total += inner.numel() * inner.element_size()
                        for extra in ("scale", "qmap"):
                            e = getattr(v, extra, None)
                            if isinstance(e, torch.Tensor):
                                total += e.numel() * e.element_size()
                        continue
                    total += v.numel() * v.element_size()
                except Exception:  # noqa
                    pass
    return total / 2**30


# ---------------------------------------------------------------------------------------------------------------
# diagnostics: kernel backends, one profiled step, memory attribution at the activation peak
# ---------------------------------------------------------------------------------------------------------------
def _find_impl(fn, prefix="fla", depth=0):
    """Walk a decorated function's closures/__wrapped__ for a callable whose module starts with `prefix`."""
    if depth > 6 or not callable(fn):
        return None
    if (getattr(fn, "__module__", "") or "").startswith(prefix):
        return fn
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if callable(v) and v is not fn:
            r = _find_impl(v, prefix, depth + 1)
            if r is not None:
                return r
    w = getattr(fn, "__wrapped__", None)
    return _find_impl(w, prefix, depth + 1) if w is not None and w is not fn else None


def kernel_backends(model):
    """Which implementation transformers' hub-kernel dispatch resolved for the GDN chunk rule / causal conv."""
    import sys
    base = model.module if hasattr(model, "module") else model
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    layer = next(l for l in base.model.layers if hasattr(l, "linear_attn"))
    mod = sys.modules[type(layer.linear_attn).__module__]
    out = {}
    for name, pkg in (("torch_chunk_gated_delta_rule", "fla"), ("causal_conv1d_fn", "causal_conv1d")):
        fn = getattr(mod, name, None)
        impl = _find_impl(fn, pkg) if fn is not None else None
        out[name] = f"{impl.__module__}.{getattr(impl, '__qualname__', '?')}" if impl is not None else "torch fallback"
    try:
        import fla
        out["fla"] = "v" + str(getattr(fla, "__version__", "?"))
    except Exception as e:  # noqa
        out["fla"] = f"ABSENT ({type(e).__name__})"
    out["attn_implementation"] = getattr(base.config, "_attn_implementation", "?")
    return out


def memory_attribution(snap, top=28):
    """Aggregate the live blocks of a torch.cuda.memory._snapshot() by the most specific model-code frame."""
    import collections
    import os
    PRIO = ("fla", "modeling_qwen3_5", "cache_utils", "sdpa_attention", "loss_utils", "functional.py", "prefix_cache",
            "_fsdp", "fullft.py", "pretrain.py")
    agg, total, n_blocks = collections.Counter(), 0, 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk["size"]; total += size; n_blocks += 1
            frames = blk.get("frames") or []
            best, best_rank = None, len(PRIO)
            for fr in frames:
                fn = fr.get("filename", "")
                for r, pat in enumerate(PRIO):
                    if pat in fn and r < best_rank:
                        best, best_rank = fr, r
                        break
            if best is None and frames:
                best = frames[0]
            key = f"{os.path.basename(best['filename'])}:{best['line']} {best['name']}" if best else "<no frames>"
            agg[key] += size
    lines = [f"[mem] {total / 2**30:.1f} GB live in {n_blocks} blocks at the probe point; attribution by most specific model-code frame:"]
    for k, v in agg.most_common(top):
        lines.append(f"[mem]  {v / 2**30:7.2f} GB  {k}")
    return "\n".join(lines)


def peak_attribution(snap, top=24):
    """Replay the allocator trace of a torch.cuda.memory._snapshot() (recording must have been on for the whole step):
    find the moment of maximal live bytes among the blocks allocated since recording started and attribute THAT live set
    by frame -- the true within-step peak, which for FSDP training sits inside the backward, not at the end of the forward."""
    import collections
    import os
    traces = snap.get("device_traces") or []
    dev = torch.cuda.current_device()
    tr = traces[dev] if len(traces) > dev else (traces[0] if traces else [])
    live, total, best, best_i = {}, 0, -1, -1
    for i, ev in enumerate(tr):
        a = ev.get("action")
        if a == "alloc":
            live[ev["addr"]] = ev["size"]; total += ev["size"]
            if total > best:
                best, best_i = total, i
        elif a in ("free_completed", "free"):
            total -= live.pop(ev["addr"], 0)
    if best_i < 0:
        return "[mem] peak attribution: no alloc events in the trace"
    PRIO = ("fla", "modeling_qwen3_5", "cache_utils", "sdpa_attention", "loss_utils", "functional.py", "checkpoint.py",
            "prefix_cache", "_fsdp", "fullft.py", "pretrain.py", "autograd")
    live = {}
    for ev in tr[: best_i + 1]:
        a = ev.get("action")
        if a == "alloc":
            live[ev["addr"]] = ev
        elif a in ("free_completed", "free"):
            live.pop(ev["addr"], None)
    agg = collections.Counter()
    for ev in live.values():
        frames = ev.get("frames") or []
        bestf, rank = None, len(PRIO)
        for fr in frames:
            fn = fr.get("filename", "")
            for r, pat in enumerate(PRIO):
                if pat in fn and r < rank:
                    bestf, rank = fr, r
                    break
        if bestf is None and frames:
            bestf = frames[0]
        key = f"{os.path.basename(bestf['filename'])}:{bestf['line']} {bestf['name']}" if bestf else "<no frames>"
        agg[key] += ev["size"]
    lines = [f"[mem] WITHIN-STEP PEAK: {best / 2**30:.1f} GB of blocks allocated since the step began were live at trace event "
             f"{best_i}/{len(tr)} (add the pre-step resident bytes for the absolute peak); attribution:"]
    for k, v in agg.most_common(top):
        lines.append(f"[mem]  {v / 2**30:7.2f} GB  {k}")
    return "\n".join(lines)


class StepProfiler:
    """Profile ONE optimizer step (rank 0): kernel table by self device time, GPU busy / NCCL fractions, and a memory
    attribution snapshot taken by ``mem_probe`` (call it after the first micro-batch's forward, before its backward =
    the activation peak). ``report`` returns the text. Everything is off when enabled=False."""

    def __init__(self, enabled, mem=True):
        self.enabled, self.mem = enabled, mem
        self.prof = None
        self.mem_text = None
        self.t0 = None

    def __enter__(self):
        if not self.enabled:
            return self
        torch.cuda.synchronize()
        self.resident_gb = torch.cuda.memory_allocated() / 2**30
        if self.mem:
            torch.cuda.memory._record_memory_history(max_entries=3_000_000)
        self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA])
        self.prof.__enter__()
        self.t0 = time.time()
        return self

    def mem_probe(self):
        if not (self.enabled and self.mem) or self.mem_text is not None:
            return
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 2**30
        snap = torch.cuda.memory._snapshot()
        self.mem_text = (f"[mem] end-of-forward probe: allocated {alloc:.1f} GB (resident at step start {self.resident_gb:.1f} GB), "
                         f"peak so far {torch.cuda.max_memory_allocated() / 2**30:.1f} GB\n" + memory_attribution(snap, top=12))

    def __exit__(self, *exc):
        if not self.enabled:
            return False
        torch.cuda.synchronize()
        self.wall = time.time() - self.t0
        self.prof.__exit__(*exc)
        if self.mem:
            try:
                self.peak_text = peak_attribution(torch.cuda.memory._snapshot())
            except Exception as e:  # noqa
                self.peak_text = f"[mem] peak attribution failed: {e!r}"
            torch.cuda.memory._record_memory_history(enabled=None)
        return False

    def report(self, rows=22):
        if not self.enabled or self.prof is None:
            return ""
        ka = self.prof.key_averages()
        def dev(e):
            return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
        dev_total = sum(dev(e) for e in ka) / 1e6
        nccl = sum(dev(e) for e in ka if "nccl" in e.key.lower()) / 1e6
        copies = sum(dev(e) for e in ka if any(t in e.key for t in ("copy_", "Copy", "_to_copy", "cast", "fill_"))) / 1e6
        gemm = sum(dev(e) for e in ka if any(t in e.key.lower() for t in ("gemm", "cutlass", "matmul", "nvjet", "sm100", "sm90"))) / 1e6
        triton = sum(dev(e) for e in ka if "triton" in e.key.lower() or "chunk_" in e.key or "fused_" in e.key) / 1e6
        lines = [f"[prof] step wall {self.wall:.3f} s | device kernel time (sum of self, all streams) {dev_total:.3f} s = "
                 f"{dev_total / self.wall:.0%} of wall | NCCL {nccl:.3f} s ({nccl / self.wall:.0%}) | GEMM-like {gemm:.3f} s "
                 f"({gemm / self.wall:.0%}) | triton/fla-like {triton:.3f} s ({triton / self.wall:.0%}) | copies/casts {copies:.3f} s",
                 "[prof] top kernels by self device time:"]
        try:
            table = ka.table(sort_by="self_device_time_total", row_limit=rows)
        except Exception:  # noqa -- older sort key name
            table = ka.table(sort_by="self_cuda_time_total", row_limit=rows)
        lines += ["[prof] " + l for l in table.splitlines()]
        if self.mem_text:
            lines.append(self.mem_text)
        if getattr(self, "peak_text", None):
            lines.append(f"[mem] resident at step start {self.resident_gb:.1f} GB; max_memory_allocated now "
                         f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            lines.append(self.peak_text)
        return "\n".join(lines)


def clip_grad_norm(params, max_norm):
    """clip_grad_norm_ over FSDP2 DTensor grads; returns the total norm as a python float."""
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(gn, DTensor):
            gn = gn.full_tensor()
    except ImportError:
        pass
    return float(gn)


@contextlib.contextmanager
def injection_probe(model, inject_layer, log=print, tag="fullft"):
    """One-batch proof that the layer-`inject_layer` injection hook fires under FSDP2: compare the marker row
    (suffix index 0 in the prefix-cache path) of the injection layer's output BEFORE the injection hook (a forward
    hook registered before it) with the next layer's INPUT (after every hook). Norm-matched addition of a unit
    vector gives ratio ~ sqrt(2 + 2 cos) ~ 1.41; ratio 1.00 means the hook did NOT fire."""
    pairs, pending = [], {}   # one (seq_len, batch, pre-norm, post-norm) per forward: the prefix forward (no injection) and the suffix forward
    L = model.model.layers

    def pre_inj(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            pending["pre"] = (h.shape[1], h.shape[0], h[:, 0].detach().float().norm(dim=-1))

    def next_in(_m, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is not None and h.shape[1] > 1 and "pre" in pending:
            Lq, B, pre = pending.pop("pre")
            pairs.append((Lq, B, pre, h[:, 0].detach().float().norm(dim=-1)))

    h1 = L[inject_layer].register_forward_hook(pre_inj)
    h2 = L[inject_layer + 1].register_forward_pre_hook(next_in, with_kwargs=True)
    try:
        yield
    finally:
        h1.remove(); h2.remove()
        if not pairs:
            log(f"[{tag}] injection check: probe captured nothing")
        for Lq, B, pre, post in pairs:
            r = post / pre.clamp_min(1e-6)
            kind = "prefix forward, no injection expected" if B == 1 and Lq > 32 else "suffix forward, marker = index 0"
            log(f"[{tag}] injection check @layer {inject_layer} [{kind}; B={B} L={Lq}]: marker-row norm ratio post/pre = "
                f"{r.mean():.3f} (min {r.min():.3f} max {r.max():.3f}; ~1.41 expected on the suffix, 1.00 = no injection) -> "
                f"{'OK' if (r.mean() > 1.2) == (kind.startswith('suffix')) else 'FAIL'}")


# ---------------------------------------------------------------------------------------------------------------
# full-model checkpoints in the base repo layout
# ---------------------------------------------------------------------------------------------------------------
def _base_snapshot_dir(model_id):
    """Local snapshot dir of the base repo (offline: resolves from HF_HOME)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id)


def prepare_nontext_shard(model_id, out_path, log=print):
    """Copy every base tensor that is NOT part of the text model (vision tower, MTP head) into one safetensors file,
    so a checkpoint directory carries the exact tensor set of the base repo. Returns the list of tensor names."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    snap = _base_snapshot_dir(model_id)
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    extra = sorted(k for k in wm if not k.startswith(_LM_PREFIX) and k != "lm_head.weight")
    if os.path.exists(out_path):
        return extra
    tensors, by_file = {}, {}
    for k in extra:
        by_file.setdefault(wm[k], []).append(k)
    t0 = time.time()
    for fn, keys in by_file.items():
        with safe_open(f"{snap}/{fn}", framework="pt", device="cpu") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k).contiguous()
    tmp = out_path + ".tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, out_path)
    log(f"[fullft] non-text base tensors ({len(extra)}, {sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30:.2f} GB) "
        f"staged at {out_path} in {time.time() - t0:.0f}s")
    return extra


def map_name(name):
    """Qwen3_5ForCausalLM param name -> base repo (Qwen3_5ForConditionalGeneration) tensor name."""
    if name.startswith("model."):
        return _LM_PREFIX + name[len("model."):]
    return name   # lm_head.weight


def save_full_ckpt(model, path, tok, model_id, is_main, world, nontext_shard=None, shard_bytes=4 * 2**30, log=print, extra_meta=None,
                   gather=None):
    """COLLECTIVE (every rank must call it): all-gather each FSDP2-sharded param (one at a time), rank 0 writes bf16
    safetensors shards in the base layout + config/tokenizer files + the non-text shard + SAVE_DONE. Atomic via
    <path>.tmp -> rename. `gather(name, param) -> full tensor` overrides the per-param gathering (sft/tp.py's
    TPModel.gather_full for tensor-parallel shards); default = DTensor.full_tensor() / the param itself."""
    from safetensors.torch import save_file
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        DTensor = ()

    t0 = time.time()
    tmp = path + ".tmp"
    if is_main:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
    snap = _base_snapshot_dir(model_id) if is_main else None
    base_keys = set(json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]) if is_main else None

    weight_map, cur, cur_bytes, parts, total_bytes, n_tensors = {}, {}, 0, [], 0, 0

    def flush():
        nonlocal cur, cur_bytes
        if not cur:
            return
        fn = f"part-{len(parts):05d}.safetensors"
        save_file(cur, f"{tmp}/{fn}", metadata={"format": "pt"})
        parts.append(fn)
        for k in cur:
            weight_map[k] = fn
        cur, cur_bytes = {}, 0

    for name, p in model.named_parameters():   # identical order on every rank -> the all-gathers line up
        with torch.no_grad():
            if gather is not None:
                full = gather(name, p)
            else:
                full = p.full_tensor() if isinstance(p, DTensor) else p.detach()
        if is_main:
            t = full.detach().to(torch.bfloat16).cpu().contiguous()
            k = map_name(name)
            assert k in base_keys, f"{name} -> {k} is not a tensor of {model_id}"
            cur[k] = t
            cur_bytes += t.numel() * t.element_size()
            total_bytes += t.numel() * t.element_size()
            n_tensors += 1
            if cur_bytes >= shard_bytes:
                flush()
        del full
    if is_main:
        flush()
        n_parts = len(parts)
        final_names = {fn: f"model-{i + 1:05d}-of-{n_parts:05d}.safetensors" for i, fn in enumerate(parts)}
        for fn, new in final_names.items():
            os.replace(f"{tmp}/{fn}", f"{tmp}/{new}")
        weight_map = {k: final_names[fn] for k, fn in weight_map.items()}
        if nontext_shard and os.path.exists(nontext_shard):
            from safetensors import safe_open
            shutil.copy(nontext_shard, f"{tmp}/model-nontext.safetensors")
            with safe_open(nontext_shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_map[k] = "model-nontext.safetensors"
        missing = base_keys - set(weight_map)
        assert not missing, f"checkpoint misses {len(missing)} base tensors, e.g. {sorted(missing)[:5]}"
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                  open(f"{tmp}/model.safetensors.index.json", "w"), indent=1)
        for fn in os.listdir(snap):   # config.json (ConditionalGeneration), tokenizer files, chat template, generation config
            if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn.startswith("."):
                continue
            src = os.path.join(snap, fn)
            if os.path.isfile(src):
                shutil.copy(os.path.realpath(src), f"{tmp}/{fn}")
        try:
            tok.save_pretrained(tmp)
        except Exception as e:  # noqa — base tokenizer files were copied above already
            log(f"[fullft] tok.save_pretrained failed ({e}); base tokenizer files kept")
        meta = {"format": "full_model_bf16_base_layout", "n_tensors_text": n_tensors, "n_parts": n_parts,
                "bytes_text": total_bytes, "saved_at": time.time(), "save_s": time.time() - t0, **(extra_meta or {})}
        json.dump(meta, open(f"{tmp}/SAVE_DONE", "w"), indent=1)   # written LAST, before the rename
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.replace(tmp, path)
        log(f"[fullft] saved full model -> {path} ({total_bytes / 2**30:.1f} GB text weights in {n_parts} shards"
            f"{' + non-text shard' if nontext_shard else ''}) in {time.time() - t0:.0f}s")
    if world > 1:
        dist.barrier()
