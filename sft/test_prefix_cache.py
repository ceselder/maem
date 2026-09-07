"""Equivalence + speed test for sft/prefix_cache.py on ONE GPU via Modal (ephemeral app `maemm-prefix-test`).

Image = sft/modal_sft.py's image with transformers replaced by the prefix-cache fork (PFX_TF_SPEC, default the
pinned fork commit) + flash-linear-attention==0.5.2. The real Qwen3.6-27B is loaded from the maemm-data volume's
HF cache with the LoRA adapter at /data/sft_mix/last5_rp/final (or a fresh rsLoRA with --fresh-lora).

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    modal run sft/test_prefix_cache.py::equiv --batch 8 --seed 0          # naive vs prefix-cached, autocast off+on
    modal run sft/test_prefix_cache.py::bench --steps 10 --compile 1      # ex/s table naive mb16 vs cached 16/64/128
    PFX_GPU=H200:1 PFX_TRITON=3.7.1 modal run ...                           # Hopper fallback (fla GDN bwd needs triton>=3.7.1)
    modal run sft/test_prefix_cache.py::share_step --batch 64 --grad-accum 8
    modal run sft/test_prefix_cache.py::bench --steps 5 \
        --configs cached_gradaccum:64x8,cached_sharedstep:64x8

Results are printed and also written locally to sft/results/prefix_cache_{equiv,bench}.json by the entrypoints.
"""

import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
FORK_SHA = "e52940e567ab9a991a1c971c1094e340233baff3"
TF_SPEC = os.environ.get("PFX_TF_SPEC", f"transformers @ git+https://github.com/ceselder/transformers@{FORK_SHA}")

app = modal.App("maemm-prefix-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        TF_SPEC,
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    .pip_install("flash-linear-attention==0.5.2")
)
_TRITON = os.environ.get("PFX_TRITON", "")
if _TRITON:
    image = image.pip_install(f"triton=={_TRITON}")
image = (
    image.env({"HF_HOME": "/data/hf_cache", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONPATH": "/pmx/helpers:/pmx"})
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "sft" / "prefix_cache.py", "/pmx/sft/prefix_cache.py")
    .add_local_file(REPO / "sft" / "pretrain.py", "/pmx/sft/pretrain.py")
)

vol = modal.Volume.from_name("maemm-data")
GPU = os.environ.get("PFX_GPU", "B200:1")
ADAPTER = "/data/sft_mix/last5_rp/final"


# ----------------------------------------------------------------------------------------------------------------
# container-side helpers
# ----------------------------------------------------------------------------------------------------------------
def _load(adapter: str, fresh_lora: bool, dtype: str = "bfloat16"):
    import sys
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    from mxf.config import MODEL, TrainConfig

    cfg = TrainConfig()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, dtype), attn_implementation="sdpa",
                                                 device_map={"": "cuda:0"})
    model.enable_input_require_grads()
    if fresh_lora or not os.path.exists(adapter):
        print(f"[load] fresh rsLoRA r{cfg.lora_r}/a{cfg.lora_alpha} (adapter={'absent ' + adapter if not fresh_lora else 'not requested'})", flush=True)
        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, use_rslora=True,
                                                 target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    else:
        print(f"[load] adapter {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    model.train()
    return tok, model


def _env_report():
    import torch, transformers, peft
    try:
        import fla
        fla_v = getattr(fla, "__version__", "?")
    except Exception as e:  # noqa
        fla_v = f"ABSENT ({e})"
    from transformers import cache_utils
    fork = hasattr(cache_utils, "_write_cached_state") and hasattr(cache_utils.LinearAttentionLayer, "batch_repeat_interleave")
    import triton
    # str(): torch.__version__ is a TorchVersion str-subclass, which the torch-less local client cannot unpickle
    rep = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__), "transformers": str(transformers.__version__),
           "transformers_fork": bool(fork), "peft": str(peft.__version__), "fla": str(fla_v), "triton": str(triton.__version__),
           "tf_spec": TF_SPEC}
    print("[env]", json.dumps(rep), flush=True)
    return rep


def _plain(results):
    """Round-trip through JSON so the return value holds only builtins (the local client has no torch)."""
    out = json.loads(json.dumps(results, default=str))
    print("RESULTS_JSON " + json.dumps(out), flush=True)   # log fallback if the client-side deserialization fails
    return out


def _random_batch(gen, B, d_model, vocab, min_tgt, max_tgt, eos):
    import torch
    vecs = torch.nn.functional.normalize(torch.randn(B, d_model, generator=gen), dim=-1)
    lens = torch.randint(min_tgt, max_tgt + 1, (B,), generator=gen).tolist()
    targets = [torch.randint(1000, min(vocab, 150_000), (n,), generator=gen).tolist() + [eos] for n in lens]
    return vecs, targets


def _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device, max_seq=192, use_cache=None,
                   injector=None, fwd=None, inject_dtype=None):
    """Exactly sft/pretrain.py's per-example padded path (L rounded to 64, <= max_seq; hook at the absolute marker).
    ``injector``/``fwd``: pretrain.py's --compile variant (FixedPositionInjector registered once + compiled forward)."""
    import contextlib
    import torch
    from mxf.config import STEER_COEFF
    from mxf.inject import hooked, make_inject_hook
    from sft.pretrain import autocast_region

    rows = [prompt_ids + list(t) for t in targets]
    B = len(rows)
    L = max(len(r) for r in rows)
    L = min(((L + 63) // 64) * 64, max_seq)
    ids = torch.full((B, L), 0, dtype=torch.long); labels = torch.full((B, L), -100, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.bool)
    for i, r in enumerate(rows):
        r = r[:L]
        ids[i, : len(r)] = torch.tensor(r); attn[i, : len(r)] = True
        labels[i, len(prompt_ids) : len(r)] = torch.tensor(r[len(prompt_ids):])
    if injector is not None:
        injector.set_vectors(torch.as_tensor(vecs).to(device, injector.vectors.dtype)); hook_cm = contextlib.nullcontext()
    else:
        hook = make_inject_hook([v[None] for v in vecs], [[marker]] * B, STEER_COEFF, device, inject_dtype or torch.bfloat16)
        hook_cm = hooked(submodule, hook)
    kw = {} if use_cache is None else {"use_cache": use_cache}
    call = fwd if fwd is not None else model
    with hook_cm, autocast_region(model, autocast_on):
        out = call(input_ids=ids.to(device), attention_mask=attn.to(device), labels=labels.to(device), **kw)
    return out, labels


def _flat_grads(params):
    import torch
    return torch.cat([p.grad.detach().flatten().float() for p in params if p.grad is not None])


def _cmp_logits(a, b):
    """a, b: [N, V] logits at matched positions. Returns dict of diffs (fp32 math)."""
    import torch
    a = a.float(); b = b.float()
    d = (a - b).abs()
    la = torch.log_softmax(a, -1); lb = torch.log_softmax(b, -1)
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "max_rel_to_scale": (d.max() / a.abs().max()).item(),
            "logprob_max_abs": (la - lb).abs().max().item(), "logprob_mean_abs": (la - lb).abs().mean().item(),
            "argmax_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item(), "n_positions": a.shape[0]}


def _cmp_grads(g1, g2):
    import torch
    cos = torch.nn.functional.cosine_similarity(g1[None], g2[None]).item()
    rel = ((g1 - g2).norm() / g1.norm()).item()
    return {"cosine": cos, "rel_l2": rel, "max_abs": (g1 - g2).abs().max().item(), "norm_ref": g1.norm().item(),
            "norm_other": g2.norm().item()}


# ----------------------------------------------------------------------------------------------------------------
# equivalence
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=5400)
def equiv_remote(adapter: str = ADAPTER, batch: int = 8, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
                 fresh_lora: bool = False, stock_demo: bool = False, compile_prefix: str = "", dtype: str = "bfloat16"):
    """dtype="float32": whole model in fp32 (fits one B200 at small batch) -- without bf16 rounding the naive and cached
    paths must agree to ~1e-5 relative, which separates 'exact algorithm + bf16 noise' from a real bug."""
    import sys
    import time
    import torch

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    env = _env_report()
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    from sft import prefix_cache as pcmod
    from sft.pretrain import autocast_region

    if stock_demo:  # show what stock transformers does on this path (image built with PFX_TF_SPEC=transformers==5.15.0)
        pcmod.check_transformers = lambda: None

    t0 = time.time()
    tok, model = _load(adapter, fresh_lora, dtype)
    print(f"[load] {time.time() - t0:.0f}s dtype={dtype}", flush=True)
    device = "cuda:0"
    tdtype = getattr(torch, dtype)
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    print(f"[prompt] len={len(prompt_ids)} marker={marker} tail_after_marker={len(prompt_ids) - marker - 1}", flush=True)
    submodule = get_layer(model, INJECT_LAYER)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"[model] trainable params {n_train / 1e6:.1f}M", flush=True)

    gen = torch.Generator().manual_seed(seed)
    vecs, targets = _random_batch(gen, batch, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
    print(f"[batch] B={batch} target lens {[len(t) for t in targets]}", flush=True)
    results = {"env": env, "prompt_len": len(prompt_ids), "marker": marker, "batch": batch, "seed": seed, "dtype": dtype,
               "target_lens": [len(t) for t in targets], "trainable_params": n_train, "runs": {}}

    pc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device, inject_dtype=tdtype)
    suffix_lens = [len(pc.suffix_prompt) + len(t) for t in targets]
    if dtype == "float32":
        compile_prefix = ""   # numerics test only

    for autocast_on in ((False,) if dtype == "float32" else (False, True)):
        tag = f"autocast_bf16={'on' if autocast_on else 'off'}" + (f" dtype={dtype}" if dtype != "bfloat16" else "")
        r = {}
        # ---- naive (reference) ----
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time()
        out_n, labels_n = _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device,
                                         inject_dtype=tdtype)
        out_n.loss.backward(); torch.cuda.synchronize()
        r["naive_time_s"] = time.time() - t0
        g_naive = _flat_grads(params)
        loss_naive = out_n.loss.item()
        # matched positions: naive absolute [marker, marker+suffix_len) <-> cached [0, suffix_len)
        naive_sel = torch.cat([out_n.logits[b, marker : marker + suffix_lens[b]] for b in range(batch)]).detach()
        del out_n

        # ---- prefix-cached ----
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        try:
            out_c = pc.forward(vecs, targets, autocast=lambda: autocast_region(model, autocast_on))
            out_c.loss.backward(); torch.cuda.synchronize()
        except Exception as e:  # noqa
            msg = f"{type(e).__name__}: {str(e)[:600]}"
            print(f"[{tag}] PREFIX-CACHED PATH FAILED: {msg}", flush=True)
            r["cached_error"] = msg
            results["runs"][tag] = r
            if stock_demo:
                continue
            raise
        r["cached_time_s"] = time.time() - t0
        g_cached = _flat_grads(params)
        loss_cached = out_c.loss.item()
        cached_sel = torch.cat([out_c.logits[b, : suffix_lens[b]] for b in range(batch)]).detach()
        assert cached_sel.shape == naive_sel.shape, (cached_sel.shape, naive_sel.shape)
        r["loss_naive"] = loss_naive; r["loss_cached"] = loss_cached; r["loss_abs_diff"] = abs(loss_naive - loss_cached)
        r["logits"] = _cmp_logits(naive_sel, cached_sel)
        r["grads"] = _cmp_grads(g_naive, g_cached)
        r["n_target_tokens"] = out_c.n_target_tokens; r["suffix_len_padded"] = out_c.suffix_len
        del out_c, cached_sel

        N = int((labels_n != -100).sum())
        half = batch // 2
        ac = lambda: autocast_region(model, autocast_on)  # noqa

        # ---- shared prefix across micro-batches (grad accumulation): ONE prefix forward, two half-batches each on a
        # copy-expanded cache, token-weighted losses, retain_graph on the prefix graph until the last one ----
        model.zero_grad(set_to_none=True)
        cache0 = pc.run_prefix(ac)
        loss_acc = 0.0; acc_sel = []
        for k, (lo, hi) in enumerate(((0, half), (half, batch))):
            out_h = pc.forward(vecs[lo:hi], targets[lo:hi], autocast=ac, prefix_cache=cache0)
            (out_h.loss * (out_h.n_target_tokens / N)).backward(retain_graph=k == 0)
            loss_acc += out_h.loss.item() * out_h.n_target_tokens / N
            acc_sel.append(torch.cat([out_h.logits[b, : suffix_lens[lo + b]] for b in range(hi - lo)]).detach())
            del out_h
        del cache0
        g_acc = _flat_grads(params); acc_sel = torch.cat(acc_sel)
        r["accum_loss_abs_diff"] = abs(loss_naive - loss_acc)
        r["accum_logits"] = _cmp_logits(naive_sel, acc_sel); r["accum_grads"] = _cmp_grads(g_naive, g_acc)
        del acc_sel, g_acc
        print(f"[{tag}] shared-prefix 2x{half}: loss |d|={r['accum_loss_abs_diff']:.2e} logits {json.dumps({k: round(v, 6) for k, v in r['accum_logits'].items()})} "
              f"grads {json.dumps({k: round(v, 6) for k, v in r['accum_grads'].items()})}", flush=True)

        # ---- compiled prefix (torch.compile reduce-overhead on the static prefix call only) ----
        if compile_prefix:
            model.zero_grad(set_to_none=True)
            pcc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device,
                                    compile_prefix=compile_prefix)
            torch.cuda.synchronize(); t0 = time.time()
            out_cc = pcc.forward(vecs, targets, autocast=ac)   # compile happens here
            torch.cuda.synchronize(); r["pfxcompile_first_call_s"] = time.time() - t0
            out_cc.loss.backward()
            g_cc = _flat_grads(params); cc_sel = torch.cat([out_cc.logits[b, : suffix_lens[b]] for b in range(batch)]).detach()
            r["pfxcompile_loss_abs_diff"] = abs(loss_naive - out_cc.loss.item())
            r["pfxcompile_logits"] = _cmp_logits(naive_sel, cc_sel); r["pfxcompile_grads"] = _cmp_grads(g_naive, g_cc)
            del out_cc, g_cc, cc_sel
            # second call = steady state timing of the compiled prefix (fwd only, synced)
            model.zero_grad(set_to_none=True)
            tm = {}
            out_cc = pcc.forward(vecs, targets, autocast=ac, timings=tm); out_cc.loss.backward(); del out_cc
            r["pfxcompile_phases_s"] = tm
            print(f"[{tag}] compiled-prefix({compile_prefix}): first call {r['pfxcompile_first_call_s']:.0f}s, steady prefix_fwd "
                  f"{1000 * tm['prefix_fwd_s']:.1f}ms; loss |d|={r['pfxcompile_loss_abs_diff']:.2e} "
                  f"logits {json.dumps({k: round(v, 6) for k, v in r['pfxcompile_logits'].items()})} "
                  f"grads {json.dumps({k: round(v, 6) for k, v in r['pfxcompile_grads'].items()})}", flush=True)
            del pcc; torch._dynamo.reset()

        # ---- noise-floor control: the SAME naive computation split into two half-batches (different GEMM shapes
        # -> bf16 kernel/tiling noise), token-weighted so the total loss/grad equals the full-batch mean ----
        model.zero_grad(set_to_none=True)
        loss_ctrl = 0.0; ctrl_sel = []
        for lo, hi in ((0, half), (half, batch)):
            out_h, labels_h = _naive_forward(model, prompt_ids, marker, vecs[lo:hi], targets[lo:hi], submodule, autocast_on, device,
                                             inject_dtype=tdtype)
            n_h = int((labels_h != -100).sum())
            (out_h.loss * (n_h / N)).backward()
            loss_ctrl += out_h.loss.item() * n_h / N
            ctrl_sel.append(torch.cat([out_h.logits[b, marker : marker + suffix_lens[lo + b]] for b in range(hi - lo)]).detach())
            del out_h
        g_ctrl = _flat_grads(params)
        ctrl_sel = torch.cat(ctrl_sel)
        r["control_loss_abs_diff"] = abs(loss_naive - loss_ctrl)
        r["control_logits"] = _cmp_logits(naive_sel, ctrl_sel)
        r["control_grads"] = _cmp_grads(g_naive, g_ctrl)
        del ctrl_sel, g_ctrl, g_cached, g_naive, naive_sel
        model.zero_grad(set_to_none=True)
        results["runs"][tag] = r
        print(f"\n[{tag}] loss naive {loss_naive:.6f} cached {loss_cached:.6f} (|d|={r['loss_abs_diff']:.2e}; control |d|={r['control_loss_abs_diff']:.2e})")
        print(f"[{tag}] logits  cached-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['logits'].items()})}")
        print(f"[{tag}] logits control(half-batches)-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['control_logits'].items()})}")
        print(f"[{tag}] grads   cached-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['grads'].items()})}")
        print(f"[{tag}] grads   control-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['control_grads'].items()})}", flush=True)
    return _plain(results)


@app.local_entrypoint()
def equiv(adapter: str = ADAPTER, batch: int = 8, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
          fresh_lora: bool = False, stock_demo: bool = False, compile_prefix: str = "", dtype: str = "bfloat16", out: str = ""):
    res = equiv_remote.remote(adapter=adapter, batch=batch, seed=seed, min_tgt=min_tgt, max_tgt=max_tgt,
                              fresh_lora=fresh_lora, stock_demo=stock_demo, compile_prefix=compile_prefix, dtype=dtype)
    default_name = "prefix_cache_equiv_stock.json" if stock_demo else ("prefix_cache_equiv_fp32.json" if dtype == "float32" else "prefix_cache_equiv.json")
    path = Path(out) if out else REPO / "sft" / "results" / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


# ----------------------------------------------------------------------------------------------------------------
# speed
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=7200)
def bench_remote(adapter: str = ADAPTER, steps: int = 10, warmup: int = 3, seed: int = 0, min_tgt: int = 8,
                 max_tgt: int = 32, configs: str = "naive:16,naive_nocache:16,cached:16,cached:64,cached:128",
                 compile_: bool = False, fresh_lora: bool = False, autocast_on: bool = True):
    import sys
    import time
    import torch

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    env = _env_report()
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    from mxf.inject import FixedPositionInjector, get_layer
    from mxf.prompts import build_prompt_ids
    from sft import prefix_cache as pcmod
    from sft.pretrain import autocast_region

    tok, model = _load(adapter, fresh_lora)
    device = "cuda:0"
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    submodule = get_layer(model, INJECT_LAYER)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)   # lr 0: identical weights for every config
    gen = torch.Generator().manual_seed(seed)
    results = {"env": env, "prompt_len": len(prompt_ids), "marker": marker, "steps": steps, "warmup": warmup,
               "autocast_bf16": autocast_on, "target_len_range": [min_tgt, max_tgt], "rows": []}

    def batch_gen(B, force_len=None):
        """Random micro-batch; force_len pins every target length (warm up the shortest/longest suffix shapes)."""
        if force_len is None:
            return _random_batch(gen, B, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
        return _random_batch(gen, B, D_MODEL, tok.vocab_size, force_len, force_len, tok.eos_token_id)

    def step_naive(B, use_cache, injector=None, fwd=None, force_len=None):
        vecs, targets = batch_gen(B, force_len)
        out, labels = _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device,
                                     use_cache=use_cache, injector=injector, fwd=fwd)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        return B, int((labels != -100).sum()), out.logits.shape[1]

    def make_naive_compiled(B):
        """pretrain.py --compile: FixedPositionInjector at the absolute marker + torch.compile(model.forward)."""
        inj = FixedPositionInjector(B, D_MODEL, marker, STEER_COEFF, device, torch.bfloat16)
        handle = submodule.register_forward_hook(inj.hook)
        fwd = torch.compile(model.forward)

        def step(force_len=None):
            return step_naive(B, None, injector=inj, fwd=fwd, force_len=force_len)

        def cleanup():
            handle.remove(); torch._dynamo.reset()
        return step, cleanup

    def make_cached(B, use_compile, n_accum=1, compile_prefix=None, compile_mlp=False, pad_multiple=8):
        """n_accum > 1: ONE prefix per optimizer step shared by n_accum micro-batches of B (token-weighted losses, so
        the step equals a single mean-loss step over n_accum*B examples). compile_prefix: torch.compile the static
        prefix call only. compile_mlp: regional torch.compile of the 64 MLP blocks (cache never enters a graph)."""
        inj = None
        undo_mlp = pcmod.compile_mlp_blocks(model) if compile_mlp else None
        if use_compile:
            inj = FixedPositionInjector(B, D_MODEL, 0, STEER_COEFF, device, torch.bfloat16)
            handle = submodule.register_forward_hook(inj.hook)
            torch._dynamo.config.cache_size_limit = 64
            fwd = torch.compile(model.forward, dynamic=True)
        pc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device,
                               persistent_injector=inj, compile_prefix=compile_prefix, pad_multiple=pad_multiple)
        if use_compile:
            class _Wrap:  # route both forwards through the compiled function
                def __call__(self, **kw):
                    return fwd(**kw)
            pc.model = pc.prefix_model = pc._prefix_fn = _Wrap()
        ac = lambda: autocast_region(model, autocast_on)  # noqa

        def step(force_len=None):
            if n_accum == 1:
                vecs, targets = batch_gen(B, force_len)
                out = pc.forward(vecs, targets, autocast=ac)
                out.loss.backward()
                n_tok, L = out.n_target_tokens, out.suffix_len
                del out
            else:
                mbs = [batch_gen(B, force_len) for _ in range(n_accum)]
                n_tot = sum(len(t) for _, tg in mbs for t in tg)
                cache0 = pc.run_prefix(ac)
                n_tok = 0; L = 0
                for k, (vecs, targets) in enumerate(mbs):
                    out = pc.forward(vecs, targets, autocast=ac, prefix_cache=cache0)
                    (out.loss * (out.n_target_tokens / n_tot)).backward(retain_graph=k < n_accum - 1)
                    n_tok += out.n_target_tokens; L = max(L, out.suffix_len)
                    del out
                del cache0
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            return B * n_accum, n_tok, L

        def cleanup():
            if use_compile:
                handle.remove()
            if undo_mlp is not None:
                undo_mlp()
            if use_compile or compile_prefix or compile_mlp:
                torch._dynamo.reset()
        return step, cleanup

    def run_case(name, B, fn, cleanup=None):
        row = {"config": name, "micro_batch": B}
        try:
            gen.manual_seed(seed)  # identical examples/lengths across compared configurations
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t_first = time.time(); fn(); torch.cuda.synchronize(); row["first_step_s"] = time.time() - t_first
            # warm up BOTH shape extremes: Triton specializes int args (seq len) on divisibility by 16, so a new
            # suffix length inside the measured window would otherwise pay a recompile
            t_w = time.time(); fn(force_len=min_tgt); fn(force_len=max_tgt); torch.cuda.synchronize()
            row["shape_warmup_s"] = time.time() - t_w
            for _ in range(max(warmup - 3, 0)):
                fn()
            torch.cuda.synchronize(); t0 = time.time()
            n_ex = n_tok = 0; seq = []
            for _ in range(steps):
                b, nt, L = fn(); n_ex += b; n_tok += nt; seq.append(L)
            torch.cuda.synchronize(); dt = time.time() - t0
            row.update({"examples_per_s": n_ex / dt, "target_tokens_per_s": n_tok / dt, "ms_per_step": 1000 * dt / steps,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "mean_seq_len": sum(seq) / len(seq)})
            # diagnostic: the same steps with a device sync after each one (per-step distribution; also tells whether
            # letting the CPU run ahead of the GPU helps or hurts this launch-heavy path)
            per = []
            for _ in range(steps):
                ts = time.time(); fn(); torch.cuda.synchronize(); per.append(time.time() - ts)
            per_s = sorted(per)
            row.update({"ms_per_step_synced": 1000 * sum(per) / len(per), "ms_step_min": 1000 * per_s[0],
                        "ms_step_median": 1000 * per_s[len(per) // 2], "ms_step_max": 1000 * per_s[-1]})
            print(f"[bench] {name:>14} mb={B:<4} {row['examples_per_s']:8.2f} ex/s  {row['target_tokens_per_s']:8.0f} tgt-tok/s  "
                  f"{row['ms_per_step']:8.1f} ms/step  peak {row['peak_mem_gb']:.1f} GB  L={row['mean_seq_len']:.0f}  first {row['first_step_s']:.1f}s  "
                  f"| synced {row['ms_per_step_synced']:.0f} ms (min {row['ms_step_min']:.0f} / med {row['ms_step_median']:.0f} / max {row['ms_step_max']:.0f})", flush=True)
        except torch.OutOfMemoryError as e:
            row["error"] = f"OOM: {str(e)[:300]}"; print(f"[bench] {name} mb={B} OOM", flush=True)
            opt.zero_grad(); torch.cuda.empty_cache()
        except Exception as e:  # noqa
            row["error"] = f"{type(e).__name__}: {str(e)[:800]}"; print(f"[bench] {name} mb={B} FAILED {row['error']}", flush=True)
            opt.zero_grad(); torch.cuda.empty_cache()
        finally:
            if cleanup:
                cleanup()
        results["rows"].append(row)

    def make_step_accum(B, n_accum, share):
        """Match pretrain.py --grad-accum: mean of micro-batch token-mean losses."""
        pc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device)
        ac = lambda: autocast_region(model, autocast_on)

        def step(force_len=None):
            accum = pcmod.PrefixGradientAccumulator(pc.run_prefix(ac)) if share else None
            n_tok = L = 0
            for _ in range(n_accum):
                vecs, targets = batch_gen(B, force_len)
                out = pc.forward(vecs, targets, autocast=ac,
                                 prefix_cache=accum.cache if accum is not None else None)
                (out.loss / n_accum).backward()
                if accum is not None:
                    accum.accumulate()
                n_tok += out.n_target_tokens
                L = max(L, out.suffix_len)
                del out
            if accum is not None:
                accum.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            return B * n_accum, n_tok, L

        return step

    for spec in configs.split(","):
        kind, Bs = spec.split(":")
        B, n_acc = (int(Bs.split("x")[0]), int(Bs.split("x")[1])) if "x" in Bs else (int(Bs), 1)
        label = f"{kind}" + (f" {B}x{n_acc}" if n_acc > 1 else "")
        if kind == "naive":
            run_case("naive", B, lambda force_len=None, B=B: step_naive(B, None, force_len=force_len))
        elif kind == "naive_nocache":
            run_case("naive_nocache", B, lambda force_len=None, B=B: step_naive(B, False, force_len=force_len))
        elif kind == "naive_compile":
            fn, cl = make_naive_compiled(B); run_case("naive_compile", B, fn, cl)
        elif kind == "cached":
            fn, cl = make_cached(B, False, n_acc); run_case(label, B * n_acc, fn, cl)
        elif kind == "cached_pfxcompile":
            fn, cl = make_cached(B, False, n_acc, compile_prefix="reduce-overhead"); run_case(label, B * n_acc, fn, cl)
        elif kind == "cached_pfxcompile_default":
            fn, cl = make_cached(B, False, n_acc, compile_prefix="default"); run_case(label, B * n_acc, fn, cl)
        elif kind == "cached_compile":
            fn, cl = make_cached(B, True, n_acc); run_case(label, B * n_acc, fn, cl)
        elif kind == "cached_mlpcompile":
            fn, cl = make_cached(B, False, n_acc, compile_mlp=True); run_case(label, B * n_acc, fn, cl)
        elif kind == "cached_pad16":
            fn, cl = make_cached(B, False, n_acc, pad_multiple=16); run_case(label, B * n_acc, fn, cl)
        elif kind in ("cached_gradaccum", "cached_sharedstep"):
            run_case(label, B * n_acc, make_step_accum(B, n_acc, kind == "cached_sharedstep"))
        else:
            raise ValueError(spec)
    if compile_:
        for B in (16, 64):
            fn, cl = make_cached(B, True); run_case("cached_compile", B, fn, cl)
    return _plain(results)


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=1800)
def share_step_remote(adapter: str = ADAPTER, batch: int = 64, grad_accum: int = 8, seed: int = 0,
                      fresh_lora: bool = False):
    """B200 BF16 loss/gradient parity: repeated prefix vs one prefix per optimizer step."""
    import sys
    import torch

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    from sft.prefix_cache import PrefixCache, PrefixGradientAccumulator
    from sft.pretrain import autocast_region

    assert batch > 0 and grad_accum > 0
    env = _env_report()
    tok, model = _load(adapter, fresh_lora)
    params = [p for p in model.parameters() if p.requires_grad]
    prompt, marker = build_prompt_ids(tok)
    pc = PrefixCache(model, prompt, marker[0], tok.pad_token_id, get_layer(model, INJECT_LAYER),
                     STEER_COEFF, "cuda:0")
    ac = lambda: autocast_region(model, True)
    gen = torch.Generator().manual_seed(seed)
    data = [_random_batch(gen, batch, D_MODEL, tok.vocab_size, 8, 32, tok.eos_token_id)
            for _ in range(grad_accum)]

    def backward(share):
        model.zero_grad(set_to_none=True)
        accum = PrefixGradientAccumulator(pc.run_prefix(ac)) if share else None
        loss = torch.zeros((), device="cuda")
        for vecs, targets in data:
            out = pc.forward(vecs, targets, autocast=ac, prefix_cache=accum.cache if share else None)
            (out.loss / grad_accum).backward()
            loss += out.loss.detach() / grad_accum
            if share:
                accum.accumulate()
            del out
        if share:
            accum.backward()
        return float(loss), _flat_grads(params).cpu()

    loss_ref, grads_ref = backward(False)
    loss_shared, grads_shared = backward(True)
    cmp = _cmp_grads(grads_ref, grads_shared)
    result = {"env": env, "batch": batch, "grad_accum": grad_accum,
              "loss_repeated": loss_ref, "loss_shared": loss_shared,
              "loss_abs_diff": abs(loss_ref - loss_shared), "gradients": cmp}
    # A shared bf16 prefix changes reduction order, not the objective. Report the
    # complete metrics and reject differences beyond the existing bf16 parity regime.
    result["ok"] = abs(loss_ref - loss_shared) < 1e-3 and cmp["cosine"] > 0.999 and cmp["rel_l2"] < 0.02
    _plain(result)
    assert result["ok"], f"shared-prefix gradient parity failed: {result}"
    return result


@app.local_entrypoint()
def share_step(adapter: str = ADAPTER, batch: int = 64, grad_accum: int = 8, seed: int = 0,
               fresh_lora: bool = False):
    result = share_step_remote.remote(adapter=adapter, batch=batch, grad_accum=grad_accum,
                                      seed=seed, fresh_lora=fresh_lora)
    out = REPO / "sft" / "results" / "prefix_share_step_equiv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


@app.local_entrypoint()
def bench(adapter: str = ADAPTER, steps: int = 10, warmup: int = 3, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
          configs: str = "naive:16,naive_nocache:16,cached:16,cached:64,cached:128", compile: int = 0,
          fresh_lora: bool = False, out: str = ""):
    res = bench_remote.remote(adapter=adapter, steps=steps, warmup=warmup, seed=seed, min_tgt=min_tgt, max_tgt=max_tgt,
                              configs=configs, compile_=bool(compile), fresh_lora=fresh_lora)
    path = Path(out) if out else REPO / "sft" / "results" / "prefix_cache_bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


# ----------------------------------------------------------------------------------------------------------------
# profiling: where does the cached step spend its time? (CUDA-synchronized phases + torch.profiler kernel table)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=5400)
def profile_remote(adapter: str = ADAPTER, batch: int = 16, steps: int = 5, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
                   pad_multiple: int = 8, fresh_lora: bool = False, autocast_on: bool = True, naive_batch: int = 16):
    import sys
    import time
    import torch

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    env = _env_report()
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    from sft import prefix_cache as pcmod
    from sft.pretrain import autocast_region

    tok, model = _load(adapter, fresh_lora)
    device = "cuda:0"
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    submodule = get_layer(model, INJECT_LAYER)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    pc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device, pad_multiple=pad_multiple)
    ac = lambda: autocast_region(model, autocast_on)  # noqa
    results = {"env": env, "batch": batch, "pad_multiple": pad_multiple, "phases_cached": [], "phases_naive": [], "phases_prefix_only": []}

    def sync():
        torch.cuda.synchronize(); return time.time()

    def cached_step(record=None):
        vecs, targets = _random_batch(gen, batch, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
        tm = {} if record is not None else None
        t0 = sync()
        out = pc.forward(vecs, targets, autocast=ac, timings=tm)
        t1 = sync(); out.loss.backward(); t2 = sync()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); t3 = sync()
        if record is not None:
            tm.update(backward_s=t2 - t1, opt_s=t3 - t2, total_s=t3 - t0, suffix_len=out.suffix_len)
            record.append(tm)

    def naive_step(record=None):
        vecs, targets = _random_batch(gen, naive_batch, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
        t0 = sync()
        out, labels = _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device)
        t1 = sync(); out.loss.backward(); t2 = sync()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); t3 = sync()
        if record is not None:
            record.append({"fwd_s": t1 - t0, "backward_s": t2 - t1, "opt_s": t3 - t2, "total_s": t3 - t0, "L": out.logits.shape[1]})

    def prefix_only_step(record=None):
        """B=1 x prefix-length forward+backward through the same stack: the per-step floor the cached path pays."""
        ids = pc._prefix_tensor
        t0 = sync()
        with ac():
            out = model(input_ids=ids, labels=ids, use_cache=False)
        t1 = sync(); out.loss.backward(); t2 = sync(); opt.zero_grad()
        if record is not None:
            record.append({"fwd_s": t1 - t0, "backward_s": t2 - t1, "total_s": t2 - t0})

    for _ in range(3):
        cached_step(); naive_step(); prefix_only_step()
    for _ in range(steps):
        cached_step(results["phases_cached"]); naive_step(results["phases_naive"]); prefix_only_step(results["phases_prefix_only"])

    def _mean(rows, k):
        return sum(r[k] for r in rows) / len(rows)
    for name, rows in (("cached", results["phases_cached"]), ("naive", results["phases_naive"]), ("prefix_only(B=1)", results["phases_prefix_only"])):
        print(f"[phases] {name:>17} " + "  ".join(f"{k}={1000 * _mean(rows, k):7.1f}ms" for k in rows[0] if k.endswith("_s")), flush=True)

    # torch.profiler: one cached step and one naive step (kernel table + CPU/GPU totals)
    from torch.profiler import ProfilerActivity, profile
    for name, fn in (("cached", cached_step), ("naive", naive_step)):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
        ka = prof.key_averages()
        cuda_total = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka) / 1e6
        cpu_total = sum(e.self_cpu_time_total for e in ka) / 1e6
        results[f"profile_{name}"] = {"self_cuda_total_s": cuda_total, "self_cpu_total_s": cpu_total}
        print(f"\n[profile] {name}: self CUDA total {cuda_total:.3f}s, self CPU total {cpu_total:.3f}s", flush=True)
        sort_key = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
        print(ka.table(sort_by=sort_key, row_limit=25), flush=True)
        print(ka.table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
    return _plain(results)


@app.local_entrypoint()
def profile(adapter: str = ADAPTER, batch: int = 16, steps: int = 5, seed: int = 0, pad_multiple: int = 8,
            fresh_lora: bool = False, naive_batch: int = 16, out: str = ""):
    res = profile_remote.remote(adapter=adapter, batch=batch, steps=steps, seed=seed, pad_multiple=pad_multiple,
                                fresh_lora=fresh_lora, naive_batch=naive_batch)
    path = Path(out) if out else REPO / "sft" / "results" / "prefix_cache_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")
