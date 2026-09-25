"""Dr. GRPO-style RL for the MAEMM inverter — HF-generate rollouts, LoRA actor, data-parallel over groups.

One step:
  1. sample B directions (a contiguous slice of WHOLE groups per DDP rank);
  2. rollout (--rollout-engine hf | vllm): G samples per direction at T=1 with unit(dir) injected
     (norm-matched) at the prompt marker through INJECT_LAYER. vllm: a per-rank vLLM engine (child
     process) generates with the CURRENT LoRA (re-saved each step -> LoRARequest) and vllm_lens
     per-request steering; old_logp is ALWAYS recomputed HF-side with the SAME hook update() uses
     (ratio == 1 on-policy), and every step the vLLM per-token logprobs are checked against the HF
     recompute (--vllm-logp-tol) so the sampler provably IS the training policy;
  3. score(): every generation re-tokenized STANDALONE (sink-prepended, no chat template) through the
     CLEAN base model (adapter disabled, no injection); reward = [log1p] max cosine between the
     READ_LAYER residual and unit(dir) over the LAST --reward-window-last kept tokens (0 = all);
  4. gates (clean-base fluency floor, distinct-token floor) subtract --gate-penalty; length beyond
     --len-penalty-start costs --len-penalty-per-tok (valid rollouts only);
  5. compute_advantages(): r - group_mean, optionally / per-group std (--std-norm) or / one
     GLOBAL-batch std with zero-variance groups dropped (--batch-norm, ScaleRL);
  6. update(): ONE clipped policy-gradient step, per-token normalized over the global batch,
     optional --entropy-coef bonus and --kl-coef capped-k3 KL to the frozen init adapter.

    torchrun --nproc_per_node=8 rl/rl.py --data-dir <pool> --init-adapter <sft> ...
    python rl/rl.py --groups-per-step 2 --group-size 4 --total-steps 2 --no-wandb   # 1-GPU smoke
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")   # collective_rpc(callable) for the steer counter

# ---- ONE visible GPU per DDP rank, set BEFORE torch is imported. Each rank and its vLLM engine child
# then address the same physical GPU as cuda:0 (vLLM maps current_device through CUDA_VISIBLE_DEVICES;
# narrowing it after CUDA init -> "Invalid device id"). The engine child re-imports this module with
# LOCAL_RANK hidden (init_vllm), so its inherited single-GPU setting is left untouched. ----
if "LOCAL_RANK" in os.environ:
    _lr = int(os.environ["LOCAL_RANK"])
    _cvd = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not _cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_lr)
    elif len(_cvd) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = _cvd[_lr]

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from mxf.config import D_MODEL, INJECT_LAYER, MODEL, MULTIMODAL, READ_LAYER, STEER_COEFF, RLConfig, TrainConfig
from mxf.inject import get_layer, hooked, make_inject_hook, read_resid
from mxf.prompts import build_prompt_ids


# ----------------------------------------------------------------------------------------------
# rollouts
# ----------------------------------------------------------------------------------------------
def _eos_ids(tok, actor):
    """Stop-token set shared by BOTH engines (tokenizer eos + generation_config eos ids)."""
    ids = set()

    def add(e):
        if isinstance(e, (list, tuple)):
            for x in e:
                ids.add(int(x))
        elif e is not None:
            ids.add(int(e))
    add(tok.eos_token_id)
    try:
        add(actor.generation_config.eos_token_id)
    except Exception:
        pass
    return ids


SCORE_STATS = {}   # side-stats from the last score() call (peak position etc.), picked up by the step log


def _trim_at_stop(g, eos_ids):
    """Keep tokens up to and INCLUDING the first stop token; drop any pad tail."""
    trimmed = []
    for t in g:
        trimmed.append(t)
        if t in eos_ids:
            break
    return trimmed if trimmed else g


@torch.no_grad()
def _old_logp(actor, submodule, prompt, row_gen, vecs, marker, tok, device):
    """Behaviour-policy logps of the sampled tokens: a no-grad forward over prompt+gen with the SAME
    inject hook update() uses (log_softmax of RAW logits gathered at the sampled tokens). Used by
    both engines so old_logp/new_logp are computed identically (ratio == 1 at step 0)."""
    bsz, p_len = len(row_gen), prompt.shape[0]
    gmax = max(len(g) for g in row_gen)
    ids_f = torch.full((bsz, p_len + gmax), tok.pad_token_id, dtype=torch.long, device=device)
    attn_f = torch.zeros((bsz, p_len + gmax), dtype=torch.long, device=device)
    ids_f[:, :p_len] = prompt
    for i, g in enumerate(row_gen):
        ids_f[i, p_len : p_len + len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        attn_f[i, : p_len + len(g)] = 1
    hook = make_inject_hook(vecs, [[marker]] * bsz, STEER_COEFF, device, torch.bfloat16)
    with hooked(submodule, hook):
        logits = actor(input_ids=ids_f, attention_mask=attn_f).logits[:, p_len - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids_f[:, p_len:, None]).squeeze(-1)  # [bsz, gmax]
    del logits
    return [lp[i, : len(g)].detach().float().cpu() for i, g in enumerate(row_gen)]


def rollout(actor, submodule, tok, prompt_ids, marker, dirs, a, device):
    """HF-generate engine: B groups x G rollouts in --rollout-chunk mini-batches. dirs: [B, d].
    Returns flat group-major lists (texts, gen_ids, old_logps); rollout i belongs to group i // G.
    The hook fires only at PREFILL (h.shape[1] > 1) and adds unit(dir) at the marker. Sampling is
    pure temperature-1 softmax (top_p=1, top_k off, no repetition penalty)."""
    G = a.group_size
    all_dirs = dirs.repeat_interleave(G, 0).to(device)          # [B*G, d] group-major
    N = all_dirs.shape[0]
    p_len = len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    eos_ids = _eos_ids(tok, actor)
    texts, gen_ids, old_lps = [], [], []
    for s in range(0, N, a.rollout_chunk):
        e = min(s + a.rollout_chunk, N)
        bsz = e - s
        inp = prompt.unsqueeze(0).expand(bsz, -1).contiguous()          # shared prompt, left-aligned
        vecs = [all_dirs[s + i : s + i + 1] for i in range(bsz)]        # one direction per row
        hook = make_inject_hook(vecs, [[marker]] * bsz, STEER_COEFF, device, torch.bfloat16)
        with hooked(submodule, hook):
            out = actor.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True,
                                 temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0,
                                 repetition_penalty=1.0, max_new_tokens=a.max_new_tokens,
                                 min_new_tokens=a.min_new_tokens, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
        row_gen = [_trim_at_stop(g, eos_ids) for g in out[:, p_len:].tolist()]
        for c0 in range(0, bsz, a.logp_chunk):
            c1 = min(c0 + a.logp_chunk, bsz)
            old_lps += _old_logp(actor, submodule, prompt, row_gen[c0:c1], vecs[c0:c1], marker, tok, device)
        gen_ids += row_gen
        texts += [tok.decode(g, skip_special_tokens=True) for g in row_gen]
    return texts, gen_ids, old_lps


def init_vllm(a, local, rank, p_len, max_seqs):
    """Per-rank vLLM engine (TP=1, child process) on THIS rank's GPU. Built AFTER the HF actor (see
    main(): vllm's import clobbers transformers' Qwen3_5 AutoConfig entry). torchrun's distributed env is hidden from the child (it
    must not join our gloo world); the rank's single visible GPU (module top) is inherited. LoRA enabled so the
    current adapter is served via LoRARequest (vLLM reads rsLoRA alpha/sqrt(r) from
    adapter_config.json); prefix caching OFF (the shared prompt's KV carries the injected
    direction — reuse across requests would leak direction A into direction B); TRITON_ATTN +
    eager are what vllm_lens' steering needs."""
    hidden = {k: os.environ.pop(k) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                                             "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT",
                                             "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE")
              if k in os.environ}
    try:
        from vllm import LLM
        llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=a.vllm_gpu_mem,
                  max_model_len=p_len + a.max_new_tokens + 8, attention_backend="TRITON_ATTN",
                  enforce_eager=True, enable_prefix_caching=False, **({'language_model_only': True} if MULTIMODAL else {}),
                  enable_lora=True, max_loras=1, max_lora_rank=64, max_num_seqs=max(int(max_seqs), 1),
                  seed=a.seed * 1000 + rank, dtype="bfloat16")
    finally:
        os.environ.update(hidden)
    return llm


@torch.no_grad()
def _marker_norm(actor, submodule, prompt, marker, device, adapter=True):
    """||h|| at the marker of the INJECT_LAYER output for the shared prompt — the exact scalar
    make_inject_hook multiplies unit(dir) by (bf16 norm, as in the hook). A per-step constant: the
    prompt is shared and causal, so every rollout's marker activation is identical."""
    cap = {}

    def grab(_m, _i, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out
    hd = submodule.register_forward_hook(grab)
    try:
        ids = prompt.unsqueeze(0)
        if adapter:
            actor(input_ids=ids, attention_mask=torch.ones_like(ids))
        else:
            with actor.disable_adapter():
                actor(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        hd.remove()
    return cap["h"][0, marker].norm().float().item()


def _save_adapter_for_vllm(actor, lora_dir):
    """Write the CURRENT 'default' adapter in the module naming vLLM expects for this model.

    The actor is AutoModelForCausalLM (paths `model.layers.N...`) but vLLM serves the
    Qwen3_5ForConditionalGeneration wrapper, whose hf_to_vllm_mapper only knows
    `model.language_model.` -> `language_model.model.`. vLLM's LoRA loader validates module names by
    SUFFIX only, so a CausalLM-named adapter passes validation and is then SILENTLY ignored — every
    per-module lookup misses and the engine samples from the BASE model (measured: mean
    |logp_vllm - logp_hf| = 1.47 nats). Renaming to `model.language_model.layers.` fixes the lookup."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    os.makedirs(lora_dir, exist_ok=True)
    sd = get_peft_model_state_dict(actor, adapter_name="default")
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.detach().to("cpu", copy=True).contiguous()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    actor.peft_config["default"].save_pretrained(lora_dir)              # adapter_config.json (r, alpha, rslora, targets)
    return len(out)


def _steer_vec(v, hnorm, marker):
    """vllm_lens SteeringVector carrying the ABSOLUTE injection coeff*||h||*unit(v) (norm_match=False):
    lens adds the vector 1:1 into the residual stream, but its own norm_match would scale by the
    norm of vLLM's split-residual COMPONENT (~12% of the full stream) — verified numerically."""
    from vllm_lens import SteeringVector
    vec = (F.normalize(v.float(), dim=0) * (hnorm * STEER_COEFF)).view(1, 1, -1).cpu()
    return SteeringVector(activations=vec, layer_indices=[INJECT_LAYER], scale=1.0, norm_match=False,
                          position_indices=[marker])


@torch.no_grad()
def _hf_logp_conditions(actor, submodule, prompt, gen_ids, dirs_rep, marker, tok, device, chunk):
    """HF logps of the SAME sampled tokens under 4 policies: {adapter on/off} x {inject hook on/off}.
    Whichever condition matches vLLM's logprobs tells us what the engine is actually running."""
    out = {}
    for ad in (True, False):
        for hk in (True, False):
            lps = []
            for s0 in range(0, len(gen_ids), chunk):
                e0 = min(s0 + chunk, len(gen_ids))
                vecs = [dirs_rep[i : i + 1] for i in range(s0, e0)]
                if hk:
                    fn = lambda: _old_logp(actor, submodule, prompt, gen_ids[s0:e0], vecs, marker, tok, device)
                else:
                    fn = lambda: _old_logp_nohook(actor, prompt, gen_ids[s0:e0], tok, device)
                if ad:
                    lps += fn()
                else:
                    with actor.disable_adapter():
                        lps += fn()
            out[(ad, hk)] = lps
    return out


@torch.no_grad()
def _old_logp_nohook(actor, prompt, row_gen, tok, device):
    bsz, p_len = len(row_gen), prompt.shape[0]
    gmax = max(len(g) for g in row_gen)
    ids_f = torch.full((bsz, p_len + gmax), tok.pad_token_id, dtype=torch.long, device=device)
    attn_f = torch.zeros((bsz, p_len + gmax), dtype=torch.long, device=device)
    ids_f[:, :p_len] = prompt
    for i, g in enumerate(row_gen):
        ids_f[i, p_len : p_len + len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        attn_f[i, : p_len + len(g)] = 1
    logits = actor(input_ids=ids_f, attention_mask=attn_f).logits[:, p_len - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids_f[:, p_len:, None]).squeeze(-1)
    del logits
    return [lp[i, : len(g)].detach().float().cpu() for i, g in enumerate(row_gen)]


def _absdiff(hf_lps, vllm_lps):
    d = [abs(h - v) for hl, vl in zip(hf_lps, vllm_lps) for h, v in zip(hl.tolist(), vl) if v is not None]
    return (sum(d) / len(d)) if d else float("nan")


@torch.no_grad()
def verify_vllm_injection(llm, actor, submodule, prompt_ids, marker, device, seed=0):
    """One-time numeric PROOF that vllm_lens steering reaches the engine and matches our HF hook's
    norm-matched formula. Capture the INJECT_LAYER residual (vllm_lens output_residual_stream) for a
    clean and a steered request (base weights, greedy, 1 token): the marker-row delta must equal
    STEER_COEFF * ||h_clean|| * unit(v) (cos ~ 1, norm ratio ~ 1); rows before the marker must be
    untouched (causal prefill is deterministic in eager mode)."""
    from vllm import SamplingParams
    g = torch.Generator().manual_seed(seed)
    v = F.normalize(torch.randn(D_MODEL, generator=g), dim=0)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    hnorm_hf = _marker_norm(actor, submodule, prompt, marker, device, adapter=False)   # probe runs BASE weights

    def run(steer):
        extra = {"output_residual_stream": [INJECT_LAYER]}
        if steer:
            extra["apply_steering_vectors"] = [_steer_vec(v, hnorm_hf, marker)]
        out = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                           [SamplingParams(temperature=0.0, max_tokens=1, extra_args=extra)], use_tqdm=False)[0]
        act = getattr(out, "activations", None)
        assert act is not None and "residual_stream" in act, "vllm_lens capture returned nothing — plugin not active?"
        return act["residual_stream"][0].float()                     # [seq, d] (one captured layer)
    h_clean, h_steer = run(False), run(True)
    delta = h_steer[marker] - h_clean[marker]
    cos = F.cosine_similarity(delta, v, dim=0).item()
    ratio = (delta.norm() / (STEER_COEFF * hnorm_hf)).item()            # injected magnitude vs the HF hook's
    hnorm_vllm = h_clean[marker].norm().item()                          # vLLM's clean marker norm vs HF's
    other = (h_steer[:marker] - h_clean[:marker]).norm(dim=-1).max().item() if marker > 0 else 0.0
    return {"cos": cos, "norm_ratio": ratio, "hnorm_hf": hnorm_hf, "hnorm_vllm": hnorm_vllm,
            "hnorm_agree": hnorm_vllm / max(hnorm_hf, 1e-6), "max_other_row_delta": other,
            "ok": cos > 0.99 and 0.95 < ratio < 1.05 and 0.97 < hnorm_vllm / max(hnorm_hf, 1e-6) < 1.03}


def rollout_vllm(llm, actor, submodule, tok, prompt_ids, marker, dirs, a, device, step, rank):
    """vLLM engine: publish the CURRENT LoRA (save -> LoRARequest with a fresh id), then ONE
    generate() call with one request per direction, n=G samples each, the direction steered at the
    marker by vllm_lens (norm_match == our hook). Token-identical to rollout(): same prompt ids,
    same stop set (stop token kept), T=1 full softmax; old_logp from the HF recompute; the vLLM
    per-token logprobs are compared against it (stats returned; main() enforces --vllm-logp-tol)."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    G, p_len = a.group_size, len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    eos_ids = _eos_ids(tok, actor)
    hnorm = _marker_norm(actor, submodule, prompt, marker, device, adapter=True)  # current policy's ||h|| at the marker
    t0 = time.time()
    lora_dir = f"/tmp/rl_lora/rank{rank}/step{step}"
    _save_adapter_for_vllm(actor, lora_dir)
    t_save = time.time() - t0
    lora_req = LoRARequest(lora_name=f"step{step}", lora_int_id=step + 1, lora_path=lora_dir)
    reqs, params = [], []
    for v in dirs:
        sv = _steer_vec(v, hnorm, marker)
        reqs.append({"prompt_token_ids": list(prompt_ids)})
        params.append(SamplingParams(n=G, temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0,
                                     repetition_penalty=1.0, max_tokens=a.max_new_tokens,
                                     min_tokens=a.min_new_tokens, stop_token_ids=sorted(eos_ids),
                                     logprobs=0, extra_args={"apply_steering_vectors": [sv]}))
    t1 = time.time()
    outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
    t_gen = time.time() - t1
    gen_ids, vllm_lps, appended = [], [], 0
    for out in outs:                                                      # request order == dirs order
        assert len(out.outputs) == G, f"expected {G} samples, got {len(out.outputs)}"
        for o in out.outputs:
            g = list(o.token_ids)
            lp = [None] * len(g)
            if o.logprobs:
                lp = [(d[t].logprob if (d is not None and t in d) else None) for d, t in zip(o.logprobs, g)]
            if o.finish_reason == "stop" and (not g or g[-1] not in eos_ids):   # engine dropped the stop token
                g.append(int(o.stop_reason) if isinstance(o.stop_reason, int) else int(tok.eos_token_id))
                lp.append(None); appended += 1
            g2 = _trim_at_stop(g, eos_ids)
            gen_ids.append(g2); vllm_lps.append(lp[: len(g2)])
    # old_logp: HF recompute with the SAME hook update() uses (chunked for memory)
    all_dirs = dirs.repeat_interleave(G, 0).to(device)
    old_lps = []
    for s in range(0, len(gen_ids), a.logp_chunk):
        e = min(s + a.logp_chunk, len(gen_ids))
        vecs = [all_dirs[i : i + 1] for i in range(s, e)]
        old_lps += _old_logp(actor, submodule, prompt, gen_ids[s:e], vecs, marker, tok, device)
    # token-similarity: vLLM's sampled-token logprobs vs the HF recompute
    diffs = torch.tensor([abs(h - v) for hl, vl in zip(old_lps, vllm_lps) for h, v in zip(hl.tolist(), vl) if v is not None])
    texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
    stats = {"rollout/vllm_hf_logp_absdiff_mean": diffs.mean().item() if len(diffs) else float("nan"),
             "rollout/vllm_hf_logp_absdiff_p99": diffs.quantile(0.99).item() if len(diffs) > 1 else float("nan"),
             "rollout/vllm_hf_logp_absdiff_max": diffs.max().item() if len(diffs) else float("nan"),
             "rollout/vllm_compared_tokens": float(len(diffs)),
             "rollout/appended_stop": float(appended), "rollout/marker_hnorm": hnorm,
             "time/lora_save_s": t_save, "time/vllm_gen_s": t_gen}
    prev = f"/tmp/rl_lora/rank{rank}/step{step - 1}"
    if os.path.isdir(prev):
        import shutil
        shutil.rmtree(prev, ignore_errors=True)
    stats["_vllm_lps"] = vllm_lps                                     # for the failure diagnosis only
    stats["_lora_req"] = lora_req
    return texts, gen_ids, old_lps, stats


# ----------------------------------------------------------------------------------------------
# reward (clean base model, standalone re-tokenization)
# ----------------------------------------------------------------------------------------------
def _distinct_fraction(input_ids, attention_mask):
    """Vectorized distinct-token fraction (special tokens included)."""
    masked = input_ids.masked_fill(~attention_mask.bool(), -1)
    ordered = masked.sort(dim=1).values
    unique = torch.ones(len(input_ids), dtype=torch.long, device=input_ids.device)
    unique += (ordered[:, 1:] != ordered[:, :-1]).sum(1)
    unique -= (~attention_mask.bool()).any(1).long()  # remove the padding sentinel, if present
    return unique.float() / attention_mask.sum(1).clamp(min=1)



def _center_mu(device):
    """The layer-42 corpus mean for the cosine reward (eval_universal.center_mu: /data/acts27b/whiten_mu.npy, cached per device;
    None when MAEMM_SCORER_CENTER=0). Imported lazily: eval_universal lives in /pmx/eval on Modal and in ../eval in the repo."""
    try:
        import eval_universal as EU
    except ModuleNotFoundError:
        for cand in ("/pmx/eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval")):
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        import eval_universal as EU
    return EU.center_mu(device)

@torch.no_grad()
def score(texts, dirs_rep, actor, tok, device, a, with_fluency=False):
    """Reward each generation through the CLEAN base model (adapter disabled, no injection).

    Tokenization matches the shared clean-base read path (eval_universal / collect_acts): a sink
    token (bos, or eos=<|endoftext|> for Qwen) is PREPENDED, add_special_tokens=False, position 0
    dropped from the reward. Returns r [n] (+ mean clean-base logp/token and distinct fraction when
    with_fluency, the gate inputs) — one forward per batch does reward and gates together."""
    SCORE_STATS.clear()
    r = torch.zeros(len(texts))
    logp = torch.full((len(texts),), -20.0) if with_fluency else None
    dis = torch.zeros(len(texts)) if with_fluency else None
    valid = [i for i, t in enumerate(texts) if t.strip()]
    prev = tok.padding_side
    tok.padding_side = "right"  # position 0 must be the sink token
    try:
        sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        for s in range(0, len(valid), a.score_batch):
            idxs = valid[s : s + a.score_batch]
            e = tok([texts[i] for i in idxs], return_tensors="pt", padding=True, truncation=True,
                    max_length=95, add_special_tokens=False).to(device)
            n = e["input_ids"].shape[0]
            enc = {"input_ids": torch.cat([torch.full((n, 1), sink, device=device, dtype=e["input_ids"].dtype),
                                           e["input_ids"]], 1),
                   "attention_mask": torch.cat([torch.ones((n, 1), device=device, dtype=e["attention_mask"].dtype),
                                                e["attention_mask"]], 1)}
            if with_fluency:
                captured = {}

                def capture(_module, _inputs, output):
                    captured["h"] = output[0] if isinstance(output, tuple) else output

                handle = get_layer(actor, READ_LAYER).register_forward_hook(capture)
                try:
                    with actor.disable_adapter():
                        logits = actor(**enc).logits[:, :-1].float()
                finally:
                    handle.remove()
                h = captured["h"].float()
                mask = enc["attention_mask"].bool()
            else:
                with actor.disable_adapter():
                    h, mask = read_resid(actor, READ_LAYER, dict(enc), pool="all")
            keep = mask.clone()
            keep[:, 0] = False                                            # drop the sink position
            if a.reward_metric == "cosine":                               # centered like every bank direction (fix 2026-09-18; see eval_universal.center_mu)
                mu = _center_mu(device)
                hh = F.normalize(h.float() - mu if mu is not None else h.float(), dim=-1)
            else:
                hh = h
            proj = torch.einsum("btd,bd->bt", hh, dirs_rep[idxs])
            sel = keep
            revcnt = sel.flip(1).cumsum(1).flip(1)                        # kept tokens from here to the end (last kept = 1)
            if a.reward_window_last > 0:                                  # HARD anti-smear: last N kept tokens only
                sel = sel & (revcnt <= a.reward_window_last)
            if a.reward_pos_penalty > 0:                                  # SOFT anti-smear: cos_t - lambda * (distance of t from the end)
                proj = proj - a.reward_pos_penalty * (revcnt - 1).clamp(min=0).to(proj.dtype)   # inside the max -> no argmax-flip discontinuity
            pf = proj.masked_fill(~sel, torch.finfo(proj.dtype).min)
            if a.reward_topk <= 1:
                _pk = pf.argmax(1)                                        # where the (position-adjusted) peak sits
                _d = (revcnt.gather(1, _pk[:, None]).squeeze(1) - 1).clamp(min=0).float()
                SCORE_STATS.setdefault("peak_dist", []).append(torch.where(keep.any(1), _d, torch.zeros_like(_d)).cpu())
            k = max(1, a.reward_topk)
            if k == 1:
                best = pf.max(1).values
            else:                                                         # mean of the top-K in the window
                topv = pf.topk(min(k, pf.shape[1]), dim=1).values
                avail = sel.sum(1, keepdim=True).clamp(min=1)
                tk = torch.arange(topv.shape[1], device=proj.device).unsqueeze(0) < avail
                topv = torch.where(tk, topv, torch.zeros_like(topv))
                best = topv.sum(1) / tk.sum(1).clamp(min=1)
            if a.log_reward:
                best = torch.log1p(best.clamp(min=-0.9))                  # diminishing returns at the high end
            r[idxs] = torch.where(keep.any(1), best, 0).cpu()
            if with_fluency and logits.shape[1]:
                targets = enc["input_ids"][:, 1:]
                token_lp = -F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                            reduction="none").view_as(targets)
                next_mask = mask[:, 1:]
                mean_lp = (token_lp * next_mask).sum(1) / next_mask.sum(1).clamp(min=1)
                # single-token rows have no next-token logprob -> keep -20 so they FAIL the floor
                logp[idxs] = torch.where(next_mask.any(1), mean_lp, torch.full_like(mean_lp, -20.0)).cpu()
                dis[idxs] = _distinct_fraction(enc["input_ids"], mask).cpu()
    finally:
        tok.padding_side = prev
    return (r, logp, dis) if with_fluency else r


# ----------------------------------------------------------------------------------------------
# advantages + update
# ----------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# inline eval (ALL ranks, no separate runner): the frozen held-out eval sets are generated with each
# rank's vLLM engine + the LoRA it already published this step (== the weights just saved as
# step_{ckpt}), sharded over ranks, and scored on the clean base with the SAME eval_universal
# functions the daemon uses (score_probe_cos / score_sae_peaks). vLLM sampling (not HF generate) is
# the only protocol difference -> distribution-identical, not bitwise, vs the daemon's numbers.
# ---------------------------------------------------------------------------------------------
def load_eval_assets(a, device, is_main):
    """Held-out eval sets + SAE, once per rank. Returns None (inline eval off) if anything is missing."""
    try:
        if "/pmx/eval" not in sys.path:
            sys.path.insert(0, "/pmx/eval")
        import eval_universal as EU
        from mxf.sae import load_sae
        es = torch.load(a.eval_cache, map_location="cpu", weights_only=False)
        sae = load_sae(path=a.eval_sae, device=device, dtype=torch.float32)
        gate = EU.configure_sae_fire(a.eval_sae)   # "fired"/"unverbalized" = the SAE's learned BatchTopK gate, not the old 1.0 cut
        if is_main:
            print(f"[eval] SAE fire threshold = learned gate {gate:.4f}", flush=True)
        assert es["meta"]["d_sae"] == sae.d_sae, f"cache d_sae {es['meta']['d_sae']} != SAE {sae.d_sae}"
        sae.W_dec = None   # decoder unused by encode_features / sae_rank_at_peaks -> free 2.7 GB next to the actor + vLLM
        fams = list(es["meta"].get("cos_families", EU.COS_FAMILIES))
        xfams = EU.extra_families(es)   # cache v2 extras (mlp / mlp_pair): cosine + fire-back, NEVER in mean_all
        for fam in fams:
            assert f"{fam}_dirs" in es, f"eval cache lacks {fam}_dirs"
        for fam in xfams:
            for suf in ("_dirs", "_neuron", "_polarity", "_corpus_max"):
                assert f"{fam}{suf}" in es, f"eval cache lacks {fam}{suf}"
        sfams = EU.slice_families(es)   # cache v3: held-out 2M-SAE features (enc + dec dirs) scored through an encoder SLICE, NEVER in mean_all
        for fam in sfams:
            for suf in ("_dirs", "_feats", "_corpus_peak"):
                assert f"{fam}{suf}" in es, f"eval cache lacks {fam}{suf}"
        assert not sfams or EU.SLICE_KEY in es, f"eval cache lists slice families {sfams} but has no {EU.SLICE_KEY}"
        if a.eval_n_per_family > 0:   # cost control: first n rows of every family (frozen order -> same subset every ckpt)
            n = a.eval_n_per_family
            for fam in fams:
                es[f"{fam}_dirs"] = es[f"{fam}_dirs"][:n]
            for fam in xfams:
                for suf in ("_dirs", "_neuron", "_polarity", "_corpus_max"):
                    es[f"{fam}{suf}"] = es[f"{fam}{suf}"][:n]
            es["sae_dirs"], es["sae_feats"] = es["sae_dirs"][:n], list(es["sae_feats"])[:n]
            es["corpus_peak"] = es["corpus_peak"][:n]
            for fam in sfams:
                es[f"{fam}_dirs"], es[f"{fam}_corpus_peak"] = es[f"{fam}_dirs"][:n], es[f"{fam}_corpus_peak"][:n]
                es[f"{fam}_feats"] = list(es[f"{fam}_feats"])[:n]
        ev = {"EU": EU, "es": es, "sae": sae, "fams": fams, "xfams": xfams, "feats": list(es["sae_feats"]),
              "cp": es["corpus_peak"].numpy().astype(np.float64), "mlp_stats": None, "mlp_chance_acts": None,
              "sfams": sfams, "slice": EU.load_sae_slice(es, device) if sfams else None}
        if sfams and is_main:
            print(f"[inline-eval] SAE-slice families {sfams}: n={[len(es[f + '_dirs']) for f in sfams]}, slice {len(ev['slice'].feats)} cols of "
                  f"F={ev['slice'].d_sae}, gate {ev['slice'].threshold:.4f} ({es[EU.SLICE_KEY].get('ae')})", flush=True)
        # cross-neuron RANK metrics of the extra (mlp) families (eval_universal.score_mlp_rank): need the corpus stats of EVERY
        # layer-42 neuron (--mlp-stats; the trainer leaves it '' = off, eval_ckpt_daemon defaults it on when the file exists)
        mlp_stats = getattr(a, "mlp_stats", "") or ""
        if xfams and mlp_stats and os.path.exists(mlp_stats):
            ev["mlp_stats"] = EU.load_mlp_neuron_stats(mlp_stats)
            ca = getattr(a, "mlp_chance_acts", "") or ""
            ev["mlp_chance_acts"] = ca if (ca and os.path.exists(os.path.join(ca, "toks.i32"))) else None
        elif xfams and mlp_stats and is_main:
            print(f"[inline-eval] --mlp-stats {mlp_stats} not found -> MLP rank metrics OFF", flush=True)
        if is_main:
            print(f"[inline-eval] ready: families {fams} n={len(es[fams[0] + '_dirs'])} (cache n={es['meta'].get('n')}) | sae feats {len(ev['feats'])} "
                  f"| extra families {xfams} n={[len(es[f + '_dirs']) for f in xfams]} (cache {a.eval_cache}) "
                  f"| bo={a.eval_bo} temp={a.eval_temp} tokens {a.eval_min_new}-{a.eval_max_new} | every {a.inline_eval_every} steps"
                  + (f" | MLP rank metrics ON ({mlp_stats}; chance windows {ev['mlp_chance_acts'] or 'OFF'})" if ev["mlp_stats"] else ""),
                  flush=True)
        return ev
    except Exception as e:  # noqa
        if is_main:
            print(f"[inline-eval] DISABLED ({type(e).__name__}: {e})", flush=True)
        return None


@torch.no_grad()
def inline_eval(llm, actor, submodule, tok, prompt_ids, marker, a, device, ckpt_step, lora_step, rank, world, EV):
    """Best-of-bo eval of every held-out family, rows i % world == rank on this rank, generation by
    this rank's vLLM engine with the LoRA published at loop step `lora_step` (weights == step_{ckpt_step}).
    Every rank ALWAYS joins the gather (errors travel as data) so a failure can't deadlock DDP.
    Rank 0 returns the reduced metrics dict (or {"error": ...}); other ranks return {}."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    EU, es, sae = EV["EU"], EV["es"], EV["sae"]
    per_dir = bool(getattr(a, "dump_per_dir", False))   # opt-in: ship the per-direction scores (+ sae best texts) under out["_perdir"]
    t0 = time.time()
    local = {}
    try:
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        eos_ids = _eos_ids(tok, actor)
        hnorm = _marker_norm(actor, submodule, prompt, marker, device, adapter=True)
        lora_req = LoRARequest(lora_name=f"step{lora_step}", lora_int_id=lora_step + 1,
                               lora_path=f"/tmp/rl_lora/rank{rank}/step{lora_step}")
        bo = a.eval_bo

        def gen(dirs_unit):
            rows = list(range(rank, len(dirs_unit), world))
            if not rows:
                return rows, []
            reqs, params = [], []
            for i in rows:
                sv = _steer_vec(F.normalize(dirs_unit[i].float(), dim=-1), hnorm, marker)
                reqs.append({"prompt_token_ids": list(prompt_ids)})
                params.append(SamplingParams(n=bo, temperature=a.eval_temp, top_p=1.0, top_k=0, min_p=0.0,
                                             repetition_penalty=1.0, max_tokens=a.eval_max_new,
                                             min_tokens=a.eval_min_new, stop_token_ids=sorted(eos_ids),
                                             seed=EU.GEN_SEED * 1000 + i,
                                             extra_args={"apply_steering_vectors": [sv]}))
            outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
            texts = []
            for out in outs:                                   # row-major: rows[j] * bo + k
                assert len(out.outputs) == bo
                for o in out.outputs:
                    texts.append(tok.decode(_trim_at_stop(list(o.token_ids), eos_ids), skip_special_tokens=True))
            return rows, texts

        for fam in EV["fams"]:
            du = es[f"{fam}_dirs"]
            rows, texts = gen(du)
            if rows:
                rd = F.normalize(torch.stack([du[i] for i in rows for _ in range(bo)]).float(), dim=-1)
                cos = EU.score_probe_cos(texts, rd, actor, tok, device).view(len(rows), bo).max(1).values
                local[fam] = {int(i): float(c) for i, c in zip(rows, cos.tolist())}
            else:
                local[fam] = {}
        du, feats = es["sae_dirs"], EV["feats"]
        rows, texts = gen(du)
        if rows:
            fl = [feats[i] for i in rows for _ in range(bo)]
            rd = F.normalize(torch.stack([du[i] for i in rows for _ in range(bo)]).float(), dim=-1)
            sc = EU.score_probe_cos(texts, rd, actor, tok, device).view(len(rows), bo).max(1).values   # cosine view of the SAE family
            local["sae_cos"] = {int(i): float(c) for i, c in zip(rows, sc.tolist())}
            acts, peaks = EU.score_sae_peaks(texts, fl, sae, actor, tok, device)      # acts [n*bo], peaks [n*bo, d]
            acts = acts.view(len(rows), bo)
            best, arg = acts.max(1)
            pk = peaks.view(len(rows), bo, -1)[torch.arange(len(rows)), arg]          # peak hidden of the best sample
            local["sae"] = {int(i): float(v) for i, v in zip(rows, best.tolist())}
            local["sae_peak"] = {int(i): pk[j].half().numpy().tobytes() for j, i in enumerate(rows)}   # fp16 bytes (≈10 KB/row)
            if per_dir:   # --dump-per-dir: all bo sample acts + the best sample's text per feature (small; gathered like the rest)
                local["sae_act_bo"] = {int(i): [float(v) for v in acts[j].tolist()] for j, i in enumerate(rows)}
                local["sae_text"] = {int(i): texts[j * bo + int(arg[j])] for j, i in enumerate(rows)}
        else:
            local["sae"], local["sae_peak"] = {}, {}
        # extra families (cache v2: layer-42 MLP neurons / co-firing pairs): cosine + clean-base fire-back (+ cross-neuron RANK
        # of the best-of-bo sample and its chance level on random corpus windows when the per-neuron stats are loaded)
        stats = EV.get("mlp_stats")
        for fi, fam in enumerate(EV.get("xfams", [])):
            du = es[f"{fam}_dirs"]
            rows, texts = gen(du)
            local[fam], local[f"{fam}_na"], local[f"{fam}_na_any"] = {}, {}, {}
            if rows:
                rr = [i for i in rows for _ in range(bo)]
                rd = F.normalize(torch.stack([du[i] for i in rr]).float(), dim=-1)
                cos = EU.score_probe_cos(texts, rd, actor, tok, device).view(len(rows), bo).max(1).values
                na_min, na_max = EU.score_mlp_fireback(texts, es[f"{fam}_neuron"][rr], es[f"{fam}_polarity"][rr],
                                                       es[f"{fam}_corpus_max"][rr], actor, tok, device)
                na_min, arg = na_min.view(len(rows), bo).max(1); na_max = na_max.view(len(rows), bo).max(1).values
                local[fam] = {int(i): float(c) for i, c in zip(rows, cos.tolist())}
                local[f"{fam}_na"] = {int(i): float(v) for i, v in zip(rows, na_min.tolist())}
                local[f"{fam}_na_any"] = {int(i): float(v) for i, v in zip(rows, na_max.tolist())}
                if stats is not None:   # rank metrics of the SAME sample that defines norm_act (the best-of-bo by the weakest member)
                    rk = EU.score_mlp_rank(texts, es[f"{fam}_neuron"][rr], stats, actor, tok, device)
                    sel = torch.arange(len(rows)) * bo + arg.cpu()
                    for key in ("rank", "rank_best5", "raw_rank", "pct", "av"):
                        local[f"{fam}_{key}"] = {int(i): rk[key][sel[j]].tolist() for j, i in enumerate(rows)}
            if stats is not None and EV.get("mlp_chance_acts"):
                # chance level: random held-out corpus windows scored exactly like the generations; depends on the base model only,
                # so it is computed once per process (per rank, its own rows) and re-sent with every eval
                key = f"{fam}_chance"
                if key not in EV.setdefault("_mlp_chance_local", {}):
                    EV["_mlp_chance_local"][key] = EU.mlp_chance_rows(fam, fi, es, stats, actor, tok, device, EV["mlp_chance_acts"],
                                                                      range(rank, len(du), world), bo, a.eval_min_new, a.eval_max_new)
                local[key] = EV["_mlp_chance_local"][key]
        # SAE-slice families (cache v3: held-out 2M-SAE features, enc + dec directions): best-of-bo target-feature act through the
        # encoder slice (the sae-family scorer) + best-of-bo max-token cosine; NOT in mean_all
        for fam in EV.get("sfams", []):
            du = es[f"{fam}_dirs"]; feats_f = es[f"{fam}_feats"]
            rows, texts = gen(du)
            local[fam], local[f"{fam}_act"] = {}, {}
            if rows:
                rr = [i for i in rows for _ in range(bo)]
                rd = F.normalize(torch.stack([du[i] for i in rr]).float(), dim=-1)
                cos = EU.score_probe_cos(texts, rd, actor, tok, device).view(len(rows), bo).max(1).values
                acts, _ = EU.score_sae_peaks(texts, [int(feats_f[i]) for i in rr], EV["slice"], actor, tok, device)
                best, arg = acts.view(len(rows), bo).max(1)
                local[fam] = {int(i): float(c) for i, c in zip(rows, cos.tolist())}
                local[f"{fam}_act"] = {int(i): float(v) for i, v in zip(rows, best.tolist())}
                if per_dir:
                    local[f"{fam}_text"] = {int(i): texts[j * bo + int(arg[j])] for j, i in enumerate(rows)}
    except Exception as e:  # noqa
        local = {"error": f"rank{rank}: {type(e).__name__}: {str(e)[:300]}"}
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return {}
    errs = [g["error"] for g in gathered if "error" in g]
    if errs:
        return {"error": " | ".join(errs)}
    merged = {}
    for g in gathered:
        for k, d in g.items():
            merged.setdefault(k, {}).update(d)
    out = {}
    for fam in EV["fams"]:
        vals = np.array([merged[fam][i] for i in sorted(merged[fam])], dtype=np.float64)
        out[f"eval/{fam}/cos"] = float(vals.mean())
    idx = sorted(merged["sae"])
    best = np.array([merged["sae"][i] for i in idx], dtype=np.float64)
    cp = EV["cp"][idx]
    na = best / np.maximum(cp, 1e-6)
    out["eval/sae/norm_act"] = float(na.mean())
    if merged.get("sae_cos"):
        out["eval/sae/cos"] = float(np.mean([merged["sae_cos"][i] for i in sorted(merged["sae_cos"])]))   # best-of-bo max-token cosine to the unit encoder column
    ranks = None
    if merged.get("sae_peak"):   # full-SAE rank of the target feature at its best sample's peak token (the ARB "rank-1 fraction")
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
    # mean_all over the cos families only (no control, no sae/cos diagnostic, no cache-v2 extra families)
    cos_keys = [k for k in out if k.startswith("eval/") and k.endswith("/cos") and k.split("/")[1] not in EU.CONTROL_FAMS and k.split("/")[1] != "sae"]   # sae/cos is a diagnostic, not a mean_all family
    out["eval/mean_all"] = float(np.mean([out[k] for k in cos_keys]))
    chance_pd = {}
    for fam in EV.get("xfams", []):
        idx = sorted(merged[fam])
        out[f"eval/{fam}/cos"] = float(np.mean([merged[fam][i] for i in idx]))
        k = int(es[f"{fam}_neuron"].shape[1])
        out.update(EU.mlp_metrics(fam, [merged[f"{fam}_na"][i] for i in idx],
                                  [merged[f"{fam}_na_any"][i] for i in idx] if k > 1 else None))
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
        out[f"eval/all/{fam}_norm_act"] = out[f"eval/{fam}/norm_act"]
        out[f"eval/all/{fam}_fired10"] = out[f"eval/{fam}/fired10"]
        if merged.get(f"{fam}_rank"):   # cross-neuron RANK metrics (eval_universal.mlp_rank_metrics) + chance level
            R = lambda key: np.asarray([merged[f"{fam}_{key}"][i] for i in idx], np.float64)
            out.update(EU.mlp_rank_metrics(fam, R("rank"), R("rank_best5"), R("raw_rank"), R("pct")))
            out[f"eval/all/{fam}_rank_le{EU.MLP_RANK_LE}"] = out[f"eval/{fam}/rank_le{EU.MLP_RANK_LE}"]
            out[f"eval/all/{fam}_mrr"] = out[f"eval/{fam}/mrr"]
            if merged.get(f"{fam}_chance"):
                cm, chance_pd[fam] = EU.mlp_chance_metrics(fam, merged[f"{fam}_chance"], k)
                out.update(cm)
    slice_pd = {}
    for fam in EV.get("sfams", []):   # cache v3 SAE-slice families (after mean_all: never part of it)
        idx = sorted(merged[fam])
        out[f"eval/{fam}/cos"] = float(np.mean([merged[fam][i] for i in idx]))
        best_f = np.array([merged[f"{fam}_act"][i] for i in idx], dtype=np.float64)
        cp_f = es[f"{fam}_corpus_peak"].numpy().astype(np.float64)[idx]
        out.update(EU.slice_metrics(fam, best_f, cp_f, EV["slice"].threshold))
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
        slice_pd[fam] = {"row": [int(i) for i in idx], "feature": [int(es[f"{fam}_feats"][i]) for i in idx], "best_act": best_f.tolist(),
                         "corpus_peak": cp_f.tolist(), "norm_act": (best_f / np.maximum(cp_f, 1e-6)).tolist(),
                         "fired": (best_f > EV["slice"].threshold).astype(int).tolist(), "cos": [merged[fam][i] for i in idx],
                         **({"best_text": [merged[f"{fam}_text"].get(i, "") for i in idx]} if merged.get(f"{fam}_text") else {})}
    for fam in EV["fams"]:
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
    out["eval/all/sae_norm_act"] = out["eval/sae/norm_act"]
    out["eval/all/sae_unverbalized"] = out["eval/sae/unverbalized_frac"]
    if per_dir:   # NOT a wandb scalar: the caller pops it (eval_ckpt_daemon -> perdir_ckpt_<step>.json)
        out["_perdir"] = per_dir_dump(EV, merged, ranks, chance_pd)
        if slice_pd:
            out["_perdir"]["slice"] = slice_pd
    out["time/inline_eval_s"] = time.time() - t0
    return out


def per_dir_dump(EV, merged, ranks, chance_pd=None):
    """--dump-per-dir: the per-direction best-of-bo scores behind every inline_eval aggregate, in eval-cache row order
    (list position == cache row 0..n-1). cos families: best-of-bo max-token cosine per direction. sae: feature id, best
    sample's raw act, corpus peak, norm_act = best/peak, fired (> SAE_FIRE), beat_corpus, cosine view, full-SAE rank + rank1
    flag, every sample's act and the best sample's text. Extra (cache v2 mlp) families: cosine + fire-back per direction.
    With the per-neuron stats loaded (--mlp-stats) the extra families also carry, per direction, the best-of-bo sample's
    cross-neuron ranks (rank / rank_best5 / raw_rank, [n, k] lists), its corpus percentile, the polarity-signed value behind
    norm_act, the neuron ids, and "chance" = the same arrays for random held-out corpus windows (+ the windows' (seq, off, len)).
    Small (11 families x 512 floats + the sae arrays, ~250 KB with texts). Never logged to wandb -- the caller pops it."""
    EU, es = EV["EU"], EV["es"]
    d = {"row_order": "list position == eval-cache row index", "cos": {}, "sae": {}, "extra": {}}
    for fam in EV["fams"]:
        d["cos"][fam] = [merged[fam][i] for i in sorted(merged[fam])]
    idx = sorted(merged["sae"])
    best = np.array([merged["sae"][i] for i in idx], dtype=np.float64)
    cp = EV["cp"][idx]
    na = best / np.maximum(cp, 1e-6)
    sd = {"row": [int(i) for i in idx], "feature": [int(EV["feats"][i]) for i in idx], "best_act": best.tolist(),
          "corpus_peak": cp.tolist(), "norm_act": na.tolist(), "fired": (best > EU.SAE_FIRE).astype(int).tolist(),
          "beat_corpus": (best > cp).astype(int).tolist()}
    pool_rows = (es["meta"].get("rows") or {}).get("sae")   # held-out-pool caches: the pool vec_idx of every sae row
    if pool_rows is not None:
        sd["pool_vec_idx"] = [int(pool_rows[i]) for i in idx]
    if merged.get("sae_cos"):
        sd["cos"] = [merged["sae_cos"][i] for i in idx]
    if ranks is not None:
        sd["rank"] = [int(r) for r in ranks]
        sd["rank1"] = [int(r == 1) for r in ranks]
    if merged.get("sae_act_bo"):
        sd["act_bo"] = [merged["sae_act_bo"][i] for i in idx]
    if merged.get("sae_text"):
        sd["best_text"] = [merged["sae_text"][i] for i in idx]
    d["sae"] = sd
    for fam in EV.get("xfams", []):
        xi = sorted(merged[fam])
        d["extra"][fam] = {"row": [int(i) for i in xi], "cos": [merged[fam][i] for i in xi],
                           "norm_act": [merged[f"{fam}_na"][i] for i in xi], "any_norm_act": [merged[f"{fam}_na_any"][i] for i in xi]}
        if merged.get(f"{fam}_rank"):
            d["extra"][fam].update({"neuron": es[f"{fam}_neuron"][xi].tolist(),
                                    "rank": [[int(x) for x in merged[f"{fam}_rank"][i]] for i in xi],
                                    "rank_best5": [[int(x) for x in merged[f"{fam}_rank_best5"][i]] for i in xi],
                                    "raw_rank": [[int(x) for x in merged[f"{fam}_raw_rank"][i]] for i in xi],
                                    "corpus_pct": [merged[f"{fam}_pct"][i] for i in xi], "best_signed_act": [merged[f"{fam}_av"][i] for i in xi],
                                    "rank_note": "rank among all d_ff layer-42 neurons by polarity*a/corpus_max at the best-of-bo sample's "
                                                 "best last-5 token (rank_best5: min over the window; raw_rank: by polarity*a, unnormalized)"})
            if chance_pd and fam in chance_pd:
                d["extra"][fam]["chance"] = chance_pd[fam]
    return d


def compute_advantages(r, n_groups, group_size, mode="none"):
    """GRPO advantages for group-major rewards r [n_groups*group_size].
    mode 'none'  = Dr. GRPO: r - group_mean (no /std);
         'group' = standard GRPO: / per-group std;
         'batch' = ScaleRL: drop zero-variance groups, / ONE std over all surviving rollouts of the
                   GLOBAL batch (all_reduce'd across DDP ranks — each rank only holds its own groups)."""
    rg = r.view(n_groups, group_size)
    adv = rg - rg.mean(1, keepdim=True)
    if mode == "group":
        adv = adv / (rg.std(1, keepdim=True) + 1e-6)
    elif mode == "batch":
        nz = (rg.std(1, keepdim=True) > 1e-6).expand(-1, group_size)
        adv = adv * nz
        stats = torch.tensor([adv[nz].double().pow(2).sum(), adv[nz].double().sum(), nz.sum()],
                             dtype=torch.float64)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(stats)                                        # gloo: CPU tensor
        n = stats[2].item()
        std = math.sqrt(max(stats[0].item() / n - (stats[1].item() / n) ** 2, 0.0)) if n > 1 else 1.0
        adv = adv / (std + 1e-6)
    elif mode != "none":
        raise ValueError(mode)
    return adv.flatten().detach()


def _ddp_sync_grads(params, total_tok):
    """All-reduce the LoRA grads across ranks as ONE flat CPU buffer over gloo. Token-weighted
    average: each rank's grad is (sum of its token grads)/local_tok, so
    sum_r grad_r*tok_r / sum_r tok_r == EXACTLY the single-GPU gradient over the union batch."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.detach().reshape(-1).float() for g in grads]
                     + [torch.ones(1, device=grads[0].device)]).cpu()
    flat.mul_(float(total_tok))                    # token-weight this rank's contribution
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(flat[-1].item())                     # /= global completion-token count
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off : off + n].view_as(g))
        off += n


def update(actor, opt, submodule, ids, attn, p_len, marker, old_lp, adv, dirs_rep, a, device):
    """ONE optimizer update. loss = sum_tokens -min(ratio*A, clip(ratio)*A) / total completion tokens
    (global per-token normalizer). ratio is TIS-capped (upper only). new_logp runs with the SAME
    inject hook as rollout. Optional entropy bonus and capped-k3 KL to the frozen 'ref' adapter."""
    n = ids.shape[0]
    gen_mask = attn[:, p_len:].bool()
    total_tok = max(int(gen_mask.sum()), 1)
    if a.loss_agg == "seq":   # each rollout weighs 1/n regardless of its length (GRPO / EasyNLA)
        w_all = gen_mask.float() / gen_mask.sum(1, keepdim=True).clamp(min=1).float() / n
    else:                     # every completion token weighs 1/total_tok (DAPO-style token-mean)
        w_all = gen_mask.float() / total_tok
    lo, hi = 1 - a.clip_eps, 1 + a.clip_eps
    loss_sum, clipped_tok, ent_sum, kl_sum, ratio_sum = 0.0, 0, 0.0, 0.0, 0.0
    # ---- KL reference logps: ONE no-grad pass over the whole batch at a large micro-batch (2 PEFT
    # adapter switches per step instead of 2 per micro-batch, and full GPU utilization) ----
    ref_lp_all = None
    if a.kl_coef > 0:
        ref_lp_all = torch.zeros_like(old_lp)
        actor.set_adapter("ref")
        try:
            with torch.no_grad():
                for s in range(0, n, a.ref_micro_batch):
                    e = min(s + a.ref_micro_batch, n)
                    b_ids, b_attn = ids[s:e].to(device), attn[s:e].to(device)
                    hook = make_inject_hook([dirs_rep[i : i + 1] for i in range(s, e)], [[marker]] * (e - s),
                                            STEER_COEFF, device, torch.bfloat16)
                    with hooked(submodule, hook):
                        ref_logits = actor(input_ids=b_ids, attention_mask=b_attn).logits[:, p_len - 1 : -1]
                    ref_lp_all[s:e] = torch.log_softmax(ref_logits.float(), -1).gather(
                        -1, b_ids[:, p_len:, None]).squeeze(-1).cpu()
                    del ref_logits
        finally:
            actor.set_adapter("default")                 # MUST restore before the trainable pass
    opt.zero_grad(set_to_none=True)
    for s in range(0, n, a.micro_batch):
        e = min(s + a.micro_batch, n)
        b_ids, b_attn = ids[s:e].to(device), attn[s:e].to(device)
        hook = make_inject_hook([dirs_rep[i : i + 1] for i in range(s, e)], [[marker]] * (e - s),
                                STEER_COEFF, device, torch.bfloat16)
        with hooked(submodule, hook):      # hook stays armed through backward: grad-ckpt recompute re-runs the injection
            logits = actor(input_ids=b_ids, attention_mask=b_attn).logits[:, p_len - 1 : -1]
            logp_full = torch.log_softmax(logits.float(), -1)
            del logits
            new_lp = logp_full.gather(-1, b_ids[:, p_len:, None]).squeeze(-1)
            m = gen_mask[s:e].to(device)
            w = w_all[s:e].to(device)
            ratio = torch.exp(new_lp - old_lp[s:e].to(device)).clamp(max=a.tis_cap)
            A = adv[s:e, None].to(device)
            loss = (-torch.minimum(ratio * A, ratio.clamp(lo, hi) * A) * w).sum()
            ent = -(logp_full.exp() * logp_full).sum(-1)                    # per-token entropy (logged always)
            ent_sum += float((ent.detach() * m).sum())
            if a.entropy_coef > 0:
                loss = loss - a.entropy_coef * (ent * w).sum()
            if a.kl_coef > 0:  # capped KL-to-init: ref logps (precomputed above) from the FROZEN init adapter
                ref_lp = ref_lp_all[s:e].to(device)
                delta = ref_lp - new_lp                                    # log(pi_ref / pi)
                kl = (torch.exp(delta) - delta - 1).clamp(0.0, a.kl_cap)   # k3 estimator, capped per token
                loss = loss + a.kl_coef * (kl * w).sum()
                kl_sum += float((kl.detach() * m).sum())
            del logp_full
            loss.backward()                              # micro-losses share the global normalizer
        loss_sum += loss.item()
        clipped_tok += int((((ratio < lo) | (ratio > hi)) & m).sum())
        ratio_sum += float((ratio.detach() * m).sum())
    params = [p for p in actor.parameters() if p.requires_grad]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        _ddp_sync_grads(params, total_tok)           # grads (hence clip + step) match on all ranks
    gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
    if math.isfinite(gn):
        opt.step()
    else:                                            # stepping Adam on nan/inf grads corrupts moments
        opt.zero_grad(set_to_none=True)
        print(f"[update] non-finite grad norm ({gn}) — skipping step", flush=True)
    return {"loss": loss_sum, "grad_norm": gn, "clipfrac": clipped_tok / total_tok,
            "entropy": ent_sum / total_tok, "kl": kl_sum / total_tok, "ratio_mean": ratio_sum / total_tok}


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def parse_args():
    cfg = RLConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / init / resume
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--bank-file", default="vecs.f32", help="direction-bank file under --data-dir")
    ap.add_argument("--direction-source", choices=("cluster", "random"), default=cfg.direction_source,
                    help="cluster: rows of the bank; random: fresh isotropic unit directions each step")
    ap.add_argument("--n-eval-dirs", type=int, default=64,
                    help="UNIQUE directions at the FRONT of the bank reserved as eval-only (never sampled)")
    ap.add_argument("--init-adapter", default=cfg.init_adapter)
    ap.add_argument("--ref-adapter", default=None, help="KL reference adapter (default: --init-adapter)")
    ap.add_argument("--step-offset", type=int, default=0,
                    help="resume: loop runs range(step_offset, total_steps); resuming from step_N -> N+1")
    ap.add_argument("--wandb-id", default=None, help="resume logging into an existing wandb run")
    ap.add_argument("--save-dir", default=cfg.save_dir)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--run-name", default=cfg.run_name)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # batch / sampling
    ap.add_argument("--groups-per-step", type=int, default=cfg.groups_per_step)
    ap.add_argument("--group-size", type=int, default=cfg.group_size)
    ap.add_argument("--total-steps", type=int, default=cfg.total_steps)
    ap.add_argument("--max-new-tokens", type=int, default=cfg.max_new_tokens)
    ap.add_argument("--min-new-tokens", type=int, default=cfg.min_new_tokens)
    ap.add_argument("--temperature", type=float, default=cfg.temperature)
    ap.add_argument("--rollout-chunk", type=int, default=32, help="hf engine: sequences per generate() call")
    ap.add_argument("--logp-chunk", type=int, default=16,
                    help="sequences per old_logp recompute forward (fp32 logits over the 248k vocab: 64 seqs ~13 GB peak)")
    ap.add_argument("--rollout-engine", choices=("hf", "vllm"), default="hf",
                    help="vllm: per-rank vLLM engine + vllm_lens steering, LoRA re-published each step")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.42,
                    help="vLLM gpu_memory_utilization (the HF actor shares the GPU; ~0.3 of it is weights)")
    ap.add_argument("--vllm-logp-tol", type=float, default=0.10,
                    help="max allowed MEAN |vLLM logp - HF logp| per sampled token; exceeding it aborts "
                         "(the sampler is not the training policy: LoRA/steering/tokenization drift)")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--ref-micro-batch", type=int, default=16, help="sequences per KL reference (no-grad) forward")
    ap.add_argument("--score-batch", type=int, default=64)
    # reward
    ap.add_argument("--reward-metric", choices=("proj", "cosine"), default="proj",
                    help="proj: max_t <h_t, unit(v)>; cosine: max_t cos(h_t, v) (scale-invariant)")
    ap.add_argument("--reward-scale", type=float, default=1.0, help="multiply raw reward before shaping")
    ap.add_argument("--log-reward", action="store_true", help="log1p-compress the (cosine) reward")
    ap.add_argument("--reward-window-last", type=int, default=0,
                    help="reward over only the LAST N kept tokens (0 = all tokens)")
    ap.add_argument("--reward-topk", type=int, default=1, help="mean of the top-K cosines in the window (1=max)")
    ap.add_argument("--reward-pos-penalty", type=float, default=0.0,
                    help="soft window: reward = max_t [cos_t - lambda*(kept tokens after t)] over ALL kept tokens "
                         "(lambda in cosine units/token, e.g. 0.01). 0 = off. Can combine with --reward-window-last.")
    ap.add_argument("--fluency-floor", type=float, default=cfg.fluency_floor,
                    help="min clean-base mean logp/token; below it the rollout fails the gate")
    ap.add_argument("--distinct-floor", type=float, default=cfg.distinct_floor,
                    help="min distinct-token fraction; below it the rollout fails the gate")
    ap.add_argument("--gate-penalty", type=float, default=cfg.gate_penalty)
    ap.add_argument("--len-penalty-start", type=int, default=cfg.len_penalty_start)
    ap.add_argument("--len-penalty-per-tok", type=float, default=cfg.len_penalty_per_tok)
    ap.add_argument("--no-gates", action="store_true",
                    help="disable the fluency + distinct-token gates (no gate penalty); the length penalty stays")
    ap.add_argument("--no-len-penalty", action="store_true", help="disable the length penalty")
    # optimization
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--clip-eps", type=float, default=cfg.clip_eps)
    ap.add_argument("--tis-cap", type=float, default=cfg.tis_cap, help="upper cap on the importance ratio")
    ap.add_argument("--adv-mode", choices=["none", "group", "batch"], default=None,
                    help="explicit advantage mode; overrides --std-norm/--batch-norm (sweep-arm override)")
    ap.add_argument("--loss-agg", choices=["token", "seq"], default="token",
                    help="token: sum over all completion tokens / total tokens (DAPO-style, long rollouts weigh more); "
                         "seq: per-rollout mean over its tokens, then mean over rollouts (GRPO / EasyNLA)")
    ap.add_argument("--trunc-reward", type=float, default=None,
                    help="fixed shaped reward for rollouts that hit --max-new-tokens without EOS (EasyNLA: -2); "
                         "they stay in the group baseline and train")
    ap.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.999), metavar=("B1", "B2"))
    ap.add_argument("--grad-ckpt", action="store_true", help="HF gradient checkpointing (non-reentrant); allows a much bigger --micro-batch")
    ap.add_argument("--inline-eval-every", type=int, default=0,
                    help="run the held-out eval suite INSIDE the trainer on all ranks every N steps (vLLM gen + clean-base "
                         "scoring, logged into this run with x-axis ckpt_step). 0 = off. vllm engine only.")
    ap.add_argument("--eval-cache", default=os.environ.get("MAEMM_EVAL_CACHE", "/data/eval_universal_ho/eval_sets_heldout.pt"),
                    help="frozen eval-set cache (env MAEMM_EVAL_CACHE); eval_sets_heldout_v2.pt adds the mlp / mlp_pair extra families")
    ap.add_argument("--eval-sae", default="/data/sae/ae.pt")
    ap.add_argument("--eval-bo", type=int, default=4)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--eval-max-new", type=int, default=64)
    ap.add_argument("--eval-min-new", type=int, default=16)
    ap.add_argument("--mlp-stats", default="",
                    help="per-neuron corpus stats of ALL layer-42 neurons (/data/mlp42/neuron_stats.npz) -> cross-neuron RANK metrics of the "
                         "cache-v2 mlp families in the inline eval (eval_universal.score_mlp_rank). '' (default) = off; eval_ckpt_daemon turns it on")
    ap.add_argument("--mlp-chance-acts", default="/data/acts27b",
                    help="acts27b dump for the random-corpus-window CHANCE level of the MLP metrics (needs --mlp-stats); '' = off")
    ap.add_argument("--no-extra-evals", action="store_true", help="skip the autointerp/locality/WildChat/adversarial inline evals")
    ap.add_argument("--eval-n-per-family", type=int, default=0,
                    help="inline eval: use only the first N directions of each family (0 = the whole cache). "
                         "512/family x 12 families x Bo4 took ~25 min per ckpt on 4 ranks; 128 -> ~6 min")
    ap.add_argument("--std-norm", action="store_true", help="advantage / per-group std (standard GRPO)")
    ap.add_argument("--batch-norm", action="store_true",
                    help="advantage / ONE global-batch std, zero-variance groups dropped (ScaleRL)")
    ap.add_argument("--entropy-coef", type=float, default=cfg.entropy_coef, help="beta in r + beta*H(pi)")
    ap.add_argument("--kl-coef", type=float, default=0.0, help="capped-k3 KL-to-init anchor (0=off)")
    ap.add_argument("--kl-cap", type=float, default=10.0, help="per-token KL clamp (nats)")
    # transcript logging (rank 0): a few rollouts per step with their target spans + rewards, so a
    # breaking policy can be read, not just measured
    ap.add_argument("--transcript-every", type=int, default=5, help="log rollout transcripts every N steps (0=off)")
    ap.add_argument("--transcript-groups", type=int, default=4, help="directions per transcript log")
    ap.add_argument("--transcript-samples", type=int, default=4, help="samples per direction per transcript log")
    # removed reward-shaping terms — fail loudly rather than silently ignore an old launcher
    ap.add_argument("--div-coef", type=float, default=0.0, help="REMOVED (must be 0)")
    ap.add_argument("--firsttok-coef", type=float, default=0.0, help="REMOVED (must be 0)")
    a = ap.parse_args()
    assert a.div_coef == 0 and a.firsttok_coef == 0, "--div-coef/--firsttok-coef were removed from the trainer"
    assert not (a.std_norm and a.batch_norm), "pick one of --std-norm / --batch-norm"
    assert a.temperature == 1.0, "sampling temp must be 1.0 so the behaviour policy == the policy old_logp measures"
    if a.no_gates:
        a.fluency_floor = a.distinct_floor = None
    if a.no_len_penalty:
        a.len_penalty_start = None
    return a


def main():
    a = parse_args()
    tr = TrainConfig()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # ---- data-parallel over groups (torchrun sets WORLD_SIZE/RANK/LOCAL_RANK). DDP_BACKEND=gloo:
    # all collectives here run on CPU tensors (NCCL deadlocked on the original box). ----
    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0)); is_main = rank == 0
    if world > 1:
        assert a.groups_per_step % world == 0, "groups_per_step must divide by world (whole groups per rank)"
        dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl"))
    torch.cuda.set_device(0)
    device = "cuda:0"                                  # the rank's single visible GPU (see module top)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)

    # ---- direction bank + held-out reservation (first --n-eval-dirs UNIQUE blocks never sampled) ----
    bank, n_vecs, eval_rows = None, None, 0
    if a.direction_source == "cluster":
        stats_p = f"{a.data_dir}/build_stats.json"
        n_vecs = (json.load(open(stats_p))["n_examples"] if os.path.exists(stats_p)
                  else os.path.getsize(f"{a.data_dir}/{a.bank_file}") // (4 * D_MODEL))
        bank = np.memmap(f"{a.data_dir}/{a.bank_file}", dtype=np.float32, mode="r", shape=(n_vecs, D_MODEL))
        if a.n_eval_dirs > 0:
            blocks, i = 0, 0
            while i < n_vecs and blocks < a.n_eval_dirs:
                row = np.asarray(bank[i]); i += 1; blocks += 1
                while i < n_vecs and np.array_equal(np.asarray(bank[i]), row):
                    i += 1
            eval_rows = i
        assert n_vecs - eval_rows >= a.groups_per_step, "bank too small after the eval reservation"

    B, G = a.groups_per_step, a.group_size            # B = GLOBAL groups/step
    Bl = B // world                                   # groups THIS rank rolls out / scores / backprops

    # ---- actor (HF + LoRA). NO gradient checkpointing: recompute would run outside the hook. ----
    actor = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": device})
    if a.init_adapter:
        actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
    else:
        actor = get_peft_model(actor, LoraConfig(
            r=tr.lora_r, lora_alpha=tr.lora_alpha, lora_dropout=0.0, use_rslora=True,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    if a.grad_ckpt:   # safe now: update() keeps the inject hook armed through backward, so the recompute re-injects
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actor.enable_input_require_grads()
        actor.base_model.model.config.use_cache = False
        if is_main:
            print("[rl] gradient checkpointing ON (non-reentrant); hook armed through backward", flush=True)
    actor.train()
    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0, eps=a.adam_eps, betas=tuple(a.adam_betas))
    optim_p = os.path.join(a.init_adapter or "", "optim.pt")
    if a.init_adapter and os.path.exists(optim_p):                  # resume AdamW moments (same param order)
        opt.load_state_dict(torch.load(optim_p, map_location="cpu"))
        if is_main:
            print(f"[resume] AdamW state restored from {optim_p}", flush=True)
    elif a.step_offset and is_main:
        print(f"[resume] WARNING: no optim.pt in {a.init_adapter} — fresh AdamW moments", flush=True)
    submodule = get_layer(actor, INJECT_LAYER)
    if a.kl_coef > 0:                                                # frozen reference for the KL anchor
        ref_src = a.ref_adapter or a.init_adapter
        assert ref_src, "--kl-coef needs --init-adapter or --ref-adapter"
        actor.load_adapter(ref_src, adapter_name="ref")
        actor.set_adapter("default")
        if is_main:
            print(f"[kl] ref adapter = {ref_src}; kl_coef={a.kl_coef} cap={a.kl_cap}", flush=True)

    # ---- vLLM engine AFTER the HF actor: importing vllm registers its vendored Qwen3_5Config with
    # transformers' AutoConfig (exist_ok=True), which breaks AutoModelForCausalLM.from_pretrained for
    # this model if it runs afterwards. Memory: the engine's util*total check is against FREE memory,
    # which still clears with the 54 GB actor resident (0.42*183 = 77 GB <= ~127 GB free). ----
    llm = None
    if a.rollout_engine == "vllm":
        t_v = time.time()
        llm = init_vllm(a, local, rank, p_len, max_seqs=Bl * G)
        if is_main:
            print(f"[vllm] engine up in {time.time() - t_v:.0f}s (rank-local, TP=1, LoRA on, prefix-cache off)", flush=True)
        chk = verify_vllm_injection(llm, actor, submodule, prompt_ids, marker, device, seed=a.seed)
        if not chk["ok"]:
            raise RuntimeError(f"vLLM steering does NOT match the HF inject hook (cos/norm-ratio): {chk}")
        if is_main:
            print(f"[vllm] injection verified: cos(delta, v) = {chk['cos']:.4f} | injected/HF magnitude = {chk['norm_ratio']:.3f} "
                  f"| ||h|| vllm/hf = {chk['hnorm_agree']:.3f} ({chk['hnorm_vllm']:.1f}/{chk['hnorm_hf']:.1f}) "
                  f"| max |delta| pre-marker rows = {chk['max_other_row_delta']:.2e}", flush=True)

    # ---- transcript targets: vec_idx -> (family, target_text) from the pool's records.jsonl ----
    tgt_map = {}
    if is_main and a.transcript_every > 0 and a.direction_source == "cluster" and os.path.exists(f"{a.data_dir}/records.jsonl"):
        with open(f"{a.data_dir}/records.jsonl") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                vi = rec.get("vec_idx", i)
                if vi not in tgt_map and rec.get("target_text") is not None:
                    tgt_map[vi] = (rec.get("family"), rec["target_text"][:240])
        print(f"[transcripts] {len(tgt_map)} target spans loaded; logging {a.transcript_groups}x{a.transcript_samples} rollouts every {a.transcript_every} steps", flush=True)

    adv_mode = a.adv_mode or ("batch" if a.batch_norm else ("group" if a.std_norm else "none"))
    use_gates = a.fluency_floor is not None or a.distinct_floor is not None
    # ---- inline eval assets (held-out sets + SAE) — after the engine, so a load failure just disables eval ----
    EV = load_eval_assets(a, device, is_main) if (a.inline_eval_every > 0 and a.rollout_engine == "vllm") else None
    # ---- extra inline evals (autointerp AUC, snippet locality, WildChat fire-prediction, adversarial confirmation):
    # GPU stage every eval period on all ranks, LLM-judge stage in a background thread on rank 0 ----
    EX, IX = None, None
    if EV is not None and not a.no_extra_evals:
        try:
            import inline_extra_evals as IX
            EX = IX.prepare_extra_eval_assets(a, device, rank, world, is_main, sae=EV["sae"])
        except Exception as e:  # noqa
            EX, IX = None, None
            if is_main:
                print(f"[extra-eval] DISABLED ({type(e).__name__}: {e})", flush=True)
    if is_main:
        print(f"[rl] {n_vecs} directions (eval-reserved rows [0,{eval_rows})) | prompt {p_len} toks, "
              f"marker @{marker} | world {world} | adv {adv_mode} | gates {use_gates} | engine {a.rollout_engine}", flush=True)
        if not a.no_wandb:
            wandb.init(project="maxact-fast", name=a.run_name, config=vars(a),
                       id=a.wandb_id or None, resume="must" if a.wandb_id else None)
            wandb.define_metric("ckpt_step")
            wandb.define_metric("eval/*", step_metric="ckpt_step")
            wandb.define_metric("extra/*", step_metric="ckpt_step")
        os.makedirs(a.save_dir, exist_ok=True)
    for _ in range(a.step_offset if a.direction_source == "cluster" else 0):
        rng.choice(n_vecs - eval_rows, size=B, replace=False)   # fast-forward the direction rng on resume

    for step in range(a.step_offset, a.total_steps):
        t0 = time.time()
        # every rank draws the SAME B directions (same seed), then takes its slice of WHOLE groups
        if a.direction_source == "random":
            dirs = F.normalize(torch.randn(B, D_MODEL, dtype=torch.float32), dim=-1)
        else:
            idx = eval_rows + np.sort(rng.choice(n_vecs - eval_rows, size=B, replace=False))
            dirs = F.normalize(torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
        if world > 1:
            dirs = dirs[rank * Bl : (rank + 1) * Bl]
        if llm is not None:
            texts, gen_ids, old_lps, rstats = rollout_vllm(llm, actor, submodule, tok, prompt_ids, marker,
                                                           dirs, a, device, step, rank)
            if is_main:
                print(f"  [vllm] |logp_vllm - logp_hf| mean {rstats['rollout/vllm_hf_logp_absdiff_mean']:.4f} "
                      f"p99 {rstats['rollout/vllm_hf_logp_absdiff_p99']:.3f} max {rstats['rollout/vllm_hf_logp_absdiff_max']:.3f} "
                      f"over {int(rstats['rollout/vllm_compared_tokens'])} toks | appended_stop {int(rstats['rollout/appended_stop'])} "
                      f"| lora save {rstats['time/lora_save_s']:.1f}s | gen {rstats['time/vllm_gen_s']:.1f}s", flush=True)
            if not (rstats["rollout/vllm_hf_logp_absdiff_mean"] <= a.vllm_logp_tol):
                # ---- diagnose WHICH policy the engine is running before dying ----
                prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
                conds = _hf_logp_conditions(actor, submodule, prompt_t, gen_ids, dirs.repeat_interleave(G, 0).to(device), marker, tok, device, a.logp_chunk)
                table = {f"adapter={'on' if ad else 'off'},hook={'on' if hk else 'off'}": round(_absdiff(l, rstats["_vllm_lps"]), 4)
                         for (ad, hk), l in conds.items()}
                # LoRA-only probe: greedy vLLM WITH the adapter, NO steering -> HF adapter-on/no-hook logps
                from vllm import SamplingParams
                po = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                                  [SamplingParams(temperature=0.0, max_tokens=24, min_tokens=24, logprobs=0)],
                                  lora_request=rstats["_lora_req"], use_tqdm=False)[0].outputs[0]
                pg = list(po.token_ids); pv = [(d[t].logprob if (d is not None and t in d) else None) for d, t in zip(po.logprobs or [], pg)]
                ph_on = _old_logp_nohook(actor, prompt_t, [pg], tok, device)
                with actor.disable_adapter():
                    ph_off = _old_logp_nohook(actor, prompt_t, [pg], tok, device)
                table["LORA-ONLY probe: vllm(lora,no-steer) vs hf(adapter on,no hook)"] = round(_absdiff(ph_on, [pv]), 4)
                table["LORA-ONLY probe: vllm(lora,no-steer) vs hf(adapter OFF,no hook)"] = round(_absdiff(ph_off, [pv]), 4)
                table["LORA-ONLY probe text"] = tok.decode(pg, skip_special_tokens=True)[:80]
                print("[vllm-diag] mean |logp_vllm - logp_hf| per HF condition:", json.dumps(table, indent=1), flush=True)
                raise RuntimeError(f"vLLM sampler is NOT the training policy: mean |logp_vllm - logp_hf| = "
                                   f"{rstats['rollout/vllm_hf_logp_absdiff_mean']:.4f} > tol {a.vllm_logp_tol} "
                                   f"(p99 {rstats['rollout/vllm_hf_logp_absdiff_p99']:.3f}); diagnosis: {table}")
            rstats = {k: v for k, v in rstats.items() if not k.startswith("_")}
        else:
            texts, gen_ids, old_lps = rollout(actor, submodule, tok, prompt_ids, marker, dirs, a, device)
            rstats = {}
        t_roll = time.time() - t0
        # ---- inline eval of the checkpoint saved at the END of the previous step: the LoRA this rank just
        # published for the rollout IS those weights, so no extra save; every rank evals its shard ----
        if EV is not None and step > a.step_offset and (step - 1) % a.inline_eval_every == 0:
            ev = inline_eval(llm, actor, submodule, tok, prompt_ids, marker, a, device, step - 1, step, rank, world, EV)
            if is_main:
                if "error" in ev:
                    print(f"  [inline-eval] FAILED for ckpt {step - 1}: {ev['error']}", flush=True)
                else:
                    print(f"  [inline-eval] ckpt {step - 1}: mean_all {ev['eval/mean_all']:.4f} | sae norm_act "
                          f"{ev['eval/sae/norm_act']:.4f} unverb {ev['eval/sae/unverbalized_frac']:.3f} | realact "
                          f"{ev.get('eval/realact/cos', float('nan')):.4f} | {ev['time/inline_eval_s']:.0f}s", flush=True)
                    if not a.no_wandb:
                        wandb.log({**ev, "ckpt_step": step - 1}, step=step - 1)
        if EX is not None and step > a.step_offset and (step - 1) % a.inline_eval_every == 0:
            ex = IX.run_extra_evals_gpu(llm, actor, submodule, tok, prompt_ids, marker, a, device, step - 1, step, rank, world, EX,
                                        _steer_vec, _marker_norm, _eos_ids(tok, actor), _trim_at_stop)
            if is_main:
                if "error" in ex:
                    print(f"  [extra-eval] FAILED for ckpt {step - 1}: {ex['error']}", flush=True)
                else:
                    print(f"  [extra-eval] ckpt {step - 1}: locality win5 {ex.get('extra/locality/win5_share', float('nan')):.3f} "
                          f"fire {ex.get('extra/locality/fire_frac', float('nan')):.3f} peak_pos {ex.get('extra/locality/peak_pos_median', float('nan')):.2f} "
                          f"| {ex.get('time/extra_eval_gpu_s', float('nan')):.0f}s -> judge stage launched", flush=True)
                    if not a.no_wandb:
                        wandb.log({**ex, "ckpt_step": step - 1}, step=step - 1)
                    try:
                        IX.launch_judge_stage(None, step - 1, EX, a)
                    except Exception as e:  # noqa
                        print(f"  [extra-eval] judge launch failed: {type(e).__name__}: {e}", flush=True)
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)             # [Bl*G, d]

        # ---- reward + shaping ----
        if use_gates:
            r, flu, dis = score(texts, dirs_rep, actor, tok, device, a, with_fluency=True)
        else:
            r = score(texts, dirs_rep, actor, tok, device, a)
        r = r * a.reward_scale
        raw_r, gate_frac = r.clone(), 1.0                      # raw_r = the TRUE cosine (logged/transcripts), before any shaping
        eos_set = set(_eos_ids(tok, actor))
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
            r = r - a.gate_penalty * (~gate).float()       # subtract (not zero): gated garbage ranks below coherent negatives
            gate_frac = gate.float().mean().item()
        if a.len_penalty_start is not None:                # length costs only for VALID rollouts (gated ones already pay)
            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids], dtype=torch.float32)
            r = r - a.len_penalty_per_tok * over * gate.float()
        adv = compute_advantages(r, Bl, G, adv_mode)

        # ---- transcripts (rank 0): what the policy is actually writing, next to the span that fired the direction ----
        _table = None
        if is_main and a.transcript_every > 0 and step % a.transcript_every == 0:
            rows_t = []
            for g in range(min(a.transcript_groups, Bl)):
                vi = int(idx[g]) if (a.direction_source == "cluster" and idx is not None) else -1
                fam, tgt = tgt_map.get(vi, (None, None))
                for j in range(min(a.transcript_samples, G)):
                    i = g * G + j
                    rows_t.append({"step": step, "group": g, "vec_idx": vi, "family": fam, "target": tgt,
                                   "text": texts[i], "cos": float(raw_r[i]) / a.reward_scale, "reward": float(r[i]),
                                   "adv": float(adv[i]), "n_tok": len(gen_ids[i])})
            with open(f"{a.save_dir}/transcripts.jsonl", "a") as f:
                for row in rows_t:
                    f.write(json.dumps(row) + "\n")
            if not a.no_wandb:
                _table = wandb.Table(columns=list(rows_t[0].keys()), data=[list(x.values()) for x in rows_t])

        # ---- pad the batch (shared prompt -> constant p_len) and update ----
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len))
        pt = torch.tensor(prompt_ids, dtype=torch.long)
        for i, (g, lp) in enumerate(zip(gen_ids, old_lps)):
            ids[i, :p_len] = pt
            ids[i, p_len : p_len + len(g)] = torch.tensor(g)
            attn[i, : p_len + len(g)] = 1
            old_lp[i, : len(g)] = lp
        stats = update(actor, opt, submodule, ids, attn, p_len, marker, old_lp, adv, dirs_rep, a, device)

        # ---- logging (reward stats aggregated over ALL ranks) ----
        secs = time.time() - t0
        mem_alloc = torch.cuda.memory_allocated(device) / 2**30
        mem_peak = torch.cuda.max_memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        n_gen = float(sum(len(g) for g in gen_ids))
        rg_std = r.view(Bl, G).std(1).mean().item()
        # ---- variance tracking (user): where does the learning signal live? ----
        _rg = raw_r.view(Bl, G)
        var_stats = torch.tensor([_rg.std(1).mean().item(),                      # mean within-group std of the RAW reward
                                  _rg.mean(1).std().item() if Bl > 1 else 0.0,   # between-group std (of group means)
                                  (_rg.std(1) < 1e-6).float().mean().item(),     # zero-variance groups (no signal)
                                  adv.std().item(), adv.abs().mean().item(),      # advantage spread actually fed to the update
                                  _rg.std(1).min().item(), _rg.std(1).max().item()], dtype=torch.float64)
        if world > 1:
            gath = [torch.zeros_like(raw_r) for _ in range(world)]
            dist.all_gather(gath, raw_r); raw_r_all = torch.cat(gath)
            gath = [torch.zeros_like(r) for _ in range(world)]
            dist.all_gather(gath, r); r_all = torch.cat(gath)
            aux = torch.tensor([n_gen, gate_frac, rg_std], dtype=torch.float64)
            dist.all_reduce(aux)                                       # equal rollouts/rank -> means are exact
            n_gen, gate_frac, rg_std = float(aux[0]), float(aux[1] / world), float(aux[2] / world)
            vmin, vmax = var_stats[5].clone(), var_stats[6].clone()
            dist.all_reduce(var_stats); var_stats /= world
            dist.all_reduce(vmin, op=dist.ReduceOp.MIN); dist.all_reduce(vmax, op=dist.ReduceOp.MAX)
            var_stats[5], var_stats[6] = vmin, vmax
        else:
            raw_r_all, r_all = raw_r, r
        log = {"reward/mean": raw_r_all.mean().item(), "reward/std": raw_r_all.std().item(),
               "reward/max": raw_r_all.max().item(), "reward/shaped_mean": r_all.mean().item(),
               "reward/within_group_std": rg_std, "reward/gate_frac": gate_frac,
               "ratio/clipfrac": stats["clipfrac"], "ratio/mean": stats["ratio_mean"],
               "policy/entropy": stats["entropy"], "policy/kl_to_init": stats["kl"],
               "loss": stats["loss"], "grad_norm": stats["grad_norm"],
               "grad_norm_did_clip": float(stats["grad_norm"] > a.max_grad_norm),
               "rollout/mean_logp": torch.cat(old_lps).mean().item(),
               "rollout/len_mean": n_gen / (B * G), "tokens_per_sec": n_gen / secs,
               "time/rollout_s": t_roll, "time/step_s": secs,
               "mem/hf_alloc_gb": mem_alloc, "mem/hf_peak_gb": mem_peak, **rstats}
        log["reward/trunc_frac"] = trunc_frac
        if SCORE_STATS.get("peak_dist"):
            _pd = torch.cat(SCORE_STATS["peak_dist"]); log["reward/peak_dist_mean"] = _pd.mean().item()
            log["reward/peak_in_last5_frac"] = (_pd <= 4).float().mean().item()
            log["reward/peak_last_frac"] = (_pd == 0).float().mean().item()      # reward argmax exactly at the last token
        log.update({"var/within_group_std_raw": float(var_stats[0]), "var/between_group_std_raw": float(var_stats[1]),
                    "var/zero_var_group_frac": float(var_stats[2]), "var/adv_std": float(var_stats[3]),
                    "var/adv_abs_mean": float(var_stats[4]), "var/group_std_min": float(var_stats[5]),
                    "var/group_std_max": float(var_stats[6]),
                    "var/signal_ratio": float(var_stats[0] / (var_stats[1] + 1e-9))})   # within/between: >1 = groups overlap, per-direction contrast dominates
        if _table is not None:
            log["rollouts/samples"] = _table
        if is_main:
            print(f"step {step:05d} | r {log['reward/mean']:.2f} (max {log['reward/max']:.1f}) | gate {gate_frac:.0%} "
                  f"| ent {log['policy/entropy']:.2f} | ratio {log['ratio/mean']:.3f} clip {log['ratio/clipfrac']:.2%} "
                  f"| len {log['rollout/len_mean']:.0f} | gnorm {log['grad_norm']:.3f}{'*CLIP' if log['grad_norm_did_clip'] else ''} "
                  f"| {log['tokens_per_sec']:.0f} tok/s | {secs:.0f}s | hf mem {mem_alloc:.0f}G (peak {mem_peak:.0f}G)"
                  + (f" | vllm-hf |dlogp| {rstats['rollout/vllm_hf_logp_absdiff_mean']:.4f} (p99 {rstats['rollout/vllm_hf_logp_absdiff_p99']:.3f}) gen {rstats['time/vllm_gen_s']:.0f}s" if rstats else ""), flush=True)
            if step % 10 == 0:
                print(f"  sample r={raw_r[0]:.2f}: {texts[0][:110]!r}", flush=True)
            if not a.no_wandb:
                wandb.log(log, step=step)
            if IX is not None and EX is not None:
                for cs, m in IX.poll_judge_results():          # judge results arrive minutes later; x-axis = ckpt_step
                    print(f"  [extra-eval] judge results for ckpt {cs}: " + " ".join(
                        f"{k.split('/')[-1]}={v:.3f}" for k, v in m.items() if k.startswith("extra/") and "auc" in k), flush=True)
                    if not a.no_wandb:
                        wandb.log({**m, "ckpt_step": cs}, step=step)
            if a.save_every and step and step % a.save_every == 0:
                actor.save_pretrained(f"{a.save_dir}/step_{step}")
                torch.save(opt.state_dict(), f"{a.save_dir}/step_{step}/optim.pt")   # keep newest two
                stale = os.path.join(a.save_dir, f"step_{step - 2 * a.save_every}", "optim.pt")
                if os.path.exists(stale):
                    os.remove(stale)
    if is_main:
        actor.save_pretrained(f"{a.save_dir}/final")
        if a.save_every:
            torch.save(opt.state_dict(), f"{a.save_dir}/final/optim.pt")
        if IX is not None and EX is not None:
            IX.wait_for_judge_stages(900)
            for cs, m in IX.poll_judge_results():
                if not a.no_wandb:
                    wandb.log({**m, "ckpt_step": cs}, step=a.total_steps)
        print("RL_DONE", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
