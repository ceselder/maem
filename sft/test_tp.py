"""TP exactness unit test on a TINY random Qwen3.5 hybrid (2 GDN + 1 attention layers, d 256), CPU / gloo:

    DDP_BACKEND=gloo torchrun --standalone --nproc_per_node=2 sft/test_tp.py [--kv-heads 2|1] [--dtype fp32]

Every rank builds the same full model (same seed), keeps an fp32 single-process REFERENCE copy and shards a second copy
with sft/tp.py (TP = world). For each case the TP loss and the GATHERED full parameter gradients are compared with the
reference to fp32 tolerance (1e-4 relative Frobenius per parameter, |loss diff| < 1e-4):
    full      plain padded full-sequence forward with labels (HF ForCausalLM loss)
    prefix    the production prefix-cache path: shared prefix fwd once, PrefixGradientAccumulator over 2 micro-batches
    groups    prefix path with loss_group_ids (per-group token-mean average = the FSDP2/DP loss normalisation)
then one AdamW step on the masters vs the reference optimizer (params after the step), the clipped global grad norm,
and gather_full(param) == the original full weights (checkpoint layout). --kv-heads 1 exercises kv-head replication
(n_kv < tp: k/v_proj shared by a rank group, gradient summed over the group).
"""
import argparse
import contextlib
import copy
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp as TP  # noqa: E402
from prefix_cache import PrefixCache, PrefixGradientAccumulator  # noqa: E402

from mxf.inject import get_layer, hooked, make_inject_hook  # noqa: E402

D = 256
VOCAB = 1024
PROMPT = list(range(10, 22))   # 12 prompt tokens; marker = 6 -> 6 shared prefix tokens, suffix = [marker] + target
MARKER = 6


def log(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def tiny_model(seed, kv_heads, dtype):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=D, intermediate_size=512, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=64,
                            linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=["linear_attention", "full_attention", "linear_attention"],
                            max_position_embeddings=256, pad_token_id=0, eos_token_id=1, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(cfg).to(dtype=dtype)
    with torch.no_grad():   # non-trivial norm weights (HF inits them to 0 = identity), so their gradients are exercised
        for n, p in model.named_parameters():
            if n.endswith("norm.weight") or "layernorm" in n:
                p.add_(0.1 * torch.randn_like(p))
    model.train()
    return model


def make_batch(seed, B, lo=3, hi=9):
    g = torch.Generator().manual_seed(seed)
    targets = [torch.randint(2, VOCAB, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist() + [1] for _ in range(B)]
    vecs = torch.randn(B, D, generator=g)
    return vecs / vecs.norm(dim=-1, keepdim=True), targets


def full_sequence_batch(targets, pad=0):
    rows = [PROMPT + t for t in targets]
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    labels = torch.full((len(rows), L), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r); mask[i, :len(r)] = True
        labels[i, len(PROMPT):len(r)] = torch.tensor(r[len(PROMPT):])
    return ids, labels, mask


def ref_grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    tol = a.tol if dtype == torch.float32 else 3e-2

    dist.init_process_group(os.environ.get("DDP_BACKEND", "gloo"))
    world, rank = dist.get_world_size(), dist.get_rank()
    device = "cpu"
    if torch.cuda.is_available() and os.environ.get("DDP_BACKEND") == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); device = f"cuda:{rank}"
    log(f"[test_tp] torch {torch.__version__} | world {world} | kv_heads {a.kv_heads} | dtype {a.dtype} | tol {tol:g} | device {device}")
    failures = []

    ref = tiny_model(a.seed, a.kv_heads, dtype).to(device)
    full_params = {n: p.detach().clone() for n, p in ref.named_parameters()}
    tpmodel = copy.deepcopy(ref)
    tpm = TP.shard_model_tp(tpmodel, world, device, log=log, fused_norm=False)
    tpm.build_masters()
    n_local = tpm.local_param_count(); n_full = sum(p.numel() for p in ref.parameters())
    log(f"[test_tp] params: full {n_full}, local {n_local} ({n_local / n_full:.3f} of full; 1/tp = {1 / world:.3f} + replicated norms)")

    # ---- gather_full(param) == original full weights ----
    bad = [n for n, p in tpmodel.named_parameters() if not torch.equal(tpm.gather_full(n, p), full_params[n])]
    log(f"[test_tp] gather_full == original weights for {len(full_params) - len(bad)}/{len(full_params)} params -> {'OK' if not bad else 'FAIL ' + str(bad[:3])}")
    if bad:
        failures.append("gather_full")

    def check(name, loss_ref, loss_tp, g_ref, g_tp, extra=""):
        cmp = TP.compare_grads(g_tp, g_ref)
        # per-param: rel Frobenius < tol, OR max|d| below the fp32 noise floor (the 4-element A_log carries a ~1e-3 gradient
        # whose fp32 summation-order noise is ~1e-7 absolute = 1e-4..1e-3 relative; fullft_smoke saw the same on FSDP2)
        abs_floor = 1e-6 if dtype == torch.float32 else 1e-2
        bad_rows = {k: v for k, v in cmp.rows.items() if v[2] >= tol and v[1] >= abs_floor}
        worst_rel = max(r[2] for r in cmp.rows.values())
        worst_rel_name = max(cmp.rows, key=lambda k: cmp.rows[k][2])
        missing = set(g_ref) ^ set(g_tp)
        ok = abs(loss_ref - loss_tp) < tol and not bad_rows and cmp.global_cos > 1 - 1e-6 and not missing
        log(f"[test_tp] {name:8s} loss ref {loss_ref:.6f} tp {loss_tp:.6f} |d| {abs(loss_ref - loss_tp):.2e} | grads: {cmp.n} params, "
            f"global cos {cmp.global_cos:.8f}, worst cos {cmp.worst_cos:.8f} ({cmp.worst_name}), worst rel {worst_rel:.2e} "
            f"({worst_rel_name}), max|d| {cmp.max_abs:.2e}{' MISSING ' + str(sorted(missing)[:3]) if missing else ''}"
            f"{' BAD ' + str(sorted(bad_rows)[:3]) if bad_rows else ''}{extra} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    vecs, targets = make_batch(a.seed + 1, B=6)
    coeff = 1.0

    # ---- case 1: plain full-sequence forward (marker injection at the absolute prompt index) ----
    ids, labels, mask = full_sequence_batch(targets)
    pos = [[len(PROMPT) - 1]] * len(targets)
    for model, tag in ((ref, "ref"), (tpmodel, "tp")):
        model.zero_grad(set_to_none=True)
        hook = make_inject_hook([v[None] for v in vecs], pos, coeff, device, dtype, mode="add_clone")
        with hooked(get_layer(model, 1), hook):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device), use_cache=False)
        out.loss.backward()
        if tag == "ref":
            l_ref, g_ref = out.loss.item(), ref_grads(ref)
        else:
            l_tp = out.loss.item()
            tpm.reduce_grads()
            g_tp = TP.gathered_grads(tpm)
    check("full", l_ref, l_tp, g_ref, g_tp)

    # ---- case 2: prefix-cache path, shared prefix over 2 micro-batches (production path) ----
    def prefix_step(model, groups_of, group_ids=None):
        pc = PrefixCache(model, PROMPT, MARKER, 0, get_layer(model, 1), coeff, device, prefix_model=model, pad_multiple=1,
                         inject_mode="add_clone", inject_dtype=dtype, keep_prefix_grad_path=False)
        model.zero_grad(set_to_none=True)
        if model is tpmodel:
            tpm.zero_grad()   # the masters' fp32 accumulators (pretrain.py: opt.zero_grad() after every step)
        shared = PrefixGradientAccumulator(pc.run_prefix(contextlib.nullcontext), extra_outputs=[pc.pop_prefix_logits()])
        loss_tot = 0.0
        from prefix_cache import expand_cache_copy
        for mi, (v, t) in enumerate(groups_of):
            cache = expand_cache_copy(shared.cache, len(t))
            idsb, labs, smask, L = pc.build_suffix(t)
            full_mask = torch.cat([torch.ones((len(t), pc.prefix_len), dtype=torch.bool), smask], dim=1).to(device)
            position_ids = torch.arange(pc.prefix_len, pc.prefix_len + L, device=device)[None].expand(len(t), -1)
            hook = make_inject_hook([x[None] for x in v], [[0]] * len(t), coeff, device, dtype, mode="add_clone")
            kw = {}
            if group_ids is not None:
                kw["loss_group_ids"] = group_ids[mi].to(device)
            with hooked(get_layer(model, 1), hook):
                out = model(input_ids=idsb.to(device), attention_mask=full_mask, position_ids=position_ids,
                            past_key_values=cache, use_cache=True, labels=labs.to(device), **kw)
            (out.loss / len(groups_of)).backward()
            shared.accumulate()
            loss_tot += out.loss.item() / len(groups_of)
            del out, cache
        shared.backward()
        return loss_tot

    mb = [(vecs[:3], targets[:3]), (vecs[3:], targets[3:])]
    l_ref = prefix_step(ref, mb); g_ref = ref_grads(ref)
    l_tp = prefix_step(tpmodel, mb); tpm.reduce_grads(); g_tp = TP.gathered_grads(tpm)
    check("prefix", l_ref, l_tp, g_ref, g_tp)

    # ---- case 3: loss_group_ids (DP-emulation normalisation): reference = average over groups of per-group token means,
    # computed with SEPARATE per-group forwards on the reference model ----
    gids = [torch.tensor([0, 0, 1]), torch.tensor([1, 1, 1])]   # micro-batch 0: rows {0,1} group 0, row 2 group 1; mb 1: all group 1
    # reference: per micro-batch, loss = mean over the (non-empty) groups of that micro-batch's per-group token-mean loss
    def prefix_step_groups_ref(model):
        pc = PrefixCache(model, PROMPT, MARKER, 0, get_layer(model, 1), coeff, device, prefix_model=model, pad_multiple=1,
                         inject_mode="add_clone", inject_dtype=dtype, keep_prefix_grad_path=False)
        model.zero_grad(set_to_none=True)
        shared = PrefixGradientAccumulator(pc.run_prefix(contextlib.nullcontext), extra_outputs=[pc.pop_prefix_logits()])
        from prefix_cache import expand_cache_copy
        loss_tot = 0.0
        for mi, (v, t) in enumerate(mb):
            groups = sorted(set(gids[mi].tolist()))
            loss_mb = 0.0
            for gq in groups:
                sel = [i for i in range(len(t)) if gids[mi][i] == gq]
                cache = expand_cache_copy(shared.cache, len(sel))
                tt = [t[i] for i in sel]; vv = v[sel]
                idsb, labs, smask, L = pc.build_suffix(tt)
                full_mask = torch.cat([torch.ones((len(tt), pc.prefix_len), dtype=torch.bool), smask], dim=1).to(device)
                position_ids = torch.arange(pc.prefix_len, pc.prefix_len + L, device=device)[None].expand(len(tt), -1)
                hook = make_inject_hook([x[None] for x in vv], [[0]] * len(tt), coeff, device, dtype, mode="add_clone")
                with hooked(get_layer(model, 1), hook):
                    out = model(input_ids=idsb.to(device), attention_mask=full_mask, position_ids=position_ids,
                                past_key_values=cache, use_cache=True, labels=labs.to(device))
                (out.loss / len(groups) / len(mb)).backward()
                shared.accumulate()
                loss_mb += out.loss.item() / len(groups)
                del out, cache
            loss_tot += loss_mb / len(mb)
        shared.backward()
        return loss_tot

    l_ref = prefix_step_groups_ref(ref); g_ref = ref_grads(ref)
    l_tp = prefix_step(tpmodel, mb, group_ids=gids); tpm.reduce_grads(); g_tp = TP.gathered_grads(tpm)
    check("groups", l_ref, l_tp, g_ref, g_tp)

    # ---- clipped global grad norm + one AdamW step (params after the step) ----
    gn_ref = float(torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0))
    gn_tp = tpm.clip_grad_norm(1.0)
    ok = abs(gn_ref - gn_tp) / gn_ref < 1e-5
    log(f"[test_tp] clip norm: ref {gn_ref:.6f} tp {gn_tp:.6f} rel {abs(gn_ref - gn_tp) / gn_ref:.2e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("clip")
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.0)
    opt_tp = torch.optim.AdamW(tpm.master_params(), lr=1e-3, weight_decay=0.0)
    opt_ref.step(); opt_tp.step(); tpm.sync_params(); opt_tp.zero_grad(); tpm.zero_grad()
    worst, worst_n = 0.0, ""
    for n, p in tpmodel.named_parameters():
        full = tpm.gather_full(n, p).float()
        r = float((full - dict(ref.named_parameters())[n].detach().float()).norm() / max(float(dict(ref.named_parameters())[n].detach().float().norm()), 1e-12))
        if r > worst:
            worst, worst_n = r, n
    ok = worst < (1e-4 if dtype == torch.float32 else 1e-2)
    log(f"[test_tp] AdamW step: params worst rel {worst:.2e} ({worst_n}) -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("adamw")
    # after the step, a second forward must see the updated bf16/fp32 module params (sync_params) -- loss must match ref
    l_ref2 = prefix_step(ref, mb); l_tp2 = prefix_step(tpmodel, mb)
    ok = abs(l_ref2 - l_tp2) < tol
    log(f"[test_tp] post-step loss: ref {l_ref2:.6f} tp {l_tp2:.6f} |d| {abs(l_ref2 - l_tp2):.2e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("post-step")

    log(f"[test_tp] DONE: {'ALL OK' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    dist.barrier()
    dist.destroy_process_group()
    if failures and rank == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
