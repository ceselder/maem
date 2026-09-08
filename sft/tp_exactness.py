"""TP-8 vs FSDP2 exactness on the REAL 27B, same micro-batch window (8xB200, two torchruns in one container):

    torchrun --standalone --nproc_per_node=8 SL/tp_exactness.py --mode fsdp --data-dir <bank> --out /tmp/tp_exact
    torchrun --standalone --nproc_per_node=8 SL/tp_exactness.py --mode tp   --data-dir <bank> --out /tmp/tp_exact

Data: the bank is dealt into 8 DP shards exactly as pretrain.py's FSDP2 path does (load_shard(rank, 8) + step_groups per
shard, --batch-size B per rank). FSDP mode: rank r trains on shard r (production path: prefix cache, --prefix-share-step,
pad-multiple 1). TP mode: every rank runs the SAME step, whose micro-batch is the concatenation of the 8 DP micro-batches
(8B rows) with `loss_group_ids` = the DP rank of each row, so the loss is the mean over the 8 groups of the per-group
token-mean loss -- exactly what FSDP2's gradient averaging over ranks optimises. Both modes clip at 1.0 and take plain
AdamW steps (constant lr) so the traces are comparable step for step.

FSDP mode dumps, at step 0 before the optimizer step, every full parameter gradient (fp32) to <out>/ref_grads/ and the
per-rank losses; TP mode gathers its full gradients, compares (cosine, max|d|, relative Frobenius; per param + global),
compares the losses, then runs `--steps` steps and writes both traces + the comparison to <out>/tp_exactness.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullft as FT  # noqa: E402
import tp as TP  # noqa: E402
from prefix_cache import PrefixCache, PrefixGradientAccumulator  # noqa: E402
from pretrain import gather_rows, load_shard, step_groups  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mxf.config import INJECT_LAYER, MODEL, STEER_COEFF  # noqa: E402
from mxf.inject import get_layer  # noqa: E402
from mxf.prompts import build_prompt_ids  # noqa: E402


def log(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fsdp", "tp"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="/tmp/tp_exact")
    ap.add_argument("--batch-size", type=int, default=32, help="per DP rank (FSDP); the TP micro-batch is 8x this")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-seq", type=int, default=160)
    ap.add_argument("--dump-grads", type=int, default=1)
    ap.add_argument("--fused-norm", type=int, default=1)
    a = ap.parse_args()

    world = int(os.environ["WORLD_SIZE"]); rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl"); torch.cuda.set_device(local)
    device = f"cuda:{local}"
    is_main = rank == 0
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    n_prompt = len(prompt_ids)

    # ---- data: the 8 DP shards + their step compositions (identical on every rank) ----
    shards = []
    for r in range(world):
        toks, vecs, _ = load_shard(a.data_dir, r, world, tok, a.max_seq)
        groups = step_groups(toks.lengths(), a.batch_size, 1, 0)
        shards.append((toks, vecs, groups))
    n_steps = min(a.steps, min(len(g) for _, _, g in shards))
    log(f"[exact:{a.mode}] {world} DP shards of {len(shards[0][0])} rows, {len(shards[0][2])} micro-batches of {a.batch_size} each; "
        f"running {n_steps} steps")

    def dp_batch(r, k):
        toks, vecs, groups = shards[r]
        rows = [toks[i] for i in groups[k][0]]
        return [t[0][n_prompt:] for t in rows], gather_rows(vecs, [t[3] for t in rows])

    # ---- model ----
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    log(f"[exact:{a.mode}] model loaded in {time.time() - t0:.0f}s")
    if a.mode == "fsdp":
        model = FT.shard_full_model(model, world, device, log=log, keep_unsharded_layers=0, prefetch=2)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        tpm = None
    else:
        tpm = TP.shard_model_tp(model, world, device, log=log, fused_norm=bool(a.fused_norm))
        params = tpm.build_masters()
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        model.no_sync = FT.FSDPNoSync()
    log(f"[exact:{a.mode}] backends {FT.kernel_backends(model)}")
    pc = PrefixCache(model, prompt_ids, mpos[0], tok.pad_token_id, get_layer(model, INJECT_LAYER), STEER_COEFF, device,
                     prefix_model=model, pad_multiple=1, inject_mode="add_clone", keep_prefix_grad_path=(a.mode == "fsdp"))
    torch.cuda.reset_peak_memory_stats()

    trace = []
    compare = None
    for k in range(n_steps):
        torch.cuda.synchronize(); t_step = time.time()
        FT.set_persistent_unshard(model, True)
        shared = PrefixGradientAccumulator(pc.run_prefix(), extra_outputs=[pc.pop_prefix_logits()])
        if a.mode == "fsdp":
            targets, vmat = dp_batch(rank, k)
            out = pc.forward(vmat, targets, prefix_cache=shared.cache)
            out.loss.backward()
            shared.accumulate()
            loss_local = out.loss.detach().float()
            n_tok = torch.tensor(float(out.n_target_tokens), device=device)
            del out
            FT.set_persistent_unshard(model, False)
            shared.backward()
            losses = [torch.zeros((), device=device) for _ in range(world)]
            dist.all_gather(losses, loss_local)
            ntoks = [torch.zeros((), device=device) for _ in range(world)]
            dist.all_gather(ntoks, n_tok)
            group_losses = [float(x) for x in losses]
            group_ntok = [int(x) for x in ntoks]
            loss_step = float(sum(group_losses) / world)
            if k == 0 and a.dump_grads:
                # every rank all-gathers each full grad (collective); rank 0 writes fp32 .pt files
                gdir = f"{a.out}/ref_grads"
                if is_main:
                    os.makedirs(gdir, exist_ok=True)
                t_d = time.time(); nbytes = 0
                for i, (name, p) in enumerate(model.named_parameters()):
                    g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad.detach()
                    if is_main:
                        g32 = g.detach().float().cpu()
                        torch.save(g32, f"{gdir}/{i:04d}.pt")
                        nbytes += g32.numel() * 4
                    del g
                if is_main:
                    json.dump([n for n, _ in model.named_parameters()], open(f"{gdir}/names.json", "w"))
                dist.barrier()
                log(f"[exact:fsdp] dumped {nbytes / 2**30:.1f} GB of fp32 reference grads to {gdir} in {time.time() - t_d:.0f}s")
            gn = FT.clip_grad_norm(params, 1.0)
        else:
            targets, vmats, gids = [], [], []
            for r in range(world):
                t_r, v_r = dp_batch(r, k)
                targets += t_r; vmats.append(v_r); gids += [r] * len(t_r)
            vmat = torch.cat(vmats)
            gid_t = torch.tensor(gids)
            out = pc.forward(vmat, targets, prefix_cache=shared.cache, model_kwargs={"loss_group_ids": gid_t})
            out.loss.backward()
            shared.accumulate()
            loss_step = float(out.loss.detach())
            # per-group token-mean losses for the trace (recomputed from the per-token losses is not exposed; use the
            # DP-rank token counts to report the group sizes instead)
            labels = out.labels
            group_ntok = [int(((labels[gid_t == r] != -100).sum())) for r in range(world)]
            group_losses = None
            del out
            shared.backward()
            tpm.reduce_grads()
            if k == 0 and a.dump_grads:
                gdir = f"{a.out}/ref_grads"
                names = json.load(open(f"{gdir}/names.json"))
                name_to_idx = {n: i for i, n in enumerate(names)}
                rows, num, den_a, den_b, max_abs = {}, 0.0, 0.0, 0.0, 0.0
                t_c = time.time()
                for name, p, m, spec in tpm.masters:
                    g_tp = TP.gathered_grads_one(tpm, name, m, spec)
                    if is_main:
                        ref = torch.load(f"{gdir}/{name_to_idx[name]:04d}.pt").to(device)
                        x, y = g_tp.double().flatten(), ref.double().flatten()
                        dot, nx, ny = float((x * y).sum()), float(x.norm()), float(y.norm())
                        d = float((x - y).abs().max()); rel = float((x - y).norm()) / max(ny, 1e-30)
                        rows[name] = dict(cos=dot / max(nx * ny, 1e-30), max_abs=d, rel=rel, norm_ref=ny, norm_tp=nx)
                        num += dot; den_a += nx * nx; den_b += ny * ny; max_abs = max(max_abs, d)
                        del ref
                    del g_tp
                if is_main:
                    worst = min(rows, key=lambda n: rows[n]["cos"])
                    compare = dict(global_cos=num / max(math.sqrt(den_a * den_b), 1e-30), worst_cos=rows[worst]["cos"], worst_name=worst,
                                   max_abs=max_abs, n_params=len(rows),
                                   n_cos_below_0999=sum(1 for r in rows.values() if r["cos"] < 0.999),
                                   n_cos_below_099=sum(1 for r in rows.values() if r["cos"] < 0.99),
                                   worst20=sorted(rows.items(), key=lambda kv: kv[1]["cos"])[:20], per_param=rows)
                    log(f"[exact:tp] GRAD COMPARE vs FSDP2 reference: {len(rows)} params, global cos {compare['global_cos']:.6f}, "
                        f"worst cos {compare['worst_cos']:.6f} ({worst}), max|d| {max_abs:.3e}, "
                        f"{compare['n_cos_below_0999']} params with cos < 0.999 ({time.time() - t_c:.0f}s)")
                    for n, r in compare["worst20"][:10]:
                        log(f"[exact:tp]   cos {r['cos']:.6f} rel {r['rel']:.3e} max|d| {r['max_abs']:.3e} |ref| {r['norm_ref']:.3e} {n}")
            gn = tpm.clip_grad_norm(1.0)
        opt.step(); opt.zero_grad()
        if tpm is not None:
            tpm.sync_params()
        torch.cuda.synchronize()
        dt = time.time() - t_step
        peak = torch.cuda.max_memory_allocated() / 2**30
        rec = dict(step=k, loss=loss_step, group_losses=group_losses, group_ntok=group_ntok, grad_norm=gn, dt=dt, peak_gb=peak)
        trace.append(rec)
        log(f"[exact:{a.mode}] step {k} loss {loss_step:.5f} gnorm {gn:.4f} | {dt:.2f} s | peak {peak:.1f} GB"
            + (f" | per-rank losses {[round(x, 4) for x in group_losses]}" if group_losses else ""))

    if is_main:
        json.dump(dict(mode=a.mode, batch_size=a.batch_size, steps=n_steps, lr=a.lr, trace=trace),
                  open(f"{a.out}/{a.mode}_trace.json", "w"), indent=1)
        if a.mode == "tp":
            ref = json.load(open(f"{a.out}/fsdp_trace.json"))
            pairs = [(r["step"], r["loss"], t["loss"]) for r, t in zip(ref["trace"], trace)]
            log("[exact] step | fsdp loss | tp loss | diff")
            for s, lf, lt in pairs:
                log(f"[exact] {s:4d} | {lf:.5f} | {lt:.5f} | {lt - lf:+.5f}")
            diffs = [abs(lt - lf) for _, lf, lt in pairs]
            summary = dict(step0_loss_fsdp=pairs[0][1], step0_loss_tp=pairs[0][2], step0_abs_diff=diffs[0],
                           max_abs_diff=max(diffs), mean_abs_diff=sum(diffs) / len(diffs),
                           fsdp_gnorm0=ref["trace"][0]["grad_norm"], tp_gnorm0=trace[0]["grad_norm"],
                           fsdp_peak_gb=max(r["peak_gb"] for r in ref["trace"]), tp_peak_gb=max(r["peak_gb"] for r in trace),
                           fsdp_mean_dt=float(np.mean([r["dt"] for r in ref["trace"][1:]])) if len(ref["trace"]) > 1 else None,
                           tp_mean_dt=float(np.mean([r["dt"] for r in trace[1:]])) if len(trace) > 1 else None,
                           grad_compare={k: v for k, v in (compare or {}).items() if k != "per_param"})
            json.dump(dict(summary=summary, pairs=pairs, fsdp=ref, tp=dict(trace=trace), grad_compare=compare),
                      open(f"{a.out}/tp_exactness.json", "w"), indent=1)
            print("TP_EXACTNESS_JSON_BEGIN"); print(json.dumps(summary)); print("TP_EXACTNESS_JSON_END", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
