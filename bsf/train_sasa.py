"""Train a SASA block-sparse subspace featurizer (BSF) on collected L42 activations (Qwen3.6-27B).

Modal-friendly copy of pmx_audit/bsf/train_sasa.py (the original that trained the Aug-21 BSF, EV->~0.8):
  * accepts meta.json "d" OR "d_model" (Modal acts27b uses "d");
  * computes whiten_zca.npy from the acts if it is missing (acts27b only shipped whiten_mu.npy);
  * --loader buffer (default): RAM shuffle-buffer of whole sequences refreshed with CONTIGUOUS reads —
    random 10 KB row reads through a network/FUSE volume are latency-bound; contiguous 5 MB sequence
    reads are not. Pool of --pool-seqs sequences (4096 seqs = 2.1M tok = 21 GB fp16), --refresh-seqs
    replaced per step. --loader random = the original uniform-row sampler (fine on local NVMe).

Group-top-k autoencoder: G groups of b dims; sparsity at the GROUP level (k groups active/token);
each group is a b-dim subspace. Inputs ZCA-whitened + unit-norm (the decisive 8B fix). Decoupled
weight decay on the factored dict = nuclear-norm rank adaptation (variational 1/2(||E||^2+||D||^2));
NO decoder renorm. Logs EV / mean eff-rank / dead-block count to wandb.

Saves: sasa.pt = {E,D,bias,G,b,d}, blocks_Q.pt = per-block orthonormal basis Q_j [d,r_j] (for projecting
activations through a subspace), + copies whitening. Q_j = orthonormal basis of the block's decoder columns.

Usage: python bsf/train_sasa.py --acts-dir /data/acts27b --out-dir /data/bsf27b_sasa --G 32768 --b 8 --k 32
"""
import argparse, os, json, time, shutil, numpy as np, torch, torch.nn as nn


class SASA(nn.Module):
    def __init__(self, d, G, b):
        super().__init__()
        self.d, self.G, self.b = d, G, b
        self.E = nn.Parameter(torch.randn(d, G * b) / d ** 0.5)
        self.D = nn.Parameter(torch.randn(G * b, d) / (G * b) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x, k):
        z = (x @ self.E).view(-1, self.G, self.b)          # [B,G,b]
        gn = z.norm(dim=-1)                                # [B,G] group activation
        idx = gn.topk(k, dim=-1).indices                   # [B,k]
        mask = torch.zeros_like(gn).scatter_(-1, idx, 1.0)
        zs = z * mask.unsqueeze(-1)
        xh = zs.reshape(-1, self.G * self.b) @ self.D + self.bias
        return xh, z, gn, mask


def compute_zca(acts3, mu, n_seq_sample, rng, dev, eps_rel=1e-5):
    """ZCA whitening matrix from a random sample of whole sequences (contiguous reads).
    zca = U diag((lam+eps)^-1/2) U^T on the covariance of (x - mu). eps = eps_rel * mean eigenvalue."""
    N, T, d = acts3.shape
    seqs = np.sort(rng.choice(N, size=min(n_seq_sample, N), replace=False))
    cov = torch.zeros(d, d, dtype=torch.float64, device=dev)
    n = 0
    for s0 in range(0, len(seqs), 64):
        blk = np.concatenate([np.asarray(acts3[s]) for s in seqs[s0:s0 + 64]], 0)  # [64*T, d] f16
        x = torch.from_numpy(blk.astype(np.float32)).to(dev).double() - mu.double()
        cov += x.T @ x
        n += x.shape[0]
    cov /= n
    lam, U = torch.linalg.eigh(cov)
    eps = eps_rel * lam.mean()
    zca = (U * (lam.clamp(min=0) + eps).rsqrt()) @ U.T
    return zca.float().cpu().numpy(), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/data/acts27b")
    ap.add_argument("--out-dir", default="/data/bsf27b_sasa")
    ap.add_argument("--G", type=int, default=32768)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--wandb", default="bsf-sasa-27b", help="wandb project ('' = off)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loader", choices=["buffer", "random"], default="buffer")
    ap.add_argument("--pool-seqs", type=int, default=4096, help="buffer loader: sequences held in RAM")
    ap.add_argument("--refresh-seqs", type=int, default=8, help="buffer loader: sequences replaced per step")
    ap.add_argument("--zca-seqs", type=int, default=2048, help="sequences sampled to estimate ZCA if missing")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    meta = json.load(open(f"{a.acts_dir}/meta.json"))
    N, T = meta["n_seq"], meta["seq_len"]
    d = meta.get("d_model", meta.get("d"))
    assert d, "meta.json needs d_model or d"
    acts3 = np.memmap(f"{a.acts_dir}/acts.f16", dtype=np.float16, mode="r", shape=(N, T, d))
    acts = acts3.reshape(N * T, d)
    ntok = N * T
    mu = torch.from_numpy(np.load(f"{a.acts_dir}/whiten_mu.npy")).float().to(dev)

    zca_path = f"{a.acts_dir}/whiten_zca.npy"
    if not os.path.exists(zca_path):
        zca_path = f"{a.out_dir}/whiten_zca.npy"           # acts dir may be read-only / shared
    if not os.path.exists(zca_path):
        t0 = time.time()
        print(f"[sasa] whiten_zca.npy missing -> estimating from {a.zca_seqs} random sequences", flush=True)
        zca_np, n_used = compute_zca(acts3, mu, a.zca_seqs, rng, dev)
        np.save(zca_path, zca_np)
        try:
            np.save(f"{a.acts_dir}/whiten_zca.npy", zca_np)    # also next to the acts if writable
        except Exception as e:  # noqa
            print(f"[sasa] (could not write zca next to acts: {e})")
        print(f"[sasa] zca ready from {n_used} tokens in {time.time()-t0:.0f}s -> {zca_path}", flush=True)
    zca = torch.from_numpy(np.load(zca_path)).float().to(dev)

    # ---------------- loaders ----------------
    if a.loader == "random":
        def load_batch():
            idx = rng.integers(0, ntok, size=a.batch)
            x = torch.from_numpy(np.asarray(acts[idx]).astype(np.float32)).to(dev)
            x = (x - mu) @ zca
            return torch.nn.functional.normalize(x, dim=-1)
    else:
        P = min(a.pool_seqs, N)
        print(f"[sasa] filling shuffle buffer: {P} seqs x {T} tok = {P*T/1e6:.1f}M tok "
              f"({P*T*d*2/1e9:.1f} GB fp16) ...", flush=True)
        t0 = time.time()
        pool = np.empty((P, T, d), dtype=np.float16)
        order = rng.permutation(N)
        for j, s in enumerate(order[:P]):
            pool[j] = acts3[s]
            if j % 512 == 0 and j:
                print(f"[sasa]   {j}/{P} ({time.time()-t0:.0f}s)", flush=True)
        cursor = P                                          # next unseen sequence in this permutation
        pool_flat = pool.reshape(P * T, d)
        print(f"[sasa] buffer ready in {time.time()-t0:.0f}s", flush=True)

        def load_batch():
            nonlocal cursor, order
            # refresh a few slots with fresh whole sequences (contiguous reads)
            for _ in range(a.refresh_seqs):
                if cursor >= N:
                    order = rng.permutation(N); cursor = 0
                pool[rng.integers(0, P)] = acts3[order[cursor]]; cursor += 1
            idx = rng.integers(0, P * T, size=a.batch)
            x = torch.from_numpy(pool_flat[idx].astype(np.float32)).to(dev)
            x = (x - mu) @ zca
            return torch.nn.functional.normalize(x, dim=-1)

    model = SASA(d, a.G, a.b).to(dev)
    # DECOUPLED weight decay (AdamW) on the factored dict E,D = nuclear-norm rank adaptation.
    # (Must NOT be added to the loss — that dwarfs recon and just crushes the weights → EV~0.)
    opt = torch.optim.AdamW([
        {"params": [model.E, model.D], "weight_decay": a.wd},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=a.lr)

    use_wandb = bool(a.wandb)
    if use_wandb:
        import wandb
        wandb.init(project=a.wandb, name=a.run_name or None,
                   config=vars(a) | {"ntok": ntok, "n_seq": N, "seq_len": T, "d": d,
                                     "dataset": meta.get("dataset"), "model": meta.get("model")})
    fired_ever = torch.zeros(a.G, device=dev)

    def save(final=False):
        torch.save({"E": model.E.detach().cpu(), "D": model.D.detach().cpu(),
                    "bias": model.bias.detach().cpu(), "G": a.G, "b": a.b, "d": d,
                    "k": a.k, "step": step}, f"{a.out_dir}/sasa.pt")

    t_last = time.time()
    for step in range(a.steps + 1):
        x = load_batch()
        xh, z, gn, mask = model(x, a.k)
        recon = ((x - xh) ** 2).sum(-1).mean()
        loss = recon                                        # WD is decoupled (in the optimizer), not here
        opt.zero_grad(); loss.backward(); opt.step()
        fired_ever += mask.sum(0).detach()

        if step % a.log_every == 0:
            with torch.no_grad():
                var = ((x - x.mean(0)) ** 2).sum(-1).mean()
                ev = 1 - recon / var
                zsel = z * mask.unsqueeze(-1)                # [B,G,b]
                C = torch.einsum("bgi,bgj->gij", zsel, zsel)  # [G,b,b] block code 2nd moments
                tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
                fro2 = (C ** 2).sum((-2, -1))
                active = mask.sum(0) > 0
                effr = (tr[active] ** 2 / fro2[active].clamp(min=1e-12)).mean()
                dead = int((fired_ever == 0).sum())
            now = time.time(); sps = a.log_every / max(now - t_last, 1e-6); t_last = now
            if use_wandb:
                wandb.log({"loss": loss.item(), "recon": recon.item(), "ev": ev.item(),
                           "eff_rank": effr.item(), "dead_blocks": dead, "l0": a.k,
                           "steps_per_s": sps, "step": step})
            print(f"[sasa] step {step} loss {loss.item():.4f} EV {ev.item():.3f} effr {effr.item():.2f} "
                  f"dead {dead} ({sps:.2f} step/s)", flush=True)

        if step and step % a.save_every == 0:
            save()

    # ---- finalize: per-block orthonormal Q from decoder columns (for subspace projection) ----
    D = model.D.detach().view(a.G, a.b, d)                  # [G,b,d]
    Q = torch.zeros(a.G, a.b, d)
    for g in range(a.G):
        q, _ = torch.linalg.qr(D[g].T)                      # [d,b] orthonormal cols
        Q[g] = q.T
    save(final=True)
    torch.save({"Q": Q.cpu(), "fired_ever": fired_ever.cpu(), "G": a.G, "b": a.b, "d": d}, f"{a.out_dir}/blocks_Q.pt")
    shutil.copy(f"{a.acts_dir}/whiten_mu.npy", f"{a.out_dir}/whiten_mu.npy")
    if os.path.abspath(zca_path) != os.path.abspath(f"{a.out_dir}/whiten_zca.npy"):
        shutil.copy(zca_path, f"{a.out_dir}/whiten_zca.npy")
    json.dump({"G": a.G, "b": a.b, "k": a.k, "d": d, "steps": a.steps, "batch": a.batch, "lr": a.lr, "wd": a.wd,
               "alive": int((fired_ever > 0).sum()), "acts_dir": a.acts_dir, "ntok_available": ntok,
               "dataset": meta.get("dataset"), "model": meta.get("model"), "layer": meta.get("layer")},
              open(f"{a.out_dir}/meta.json", "w"), indent=1)
    print(f"[sasa] DONE -> {a.out_dir} | alive blocks {int((fired_ever>0).sum())}/{a.G}", flush=True)


if __name__ == "__main__":
    main()
