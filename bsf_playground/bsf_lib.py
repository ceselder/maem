"""Block-sparse subspace featurizer (SASA / "BSF") math for Qwen3.6-27B layer 42, shared by the stats job and the playground.

Conventions (== data/bsf_verbatim_worker.py / modal_bank_everything.py, verbatim):
    xc = x - mu_bsf                      centered RAW layer-42 residual (x = resid_post after block 42, BOS dropped)
    y  = xc @ zca                        ZCA-whitened (unnormalized);  y_n = normalize(y) is what the encoder sees
    gn_g = || (y_n @ E).view(G, b)[g] || block g's encoder activation; the k=32 largest per token are "active"
    c_g  = Q[g] @ y                      block g's 8 subspace coordinates (Q[g]: [b, d] orthonormal rows, whitened space)
    x_g  = (c_g @ Q[g]) @ zca_inv        the block's additive component of xc in RESIDUAL space; dir = unit(x_g)
A block's "PCs" are the principal axes of c_g over corpus tokens where the block is active (mean m_g, eigvecs V_g [b, b] columns
sorted by variance, eigvals lam_g). Steering along PC i: c(alpha) = m_g + alpha * sqrt(lam_i) * V_g[:, i];  feeding a PC directly:
c = +-V_g[:, i].  Either code -> residual direction via `dir_from_code`.
"""
import numpy as np
import torch
import torch.nn.functional as F

D = 5120


class BSF:
    def __init__(self, bsf_dir, device="cuda", e_dtype=torch.float32):
        s = torch.load(f"{bsf_dir}/sasa.pt", map_location="cpu", weights_only=False)
        self.G, self.b, self.k = int(s["G"]), int(s["b"]), int(s["k"])
        self.E = s["E"].to(device, e_dtype)                       # [d, G*b]
        self.Q = torch.load(f"{bsf_dir}/blocks_Q.pt", map_location="cpu", weights_only=False)["Q"].to(device, torch.float32)   # [G, b, d]
        self.mu = torch.from_numpy(np.load(f"{bsf_dir}/whiten_mu.npy").astype(np.float32)).to(device)
        zca = np.load(f"{bsf_dir}/whiten_zca.npy").astype(np.float64)
        self.zca = torch.from_numpy(zca.astype(np.float32)).to(device)
        self.zca_inv = torch.from_numpy(np.linalg.inv(zca).astype(np.float32)).to(device)
        self.device = device

    @torch.no_grad()
    def whiten(self, x):
        """x [n, d] raw fp32 -> y [n, d] whitened (unnormalized)."""
        return (x.float() - self.mu) @ self.zca

    @torch.no_grad()
    def encode(self, y, chunk=4096):
        """y [n, d] whitened -> gn [n, G] block activations (encoder, unit-normed input)."""
        out = []
        yn = F.normalize(y, dim=-1)
        for c0 in range(0, yn.shape[0], chunk):
            z = (yn[c0:c0 + chunk].to(self.E.dtype) @ self.E).view(-1, self.G, self.b)
            out.append(z.float().norm(dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def coords(self, y, blocks):
        """y [n, d] whitened; blocks [n] or [n, R] int64 -> c [n, (R,) b] subspace coordinates."""
        Qb = self.Q[blocks]                                       # [n, (R,) b, d]
        if Qb.dim() == 3:
            return torch.einsum("nd,nbd->nb", y, Qb)
        return torch.einsum("nd,nrbd->nrb", y, Qb)

    @torch.no_grad()
    def dir_from_code(self, block, code):
        """code [b] (or [m, b]) subspace coordinates of `block` -> unit residual-space direction(s) [d] / [m, d]."""
        code = torch.as_tensor(code, dtype=torch.float32, device=self.device)
        yb = code @ self.Q[int(block)]                            # [.., d] component in whitened space
        xb = yb @ self.zca_inv                                    # residual space (centered)
        return F.normalize(xb, dim=-1)

    @torch.no_grad()
    def dir_from_act(self, x, block):
        """x [d] raw act -> unit direction of its component in `block`'s subspace (the bank builder's mint)."""
        y = self.whiten(x.view(1, -1))[0]
        return self.dir_from_code(block, self.Q[int(block)] @ y)


def pca_block(mean, cov):
    """mean [b], cov [b, b] (of the coordinates on active tokens) -> (V [b, b] eigvecs as columns, lam [b]) sorted desc."""
    lam, V = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0, None), V[:, order]
    # sign convention: make the mean's projection on each PC non-negative (so +alpha moves "along the block's typical side")
    sgn = np.sign(V.T @ np.asarray(mean, dtype=np.float64)); sgn[sgn == 0] = 1
    return (V * sgn).astype(np.float32), lam.astype(np.float32)


def eff_rank(lam):
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0, None)
    s = lam.sum()
    return float(s * s / (lam * lam).sum()) if s > 0 else 0.0
