"""Universal held-out eval suite for the Qwen3.6-27B direction->text inverter (wandb-facing).

Evaluates SIX frozen held-out direction families with the EXACT SFT/RL inject recipe (unit dir
injected norm-matched at INJECT_LAYER on the trailing ' ?' marker, best-of-bo sampling) and the
EXACT eval_dirs scoring protocol (clean base via actor.disable_adapter(), re-encode @READ_LAYER,
sink-prepended + 10x-median norm-filtered, max over content tokens):

  1. probe    held-out rows of the RL probe bank (probes.f32)      -> eval/probe/cos
  2. sae      held-out SAE features (unit encoder columns)         -> eval/sae/{norm_act,fired,
              beat_corpus,mean_rank,rank1_frac,mrr}
  3. jlens    J-lens steering vectors unit(W_U[t] @ J_42) for
              random vocab tokens t (workspace-paper direction)    -> eval/jlens/cos
  4. cluster  see NOTE below — probe-bank rows DISJOINT from (1)   -> eval/cluster/cos
  5. random   isotropic Gaussian unit dirs — OFF-manifold CONTROL  -> eval/random/cos
  6. realact  real L42 token activations from a HELD-OUT slice
              (last 5% of sequences) of the bsf27b acts dump       -> eval/realact/cos

  CONTROL: eval/random/cos must stay ~= 1/sqrt(D_MODEL) = 1/sqrt(5120) ~= 0.014. Random dirs are
  off-manifold and uninvertible; if this number rises, the metric is being gamed (e.g. the scorer
  is picking up a shared high-norm direction, or the norm filter broke) — distrust every family.

  NOTE on the cluster family: the clustering stage's centroids.npy is in ZCA-whitened +
  L2-normalized space (meta cluster_space="zca_whiten_l2norm"), NOT raw residual space, so it
  CANNOT be injected as-is (same caveat as SL/build_dom_probes.py). There is no raw-space
  cluster-centroid direction file under /root/pmx/data, so the cluster family reuses the RL
  probe bank (cluster-probe directions ARE the per-cluster raw-space directions) with a sample
  DISJOINT from family (1) — i.e. it is a second independent probe sample, kept as a separate
  curve so the wandb panel layout survives if a true centroid file lands later.

HELD-OUT MODE (--heldout-pool, PREFERRED for the universal inverter): the families above are
FRESH-SAMPLED from the same sources the training bank draws from, so they OVERLAP pool_train
(the sae family is ~100% train-contaminated). Passing --heldout-pool <dir> (e.g.
data/pool_universal/pool_heldout, built train-disjoint per family by SL/build_universal_bank.py)
instead evals the pool's own families {bsf, realact, sae, jlens, cluster} — directions are the
pool's vecs.f32 rows, sae feature ids come from the records — plus the synthetic random control.
Cosine families become [bsf, realact, jlens, cluster, random] (probe exists only in the legacy
fresh-sample mode) and the cache lives at <cache>/eval_sets_heldout.pt so the two modes never
share a cache. Scoring/metrics are IDENTICAL in both modes.

Every family is sampled ONCE with a fixed seed and cached to <cache>/eval_sets.pt; all later
evals reuse the identical directions (and identical generation RNG via a forked, fixed seed), so
the wandb curves are low-variance and checkpoint-comparable. run_eval() is a clean function with
no wandb calls — import it from the trainer and log the returned flat dict:

    from eval_universal import build_eval_sets, run_eval
    es = build_eval_sets(cache_path, sae, wu, j42, probe_bank, acts_dir, n, dev)   # cached
    wandb.log(run_eval(actor, tok, prompt_ids, marker, sub, es, sae,
                       bo=4, temp=1.0, max_new=64, min_new=16, dev=dev), step=step)

Standalone:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/eval_universal.py \
        --adapter ckpts/rl/final --sae-path ... --maxacts-path ... \
        --heldout-pool data/pool_heldout --out eval_universal.json --wandb uni-inverter
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from mxf.config import MODEL, D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF, WHITEN_MU
from mxf.inject import get_layer, make_inject_hook, hooked, read_resid
from mxf.prompts import build_prompt_ids
from mxf.sae import load_sae, load_max_acts

NORM_FILTER_MULT = 10.0   # same as eval_dirs: drop re-encoded tokens with norm > 10x batch median
SAE_FIRE_LEGACY = 1.0     # the OLD arbitrary "fired" cut (raw act > 1.0); still logged as eval/sae/{fired,unverbalized}_1p0 for continuity
SAE_FIRE = 1.654          # "fired" = the SAE's own learned BatchTopK gate (ceselder/qwen36-27b-sae-l42: threshold 1.6539); overwritten with
                          # the exact value from the checkpoint by configure_sae_fire(path). Everything that reads EU.SAE_FIRE inherits it.


def sae_gate(path):
    """Learned firing threshold stored in a dictionary_learning BatchTopK checkpoint (key "threshold"); None if absent."""
    try:
        params = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        params = torch.load(path, map_location="cpu", weights_only=False)
    t = params.get("threshold") if isinstance(params, dict) else None
    return float(t.item() if hasattr(t, "item") else t) if t is not None else None


def configure_sae_fire(path):
    """Set the module-wide fire threshold to the checkpoint's gate (call once after load_sae). Returns the value in force."""
    global SAE_FIRE
    g = sae_gate(path) if path else None
    if g is not None and g > 0:
        SAE_FIRE = g
    return SAE_FIRE
GEN_SEED = 1234           # fixed sampling noise per eval (forked RNG — does not touch trainer RNG)
HELDOUT_FRAC = 0.05       # realact: last 5% of acts.f16 sequences are eval-only (never trained on)

COS_FAMILIES = ["probe", "jlens", "cluster", "random", "realact"]   # legacy fresh-sample mode
COS_FAMILIES_HELDOUT = ["bsf", "realact", "jlens", "cluster", "random"]   # held-out-pool mode
HELDOUT_POOL_FAMILIES = ["bsf", "realact", "sae", "jlens", "cluster"]     # record families in pool_heldout
CONTROL_FAMS = {"random"}   # lower-is-better control(s): logged per-family, EXCLUDED from mean_all
# EXTRA families (eval cache v2, data/mlp42_bank_worker.py): meta["extra_families"] lists cosine families scored IN
# ADDITION to meta["cos_families"] and NEVER folded into eval/mean_all, so old and new runs stay comparable. Each
# extra family <fam> ships <fam>_dirs [n, d] plus <fam>_neuron / <fam>_polarity / <fam>_corpus_max [n, k] (k members
# per direction: 1 for "mlp" single neurons, 2 for "mlp_pair" composites) for the layer-42 MLP fire-back metric.
EXTRA_FAMS_KEY = "extra_families"
MLP_LAST_K = 5                       # fire-back: max over the LAST 5 kept tokens of polarity * a_i / corpus_max_i
MLP_FIRE_LEVELS = (0.10, 0.25, 0.50)  # -> eval/<fam>/fired10 / fired25 / fired50
# RANK metrics (the SAE family's rank1_frac / rank_le5 / mean_rank / mrr mirrored for MLP neurons): the paired neuron's rank
# among ALL d_ff layer-42 neurons by NORMALIZED value polarity_j * a_j / corpus_max_j at the token (of the last MLP_LAST_K
# kept) where its own normalized value peaks -- rank 1 = the single most (relatively) active neuron there. Needs corpus
# statistics for EVERY neuron (MLP_STATS_DEFAULT = data/mlp42_neurons_worker.py's stats: polarity, max|a|, log-histograms).
MLP_RANK_LE = 10                                   # eval/<fam>/rank_le10
MLP_TOP_PCT = 0.99                                 # eval/<fam>/top1pct_frac: achieved value above the neuron's own 99th corpus percentile
MLP_STATS_DEFAULT = "/data/mlp42/neuron_stats.npz"  # per-neuron corpus stats of all 17,408 neurons (1.02M FineFineWeb tokens)
MLP_CHANCE_ACTS_DEFAULT = "/data/acts27b"          # random corpus windows (held-out tail rows) for the chance level of the rank metrics
MLP_CHANCE_SEED = 20260908                         # window draw seed (per eval row, so sharding over ranks never changes the draw)
ENV_EVAL_CACHE = "MAEMM_EVAL_CACHE"  # env override of the eval-cache path (rl.py / rl_disagg.py / eval_ckpt_daemon.py --eval-cache default)


def extra_families(eval_sets):
    """Extra (non-mean_all) cosine families present in an eval-set cache — [] for v1 caches."""
    return list(eval_sets["meta"].get(EXTRA_FAMS_KEY, []))


# SAE-SLICE families (eval cache v3, eval/modal_build_eval_v3.py): held-out features of a SECOND SAE (the 2,097,152-feature
# layer-42 SAE /data/sae2m/trainer_0/ae.pt) whose full encoder (21.5 GB bf16) is never loaded by an evaluator. The cache ships,
# per family <fam>, <fam>_dirs [n, d] (unit ENCODER columns for "sae2m_enc", unit DECODER rows for "sae2m_dec" -- the SAME held-out
# features), <fam>_feats [n] (feature ids) and <fam>_corpus_peak [n] (top activation over the 1.0B-token max-acts scan), plus ONE
# shared eval_sets["sae_slice"] = {feats [m] sorted, W_enc [d, m], b_enc [m], b_dec [d], threshold, F, ae} holding just the eval
# features' encoder columns. Scoring == the 131k "sae" family (relu((h - b_dec) . W_enc[:, f] + b_enc[f]), max over kept tokens,
# best-of-bo): eval/<fam>/{cos, norm_act, fired, beat_corpus, unverbalized_frac, unverbalized_p10, gate}; "fired" = the 2M SAE's
# own learned gate. No cross-feature rank metrics (they need every column). NEVER in eval/mean_all (cache comparability).
SLICE_FAMS_KEY = "sae_slice_families"
SLICE_KEY = "sae_slice"


def slice_families(eval_sets):
    """SAE-slice families present in an eval-set cache — [] for v1/v2 caches."""
    return list(eval_sets["meta"].get(SLICE_FAMS_KEY, []))


class SliceSAE:
    """Encoder columns of a subset of features of a (huge) BatchTopK SAE, exposing BatchTopKSAE.encode_features so
    score_sae_peaks scores it unchanged. Feature ids are the FULL SAE's ids; the slice maps them to columns."""

    def __init__(self, feats, W_enc, b_enc, b_dec, threshold, F, device):
        self.feats = torch.as_tensor(feats, dtype=torch.long).cpu()            # sorted [m]
        assert bool((self.feats[1:] > self.feats[:-1]).all()), "slice feats must be sorted and unique"
        self.W_enc = torch.as_tensor(W_enc).float().to(device)                 # [d, m]
        self.b_enc = torch.as_tensor(b_enc).float().to(device)                 # [m]
        self.b_dec = torch.as_tensor(b_dec).float().to(device)                 # [d]
        self.threshold = float(threshold); self.d_sae = int(F); self.d_in = int(self.W_enc.shape[0])
        assert self.W_enc.shape[1] == len(self.feats) == len(self.b_enc), (self.W_enc.shape, len(self.feats), len(self.b_enc))

    def cols(self, feature_ids):
        f = torch.as_tensor(feature_ids, dtype=torch.long).cpu()
        pos = torch.searchsorted(self.feats, f)
        assert bool((pos < len(self.feats)).all()) and bool((self.feats[pos.clamp(max=len(self.feats) - 1)] == f).all()), \
            "feature id not in the SAE slice"
        return pos.to(self.W_enc.device)

    def encode_features(self, acts_BLD, feature_ids):
        idx = self.cols(feature_ids)
        return torch.relu((acts_BLD.to(self.W_enc.dtype) - self.b_dec) @ self.W_enc[:, idx] + self.b_enc[idx])


def load_sae_slice(eval_sets, device):
    """The cache's sae_slice dict -> SliceSAE on `device` (None when the cache has no slice families)."""
    if not slice_families(eval_sets):
        return None
    sl = eval_sets[SLICE_KEY]
    return SliceSAE(sl["feats"], sl["W_enc"], sl["b_enc"], sl["b_dec"], sl["threshold"], sl["F"], device)


def slice_metrics(fam, best_act, corpus_peak, gate):
    """The sae-family firing metrics for a slice family: best_act / corpus_peak np [n], gate = that SAE's learned threshold."""
    best = np.asarray(best_act, np.float64); cp = np.asarray(corpus_peak, np.float64)
    na = best / np.maximum(cp, 1e-6)
    out = {f"eval/{fam}/norm_act": float(na.mean()), f"eval/{fam}/fired": float(np.mean(best > gate)),
           f"eval/{fam}/beat_corpus": float(np.mean(best > cp)), f"eval/{fam}/unverbalized_frac": float(np.mean(best <= gate)),
           f"eval/{fam}/unverbalized_p10": float(np.mean(na < 0.10)), f"eval/{fam}/gate": float(gate),
           f"eval/{fam}/best_act_mean": float(best.mean())}
    out[f"eval/all/{fam}_norm_act"] = out[f"eval/{fam}/norm_act"]; out[f"eval/all/{fam}_unverbalized"] = out[f"eval/{fam}/unverbalized_frac"]
    return out


@torch.no_grad()
def eval_sae_slice_family(tag, dirs_unit, feats, slice_sae, actor, tok, prompt_ids, marker, sub, dev,
                          bo, temp, max_new, min_new, gen_chunk, keep_texts=False):
    """Slice family: best-of-bo max-token target-feature act (via the slice) AND best-of-bo max-token cosine to the injected
    direction. Returns (best_act np [N], best_cos np [N]) (+ best-by-act sample texts when keep_texts)."""
    n = len(feats)
    best = np.full(n, -1e9); best_cos = np.full(n, -1e9); best_txt = [""] * n
    for rows, texts in _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev, bo, temp, max_new, min_new, gen_chunk):
        acts, _ = score_sae_peaks(texts, [feats[i] for i in rows], slice_sae, actor, tok, dev)
        rdirs = F.normalize(torch.stack([dirs_unit[i] for i in rows]), dim=-1)
        cos = score_probe_cos(texts, rdirs, actor, tok, dev).numpy().astype(np.float64)
        np.maximum.at(best_cos, rows, cos)
        for j, i in enumerate(rows):
            if acts[j].item() > best[i]:
                best[i] = acts[j].item(); best_txt[i] = texts[j]
    return (best, best_cos, best_txt) if keep_texts else (best, best_cos)


# ---------------------------------------------------------------------------------------------
# scoring (protocol identical to SL/eval_dirs.py; duplicated so this module imports standalone)
# ---------------------------------------------------------------------------------------------

def _reencode(gen_texts, actor, tok, device, sbatch=32):
    """Yield (s, h [B,T,d] fp32, keep [B,T] bool) on the CLEAN base @READ_LAYER, sink-prepended +
    norm-filtered, in sub-batches. EXACT copy of eval_dirs._reencode."""
    prev = tok.padding_side; tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    try:
        for s in range(0, len(gen_texts), sbatch):
            batch = [t if t.strip() else " " for t in gen_texts[s:s + sbatch]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=95, add_special_tokens=False).to(device)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=device, dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=device, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], 1)
            with actor.disable_adapter():
                h, mask = read_resid(actor, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
            keep = mask.clone(); keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            yield s, h, keep
    finally:
        tok.padding_side = prev


# ---------------------------------------------------------------------------------------------
# scorer centering (fix 2026-09-18). Every target direction in the suite is MEAN-FREE: the training banks store
# unit(act - mu) (data/collect_bank_worker.py, collect_fullctx_worker.py; mu = the layer-42 corpus mean, ||mu|| = 61,
# one coordinate = 59.6) and every eval-cache family has cos(dir, mu) ~ 0. The model-side activation must be centered
# the same way before the cosine. Until this fix the scorer normalised the RAW activation, so (d _|_ mu) every cosine
# was the true centered cosine times ||h - mu|| / ||h||: the SOURCE TEXT itself scored ~.47 (short windows) / ~.55
# (full-document tails) instead of 1, and every reported mean_all / RL reward was compressed by that token-dependent
# factor. MAEMM_SCORER_CENTER=0 reproduces the legacy raw metric (only for re-deriving old numbers).
# ---------------------------------------------------------------------------------------------
CENTER_MU_PATH = os.environ.get("MAEMM_WHITEN_MU", WHITEN_MU)   # model profile default (27B: /data/acts27b/whiten_mu.npy)
_CENTER_MU = {}


def scorer_centered():
    return os.environ.get("MAEMM_SCORER_CENTER", "1") != "0"


def center_mu(device):
    """[d] float32 corpus mean on `device` (cached per device), or None when centering is switched off. Hard-fails when the
    mean file is missing: the scorer must never fall back to the legacy raw cosine silently."""
    if not scorer_centered():
        return None
    key = str(device)
    if key not in _CENTER_MU:
        if not os.path.exists(CENTER_MU_PATH):
            raise FileNotFoundError(f"scorer centering needs the layer-42 corpus mean at {CENTER_MU_PATH} (env MAEMM_WHITEN_MU); "
                                    f"MAEMM_SCORER_CENTER=0 only to reproduce legacy raw-cosine numbers")
        mu = torch.from_numpy(np.load(CENTER_MU_PATH).astype(np.float32))
        assert mu.shape == (D_MODEL,), f"whiten_mu shape {tuple(mu.shape)} != ({D_MODEL},)"
        _CENTER_MU[key] = mu.to(device)
    return _CENTER_MU[key]


def scorer_protocol():
    """What the cosine scorer did, for every metrics json / transcript: centered or legacy-raw, and which mean."""
    if not scorer_centered():
        return {"centered": False, "mu_path": None, "mu_norm": None, "metric": "cos(unit(h), d)  [LEGACY: raw activation, compressed by ||h-mu||/||h||]"}
    mu = center_mu("cpu")
    return {"centered": True, "mu_path": CENTER_MU_PATH, "mu_norm": round(float(mu.norm()), 3), "metric": "cos(unit(h - mu), d)"}


@torch.no_grad()
def score_probe_cos(gen_texts, dirs, actor, tok, device):
    """max-over-content-token cosine(h_t, dir_i) for gen i; dirs [N,d] unit. == eval_dirs."""
    out = torch.zeros(len(gen_texts))
    for s, h, keep in _reencode(gen_texts, actor, tok, device):
        d = dirs[s:s + h.shape[0]].to(device).float()                       # [b,d] unit (mean-free)
        mu = center_mu(device)
        hn = F.normalize(h.float() - mu if mu is not None else h.float(), dim=-1)   # [b,T,d]  centered like every target
        cos = torch.einsum("btd,bd->bt", hn, d)                             # [b,T]
        out[s:s + h.shape[0]] = cos.masked_fill(~keep, -1.0).max(1).values.float().cpu()
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def _reencode_mlp_full(gen_texts, actor, tok, device, sbatch=32):
    """_reencode that ALSO returns EVERY layer-42 MLP neuron value (the down_proj input, [b, T, d_ff] fp32) of each row.
    Yields (s, h [b,T,d] fp32, keep [b,T], a [b,T,d_ff] fp32, lastk [b,T]) with lastk = keep restricted to the last MLP_LAST_K
    positions of the row (sink counted in the length, as in the verbalization scorer of data/mlp42_neurons_worker.py). Same
    tokenization / sink / 10x-median norm filter as _reencode. _reencode_mlp (the fire-back scorer) gathers the paired
    neurons from this; score_mlp_rank ranks them against all d_ff columns."""
    layer = get_layer(actor, READ_LAYER)
    down = layer.mlp.down_proj
    prev = tok.padding_side; tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    try:
        for s in range(0, len(gen_texts), sbatch):
            batch = [t if t.strip() else " " for t in gen_texts[s:s + sbatch]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=95, add_special_tokens=False).to(device)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=device, dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=device, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], 1)
            cap = {}

            def pre(_m, inp):
                cap["a"] = inp[0]

            def post(_m, _i, out):
                cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
                raise _Stop

            h1 = down.register_forward_pre_hook(pre); h2 = layer.register_forward_hook(post)
            try:
                with actor.disable_adapter():
                    actor(input_ids=ids, attention_mask=am)
            except _Stop:
                pass
            finally:
                h1.remove(); h2.remove()
            h = cap["h"]
            a = cap["a"].float()                                                    # [b, T, d_ff]
            keep = am.bool().clone(); keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            L = am.sum(1)                                                            # incl. sink
            pos = torch.arange(h.shape[1], device=device)[None, :]
            lastk = keep & (pos >= (L - MLP_LAST_K)[:, None])
            yield s, h, keep, a, lastk
    finally:
        tok.padding_side = prev


@torch.no_grad()
def _reencode_mlp(gen_texts, neuron_ids, actor, tok, device, sbatch=32):
    """_reencode that ALSO returns the layer-42 MLP neuron values (down_proj input) of each row's k paired neurons.
    neuron_ids: LongTensor / list [N, k]. Yields (s, h [b,T,d] fp32, keep [b,T], a_sel [b,k,T] fp32, lastk [b,T]) with
    lastk = keep restricted to the last MLP_LAST_K positions of the row (sink counted in the length, as in the
    verbalization scorer of data/mlp42_neurons_worker.py). Same tokenization / sink / 10x-median norm filter as _reencode
    (gathered from _reencode_mlp_full; the values are the same fp32 tensor's columns)."""
    nid_all = torch.as_tensor(neuron_ids, dtype=torch.long)
    if nid_all.dim() == 1:
        nid_all = nid_all[:, None]
    for s, h, keep, a, lastk in _reencode_mlp_full(gen_texts, actor, tok, device, sbatch):
        B = a.shape[0]
        nid = nid_all[s:s + B].to(device)                                           # [b, k]
        a_sel = a[torch.arange(B, device=device)[:, None], :, nid]                  # [b, k, T]
        yield s, h, keep, a_sel, lastk


@torch.no_grad()
def score_mlp_fireback(gen_texts, neuron_ids, polarity, corpus_max, actor, tok, device):
    """Layer-42 MLP fire-back of the generated texts on the CLEAN base: per text and paired neuron, max over the last
    MLP_LAST_K kept tokens of polarity * a_i / corpus_max_i. Returns (na_min [N], na_max [N]) cpu fp32 = the WEAKEST and
    the STRONGEST member's normalized fire-back (k=1 singles: identical). neuron_ids / polarity / corpus_max: [N, k]."""
    pol_all = torch.as_tensor(polarity, dtype=torch.float32); cm_all = torch.as_tensor(corpus_max, dtype=torch.float32)
    if pol_all.dim() == 1:
        pol_all = pol_all[:, None]; cm_all = cm_all[:, None]
    na_min = torch.zeros(len(gen_texts)); na_max = torch.zeros(len(gen_texts))
    for s, h, keep, a_sel, lastk in _reencode_mlp(gen_texts, neuron_ids, actor, tok, device):
        b = h.shape[0]
        pol = pol_all[s:s + b].to(device); cm = cm_all[s:s + b].to(device).clamp_min(1e-6)
        av = (a_sel * pol[:, :, None]).masked_fill(~lastk[:, None, :], -float("inf")).max(2).values   # [b, k]
        av = torch.where(torch.isfinite(av), av, torch.zeros_like(av))                                  # no kept token -> 0
        na = av / cm
        na_min[s:s + b] = na.min(1).values.float().cpu(); na_max[s:s + b] = na.max(1).values.float().cpu()
    return na_min, na_max


def mlp_metrics(fam, best_na, best_na_any=None):
    """eval/<fam>/{norm_act, fired10, fired25, fired50} from best-of-bo normalized fire-back per direction (np [n]);
    for k>1 families also any_* from the strongest member."""
    best_na = np.asarray(best_na, np.float64)
    out = {f"eval/{fam}/norm_act": float(best_na.mean())}
    for lv in MLP_FIRE_LEVELS:
        out[f"eval/{fam}/fired{int(round(lv * 100))}"] = float(np.mean(best_na >= lv))
    if best_na_any is not None:
        best_na_any = np.asarray(best_na_any, np.float64)
        out[f"eval/{fam}/any_norm_act"] = float(best_na_any.mean())
        for lv in MLP_FIRE_LEVELS:
            out[f"eval/{fam}/any_fired{int(round(lv * 100))}"] = float(np.mean(best_na_any >= lv))
    return out


# ---------------------------------------------------------------------------------------------
# MLP RANK metrics: the paired neuron's rank among ALL d_ff layer-42 neurons (+ within-neuron corpus percentile, + chance)
# ---------------------------------------------------------------------------------------------

def load_mlp_neuron_stats(path=MLP_STATS_DEFAULT):
    """Per-neuron corpus statistics of EVERY layer-42 MLP neuron (data/mlp42_neurons_worker.py `stats` stage: 4,000 x 256
    FineFineWeb train windows = 1.02M tokens): polarity (sign of the extreme corpus value), corpus max|a| (the eval cache's
    <fam>_corpus_max IS max_abs[neuron]; polarity likewise) and the sign-resolved log10|a| histograms (16 bins/decade over
    [1e-4, 1e3) + under/overflow), re-keyed by FIRING side: hist_same = tokens with polarity * a > 0, hist_opp = polarity * a
    < 0. Returns a dict of numpy arrays (+ torch views of polarity / corpus_max for the scorer)."""
    z = np.load(path)
    pol = z["polarity"].astype(np.float32); cmax = z["max_abs"].astype(np.float32)
    hp, hn = z["hist_pos"].astype(np.float64), z["hist_neg"].astype(np.float64)                # [NB, N]
    pos_pol = pol[None, :] > 0
    return {"path": path, "N": int(z["N"]), "n_tok": float(z["n_tok"]), "polarity": pol, "corpus_max": cmax,
            "hist_same": np.where(pos_pol, hp, hn), "hist_opp": np.where(pos_pol, hn, hp),
            "n_zero": z["n_zero"].astype(np.float64), "hist_lo": float(z["hist_lo"]), "hist_bpd": int(z["hist_bpd"]),
            "polarity_t": torch.from_numpy(pol.copy()), "corpus_max_t": torch.from_numpy(cmax.copy())}


def _hist_count_below(hist, nid, mag, lo, bpd):
    """#tokens (one side's histogram, [NB, N]: bin 0 = |a| < 10^lo, bin b in 1..NB-2 = [10^(lo+(b-1)/bpd), 10^(lo+b/bpd)),
    bin NB-1 = overflow) of neurons nid [n] with |a| < mag [n]; log-uniform interpolation inside the straddling bin."""
    NB = hist.shape[0]
    mag = np.asarray(mag, np.float64); nid = np.asarray(nid, np.int64)
    pos = (np.log10(np.maximum(mag, 1e-300)) - lo) * bpd
    b = np.clip(np.floor(pos).astype(np.int64) + 1, 0, NB - 1)
    frac = np.where(pos < 0, 0.0, np.where(b >= NB - 1, 0.0, pos - np.floor(pos)))
    csum = np.concatenate([np.zeros((1, hist.shape[1])), np.cumsum(hist, axis=0)], 0)   # csum[b] = tokens in bins < b
    return csum[b, nid] + hist[b, nid] * frac


def mlp_corpus_percentile(vals, nid, stats):
    """Within-neuron corpus percentile of achieved polarity-signed values: P_corpus(polarity * a <= v) for neuron nid, from
    the stats histograms. vals, nid: array-like [n]. Returns np float64 [n] in [0, 1] (0.99 = the value beats 99% of the
    neuron's own 1.02M corpus tokens; a norm_act of 1.0 is by construction ~1.0)."""
    vals = np.asarray(vals, np.float64); nid = np.asarray(nid, np.int64)
    lo, bpd = stats["hist_lo"], stats["hist_bpd"]
    n_opp = stats["hist_opp"].sum(0)[nid]; n_zero = stats["n_zero"][nid]
    pos = vals > 0
    cnt = np.empty_like(vals)
    # v > 0: every opposite-side token, the exact zeros, and the same-side tokens with |a| < v
    cnt[pos] = n_opp[pos] + n_zero[pos] + _hist_count_below(stats["hist_same"], nid[pos], vals[pos], lo, bpd)
    # v <= 0: the opposite-side tokens with |a| >= |v|
    cnt[~pos] = n_opp[~pos] - _hist_count_below(stats["hist_opp"], nid[~pos], -vals[~pos], lo, bpd)
    return np.clip(cnt / stats["n_tok"], 0.0, 1.0)


@torch.no_grad()
def score_mlp_rank(gen_texts, neuron_ids, stats, actor, tok, device, sbatch=32):
    """Cross-neuron RANK of each text's paired neuron(s) among ALL d_ff layer-42 neurons on the CLEAN base (the MLP analogue
    of sae_rank_at_peaks). Per text and member, over the last MLP_LAST_K kept tokens:
      av          polarity_i * a_i at the member's best token (max over the window; == norm_act * corpus_max_i)
      na          av / corpus_max_i  (identical to score_mlp_fireback's per-member value)
      rank        1 + #neurons j with polarity_j * a_j / corpus_max_j > the member's normalized value, AT ITS BEST TOKEN
      rank_best5  the smallest such rank over the window's tokens (rank recomputed at every kept last-k token)
      raw_rank    same best token, ranked by the polarity-signed RAW value polarity_j * a_j (no normalization: dense
                  high-magnitude neurons dominate -- the contrast metric)
      pct         within-neuron corpus percentile of av (mlp_corpus_percentile)
    No kept token in the window -> av = na = 0, ranks = d_ff (worst), pct = P(polarity * a <= 0). Returns a dict of cpu
    tensors [N_text, k] (float32 av/na/pct, int64 ranks). neuron_ids: [N, k] or [N]."""
    nid_all = torch.as_tensor(neuron_ids, dtype=torch.long)
    if nid_all.dim() == 1:
        nid_all = nid_all[:, None]
    n, k = nid_all.shape
    pol = stats["polarity_t"].to(device); cm = stats["corpus_max_t"].to(device).clamp_min(1e-6)
    N = pol.numel()
    out = {"av": torch.zeros(n, k), "na": torch.zeros(n, k), "rank": torch.full((n, k), N, dtype=torch.long),
           "rank_best5": torch.full((n, k), N, dtype=torch.long), "raw_rank": torch.full((n, k), N, dtype=torch.long)}
    for s, h, keep, a, lastk in _reencode_mlp_full(gen_texts, actor, tok, device, sbatch):
        b = a.shape[0]
        a_s = a * pol                                                       # [b, T, N] polarity-signed raw values
        a_n = a_s / cm                                                      # [b, T, N] normalized (corpus max = 1)
        nid = nid_all[s:s + b].to(device)                                   # [b, k]
        bi = torch.arange(b, device=device)
        has = lastk.any(1)                                                  # [b] a kept token in the window at all
        for j in range(k):
            nj = nid[:, j]
            tj = a_n[bi, :, nj]                                             # [b, T] the member's normalized value per token
            best_v, best_t = tj.masked_fill(~lastk, -float("inf")).max(1)  # its best token in the window
            best_t = torch.where(has, best_t, torch.zeros_like(best_t))
            row_n = a_n[bi, best_t]                                         # [b, N] every neuron, normalized, at that token
            r_norm = (row_n > row_n[bi, nj][:, None]).sum(1) + 1
            row_r = a_s[bi, best_t]                                         # [b, N] every neuron, raw signed, at that token
            r_raw = (row_r > row_r[bi, nj][:, None]).sum(1) + 1
            r_all = (a_n > tj[:, :, None]).sum(2) + 1                       # [b, T] the member's rank at every token
            r_best = r_all.masked_fill(~lastk, N + 1).min(1).values
            av = torch.where(has, a_s[bi, best_t, nj], torch.zeros_like(best_v))
            out["av"][s:s + b, j] = av.float().cpu()
            out["na"][s:s + b, j] = torch.where(has, best_v, torch.zeros_like(best_v)).float().cpu()
            out["rank"][s:s + b, j] = torch.where(has, r_norm, torch.full_like(r_norm, N)).cpu()
            out["raw_rank"][s:s + b, j] = torch.where(has, r_raw, torch.full_like(r_raw, N)).cpu()
            out["rank_best5"][s:s + b, j] = torch.where(has, r_best, torch.full_like(r_best, N)).cpu()
        del a_s, a_n
    out["pct"] = torch.from_numpy(mlp_corpus_percentile(out["av"].numpy().reshape(-1), nid_all.numpy().reshape(-1), stats)
                                  .reshape(n, k).astype(np.float32))
    return out


def mlp_rank_metrics(fam, rank, rank_best5, raw_rank, pct, prefix=""):
    """eval/<fam>/<prefix>{rank1_frac, rank_le10, mean_rank, median_rank, mrr} (+ best5_* over the whole last-k window, raw_*
    by raw magnitude) and corpus_pct_mean / top1pct_frac from per-direction [n, k] arrays (best-of-bo sample). k = 1 singles:
    plain shares. k > 1 (pairs): rank1_frac / rank_le10 / mrr / top1pct_frac are MEMBER-level shares (over 2n members),
    mean_rank / median_rank average the two members' ranks first (one number per pair), and both_rank_le10 / any_rank_le10 /
    both_top1pct_frac are the pair-level flags."""
    p = f"eval/{fam}/{prefix}"
    rank = np.asarray(rank, np.float64).reshape(len(rank), -1)
    k = rank.shape[1]
    out = {}

    def block(name, r):
        r = np.asarray(r, np.float64).reshape(len(r), -1)
        flat = r.reshape(-1)
        out[f"{p}{name}rank1_frac"] = float(np.mean(flat == 1))
        out[f"{p}{name}rank_le{MLP_RANK_LE}"] = float(np.mean(flat <= MLP_RANK_LE))
        out[f"{p}{name}mean_rank"] = float(np.mean(r.mean(1)))
        out[f"{p}{name}median_rank"] = float(np.median(r.mean(1)))
        out[f"{p}{name}mrr"] = float(np.mean(1.0 / flat))
        if k > 1:
            out[f"{p}{name}both_rank_le{MLP_RANK_LE}"] = float(np.mean((r <= MLP_RANK_LE).all(1)))
            out[f"{p}{name}any_rank_le{MLP_RANK_LE}"] = float(np.mean((r <= MLP_RANK_LE).any(1)))

    block("", rank); block("best5_", rank_best5); block("raw_", raw_rank)
    pct = np.asarray(pct, np.float64).reshape(len(pct), -1)
    out[f"{p}corpus_pct_mean"] = float(pct.mean())
    out[f"{p}top1pct_frac"] = float(np.mean(pct >= MLP_TOP_PCT))
    if k > 1:
        out[f"{p}both_top1pct_frac"] = float(np.mean((pct >= MLP_TOP_PCT).all(1)))
    return out


def mlp_chance_windows(acts_dir, rows, bo, min_len, max_len, fam_idx=0, seed=MLP_CHANCE_SEED):
    """Random corpus windows for the CHANCE level of the MLP metrics: bo windows per eval row (same best-of-bo protocol as the
    generations, each window a uniform 16-64-token span, like the 16-64-token generations) from the HELD-OUT tail rows of
    the acts27b dump (>= ceil(0.95 n_seq): the same rows the realact eval family comes from; every neuron statistic, scan
    and bank used rows below). The draw is seeded PER ROW ([seed, fam_idx, row]) so sharding over ranks never changes it.
    Returns (spec [len(rows)*bo, 3] int64 of (seq, offset, length), ids: list of token-id lists), row-major rows x bo."""
    meta = json.load(open(os.path.join(acts_dir, "meta.json")))
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    lo = int(math.ceil((1.0 - HELDOUT_FRAC) * NS))
    toks = np.memmap(os.path.join(acts_dir, "toks.i32"), dtype=np.int32, mode="r", shape=(NS, T))
    spec, ids = [], []
    for i in rows:
        rng = np.random.default_rng([seed, int(fam_idx), int(i)])
        for _ in range(bo):
            L = int(rng.integers(min_len, max_len + 1))
            r = int(rng.integers(lo, NS)); o = int(rng.integers(0, T - L + 1))
            spec.append((r, o, L)); ids.append([int(x) for x in toks[r, o:o + L]])
    return np.asarray(spec, np.int64).reshape(-1, 3), ids


@torch.no_grad()
def mlp_chance_rows(fam, fam_idx, es, stats, actor, tok, device, acts_dir, rows, bo, min_len, max_len):
    """The MLP fire-back AND rank metrics of RANDOM corpus text, per eval row of family `fam` (chance level): bo random
    windows (mlp_chance_windows) stand in for the bo generations; the best-of-bo sample is picked exactly as for
    generations (max over samples of the weakest member's normalized fire-back). Returns {row: {"na", "na_any", "rank",
    "rank_best5", "raw_rank", "pct", "av": [k], "win": [[seq, off, len]] * bo}} for the given rows (sharding-safe)."""
    rows = [int(i) for i in rows]
    if not rows:
        return {}
    spec, ids = mlp_chance_windows(acts_dir, rows, bo, min_len, max_len, fam_idx=fam_idx)
    texts = [tok.decode(x, skip_special_tokens=True) for x in ids]
    rr = [i for i in rows for _ in range(bo)]
    nid = es[f"{fam}_neuron"][rr]
    na_min, na_max = score_mlp_fireback(texts, nid, es[f"{fam}_polarity"][rr], es[f"{fam}_corpus_max"][rr], actor, tok, device)
    rk = score_mlp_rank(texts, nid, stats, actor, tok, device)
    na_min = na_min.view(len(rows), bo); na_max = na_max.view(len(rows), bo)
    best, arg = na_min.max(1)
    sel = torch.arange(len(rows)) * bo + arg
    out = {}
    for j, i in enumerate(rows):
        out[i] = {"na": float(best[j]), "na_any": float(na_max[j].max()), "av": rk["av"][sel[j]].tolist(),
                  "rank": rk["rank"][sel[j]].tolist(), "rank_best5": rk["rank_best5"][sel[j]].tolist(),
                  "raw_rank": rk["raw_rank"][sel[j]].tolist(), "pct": rk["pct"][sel[j]].tolist(),
                  "win": spec[j * bo:(j + 1) * bo].tolist()}
    return out


def mlp_chance_metrics(fam, ch, k):
    """Aggregate mlp_chance_rows output ({row: {...}}) -> eval/<fam>/chance_* scalars (+ the per-direction arrays for the
    per-direction dump), rows in sorted order."""
    idx = sorted(ch)
    A = lambda key: np.asarray([ch[i][key] for i in idx], np.float64).reshape(len(idx), -1)
    na = A("na")[:, 0]
    m = mlp_rank_metrics(fam, A("rank"), A("rank_best5"), A("raw_rank"), A("pct"), prefix="chance_")
    m[f"eval/{fam}/chance_norm_act"] = float(na.mean())
    for lv in MLP_FIRE_LEVELS:
        m[f"eval/{fam}/chance_fired{int(round(lv * 100))}"] = float(np.mean(na >= lv))
    if k > 1:
        m[f"eval/{fam}/chance_any_norm_act"] = float(A("na_any")[:, 0].mean())
    pd = {"row": idx, "norm_act": na.tolist(), "rank": A("rank").astype(int).tolist(), "rank_best5": A("rank_best5").astype(int).tolist(),
          "raw_rank": A("raw_rank").astype(int).tolist(), "corpus_pct": A("pct").tolist(), "best_signed_act": A("av").tolist(),
          "windows": [ch[i]["win"] for i in idx], "windows_note": "(acts27b seq, offset, length) of the bo random held-out windows per row"}
    return m, pd


@torch.no_grad()
def score_sae_peaks(gen_texts, feat_ids, sae, actor, tok, device):
    """Per-text max-over-content-token activation of ITS paired feature (same numbers as
    eval_dirs.score_sae) PLUS the clean-base L42 hidden state at that peak token, for the
    full-SAE rank metric. Returns (acts [N] cpu, peak_h [N,d] fp32 cpu)."""
    acts = torch.zeros(len(gen_texts))
    peaks = torch.zeros(len(gen_texts), D_MODEL)
    for s, h, keep in _reencode(gen_texts, actor, tok, device):
        b = h.shape[0]
        per = sae.encode_features(h, feat_ids[s:s + b])                     # [b,T,b]
        bi = torch.arange(b, device=per.device)
        a = per[bi, :, bi].masked_fill(~keep, -1.0)                         # -1 so argmax skips masked
        best, tpos = a.max(1)
        acts[s:s + b] = best.clamp(min=0.0).float().cpu()                   # == masked_fill(0).max
        peaks[s:s + b] = h[bi, tpos].float().cpu()
    return acts, peaks


@torch.no_grad()
def sae_rank_at_peaks(sae, peak_h, feats, chunk=256):
    """Rank of each paired feature among ALL features at its peak token. BatchTopKSAE exposes no
    full-encode method, so this is sae.encode_features generalized to every column:
    relu((x - b_dec) @ W_enc + b_enc) — pre-topk post-ReLU, same convention as scoring.
    rank n = 1 + #features strictly more active; n == 1 => the paired feature is the single
    most-active feature at its own peak token. Returns int64 np array [N]."""
    feats_t = torch.as_tensor(feats, dtype=torch.long)
    ranks = torch.zeros(len(feats), dtype=torch.long)
    for s in range(0, len(feats), chunk):
        x = peak_h[s:s + chunk].to(sae.W_enc.device, sae.W_enc.dtype)       # [c,d]
        full = torch.relu((x - sae.b_dec) @ sae.W_enc + sae.b_enc)          # [c,F]
        af = full.gather(1, feats_t[s:s + chunk].to(full.device).unsqueeze(1))   # [c,1]
        ranks[s:s + chunk] = ((full > af).sum(1) + 1).cpu()
    return ranks.numpy()


# ---------------------------------------------------------------------------------------------
# generation (recipe identical to eval_dirs.run_set)
# ---------------------------------------------------------------------------------------------

def _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                 bo, temp, max_new, min_new, gen_chunk):
    """Yield (rows, texts): each dir injected bo times, fb = gen_chunk//bo dirs per GPU batch."""
    n = len(dirs_unit)
    fb = max(1, gen_chunk // bo)
    for s in range(0, n, fb):
        rows = [i for i in range(s, min(s + fb, n)) for _ in range(bo)]
        vecs = [dirs_unit[i:i + 1] for i in rows]
        hook = make_inject_hook(vecs, [[marker]] * len(rows), STEER_COEFF, dev, torch.bfloat16, mode="add")
        ids = torch.tensor([list(prompt_ids)] * len(rows), device=dev)
        with hooked(sub, hook):
            gen = actor.generate(ids, do_sample=True, temperature=temp, top_p=1.0, top_k=0, min_p=0.0,
                                 max_new_tokens=max_new, min_new_tokens=min_new,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, len(prompt_ids):], skip_special_tokens=True)
        if s % (fb * 10) == 0:
            print(f"  [{tag}] {min(s + fb, n)}/{len(dirs_unit)}", flush=True)
        yield rows, texts


@torch.no_grad()
def eval_cos_family(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                    bo, temp, max_new, min_new, gen_chunk):
    """Best-of-bo max-token cosine per direction. Returns np [N]."""
    best = np.full(len(dirs_unit), -1e9)
    for rows, texts in _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                                    bo, temp, max_new, min_new, gen_chunk):
        rdirs = F.normalize(torch.stack([dirs_unit[i] for i in rows]), dim=-1)
        cos = score_probe_cos(texts, rdirs, actor, tok, dev)
        np.maximum.at(best, rows, cos.numpy().astype(np.float64))
    return best


@torch.no_grad()
def eval_sae_family(dirs_unit, feats, sae, actor, tok, prompt_ids, marker, sub, dev,
                    bo, temp, max_new, min_new, gen_chunk, keep_texts=False):
    """Best-of-bo max-token target-feature act per feature + full-SAE rank at the best gen's peak
    token. Returns (best_act np [N], ranks np int64 [N]) -- plus the best sample's text per feature
    (list [N]) when keep_texts (the per-direction dump)."""
    n = len(feats)
    best = np.full(n, -1e9)
    peak_h = torch.zeros(n, D_MODEL)
    best_txt = [""] * n
    for rows, texts in _gen_batches("sae", dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                                    bo, temp, max_new, min_new, gen_chunk):
        acts, peaks = score_sae_peaks(texts, [feats[i] for i in rows], sae, actor, tok, dev)
        for j, i in enumerate(rows):
            if acts[j].item() > best[i]:
                best[i] = acts[j].item()
                peak_h[i] = peaks[j]
                best_txt[i] = texts[j]
    ranks = sae_rank_at_peaks(sae, peak_h, feats)
    return (best, ranks, best_txt) if keep_texts else (best, ranks)


@torch.no_grad()
def eval_mlp_family(tag, dirs_unit, neuron, polarity, corpus_max, actor, tok, prompt_ids, marker, sub, dev,
                    bo, temp, max_new, min_new, gen_chunk, stats=None):
    """Extra (MLP neuron) family: best-of-bo max-token cosine (the standard metric) AND best-of-bo normalized fire-back
    (weakest member / strongest member). Returns (best_cos [N], best_na [N], best_na_any [N]) np arrays -- plus, with
    `stats` (load_mlp_neuron_stats), a 4th item: the RANK metrics of the best-of-bo sample (the one behind best_na) as a dict
    of np arrays [N, k] {rank, rank_best5, raw_rank, pct, av} (score_mlp_rank)."""
    n = len(dirs_unit)
    best_cos = np.full(n, -1e9); best_na = np.full(n, -1e9); best_any = np.full(n, -1e9)
    k = int(torch.as_tensor(neuron).reshape(n, -1).shape[1])
    rk = {key: np.zeros((n, k), np.int64 if "rank" in key else np.float64) for key in ("rank", "rank_best5", "raw_rank", "pct", "av")}
    for rows, texts in _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                                    bo, temp, max_new, min_new, gen_chunk):
        rdirs = F.normalize(torch.stack([dirs_unit[i] for i in rows]), dim=-1)
        cos = score_probe_cos(texts, rdirs, actor, tok, dev)
        na_min, na_max = score_mlp_fireback(texts, neuron[rows], polarity[rows], corpus_max[rows], actor, tok, dev)
        if stats is not None:   # rank of the sample that sets each direction's best_na (== its norm_act)
            r = score_mlp_rank(texts, neuron[rows], stats, actor, tok, dev)
            for j, i in enumerate(rows):
                if na_min[j].item() > best_na[i]:
                    for key in rk:
                        rk[key][i] = r[key][j].numpy()
        np.maximum.at(best_cos, rows, cos.numpy().astype(np.float64))
        np.maximum.at(best_na, rows, na_min.numpy().astype(np.float64))
        np.maximum.at(best_any, rows, na_max.numpy().astype(np.float64))
    return (best_cos, best_na, best_any, rk) if stats is not None else (best_cos, best_na, best_any)


# ---------------------------------------------------------------------------------------------
# frozen eval sets
# ---------------------------------------------------------------------------------------------

def _load_heldout_pool(pool_dir):
    """Load one SL/build_universal_bank.py pool: records.jsonl rows (list of dicts, file order ==
    vec_idx order) + vecs.f32 memmap [N, D_MODEL] fp32 unit rows (row i == vec_idx i)."""
    vp, rp = os.path.join(pool_dir, "vecs.f32"), os.path.join(pool_dir, "records.jsonl")
    if not (os.path.exists(vp) and os.path.exists(rp)):
        raise FileNotFoundError(f"held-out pool {pool_dir} is missing vecs.f32 / records.jsonl")
    sz = os.path.getsize(vp)
    assert sz % (4 * D_MODEL) == 0, f"{vp} size {sz} is not a multiple of 4*{D_MODEL}"
    vecs = np.memmap(vp, dtype=np.float32, mode="r", shape=(sz // (4 * D_MODEL), D_MODEL))
    recs = [json.loads(l) for l in open(rp)]
    assert len(recs) == vecs.shape[0], (f"{rp} has {len(recs)} records but vecs.f32 has "
                                        f"{vecs.shape[0]} rows — pool is corrupt/truncated")
    return recs, vecs


def _build_heldout_eval_sets(cache_path, sae, heldout_pool, n, seed, maxacts_path):
    """HELD-OUT mode: eval families come from a build_universal_bank.py pool_heldout (train-
    disjoint per family) instead of fresh-sampling the sources the training bank draws from.

    Selection is a SEEDED SAMPLE per family (np.random.default_rng(seed), families drawn in fixed
    HELDOUT_POOL_FAMILIES order): min(n, avail) record rows without replacement, then sorted by
    vec_idx — frozen across runs and memmap-friendly. (First-n-by-vec_idx would over-concentrate
    bsf/sae in a few blocks/features because the pool writer emits those families grouped.)
    Directions are the pool's own vecs.f32 rows (unit on disk; re-normalized defensively). The sae
    family also keeps the records' "feature" ids ALIGNED to the selected rows (one index array
    selects both) and computes corpus_peak for exactly those ids. The random control stays
    synthetic Gaussian (seeded) — in no pool, never trained on."""
    pool_key = os.path.abspath(heldout_pool)
    if os.path.exists(cache_path):
        es = torch.load(cache_path, map_location="cpu", weights_only=False)
        mt = es["meta"]
        if mt.get("heldout_pool") != pool_key or mt["n"] != n or mt["seed"] != seed:
            raise ValueError(
                f"eval-set cache {cache_path} was built with heldout_pool={mt.get('heldout_pool')} "
                f"n={mt.get('n')} seed={mt.get('seed')}, requested heldout_pool={pool_key} n={n} "
                f"seed={seed}; delete the cache to resample (WARNING: breaks curve comparability)")
        if mt["d_sae"] != sae.d_sae:
            raise ValueError(f"eval-set cache {cache_path} was built for d_sae={mt['d_sae']} but "
                             f"the loaded SAE has d_sae={sae.d_sae} — wrong SAE or stale cache")
        print(f"[eval-sets] loaded HELD-OUT cache {cache_path} (pool={pool_key}, n={n}, "
              f"seed={seed})", flush=True)
        return es

    recs, vecs = _load_heldout_pool(heldout_pool)
    by_fam = {}
    for i, r in enumerate(recs):
        by_fam.setdefault(r["family"], []).append(i)
    rng = np.random.default_rng(seed)   # families drawn in FIXED order — do not reorder
    es, rows_meta = {}, {}
    for fam in HELDOUT_POOL_FAMILIES:
        rows = by_fam.get(fam, [])
        if not rows:
            raise ValueError(f"held-out pool {heldout_pool} has no '{fam}' records "
                             f"(families present: {sorted(by_fam)})")
        take = min(n, len(rows))
        if take < n:
            print(f"[eval-sets] WARN held-out family '{fam}' has only {take} rows < n={n} — "
                  "using all of them", flush=True)
        sel = sorted(int(rows[j]) for j in rng.choice(len(rows), take, replace=False))
        vidx = np.array([recs[i]["vec_idx"] for i in sel], np.int64)
        es[f"{fam}_dirs"] = F.normalize(
            torch.from_numpy(np.asarray(vecs[vidx]).astype(np.float32)), dim=-1)
        rows_meta[fam] = vidx.tolist()
        if fam == "sae":
            feats = [int(recs[i]["feature"]) for i in sel]    # SAME sel as the dirs rows — aligned
            assert max(feats) < sae.d_sae, (f"held-out sae feature id {max(feats)} >= "
                                            f"d_sae {sae.d_sae} — pool built for a different SAE")
            es["sae_feats"] = feats
            ma = load_max_acts(path=maxacts_path)["max_acts"]
            es["corpus_peak"] = ma.reshape(ma.shape[0], -1).max(1).values.float()[
                torch.tensor(feats)].cpu()

    # random control: synthetic Gaussian unit dirs (seeded) — off-manifold, in no pool.
    g = torch.Generator().manual_seed(seed)
    es["random_dirs"] = F.normalize(torch.randn(n, D_MODEL, generator=g), dim=-1)

    es["meta"] = {"mode": "heldout_pool", "heldout_pool": pool_key, "n": n, "seed": seed,
                  "d_sae": sae.d_sae, "cos_families": list(COS_FAMILIES_HELDOUT),
                  "pool_rows": len(recs), "rows": rows_meta}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(es, cache_path)
    print("[eval-sets] built + cached HELD-OUT " + cache_path + ": "
          + " ".join(f"{f}={len(rows_meta[f])}" for f in HELDOUT_POOL_FAMILIES)
          + f" random={n} (pool {pool_key}, {len(recs)} rows, seed={seed})", flush=True)
    return es


def build_eval_sets(cache_path, sae, wu, j42, probe_bank_path, acts_dir, n, dev,
                    seed=0, maxacts_path=None, heldout_pool=None):
    """Sample all six frozen direction families ONCE (fixed seed) and cache to cache_path; later
    calls load the cache so every eval scores the IDENTICAL directions (low-variance curves).

    wu: base.lm_head.weight.detach().float() [vocab, d].  j42: lens.pt["J"][READ_LAYER] [d, d].
    maxacts_path: load_max_acts path for the SAE corpus peaks (None -> HF hub default).
    heldout_pool: path to a build_universal_bank.py pool_heldout dir — switches to HELD-OUT mode
    (families {bsf, realact, sae, jlens, cluster} from the pool + the synthetic random control;
    wu/j42/probe_bank_path/acts_dir/dev are IGNORED and may be None). Use a heldout-specific
    cache_path (e.g. eval_sets_heldout.pt) so the two modes never share a cache.
    Returns {probe_dirs, sae_feats, sae_dirs, corpus_peak, jlens_dirs, cluster_dirs, random_dirs,
    realact_dirs, meta} — all dirs [n, d] fp32 UNIT rows on cpu ({fam}_dirs keyed by
    meta["cos_families"] + sae in held-out mode)."""
    if heldout_pool is not None:
        return _build_heldout_eval_sets(cache_path, sae, heldout_pool, n, seed, maxacts_path)
    if os.path.exists(cache_path):
        es = torch.load(cache_path, map_location="cpu", weights_only=False)
        if es["meta"].get("heldout_pool") is not None:
            raise ValueError(f"eval-set cache {cache_path} is a HELD-OUT cache (pool="
                             f"{es['meta']['heldout_pool']}) but heldout_pool was not requested — "
                             "wrong cache path")
        if es["meta"]["n"] != n or es["meta"]["seed"] != seed:
            raise ValueError(f"eval-set cache {cache_path} was built with n={es['meta']['n']} "
                             f"seed={es['meta']['seed']}, requested n={n} seed={seed}; delete the "
                             "cache to resample (WARNING: breaks curve comparability)")
        if es["meta"]["d_sae"] != sae.d_sae:
            raise ValueError(f"eval-set cache {cache_path} was built for d_sae={es['meta']['d_sae']} "
                             f"but the loaded SAE has d_sae={sae.d_sae} — wrong SAE or stale cache")
        print(f"[eval-sets] loaded cache {cache_path} (n={n}, seed={seed})", flush=True)
        return es
    rng = np.random.default_rng(seed)   # families drawn in FIXED order — do not reorder

    # (1)+(4) probe + cluster: two DISJOINT samples from the RL probe bank (see module docstring
    # for why the cluster family falls back to the probe bank: centroids.npy is whitened-space).
    nrows = os.path.getsize(probe_bank_path) // (4 * D_MODEL)
    assert nrows >= 2 * n, f"probe bank has {nrows} rows < 2*n={2 * n} (probe+cluster disjoint split)"
    bank = np.memmap(probe_bank_path, dtype=np.float32, mode="r", shape=(nrows, D_MODEL))
    perm = rng.choice(nrows, 2 * n, replace=False)
    pidx, cidx = np.sort(perm[:n]), np.sort(perm[n:])
    probe_dirs = F.normalize(torch.from_numpy(np.asarray(bank[pidx]).astype(np.float32)), dim=-1)
    cluster_dirs = F.normalize(torch.from_numpy(np.asarray(bank[cidx]).astype(np.float32)), dim=-1)

    # (2) SAE features: unit encoder columns + corpus peak act per feature (eval_dirs recipe).
    feats = np.sort(rng.choice(sae.d_sae, min(n, sae.d_sae), replace=False)).tolist()
    sae_dirs = sae.enc_dirs(feats).float().cpu()
    ma = load_max_acts(path=maxacts_path)["max_acts"]
    corpus_peak = ma.reshape(ma.shape[0], -1).max(1).values.float()[torch.tensor(feats)].cpu()

    # (3) J-lens: unit(W_U[t] @ J_42) for random vocab tokens (serve_playground direction).
    vocab = wu.shape[0]
    tids = np.sort(rng.choice(vocab, min(n, vocab), replace=False))
    tv = torch.as_tensor(tids, dtype=torch.long, device=wu.device)
    jlens_dirs = F.normalize(wu[tv].float() @ j42.to(wu.device).float(), dim=-1).cpu()

    # (5) random: isotropic Gaussian unit dirs — off-manifold control (expect cos ~ 1/sqrt(d)).
    g = torch.Generator().manual_seed(seed)
    random_dirs = F.normalize(torch.randn(n, D_MODEL, generator=g), dim=-1)

    # (6) realact: real L42 token acts from the HELD-OUT tail (last 5% of sequences) of the
    # bsf27b acts dump — on-manifold targets never seen in training. Skip token 0 (attention-sink
    # norms) and norm-filter 10x-median, mirroring the scoring protocol; sample with margin.
    meta_a = json.load(open(os.path.join(acts_dir, "meta.json")))
    n_seq, seq_len, d = meta_a["n_seq"], meta_a["seq_len"], meta_a["d_model"]
    assert d == D_MODEL, f"acts d_model {d} != D_MODEL {D_MODEL}"
    acts = np.memmap(os.path.join(acts_dir, "acts.f16"), dtype=np.float16, mode="r",
                     shape=(n_seq, seq_len, d))
    lo = int(np.ceil((1.0 - HELDOUT_FRAC) * n_seq))
    sidx = rng.integers(lo, n_seq, size=4 * n)
    tpos = rng.integers(1, seq_len, size=4 * n)
    X = torch.from_numpy(np.stack([np.asarray(acts[si, ti]) for si, ti in zip(sidx, tpos)])
                         .astype(np.float32))
    nrm = X.norm(dim=-1)
    ok = (nrm > 0) & (nrm <= NORM_FILTER_MULT * nrm.median())
    assert int(ok.sum()) >= n, f"only {int(ok.sum())} realact rows survive the norm filter (< n={n})"
    realact_dirs = F.normalize(X[ok][:n], dim=-1)

    es = {"probe_dirs": probe_dirs, "cluster_dirs": cluster_dirs,
          "sae_feats": feats, "sae_dirs": sae_dirs, "corpus_peak": corpus_peak,
          "jlens_dirs": jlens_dirs, "random_dirs": random_dirs, "realact_dirs": realact_dirs,
          "meta": {"mode": "fresh_sample", "heldout_pool": None,
                   "cos_families": list(COS_FAMILIES),
                   "n": n, "seed": seed, "probe_bank": probe_bank_path, "acts_dir": acts_dir,
                   "probe_rows": pidx.tolist(), "cluster_rows": cidx.tolist(),
                   "jlens_tids": tids.tolist(), "d_sae": sae.d_sae, "vocab": int(vocab),
                   "realact_lo": lo, "heldout_frac": HELDOUT_FRAC}}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(es, cache_path)
    print(f"[eval-sets] built + cached {cache_path}: 6 families x n={n} (seed={seed}); "
          f"probe bank {nrows} rows, d_sae {sae.d_sae}, vocab {vocab}, realact seqs [{lo},{n_seq})",
          flush=True)
    return es


# ---------------------------------------------------------------------------------------------
# eval entry point (imported by the trainer — no wandb in here, caller logs the returned dict)
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def run_eval(actor, tok, prompt_ids, marker, sub, eval_sets, sae, bo, temp, max_new, min_new, dev,
             gen_chunk=64, sae_fire=None, per_dir=False, mlp_stats=None, chance_acts_dir=None):
    """Run every family in eval_sets (meta["cos_families"] cosine families + the sae metric
    family); return a FLAT {wandb scalar name: float} dict. Generation RNG is forked + fixed
    (GEN_SEED) so repeat evals of the same checkpoint are deterministic and the trainer's RNG
    stream is untouched.
    per_dir=True ADDS out["_perdir"] (NOT a scalar -- pop it before wandb.log): the per-direction
    best-of-bo scores behind every aggregate in eval-cache row order -- {"cos": {fam: [n]},
    "sae": {row, feature, best_act, corpus_peak, norm_act, fired, beat_corpus, rank, rank1,
    best_text}, "extra": {fam: {cos, norm_act, any_norm_act}}}. Default output is unchanged.
    mlp_stats (load_mlp_neuron_stats) ADDS the cross-neuron RANK metrics of the extra (mlp) families -- eval/<fam>/{rank1_frac,
    rank_le10, mean_rank, median_rank, mrr, best5_*, raw_*, corpus_pct_mean, top1pct_frac} (mlp_rank_metrics) -- and, with
    chance_acts_dir, their chance level on random held-out corpus windows (eval/<fam>/chance_*). Every pre-existing key is
    unchanged."""
    if sae_fire is None:
        sae_fire = SAE_FIRE          # the SAE gate set by configure_sae_fire(), or the module default

    was_training = actor.training
    actor.eval()
    fork_devs = [dev] if str(dev).startswith("cuda") else []
    # legacy fresh-sample caches predate meta["cos_families"] -> default to the legacy list
    fams = list(eval_sets["meta"].get("cos_families", COS_FAMILIES))
    xfams = extra_families(eval_sets)
    out = {}
    pd = {"row_order": "list position == eval-cache row index", "cos": {}, "sae": {}, "extra": {}}
    try:
        with torch.random.fork_rng(devices=fork_devs):
            torch.manual_seed(GEN_SEED)
            gen_args = (actor, tok, prompt_ids, marker, sub, dev, bo, temp, max_new, min_new, gen_chunk)
            for fam in fams:
                best = eval_cos_family(fam, eval_sets[f"{fam}_dirs"], *gen_args)
                out[f"eval/{fam}/cos"] = float(best.mean())
                pd["cos"][fam] = best.tolist()
            sae_res = eval_sae_family(eval_sets["sae_dirs"], eval_sets["sae_feats"], sae, *gen_args, keep_texts=per_dir)
            best_act, ranks = sae_res[0], sae_res[1]
            cp = eval_sets["corpus_peak"].numpy().astype(np.float64)
            r = ranks.astype(np.float64)
            out["eval/sae/norm_act"] = float(np.mean(best_act / np.maximum(cp, 1e-6)))
            out["eval/sae/fired"] = float(np.mean(best_act > sae_fire))
            out["eval/sae/beat_corpus"] = float(np.mean(best_act > cp))
            out["eval/sae/mean_rank"] = float(r.mean())
            out["eval/sae/rank1_frac"] = float(np.mean(ranks == 1))
            out["eval/sae/mrr"] = float(np.mean(1.0 / r))
            na = best_act / np.maximum(cp, 1e-6)
            out["eval/sae/unverbalized_frac"] = float(np.mean(best_act <= sae_fire))  # cannot be made to fire at all
            out["eval/sae/unverbalized_p10"] = float(np.mean(na < 0.10))  # inversion reached <10pct of corpus peak
            out["eval/sae/gate"] = float(sae_fire)                                     # the threshold "fired" refers to
            out["eval/sae/fired_1p0"] = float(np.mean(best_act > SAE_FIRE_LEGACY))     # legacy 1.0 cut (pre-Sep-15 series)
            out["eval/sae/unverbalized_1p0"] = float(np.mean(best_act <= SAE_FIRE_LEGACY))
            if per_dir:
                pd["sae"] = {"row": list(range(len(best_act))), "feature": [int(f) for f in eval_sets["sae_feats"]],
                             "best_act": best_act.tolist(), "corpus_peak": cp.tolist(), "norm_act": na.tolist(),
                             "fired": (best_act > sae_fire).astype(int).tolist(), "beat_corpus": (best_act > cp).astype(int).tolist(),
                             "rank": [int(x) for x in ranks], "rank1": [int(x == 1) for x in ranks], "best_text": sae_res[2]}
                pool_rows = (eval_sets["meta"].get("rows") or {}).get("sae")
                if pool_rows is not None:
                    pd["sae"]["pool_vec_idx"] = [int(x) for x in pool_rows]
            # extra families (cache v2: layer-42 MLP neurons / co-firing pairs): cosine + fire-back, NOT in mean_all
            for fi, fam in enumerate(xfams):
                k = eval_sets[f"{fam}_neuron"].shape[1]
                res = eval_mlp_family(fam, eval_sets[f"{fam}_dirs"], eval_sets[f"{fam}_neuron"],
                                      eval_sets[f"{fam}_polarity"], eval_sets[f"{fam}_corpus_max"], *gen_args, stats=mlp_stats)
                bc, bn, ba = res[0], res[1], res[2]
                out[f"eval/{fam}/cos"] = float(bc.mean())
                out.update(mlp_metrics(fam, bn, ba if k > 1 else None))
                pd["extra"][fam] = {"cos": bc.tolist(), "norm_act": bn.tolist(), "any_norm_act": ba.tolist()}
                if mlp_stats is not None:
                    rk = res[3]
                    out.update(mlp_rank_metrics(fam, rk["rank"], rk["rank_best5"], rk["raw_rank"], rk["pct"]))
                    pd["extra"][fam].update({"neuron": eval_sets[f"{fam}_neuron"].tolist(), "rank": rk["rank"].tolist(),
                                             "rank_best5": rk["rank_best5"].tolist(), "raw_rank": rk["raw_rank"].tolist(),
                                             "corpus_pct": rk["pct"].tolist(), "best_signed_act": rk["av"].tolist()})
                    if chance_acts_dir:
                        ch = mlp_chance_rows(fam, fi, eval_sets, mlp_stats, actor, tok, dev, chance_acts_dir,
                                             range(len(bn)), bo, min_new, max_new)
                        cm, cpd = mlp_chance_metrics(fam, ch, k)
                        out.update(cm)
                        pd["extra"][fam]["chance"] = cpd
            # SAE-slice families (cache v3: held-out features of the 2M SAE, encoder + decoder directions): NOT in mean_all
            sfams = slice_families(eval_sets)
            if sfams:
                sl = load_sae_slice(eval_sets, dev)
                pd["slice"] = {}
                for fam in sfams:
                    feats_f = [int(f) for f in eval_sets[f"{fam}_feats"]]
                    res = eval_sae_slice_family(fam, eval_sets[f"{fam}_dirs"], feats_f, sl, *gen_args, keep_texts=per_dir)
                    cp_f = eval_sets[f"{fam}_corpus_peak"].numpy().astype(np.float64)
                    out[f"eval/{fam}/cos"] = float(res[1].mean())
                    out.update(slice_metrics(fam, res[0], cp_f, sl.threshold))
                    pd["slice"][fam] = {"feature": feats_f, "best_act": res[0].tolist(), "corpus_peak": cp_f.tolist(), "cos": res[1].tolist(),
                                        "fired": (res[0] > sl.threshold).astype(int).tolist(), **({"best_text": res[2]} if per_dir else {})}
    finally:
        if was_training:
            actor.train()
    if per_dir:
        out["_perdir"] = pd
    # mean_all = mean over the HIGHER-IS-BETTER cos families only. `random` is the control
    # (should stay ~0.03, LOWER is better) — folding it in would drag the mean down and move
    # mean_all the WRONG way if the control ever degraded. It stays logged separately. Extra
    # families (xfams) are NOT included either, so mean_all is comparable across cache versions.
    out["eval/mean_all"] = float(np.mean([out[f"eval/{f}/cos"] for f in fams
                                          if f not in CONTROL_FAMS]))
    # headline mirror group — one wandb panel with every family side by side
    # legacy mode: probe/jlens/cluster/random/realact _cos; held-out mode: bsf replaces probe
    for fam in fams + xfams + slice_families(eval_sets):
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
    out["eval/all/sae_norm_act"] = out["eval/sae/norm_act"]
    out["eval/all/sae_unverbalized"] = out["eval/sae/unverbalized_frac"]
    out["eval/all/sae_mean_rank"] = out["eval/sae/mean_rank"]
    for fam in xfams:
        out[f"eval/all/{fam}_norm_act"] = out[f"eval/{fam}/norm_act"]
        out[f"eval/all/{fam}_fired10"] = out[f"eval/{fam}/fired10"]
        if f"eval/{fam}/rank_le{MLP_RANK_LE}" in out:
            out[f"eval/all/{fam}_rank_le{MLP_RANK_LE}"] = out[f"eval/{fam}/rank_le{MLP_RANK_LE}"]
            out[f"eval/all/{fam}_mrr"] = out[f"eval/{fam}/mrr"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--sae-path", required=True)
    ap.add_argument("--maxacts-path", required=True)
    ap.add_argument("--lens-path", default="/root/pmx/lenses/qwen3.6-27b/j-lens/lens.pt")
    ap.add_argument("--acts-dir", default="/root/pmx/bsf27b/acts")
    ap.add_argument("--probe-bank", default="/root/pmx/data/pool_rl_1M/probes.f32")
    ap.add_argument("--heldout-pool", default=None,
                    help="path to a build_universal_bank.py pool_heldout dir (records.jsonl + "
                         "vecs.f32). When set, eval families come from this train-DISJOINT pool "
                         "{bsf,realact,sae,jlens,cluster} + the synthetic random control, instead "
                         "of fresh-sampling sources that overlap pool_train")
    ap.add_argument("--cache", default="/root/pmx/data/eval_universal",
                    help="cache DIR; frozen eval sets live at <cache>/eval_sets.pt "
                         "(<cache>/eval_sets_heldout.pt with --heldout-pool)")
    ap.add_argument("--cache-file", default=os.environ.get(ENV_EVAL_CACHE),
                    help=f"FULL path of an existing eval-set cache (overrides --cache's file name; env {ENV_EVAL_CACHE}), "
                         "e.g. .../eval_sets_heldout_v2.pt with the extra mlp / mlp_pair families")
    ap.add_argument("--n", type=int, default=1024, help="directions per family")
    ap.add_argument("--bo", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--min-new-tokens", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--sae-fire", type=float, default=None, help="raw-activation cut for fired/unverbalized; default = the SAE checkpoint's learned gate")
    ap.add_argument("--out", default=None, help="optional JSON dump of the metric dict")
    ap.add_argument("--dump-per-dir", action="store_true",
                    help="ALSO write <out>.perdir.json (needs --out): per-direction best-of-bo scores behind every aggregate "
                         "(cos per direction per family; sae feature id / best act / norm_act / fired / rank / best text); "
                         "the --out json and the wandb row are unchanged")
    ap.add_argument("--mlp-stats", default=MLP_STATS_DEFAULT,
                    help="per-neuron corpus stats of ALL layer-42 neurons (data/mlp42_neurons_worker.py neuron_stats.npz) -> the "
                         "extra mlp families' cross-neuron RANK metrics + corpus percentile; missing file = rank metrics skipped")
    ap.add_argument("--mlp-chance-acts", default=MLP_CHANCE_ACTS_DEFAULT,
                    help="acts27b dump (toks.i32 + meta.json) for the random-corpus-window CHANCE level of the MLP metrics ('' = off)")
    ap.add_argument("--wandb", default=None, help="optional wandb project; init + log one step")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    dev = "cuda:0"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok); marker = mpos[0]
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    actor = PeftModel.from_pretrained(base, a.adapter, is_trainable=False); actor.eval()
    sub = get_layer(actor, INJECT_LAYER)
    wu = base.lm_head.weight.detach().float()                                # [vocab, d] on dev
    j42 = None if a.heldout_pool else \
        torch.load(a.lens_path, map_location="cpu", weights_only=False)["J"][READ_LAYER].float().to(dev)
    sae = load_sae(path=a.sae_path, device=dev, dtype=torch.float32)
    if a.sae_fire is None:
        a.sae_fire = configure_sae_fire(a.sae_path)
        print(f"[eval] SAE fire threshold = learned gate {a.sae_fire:.4f}", flush=True)
    print(f"[eval-universal] adapter={a.adapter} | n={a.n} bo={a.bo} | WU {tuple(wu.shape)} | "
          + (f"heldout pool {a.heldout_pool}" if a.heldout_pool
             else f"J{READ_LAYER} {tuple(j42.shape)}"), flush=True)

    cache_path = a.cache_file or os.path.join(a.cache,
                                             "eval_sets_heldout.pt" if a.heldout_pool else "eval_sets.pt")
    if a.cache_file:
        assert os.path.exists(a.cache_file), f"--cache-file {a.cache_file} does not exist"
    es = build_eval_sets(cache_path, sae, wu, j42, a.probe_bank, a.acts_dir, a.n, dev,
                         seed=0, maxacts_path=a.maxacts_path, heldout_pool=a.heldout_pool)
    stats = None
    if extra_families(es):
        if a.mlp_stats and os.path.exists(a.mlp_stats):
            stats = load_mlp_neuron_stats(a.mlp_stats)
        print(f"[eval-universal] extra families {extra_families(es)} (cosine + MLP fire-back; not in mean_all) | rank metrics "
              f"{'ON (' + a.mlp_stats + ')' if stats else 'OFF (no ' + str(a.mlp_stats) + ')'}", flush=True)
    m = run_eval(actor, tok, prompt_ids, marker, sub, es, sae, a.bo, a.temp,
                 a.max_new_tokens, a.min_new_tokens, dev, gen_chunk=a.gen_chunk, sae_fire=a.sae_fire,
                 per_dir=a.dump_per_dir, mlp_stats=stats,
                 chance_acts_dir=a.mlp_chance_acts if (stats and a.mlp_chance_acts and os.path.exists(a.mlp_chance_acts)) else None)
    perdir = m.pop("_perdir", None)
    print("=== EVAL-UNIVERSAL ===\n" + json.dumps(m, indent=1), flush=True)
    if a.out:
        json.dump({"adapter": a.adapter, "n": a.n, "bo": a.bo, "metrics": m}, open(a.out, "w"), indent=1)
        print(f"EVAL_UNIVERSAL_DONE {a.out}", flush=True)
        if perdir is not None:
            pp = a.out + ".perdir.json"
            json.dump({"adapter": a.adapter, "n": a.n, "bo": a.bo, "cache": cache_path, "aggregates": m, "perdir": perdir}, open(pp, "w"))
            print(f"EVAL_UNIVERSAL_PERDIR {pp}", flush=True)
    if a.wandb:
        import wandb
        wandb.init(project=a.wandb, name=a.run_name or os.path.basename(a.adapter.rstrip("/")),
                   config={"adapter": a.adapter, "n": a.n, "bo": a.bo, "temp": a.temp,
                           "max_new_tokens": a.max_new_tokens, "min_new_tokens": a.min_new_tokens,
                           "cache": cache_path, "heldout_pool": a.heldout_pool})
        wandb.log(m)
        wandb.finish()


if __name__ == "__main__":
    main()
