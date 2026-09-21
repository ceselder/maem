"""BSF playground (Modal + Gradio, one B200): explore the Qwen3.6-27B layer-42 block-sparse subspace featurizer (SASA, 32,768 blocks x 8
dims) through the activation->text inverter. Pick a block, see where it fires in the corpus and what its 8 principal axes are, then
feed a PC straight into the inverter or steer the block's mean code along a PC and read what the model writes; every rollout is
re-encoded on the clean base and scored (cosine to the injected direction, whether the block fires, where along the PC it landed).

    cd ~/maemm-pub-simple2m && MODAL_PROFILE=safety-sahan modal deploy bsf_playground/modal_bsf_playground.py
    MODAL_PROFILE=safety-sahan modal run bsf_playground/modal_bsf_playground.py::stats --n-seq 3000     # once: block statistics (~20 min)
    open https://safety-sahan--maemm-bsf-playground-playground-web.modal.run   (Gradio auth: PLAYGROUND_USER / PLAYGROUND_PASS)
"""
import json
import os
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App(os.environ.get("BSF_PG_APP", "maemm-bsf-playground"))
GPU = os.environ.get("BSF_PG_GPU", "B200:1")
BSF_DIR = "/data/bsf27b_1b"
ACTS_DIR = "/data/acts27b_fresh"                                          # 80k x 512 raw layer-42 acts (FineFineWeb) + toks
CKPT = os.environ.get("BSF_PG_CKPT", "/data/ckpts_rl_distill_fresh2m_bo16_8x2048_anywin/step_150")   # best inverter (raw mean_all .426)
STATS = f"{BSF_DIR}/pg_stats.pt"
LABELS = f"{BSF_DIR}/pg_labels.json"
TOPC = 24                                                                  # top contexts kept per block
D = 5120

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "wandb==0.28.2", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")            # wandb: rl_hf imports it at module level
    .pip_install("flash-linear-attention==0.5.2")
    .pip_install("gradio>=5.0,<6", "fastapi[standard]", "anthropic")
    .env({"PYTHONPATH": "/pmx/helpers:/pmx/eval:/pmx/RL:/pmx/bsf_playground", "HF_HOME": "/data/hf_cache", "HF_HUB_OFFLINE": "1",
          "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "eval" / "eval_universal.py", "/pmx/eval/eval_universal.py")
    .add_local_file(REPO / "bsf_playground" / "bsf_lib.py", "/pmx/bsf_playground/bsf_lib.py")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)


# ========================================================================================================= block statistics
@app.function(image=image, gpu=GPU, cpu=8, memory=96 * 1024, volumes={"/data": vol}, timeout=4 * 3600)
def stats(n_seq: int = 3000, seed: int = 0, chunk_seq: int = 32, norm_mult: float = 10.0, overwrite: bool = False):
    """Per-block statistics over a random sample of stored sequences: active count, mean/cov of the 8 subspace coordinates on active
    tokens (-> PCs), mean activation, top-`TOPC` firing contexts (seq, pos, gn), token-frequency labels. Writes pg_stats.pt."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from bsf_lib import BSF, pca_block, eff_rank
    from mxf.config import MODEL
    vol.reload()
    if os.path.exists(STATS) and not overwrite:
        return {"exists": STATS}
    t0 = time.time(); log = lambda m: print(f"[stats +{time.time() - t0:5.0f}s] {m}", flush=True)
    dev = "cuda"
    bsf = BSF(BSF_DIR, dev)
    G, b, k = bsf.G, bsf.b, bsf.k
    meta = json.load(open(f"{ACTS_DIR}/meta.json"))
    N, L = int(meta["n_seq"]), int(meta["seq_len"])
    acts = np.memmap(f"{ACTS_DIR}/acts.f16", np.float16, "r", shape=(N, L, D))
    toks = np.memmap(f"{ACTS_DIR}/toks.i32", np.int32, "r", shape=(N, L))
    rng = np.random.default_rng(seed)
    seq_ids = np.sort(rng.choice(N, size=min(n_seq, N), replace=False))
    log(f"BSF G={G} b={b} k={k}; store {N}x{L}; sampling {len(seq_ids)} seqs = {len(seq_ids) * L / 1e6:.2f}M tokens")
    count = torch.zeros(G, dtype=torch.float64, device=dev)
    sum_c = torch.zeros(G, b, dtype=torch.float64, device=dev)
    sum_cc = torch.zeros(G, b * b, dtype=torch.float64, device=dev)
    sum_gn = torch.zeros(G, dtype=torch.float64, device=dev)
    top_val = torch.full((G, TOPC), -1.0, device=dev); top_seq = torch.full((G, TOPC), -1, dtype=torch.long, device=dev)
    top_pos = torch.full((G, TOPC), -1, dtype=torch.long, device=dev)
    Qflat = bsf.Q.view(G * b, D)
    n_tok = n_kept = 0
    for s0 in range(0, len(seq_ids), chunk_seq):
        sid = seq_ids[s0: s0 + chunk_seq]
        x = torch.from_numpy(np.ascontiguousarray(acts[sid])).to(dev).float().view(-1, D)          # [S*L, d]
        xc = x - bsf.mu
        nrm = xc.norm(dim=-1).view(len(sid), L)
        keep = (nrm <= norm_mult * nrm.median(dim=1, keepdim=True).values).view(-1)                     # the suite's norm filter
        y = xc @ bsf.zca
        seq_of = torch.as_tensor(np.repeat(sid, L), device=dev); pos_of = torch.arange(L, device=dev).repeat(len(sid))
        for c0 in range(0, y.shape[0], 1024):
            yy, kk = y[c0:c0 + 1024], keep[c0:c0 + 1024]
            if not kk.any():
                continue
            yy = yy[kk]; sq = seq_of[c0:c0 + 1024][kk]; ps = pos_of[c0:c0 + 1024][kk]
            gn = bsf.encode(yy)                                                                         # [n, G]
            tv, ti = gn.topk(k, dim=-1)                                                                  # active blocks
            coords_all = (yy @ Qflat.T).view(-1, G, b)                                                   # [n, G, b]
            cc = torch.gather(coords_all, 1, ti[:, :, None].expand(-1, -1, b))                           # [n, k, b]
            ids = ti.reshape(-1); cf = cc.reshape(-1, b); vf = tv.reshape(-1)
            count.index_add_(0, ids, torch.ones_like(vf, dtype=torch.float64))
            sum_c.index_add_(0, ids, cf.double()); sum_cc.index_add_(0, ids, (cf[:, :, None] * cf[:, None, :]).reshape(-1, b * b).double())
            sum_gn.index_add_(0, ids, vf.double())
            # top contexts: merge the existing table with this chunk's active pairs, keep the TOPC best per block
            all_ids = torch.cat([torch.arange(G, device=dev).repeat_interleave(TOPC), ids])
            all_val = torch.cat([top_val.reshape(-1), vf]); all_seq = torch.cat([top_seq.reshape(-1), sq.repeat_interleave(k)])
            all_pos = torch.cat([top_pos.reshape(-1), ps.repeat_interleave(k)])
            key = all_ids.double() * 1e4 + (1e3 - all_val.double().clamp(max=999.0))                     # block asc, val desc
            order = torch.argsort(key); all_ids, all_val, all_seq, all_pos = all_ids[order], all_val[order], all_seq[order], all_pos[order]
            starts = torch.searchsorted(all_ids, torch.arange(G, device=dev))
            rank = torch.arange(all_ids.numel(), device=dev) - starts[all_ids]
            sel = rank < TOPC
            top_val = all_val[sel].view(G, TOPC); top_seq = all_seq[sel].view(G, TOPC); top_pos = all_pos[sel].view(G, TOPC)
            n_kept += int(kk.sum())
        n_tok += y.shape[0]
        if (s0 // chunk_seq) % 10 == 0:
            log(f"{s0 + len(sid)}/{len(seq_ids)} seqs, {n_kept:,} tokens kept of {n_tok:,} | blocks seen {(count > 0).sum().item()}")
    cnt = count.clamp(min=1)
    mean_c = (sum_c / cnt[:, None]).float()
    cov = (sum_cc / cnt[:, None]).view(G, b, b) - (sum_c / cnt[:, None])[:, :, None] * (sum_c / cnt[:, None])[:, None, :]
    mean_c_np, cov_np = mean_c.cpu().numpy(), cov.float().cpu().numpy()
    V = np.zeros((G, b, b), np.float32); lam = np.zeros((G, b), np.float32); er = np.zeros(G, np.float32)
    for g in range(G):
        if count[g] >= 8:
            V[g], lam[g] = pca_block(mean_c_np[g], cov_np[g]); er[g] = eff_rank(lam[g])
        else:
            V[g] = np.eye(b, dtype=np.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    toks_sub = np.ascontiguousarray(toks[seq_ids])                                                   # [n_seq, L]
    seq_row = {int(s): i for i, s in enumerate(seq_ids)}
    labels = []
    ts, tp = top_seq.cpu().numpy(), top_pos.cpu().numpy()
    for g in range(G):
        toks_g = [int(toks_sub[seq_row[int(s)], int(p)]) for s, p in zip(ts[g], tp[g]) if s >= 0]
        if not toks_g:
            labels.append(""); continue
        u, c = np.unique(np.array(toks_g), return_counts=True); o = np.argsort(-c)[:4]
        labels.append(" ".join(f"{tok.decode([int(u[i])])!r}x{int(c[i])}" for i in o))
    out = {"G": G, "b": b, "k": k, "n_seq": int(len(seq_ids)), "seq_ids": torch.as_tensor(seq_ids), "n_tokens": n_tok, "n_kept": n_kept, "norm_mult": norm_mult,
           "count": count.float().cpu(), "frac_active": (count / max(n_kept, 1)).float().cpu(), "mean_gn": (sum_gn / cnt).float().cpu(),
           "mean_c": mean_c.cpu(), "cov": torch.from_numpy(cov_np), "V": torch.from_numpy(V), "lam": torch.from_numpy(lam), "eff_rank": torch.from_numpy(er),
           "top_val": top_val.cpu(), "top_seq": top_seq.cpu(), "top_pos": top_pos.cpu(), "toks": torch.from_numpy(toks_sub), "labels_tok": labels,
           "acts_dir": ACTS_DIR, "bsf_dir": BSF_DIR, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": round(time.time() - t0)}
    torch.save(out, STATS); vol.commit()
    log(f"DONE: {n_kept:,} tokens; blocks active >=8 times: {(count >= 8).sum().item()}; median eff-rank {np.median(er[count.cpu().numpy() >= 8]):.2f}")
    return {"n_tokens": n_kept, "blocks_active": int((count >= 8).sum()), "wall_s": round(time.time() - t0)}


# ================================================================================================================ playground
@app.cls(image=image, gpu=GPU, cpu=8, memory=64 * 1024, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-anthropic")],
         timeout=3600, scaledown_window=3600, min_containers=int(os.environ.get("BSF_PG_MIN", "1")), max_containers=1)
@modal.concurrent(max_inputs=8)
class Playground:
    @modal.enter()
    def load(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        import rl_hf as R
        import rl_disagg as DG
        from bsf_lib import BSF
        from mxf.config import INJECT_LAYER, READ_LAYER, MODEL
        from mxf.prompts import build_prompt_ids
        self.torch, self.np, self.R, self.DG = torch, np, R, DG
        self.READ_LAYER, self.INJECT_LAYER, self.MODEL = READ_LAYER, INJECT_LAYER, MODEL
        vol.reload()
        t0 = time.time(); self.log = lambda m: print(f"[pg +{time.time() - t0:5.0f}s] {m}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.prompt_ids, mpos = build_prompt_ids(self.tok); self.marker = mpos[0]
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
        DG._truncate_scorer(base, READ_LAYER + 1)
        self.scorer = DG.BaseActor(base); self.scorer.eval()
        self.eos_ids = R._eos_ids(self.tok, DG._GenCfgStub(GenerationConfig.from_pretrained(MODEL)))
        self.log(f"clean-base scorer ready ({torch.cuda.memory_allocated() / 2**30:.1f} GB)")
        import vllm as _vllm
        _LLM = _vllm.LLM

        class _QuantLLM(_LLM):
            def __init__(self, *a, **kw):
                kw.setdefault("quantization", "fp8"); super().__init__(*a, **kw)
        _vllm.LLM = _QuantLLM
        self.ea = SimpleNamespace(stock_lens_hook=False, vllm_gpu_mem=0.42, max_new_tokens=128, min_new_tokens=4, seed=0, gdn_prefill_backend="triton",
                                  engine_model=CKPT, engine_lora=False, policy_base=None, max_num_batched_tokens=16384, group_size=8, temperature=1.0)
        self.llm = DG._build_engine(self.ea, 0, len(self.prompt_ids), 256, True, "pg")
        self.hnorm = self._marker_norm()
        chk = DG._verify_injection(self.llm, self.prompt_ids, self.marker, self.hnorm, "pg", seed=0)
        self.log(f"engine up; injection check cos {chk['cos']:.4f} ratio {chk['norm_ratio']:.3f} -> {'OK' if chk['ok'] else 'FAIL'}")
        self.bsf = BSF(BSF_DIR, "cuda", e_dtype=torch.bfloat16)
        self.stats = torch.load(STATS, map_location="cpu", weights_only=False) if os.path.exists(STATS) else None
        self.labels = json.load(open(LABELS)) if os.path.exists(LABELS) else {}
        self.log(f"BSF G={self.bsf.G}; stats {'loaded (' + str(self.stats['n_kept']) + ' tokens)' if self.stats else 'NOT READY'} | {torch.cuda.memory_allocated() / 2**30:.1f} GB resident")

    # ------------------------------------------------------------------------------------------------------------ core ops
    def _marker_norm(self):
        from vllm import SamplingParams
        out = self.llm.generate([{"prompt_token_ids": list(self.prompt_ids)}], [SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [self.INJECT_LAYER]})], use_tqdm=False)[0]
        return out.activations["residual_stream"][0].float()[self.marker].norm().item()

    def generate(self, dirs, n, temperature, max_new, coeff=1.0, seed=0):
        """dirs [m, d] unit (torch, any device) -> list (per dir) of n texts. Steering = the RL protocol (h + coeff*||h||*unit(v) at the marker)."""
        import pickle
        from vllm import SamplingParams
        torch = self.torch
        dirs = torch.as_tensor(dirs, dtype=torch.float32).cpu()
        keys = [f"pg{int(time.time() * 1e3) % 10**9}_{i}" for i in range(len(dirs))]
        payload = {k: [self.R._steer_vec(v, self.hnorm * float(coeff), self.marker)] for k, v in zip(keys, dirs)}
        self.llm.collective_rpc("set_steering_data_many", args=(pickle.dumps(payload),))
        params = [SamplingParams(n=int(n), temperature=float(temperature) if temperature > 0 else 0.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=int(max_new), min_tokens=4,
                                 stop_token_ids=sorted(self.eos_ids), seed=int(seed) + i if seed else None, extra_args={"_steering_id": k}) for i, k in enumerate(keys)]
        try:
            outs = self.llm.generate([{"prompt_token_ids": list(self.prompt_ids)} for _ in keys], params, use_tqdm=False)
        finally:
            self.llm.collective_rpc("clear_steering_data_many", args=(keys,))
        res = []
        for out in outs:
            texts = []
            for o in out.outputs:
                g = list(o.token_ids)
                texts.append(self.tok.decode(self.R._trim_at_stop(g, self.eos_ids), skip_special_tokens=True))
            res.append(texts)
        return res

    def read(self, texts):
        """clean-base layer-42 residuals: [(tokens, h [T,d] fp32, keep [T] bool)], sink prepended and dropped, max 95 tokens, 10x-median norm filter."""
        from mxf.inject import read_resid
        torch = self.torch
        out = []
        tok = self.tok; sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        tok.padding_side = "right"
        with torch.inference_mode():
            for s in range(0, len(texts), 16):
                tt = [t if t.strip() else " " for t in texts[s:s + 16]]
                e = tok(tt, return_tensors="pt", padding=True, truncation=True, max_length=95, add_special_tokens=False)
                ids = torch.cat([torch.full((len(tt), 1), sink), e["input_ids"]], 1).cuda(); am = torch.cat([torch.ones(len(tt), 1, dtype=e["attention_mask"].dtype), e["attention_mask"]], 1).cuda()
                h, mask = read_resid(self.scorer, self.READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
                for i in range(len(tt)):
                    n = int(mask[i].sum()); hh = h[i, 1:n].float()
                    nrm = hh.norm(dim=-1); keep = nrm <= 10.0 * nrm.median()
                    out.append((tok.convert_ids_to_tokens(ids[i, 1:n].tolist()), hh, keep))
        return out

    def analyze(self, texts, direction, block, pc=None):
        """Per rollout: raw & centered cosine (max over kept tokens) to `direction`, block activation gn (max) + whether it is in the top-k
        somewhere, and the PC score (sigma units) of the block's coordinates at the block's peak token."""
        torch = self.torch; np = self.np
        st = self.stats; b = self.bsf
        rows = []
        d = torch.as_tensor(direction, dtype=torch.float32, device="cuda")
        for toks, h, keep in self.read(texts):
            if not keep.any():
                rows.append({"cos": 0.0, "cos_c": 0.0, "gn": 0.0, "fires": False, "rank": None, "pc": None, "peak_tok": "", "toks": toks, "tok_gn": []}); continue
            hc = h - b.mu
            cos = torch.nn.functional.normalize(h, dim=-1) @ d; cosc = torch.nn.functional.normalize(hc, dim=-1) @ d
            cos[~keep] = -2; cosc[~keep] = -2
            y = hc @ b.zca; gn = b.encode(y)                                        # [T, G]
            gb = gn[:, block].clone(); gb[~keep] = -1
            pk = int(gb.argmax()); rank = int((gn[pk] > gn[pk, block]).sum()) + 1
            row = {"cos": round(float(cos.max()), 3), "cos_c": round(float(cosc.max()), 3), "gn": round(float(gb[pk]), 3), "fires": rank <= b.k, "rank": rank,
                   "peak_tok": toks[pk], "peak_i": pk, "toks": toks, "tok_gn": [round(float(v), 2) for v in gb.tolist()], "pc": None}
            if st is not None and pc is not None:
                c = b.Q[block] @ y[pk]                                                # [8] coordinates at the peak token
                m = st["mean_c"][block].cuda(); V = st["V"][block].cuda(); lam = st["lam"][block].cuda()
                sc = ((c - m) @ V) / lam.clamp(min=1e-6).sqrt()                      # sigma units along each PC
                row["pc"] = round(float(sc[pc]), 2); row["pc_all"] = [round(float(v), 2) for v in sc.tolist()]
            rows.append(row)
        return rows

    # ------------------------------------------------------------------------------------------------------------ block info
    def block_card(self, g):
        st = self.stats; np = self.np
        if st is None:
            return "stats job not finished yet — run `modal run ...::stats`, then click 'reload stats'.", ""
        g = int(g) % self.bsf.G
        cnt = int(st["count"][g]); fa = float(st["frac_active"][g]); er = float(st["eff_rank"][g]); lam = st["lam"][g].numpy(); tot = max(float(lam.sum()), 1e-9)
        pcs = " · ".join(f"PC{i + 1} {100 * lam[i] / tot:.0f}%" for i in range(self.bsf.b))
        lab = self.labels.get(str(g), {}).get("label", "")
        md = (f"### block {g}  \n**active** {cnt:,} of {st['n_kept']:,} tokens ({100 * fa:.3f}%, i.e. ~{fa * self.bsf.k * 100 / self.bsf.k:.3f}% of tokens carry it in their top-{self.bsf.k}) · "
              f"**mean activation** {float(st['mean_gn'][g]):.3f} · **effective rank** {er:.2f} / {self.bsf.b}  \n**variance along PCs:** {pcs}  \n"
              f"**correlation label (peak tokens of the top contexts):** {st['labels_tok'][g]}" + (f"  \n**Sonnet label:** {lab}" if lab else ""))
        html = ['<div style="font-family:ui-monospace,monospace;font-size:12.5px;line-height:1.55">']
        tv, ts, tp = st["top_val"][g], st["top_seq"][g], st["top_pos"][g]
        rows_ = {int(s): i for i, s in enumerate(st["seq_ids"].tolist())}
        shown = 0
        for v, s, p in zip(tv.tolist(), ts.tolist(), tp.tolist()):
            if s < 0 or shown >= 14:
                continue
            row = st["toks"][rows_[s]].tolist(); lo, hi = max(0, p - 14), min(len(row), p + 5)
            pre = self.tok.decode(row[lo:p]); pk = self.tok.decode([row[p]]); post = self.tok.decode(row[p + 1:hi])
            html.append(f'<div><span style="color:#888">{v:.2f}</span> …{_esc(pre)}<mark style="background:#f7c9a3">{_esc(pk)}</mark>{_esc(post)}…</div>')
            shown += 1
        html.append("</div>")
        return md, "".join(html)

    def sonnet_label(self, g):
        """Ask Sonnet 5 for a short label from the block's top contexts (cached in pg_labels.json)."""
        st = self.stats
        if st is None:
            return "stats not ready"
        g = int(g) % self.bsf.G
        if str(g) in self.labels:
            return self.labels[str(g)]["label"]
        import anthropic
        rows_ = {int(s): i for i, s in enumerate(st["seq_ids"].tolist())}
        ctx = []
        for s, p in zip(st["top_seq"][g].tolist(), st["top_pos"][g].tolist()):
            if s < 0:
                continue
            row = st["toks"][rows_[s]].tolist(); lo, hi = max(0, p - 16), min(len(row), p + 4)
            ctx.append(self.tok.decode(row[lo:p]) + " <<" + self.tok.decode([row[p]]) + ">> " + self.tok.decode(row[p + 1:hi]))
        if not ctx:
            return "(block never active in the sample)"
        hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)
        prompt = ("Below are text snippets where one internal feature of a language model fires; the token where it fires most is marked <<like this>>. "
                  "Give a short label (<= 12 words) for what the feature responds to, based only on these correlations, then one sentence of evidence. "
                  "Format: LABEL: ... | EVIDENCE: ...\n\n" + "\n".join(f"- {c}" for c in ctx[:20]))
        msg = client.messages.create(model="claude-sonnet-5", max_tokens=200, messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        self.labels[str(g)] = {"label": text, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            json.dump(self.labels, open(LABELS, "w"), indent=1); vol.commit()
        except Exception as e:  # noqa
            print(f"[pg] label cache write failed: {e}", flush=True)
        return text

    def pick_block(self, mode):
        st = self.stats; np = self.np
        if st is None:
            return 0
        cnt = st["count"].numpy(); er = st["eff_rank"].numpy()
        rng = np.random.default_rng()
        if mode == "frequent":
            cand = np.nonzero(cnt >= np.quantile(cnt[cnt > 0], 0.9))[0]
        elif mode == "high-rank":
            cand = np.nonzero((cnt >= 200) & (er >= 3.0))[0]
        else:
            cand = np.nonzero(cnt >= 50)[0]
        return int(rng.choice(cand)) if len(cand) else int(rng.integers(self.bsf.G))

    def code_for(self, g, mode, pc, alpha):
        """-> (code [8], description). mode: 'steer' (mean + alpha*sigma*PC), 'pc' (+-PC axis), 'mean' (block mean code)."""
        st = self.stats; np = self.np
        g = int(g) % self.bsf.G
        if st is None:
            V = np.eye(self.bsf.b, dtype=np.float32); m = np.zeros(self.bsf.b, np.float32); lam = np.ones(self.bsf.b, np.float32)
        else:
            V, m, lam = st["V"][g].numpy(), st["mean_c"][g].numpy(), st["lam"][g].numpy()
        pc = int(pc) - 1
        if mode == "pc":
            code = np.sign(alpha if alpha != 0 else 1.0) * V[:, pc]; desc = f"block {g}: PC{pc + 1} axis fed directly ({'+' if alpha >= 0 else '-'}V{pc + 1})"
        elif mode == "mean":
            code = m; desc = f"block {g}: mean code"
        else:
            code = m + float(alpha) * float(np.sqrt(max(lam[pc], 1e-12))) * V[:, pc]; desc = f"block {g}: mean + {alpha:+.1f}σ along PC{pc + 1}"
        return code.astype(np.float32), desc

    # ------------------------------------------------------------------------------------------------------------ UI actions
    def ui_block(self, g):
        md, html = self.block_card(g)
        return md, html

    def ui_invert(self, g, mode, pc, alpha, n, temperature, max_new, coeff, seed):
        if self.stats is None and mode != "pc":
            return "stats not ready (run the stats job, then 'reload stats')", [], ""
        g = int(g) % self.bsf.G; pc_i = int(pc) - 1
        code, desc = self.code_for(g, mode, pc, alpha)
        d = self.bsf.dir_from_code(g, code)
        t0 = time.time()
        texts = self.generate(d.view(1, -1), n, temperature, max_new, coeff, seed)[0]
        rows = self.analyze(texts, d, g, pc_i)
        order = sorted(range(len(rows)), key=lambda i: -rows[i]["cos"])
        table = [[i + 1, rows[j]["cos"], rows[j]["cos_c"], rows[j]["gn"], "yes" if rows[j]["fires"] else f"no (rank {rows[j]['rank']})", rows[j]["pc"], rows[j]["peak_tok"], texts[j]] for i, j in enumerate(order)]
        summ = (f"**{desc}** — {len(texts)} rollouts in {time.time() - t0:.1f}s · best raw cos {rows[order[0]]['cos']:.3f} · block fires in {sum(r['fires'] for r in rows)}/{len(rows)} rollouts"
                + (f" · PC{pc_i + 1} score at peak (σ units): mean {self.np.mean([r['pc'] for r in rows if r['pc'] is not None]):+.2f}" if any(r["pc"] is not None for r in rows) else ""))
        return summ, table, _rollout_html(rows, order)

    def ui_sweep(self, g, pc, n, temperature, max_new, coeff, seed, alphas="-2,-1,0,1,2"):
        if self.stats is None:
            return "stats not ready", []
        g = int(g) % self.bsf.G; pc_i = int(pc) - 1
        als = [float(a) for a in str(alphas).split(",") if a.strip()]
        codes = [self.code_for(g, "steer", pc, a)[0] for a in als]
        dirs = self.bsf.dir_from_code(g, self.np.stack(codes))
        t0 = time.time()
        per_dir = self.generate(dirs, n, temperature, max_new, coeff, seed)
        table = []
        for a, dvec, texts in zip(als, dirs, per_dir):
            rows = self.analyze(texts, dvec, g, pc_i)
            j = max(range(len(rows)), key=lambda i: rows[i]["cos"])
            pcs = [r["pc"] for r in rows if r["pc"] is not None]
            table.append([f"{a:+.1f}σ", rows[j]["cos"], f"{sum(r['fires'] for r in rows)}/{len(rows)}", round(float(self.np.mean(pcs)), 2) if pcs else None, rows[j]["pc"], texts[j]])
        return f"block {g}, PC{pc_i + 1}: {len(als)} directions × {n} rollouts in {time.time() - t0:.1f}s. Columns: best raw cos, fires, mean PC score of all rollouts, PC score of the best, best rollout.", table

    def ui_text_blocks(self, text, which):
        """Which blocks fire at a token of a typed text (last token or the max over tokens)."""
        st = self.stats
        toks, h, keep = self.read([text])[0]
        y = (h - self.bsf.mu) @ self.bsf.zca; gn = self.bsf.encode(y)
        if which == "last kept token":
            idx = int(self.torch.nonzero(keep)[-1]) if keep.any() else len(toks) - 1; v = gn[idx]
        else:
            v = gn.masked_fill(~keep[:, None], 0).amax(0); idx = None
        tv, ti = v.topk(16)
        rows = [[int(g), round(float(a), 3), (st["labels_tok"][int(g)] if st is not None else ""), self.labels.get(str(int(g)), {}).get("label", "")] for a, g in zip(tv.tolist(), ti.tolist())]
        return f"{len(toks)} tokens" + (f", read at token {idx} {toks[idx]!r}" if idx is not None else ", max over tokens"), rows

    def ui_reload_stats(self):
        vol.reload()
        if os.path.exists(STATS):
            self.stats = self.torch.load(STATS, map_location="cpu", weights_only=False); return f"stats loaded: {self.stats['n_kept']:,} tokens, created {self.stats['created']}"
        return "stats file not present yet"

    # ------------------------------------------------------------------------------------------------------------ web app
    @modal.asgi_app()
    def web(self):
        import gradio as gr
        from fastapi import FastAPI
        b = self.bsf.b
        with gr.Blocks(title="BSF playground · Qwen3.6-27B layer 42") as demo:
            gr.Markdown(f"## BSF playground — Qwen3.6-27B layer-42 subspace featurizer (32,768 blocks × 8 dims, k=32) × the distilled RL inverter (`{CKPT.split('/')[-2]}/{CKPT.split('/')[-1]}`)\n"
                        "Blocks are 8-dim subspaces of the (ZCA-whitened) residual stream. *PCs* are the principal axes of a block's coordinates over corpus tokens where it is active. "
                        "Directions are mapped back to residual space (bank-builder convention) and injected at layer 1 on the marker; rollouts are re-encoded on the clean base. "
                        "Cosines: **raw** = the suite's scorer, **centered** = cos(unit(h − μ), d).")
            with gr.Row():
                g_in = gr.Number(value=self.pick_block("high-rank"), label="block id", precision=0)
                b_freq = gr.Button("random frequent block"); b_rank = gr.Button("random high-rank block (eff-rank ≥ 3)"); b_any = gr.Button("random active block")
                reload_b = gr.Button("reload stats"); reload_out = gr.Markdown()
            with gr.Row():
                n_s = gr.Slider(1, 16, value=8, step=1, label="rollouts per direction"); temp_s = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="temperature")
                len_s = gr.Slider(8, 128, value=48, step=8, label="max new tokens"); coeff_s = gr.Slider(0.25, 3.0, value=1.0, step=0.05, label="injection strength (× ‖h‖)")
                seed_s = gr.Number(value=0, label="seed (0 = random)", precision=0)
            with gr.Tab("Block card"):
                card_md = gr.Markdown(); card_html = gr.HTML()
                with gr.Row():
                    lab_btn = gr.Button("label this block with Sonnet 5 (from its top contexts)"); lab_out = gr.Textbox(label="Sonnet label", lines=2)
            with gr.Tab("Steer the block mean along a PC"):
                with gr.Row():
                    pc_s = gr.Slider(1, b, value=1, step=1, label="PC"); alpha_s = gr.Slider(-3, 3, value=1.0, step=0.25, label="α (σ units along the PC)")
                    go_steer = gr.Button("Invert mean + α·σ·PC", variant="primary"); go_mean = gr.Button("Invert the mean code (α = 0)")
                steer_md = gr.Markdown()
                steer_tbl = gr.Dataframe(headers=["#", "cos raw", "cos centered", "block act", "block fires", "PC score σ", "peak tok", "rollout"], wrap=True)
                steer_html = gr.HTML()
                with gr.Row():
                    alphas_in = gr.Textbox(value="-2,-1,0,1,2", label="sweep α values"); go_sweep = gr.Button("Sweep α along this PC", variant="primary")
                sweep_md = gr.Markdown(); sweep_tbl = gr.Dataframe(headers=["α", "best cos raw", "fires", "mean PC score", "PC score of best", "best rollout"], wrap=True)
            with gr.Tab("Feed a PC axis directly"):
                with gr.Row():
                    pc2_s = gr.Slider(1, b, value=1, step=1, label="PC"); sign_s = gr.Radio(["+", "−"], value="+", label="sign"); go_pc = gr.Button("Invert the PC axis", variant="primary")
                pc_md = gr.Markdown(); pc_tbl = gr.Dataframe(headers=["#", "cos raw", "cos centered", "block act", "block fires", "PC score σ", "peak tok", "rollout"], wrap=True); pc_html = gr.HTML()
            with gr.Tab("Custom code"):
                code_in = gr.Textbox(value=",".join(["1"] + ["0"] * (b - 1)), label=f"{b} subspace coordinates (comma-separated) — mapped through the block's basis")
                go_code = gr.Button("Invert this code", variant="primary"); code_md = gr.Markdown()
                code_tbl = gr.Dataframe(headers=["#", "cos raw", "cos centered", "block act", "block fires", "PC score σ", "peak tok", "rollout"], wrap=True); code_html = gr.HTML()
            with gr.Tab("Text → blocks"):
                txt = gr.Textbox(lines=3, label="text (clean-base layer-42 activations)"); which = gr.Radio(["last kept token", "max over tokens"], value="last kept token", label="read at")
                go_txt = gr.Button("Which blocks fire?", variant="primary"); txt_md = gr.Markdown()
                txt_tbl = gr.Dataframe(headers=["block", "activation", "correlation label", "Sonnet label (if cached)"], wrap=True)

            def _custom(g, code_txt, n, t, mx, cf, sd):
                try:
                    code = self.np.array([float(x) for x in str(code_txt).split(",")], dtype=self.np.float32); assert code.shape == (b,)
                except Exception:
                    return f"need {b} comma-separated numbers", [], ""
                g = int(g) % self.bsf.G; d = self.bsf.dir_from_code(g, code); texts = self.generate(d.view(1, -1), n, t, mx, cf, sd)[0]
                rows = self.analyze(texts, d, g, 0); order = sorted(range(len(rows)), key=lambda i: -rows[i]["cos"])
                return f"block {g}, custom code {code.round(2).tolist()}", [[i + 1, rows[j]["cos"], rows[j]["cos_c"], rows[j]["gn"], "yes" if rows[j]["fires"] else f"no (rank {rows[j]['rank']})", rows[j]["pc"], rows[j]["peak_tok"], texts[j]] for i, j in enumerate(order)], _rollout_html(rows, order)

            g_in.change(self.ui_block, [g_in], [card_md, card_html])
            b_freq.click(lambda: self.pick_block("frequent"), None, g_in); b_rank.click(lambda: self.pick_block("high-rank"), None, g_in); b_any.click(lambda: self.pick_block("any"), None, g_in)
            reload_b.click(self.ui_reload_stats, None, reload_out)
            lab_btn.click(self.sonnet_label, [g_in], lab_out)
            go_steer.click(lambda g, pc, a, n, t, mx, cf, sd: self.ui_invert(g, "steer", pc, a, n, t, mx, cf, sd), [g_in, pc_s, alpha_s, n_s, temp_s, len_s, coeff_s, seed_s], [steer_md, steer_tbl, steer_html])
            go_mean.click(lambda g, pc, n, t, mx, cf, sd: self.ui_invert(g, "mean", pc, 0.0, n, t, mx, cf, sd), [g_in, pc_s, n_s, temp_s, len_s, coeff_s, seed_s], [steer_md, steer_tbl, steer_html])
            go_sweep.click(self.ui_sweep, [g_in, pc_s, n_s, temp_s, len_s, coeff_s, seed_s, alphas_in], [sweep_md, sweep_tbl])
            go_pc.click(lambda g, pc, sg, n, t, mx, cf, sd: self.ui_invert(g, "pc", pc, 1.0 if sg == "+" else -1.0, n, t, mx, cf, sd), [g_in, pc2_s, sign_s, n_s, temp_s, len_s, coeff_s, seed_s], [pc_md, pc_tbl, pc_html])
            go_code.click(_custom, [g_in, code_in, n_s, temp_s, len_s, coeff_s, seed_s], [code_md, code_tbl, code_html])
            go_txt.click(self.ui_text_blocks, [txt, which], [txt_md, txt_tbl])
            demo.load(self.ui_block, [g_in], [card_md, card_html])
        demo.queue(max_size=16)
        api = FastAPI()
        auth = (os.environ.get("PLAYGROUND_USER", "claude"), os.environ.get("PLAYGROUND_PASS", "claube"))
        return gr.mount_gradio_app(api, demo, path="/", auth=auth)


def _esc(s):
    import html
    return html.escape(s)


def _rollout_html(rows, order):
    """Per-token block activation as background intensity; peak token outlined."""
    out = ['<div style="font-family:ui-monospace,monospace;font-size:12.5px;line-height:1.7">']
    for rank_i, j in enumerate(order[:16]):
        r = rows[j]; g = r.get("tok_gn", []); mx = max([v for v in g if v > 0] + [1e-6])
        spans = []
        for i, (t, v) in enumerate(zip(r["toks"], g)):
            a = max(0.0, min(1.0, v / mx)) if mx > 0 else 0.0
            style = f"background:rgba(220,90,60,{0.85 * a:.2f})" + (";outline:2px solid #333" if i == r.get("peak_i") else "")
            spans.append(f'<span style="{style}" title="block act {v:.2f}">{_esc(t.replace("Ġ", " ").replace("Ċ", "⏎"))}</span>')
        out.append(f'<div><b>#{rank_i + 1}</b> <span style="color:#888">cos {r["cos"]:.3f} · act {r["gn"]:.2f}</span> {"".join(spans)}</div>')
    out.append("</div>")
    return "".join(out)
