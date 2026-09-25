"""Modal app `maemm-mix-5m-bank`: compose the ~5M-row, 8-family SFT MIDTRAIN bank /data/banks/mix_5m from finalized source banks,
with per-family row caps and multiple sources per family (bank-5m). Output is the exact bank format sft/pretrain.py + rl/rl.py read:
    vecs.f32          [N, 5120] float32 UNIT rows (f16 sources are up-cast and re-normalized)
    records.jsonl     line i == vec_idx i: the source record + {"vec_idx": i, "src_bank", "src_vec_idx"}; "family" preserved
    build_stats.json  n_examples, families, per-source provenance, windows-per-feature / rows-per-neuron summaries, leak table
    meta.json         same + trainer args
Rows are a seeded global shuffle. Default spec (== the 5M bank):
    realact       1,000,000  uniformly over the 11 UNUSED realact_short_20m_{n..x} parts (9M rows each; NOT in the 104M pretrain
                             corpus = realact_short_50m_all + parts h..m), proportional to part size; f16 -> f32
    realact_long  <= 1,000,000  /data/banks/everything_5m_fresh  (fresh 512-token store, deep positions)
    sae           <= 1,000,000  /data/banks/everything_5m_fresh  (unit encoder cols x fresh max-activating windows)
    bsf           all           /data/banks/everything_5m_fresh  (fresh store block projections)
    cluster       all           /data/banks/everything_5m_fresh  (probes; the fixed 241,741 cap)
    mlp / mlp_pair / mlp_triple  all  /data/banks/mlp42_5m_fresh (fresh-store scan)
Leak guard (asserted, GPU): max cosine of EVERY output row vs EVERY direction of the v2 eval cache (all `*_dirs` families, 14) and
every /data/pool_heldout row < 0.999 (the same rule as the source builders; rows above it are dropped and reported — the
realact parts were hash-excluded from eval documents but never direction-checked before). Per-(family x reference) max-cos
table stored for the report.
Peak anchoring (`trim_to_peak="cluster"`, comma-separated families): every SELECTED row of those families whose record carries the
last5_rp probe fields `peak_idx` (0-based argmax of the direction's per-token cosine over tok(target_text, add_special_tokens=False),
sink position dropped) and `n_tok` (that token count) is cut to its first peak_idx+1 tokens, so the peak IS the last token. Exact by
causality (activations at positions <= t never depend on later tokens) -> no re-forward. Rows whose text does not re-tokenize to n_tok
tokens, whose cut does not decode->re-encode to the identical ids, or that would keep < 8 tokens are dropped at SELECTION time, so
records.jsonl line i == vec_idx i is untouched. Counters: build_stats.json / meta.json["trim_to_peak"][family]; `trim_check` previews
them on a sample without writing anything. The app name is read from env MAEMM_MIX5M_APP (default maemm-mix-5m-bank).
    MODAL_PROFILE=safety-sahan MAEMM_MIX5M_APP=maemm-mix-5m-bank2 modal deploy data/modal_mix_5m_bank.py
    python -c "import modal; print(modal.Function.from_name('maemm-mix-5m-bank2','build').spawn(trim_to_peak='cluster').object_id)"
"""
import json
import os

import modal
from pathlib import Path

APP_NAME = os.environ.get("MAEMM_MIX5M_APP", "maemm-mix-5m-bank")
app = modal.App(APP_NAME)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.4.6", "transformers==5.15.0", "huggingface_hub==1.27.0", "tokenizers==0.22.2", "safetensors==0.8.0", "hf_xet")
    .env({"HF_HOME": "/data/hf_cache"})       # tokenizer loads OFFLINE from the volume's HF cache (same pins/cache as the rest of the suite)
    .add_local_dir(Path(__file__).resolve().parent.parent / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])   # model profile only
)


def _mxf_config():
    """mxf/config.py loaded standalone (container: /pmx/helpers/mxf; deploy host: the repo) -> the model profile."""
    import importlib.util
    for p in ("/pmx/helpers/mxf/config.py", str(Path(__file__).resolve().parent.parent / "mxf" / "config.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("mxf_config_standalone", p)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            return m
    raise FileNotFoundError("mxf/config.py")


_CFG = _mxf_config()
MODEL = _CFG.MODEL                            # model profile (27B: Qwen/Qwen3.6-27B)
TRIM_MIN_TOK = 8                              # trim_to_peak: a trimmed target must keep at least this many tokens
SMALL_GPUS = ["H100", "A100-80GB", "L40S", "A100-40GB"]   # leak check only (a few GB); the rest is I/O
D = _CFG.D_MODEL                              # 27B: 5120; qwen3-8b: 4096
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
POOL_HELDOUT = "/data/pool_heldout"
LEAK_COS = 0.999
UNUSED_PARTS = [f"/data/banks/realact_short_20m_{c}" for c in "nopqrstuvwx"]      # NOT in the 104M pretrain corpus
PRETRAIN_PARTS = ["realact_short_50m_all(= 20m,b,c,d,e,f,g)", "h", "i", "j", "k", "l", "m"]
DEFAULT_SPEC = [
    {"name": "realact_parts", "banks": UNUSED_PARTS, "dtype": "f16", "families": ["realact"], "n": 1_000_000, "all_one_family": "realact",
     "fresh": "unused realact_short_20m parts (never trained on)"},
    {"name": "everything_5m_fresh", "banks": ["/data/banks/everything_5m_fresh"], "dtype": "f32",
     "families": ["realact_long", "sae", "sae_dec", "bsf", "cluster"], "n": {"realact_long": 1_000_000, "sae": 1_000_000, "sae_dec": 1_000_000}, "fresh": "fresh store /data/acts27b_fresh (cluster: fixed probe set)"},
    {"name": "mlp42_5m_fresh", "banks": ["/data/banks/mlp42_5m_fresh"], "dtype": "f32", "families": ["mlp", "mlp_pair", "mlp_triple"], "n": None,
     "fresh": "fresh store /data/acts27b_fresh scan"},
]


def _log(msg):
    import time
    print(f"[mix5m {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pread_full(fd, n, off):
    buf = bytearray(n); mv = memoryview(buf); got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


def _scan_records(bank, all_one_family=None):
    """-> (n_rows, family per line (np.str_), line->vec_idx map or None if line i == vec_idx i, raw lines or None).
    Banks built by our builders keep line i == vec_idx i and are fully parsed (lines kept). The realact parts (9M lines, shuffled
    line order, single family) are scanned with a cheap vec_idx parse only; their selected lines are re-read later."""
    import numpy as np
    st = json.load(open(f"{bank}/build_stats.json"))
    n = int(st["n_examples"])
    if all_one_family:
        assert st["families"] == {all_one_family: n}, (bank, st["families"])
        line_of_vec = np.full(n, -1, np.int64)
        with open(f"{bank}/records.jsonl") as fh:
            for i, line in enumerate(fh):
                assert line.startswith('{"vec_idx": '), (bank, i, line[:40])
                line_of_vec[int(line[12:line.index(",", 12)])] = i
        assert (line_of_vec >= 0).all(), f"{bank}: records.jsonl does not cover every vec_idx"
        return n, None, line_of_vec, None
    fams, lines = [], []
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            assert int(r["vec_idx"]) == i, (bank, i, r["vec_idx"])
            fams.append(r["family"]); lines.append(line.rstrip("\n"))
    assert len(lines) == n, (bank, len(lines), n)
    return n, np.array(fams), None, lines


def _load_tok():
    """The Qwen tokenizer from the volume's HF cache, offline (local_files_only: never downloads)."""
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    snap = f"/data/hf_cache/hub/models--{MODEL.replace('/', '--')}"
    _log(f"tokenizer {type(tok).__name__} vocab={len(tok)} <- {snap} (exists={os.path.exists(snap)})")
    return tok


def _trim_plan(tok, texts, peak_idx, n_tok, min_tok=TRIM_MIN_TOK, batch=4096):
    """trim_to_peak pre-pass over one family's candidate rows (parallel lists; peak_idx / n_tok may be None). Positions are 0-based over
    tok(text, add_special_tokens=False) == the last5_rp scoring protocol with the sink position dropped. Returns
    (keep: bool[n], new_text: {j: trimmed text}, counters, reason: str[n]). Per row:
        no integer peak_idx / n_tok          keep verbatim   no_peak_info
        peak_idx outside [0, n_tok)          drop            peak_out_of_range
        len(ids) != n_tok                    drop            len_mismatch   (text does not re-tokenize as the scorer saw it)
        peak_idx == n_tok - 1                keep verbatim   already_last
        cut = ids[:peak_idx+1]; decode(skip_special_tokens=False, clean_up_tokenization_spaces=False) must re-encode to cut exactly
                                             else drop       roundtrip_fail
        len(cut) < min_tok                   drop            too_short
        else                                 trimmed         (tokens_removed = n_tok - peak_idx - 1)"""
    import numpy as np
    n = len(texts)
    keep = np.ones(n, bool); reason = np.full(n, "", dtype=object); new_text = {}
    c = {k: 0 for k in ("n", "no_peak_info", "peak_out_of_range", "len_mismatch", "already_last", "roundtrip_fail", "too_short", "trimmed")}
    c["n"] = n
    removed, trimmed_len, from_end, cut_len_all = [], [], [], []

    def is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    for b0 in range(0, n, batch):
        idx = list(range(b0, min(n, b0 + batch)))
        enc = tok([texts[j] for j in idx], add_special_tokens=False)["input_ids"]
        cand = []                                                        # (j, cut ids)
        for j, ids in zip(idx, enc):
            p, L = peak_idx[j], n_tok[j]
            if not (is_int(p) and is_int(L)):
                c["no_peak_info"] += 1; reason[j] = "no_peak_info"; continue
            if not 0 <= p < L:
                c["peak_out_of_range"] += 1; keep[j] = False; reason[j] = "peak_out_of_range"; continue
            if len(ids) != L:
                c["len_mismatch"] += 1; keep[j] = False; reason[j] = "len_mismatch"; continue
            from_end.append(L - 1 - p)
            if p == L - 1:
                c["already_last"] += 1; reason[j] = "already_last"; continue
            cut_len_all.append(p + 1)
            cand.append((j, ids[:p + 1]))
        if cand:
            dec = tok.batch_decode([cut for _, cut in cand], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            re_enc = tok(dec, add_special_tokens=False)["input_ids"]
            for (j, cut), txt, ids2 in zip(cand, dec, re_enc):
                if ids2 != cut:
                    c["roundtrip_fail"] += 1; keep[j] = False; reason[j] = "roundtrip_fail"; continue
                if len(cut) < min_tok:
                    c["too_short"] += 1; keep[j] = False; reason[j] = "too_short"; continue
                new_text[j] = txt; reason[j] = "trimmed"
                c["trimmed"] += 1; removed.append(n_tok[j] - len(cut)); trimmed_len.append(len(cut))
    c["kept"] = int(keep.sum()); c["dropped"] = int(n - keep.sum())
    c["mean_tokens_removed"] = round(float(np.mean(removed)), 3) if removed else None
    c["trimmed_len"] = ({"min": int(min(trimmed_len)), "mean": round(float(np.mean(trimmed_len)), 2), "median": float(np.median(trimmed_len)),
                         "max": int(max(trimmed_len))} if trimmed_len else None)
    fe = np.array(from_end, np.int64)
    c["peak_from_end_hist_before"] = {str(int(k)): int(v) for k, v in zip(*np.unique(fe, return_counts=True))} if len(fe) else {}
    cl = np.array(cut_len_all, np.int64)
    c["cut_len_hist_all_candidates"] = {str(int(k)): int(v) for k, v in zip(*np.unique(cl, return_counts=True))} if len(cl) else {}
    c["too_short_rate_at_floor"] = {str(m): round(float((cl < m).mean()), 4) for m in (4, 6, 8, 10, 12)} if len(cl) else {}
    c["min_tok"] = min_tok; c["tokenizer"] = MODEL
    return keep, new_text, c, reason


# MIX_MEMORY_MIB / MIX_DISK_MIB (deploy-time): the build stages a float32 [N, 5120] memmap on local disk = N x 20 KiB (8M rows -> 164 GB;
# 20M rows -> 410 GB, which killed the 128 GB / 512 GiB default twice on 2026-09-18 with "Server has lost track of input").
@app.function(image=image, gpu=SMALL_GPUS, cpu=16, memory=int(os.environ.get("MIX_MEMORY_MIB", 131072)), ephemeral_disk=int(os.environ.get("MIX_DISK_MIB", 512 * 1024)),
              volumes={"/data": vol}, timeout=8 * 3600)
def build(out_name: str = "mix_5m", spec_json: str = "", seed: int = 2030, overwrite: bool = False, threads: int = 48, exclude_json: str = "",
          trim_to_peak: str = "", trim_min_tok: int = TRIM_MIN_TOK, dense_frac: float = 0.5, eval_cache: str = EVAL_CACHE_V2):
    """exclude_json: JSON dict {source bank: path of a rows file} — rows listed under families[*].fail_vec_idx of that file are removed from
    the selection pool of that bank. Used for (a) the SAE end-anchor filter (end_anchor_rows.json: rows whose feature does not peak within
    the last 2 tokens when the target is re-tokenized standalone) and (b) the MLP eval-pair exclusion (eval_cos_rows.json: rows whose
    direction has cos > 0.99 to an eval mlp/mlp_pair direction or whose neuron set contains a train neuron dominating an eval pair).
    trim_to_peak: comma-separated families (e.g. "cluster") whose targets are cut to end exactly at the peak token (see module doc /
    _trim_plan). Applied during SELECTION (rows that cannot be trimmed safely are dropped before caps/sampling), the trimmed texts are
    substituted when records.jsonl is written; counters land in build_stats.json["trim_to_peak"]. trim_min_tok: a trimmed target must keep
    at least this many tokens (else the row is dropped, counted too_short)."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    import torch.nn.functional as F
    T0 = time.time()
    vol.reload()
    spec = json.loads(spec_json) if spec_json else DEFAULT_SPEC
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already finalized (overwrite=False)")
    dev = "cuda:0"
    rng = np.random.default_rng(seed)
    stage = "/root/mix"; os.makedirs(stage, exist_ok=True)

    # ---- optional per-bank row exclusions (end-anchor filter) ----
    excl_rows, excl_info = {}, {}
    for bank_p, fpath in (json.loads(exclude_json) if exclude_json else {}).items():
        ea = json.load(open(fpath))
        excl_rows[bank_p] = np.unique(np.concatenate([np.array(d["fail_vec_idx"], np.int64) for d in ea["families"].values()] or [np.zeros(0, np.int64)]))
        excl_info[bank_p] = {"file": fpath, "rule": ea.get("rule"), "n_excluded_rows": int(len(excl_rows[bank_p])),
                             "families": {f: {k: v for k, v in d.items() if k != "fail_vec_idx"} for f, d in ea["families"].items()}}
        _log(f"exclusions for {bank_p}: {len(excl_rows[bank_p])} rows ({ea.get('rule')})")
    # ---- optional peak anchoring (trim_to_peak): tokenizer + per-(bank,row) replacement texts, decided at selection time ----
    trim_fams = {x.strip() for x in trim_to_peak.split(",") if x.strip()}
    tok = _load_tok() if trim_fams else None
    trim_text, trim_stats = {}, {}                           # (bid, src row) -> (text, new n_tok, old n_tok); family -> [per-bank counters]
    # ---- sources: per-bank row lists per family, caps ----
    picks_bank, picks_row, picks_fam = [], [], []            # parallel lists -> arrays
    bank_ids, bank_info, src_lines, src_line_of_vec, per_source = [], {}, {}, {}, {}
    for grp in spec:
        fams = grp["families"]; cap = grp.get("n")
        rows_by_fam = {f: [] for f in fams}                    # fam -> list of (bank_id, row)
        for bank in grp["banks"]:
            t0 = time.time()
            assert os.path.exists(f"{bank}/build_stats.json") and os.path.exists(f"{bank}/vecs.{grp['dtype']}"), f"missing source {bank}"
            n, fam_arr, line_of_vec, lines = _scan_records(bank, grp.get("all_one_family"))
            bsz = os.path.getsize(f"{bank}/vecs.{grp['dtype']}")
            assert bsz == n * D * (2 if grp["dtype"] == "f16" else 4), f"{bank}: vecs size {bsz} != {n} rows"
            bid = len(bank_ids); bank_ids.append(bank)
            bank_info[bid] = {"bank": bank, "dtype": grp["dtype"], "n": n, "group": grp["name"]}
            if lines is not None:
                src_lines[bid] = lines
            if line_of_vec is not None:
                src_line_of_vec[bid] = line_of_vec
            for f in fams:
                rows = np.arange(n) if fam_arr is None else np.flatnonzero(fam_arr == f)
                if bank in excl_rows and len(excl_rows[bank]):
                    n_before = len(rows); rows = rows[~np.isin(rows, excl_rows[bank])]
                    if n_before != len(rows):
                        _log(f"  {bank} {f}: {n_before - len(rows)} rows excluded by the end-anchor filter -> {len(rows)}")
                if f in trim_fams and len(rows):
                    assert lines is not None, f"trim_to_peak needs fully parsed records; {bank} is an all_one_family (cheap-scan) bank"
                    t1 = time.time()
                    recs = [json.loads(lines[r]) for r in rows.tolist()]
                    keep_m, new_t, cnt, _ = _trim_plan(tok, [q["target_text"] for q in recs], [q.get("peak_idx") for q in recs],
                                                       [q.get("n_tok") for q in recs], min_tok=trim_min_tok)
                    for jj, txt in new_t.items():
                        trim_text[(bid, int(rows[jj]))] = (txt, int(recs[jj]["peak_idx"]) + 1, int(recs[jj]["n_tok"]))
                    rows = rows[keep_m]
                    trim_stats.setdefault(f, []).append({"bank": bank, **cnt})
                    _log(f"  {bank} {f}: trim_to_peak -> {len(rows)} rows kept; {json.dumps({k: cnt[k] for k in ('n', 'already_last', 'trimmed', 'len_mismatch', 'roundtrip_fail', 'too_short', 'no_peak_info', 'peak_out_of_range', 'mean_tokens_removed')})} ({time.time() - t1:.0f}s)")
                    del recs
                rows_by_fam[f].append((bid, rows))
            _log(f"source {bank}: {n} rows, families {({f: sum(len(r) for b, r in rows_by_fam[f] if b == bid) for f in fams})} ({time.time() - t0:.0f}s)")
        for f in fams:
            avail = sum(len(r) for _, r in rows_by_fam[f])
            want = None
            if isinstance(cap, int):
                want = cap
            elif isinstance(cap, dict):
                want = cap.get(f)
            take_total = avail if want is None else min(int(want), avail)
            # proportional allocation over the group's banks (largest remainder), then a seeded sample inside each bank
            sizes = np.array([len(r) for _, r in rows_by_fam[f]], np.int64)
            if take_total == avail:
                alloc = sizes.copy()
            else:
                raw = take_total * sizes / sizes.sum(); alloc = np.floor(raw).astype(np.int64)
                rem = take_total - alloc.sum()
                for q in np.argsort(-(raw - np.floor(raw)))[:rem]:
                    alloc[q] += 1
            for (bid, rows), k in zip(rows_by_fam[f], alloc.tolist()):
                sel = np.sort(rng.permutation(rows)[:k]) if k < len(rows) else rows
                picks_bank.append(np.full(len(sel), bid, np.int64)); picks_row.append(sel); picks_fam.extend([f] * len(sel))
            per_source[f"{grp['name']}/{f}"] = {"available": int(avail), "taken": int(take_total), "requested": want, "banks": grp["banks"],
                                                "per_bank": {bank_ids[b]: int(k) for (b, _), k in zip(rows_by_fam[f], alloc.tolist())}, "fresh": grp.get("fresh")}
            _log(f"family {f} <- {grp['name']}: take {take_total} of {avail}")
    pb = np.concatenate(picks_bank); pr = np.concatenate(picks_row); pf = np.array(picks_fam)
    N = len(pb)
    perm = rng.permutation(N)                                   # output row i <- staged pick perm[i]
    pb, pr, pf = pb[perm], pr[perm], pf[perm]
    _log(f"{N} rows selected from {len(bank_ids)} source banks; shuffled ({time.time() - T0:.0f}s)")

    # ---- vectors: per bank, gather rows sorted by source row (sequential-ish reads), scatter into the local staging memmap ----
    vecs = np.memmap(f"{stage}/vecs.f32", np.float32, "w+", shape=(N, D))
    norm_stats = {"min": 9.0, "max": 0.0}
    for bid, bank in enumerate(bank_ids):
        t0 = time.time()
        info = bank_info[bid]
        idx = np.flatnonzero(pb == bid)                          # output rows fed by this bank
        order = idx[np.argsort(pr[idx], kind="stable")]
        src_rows = pr[order]
        row_b = D * (2 if info["dtype"] == "f16" else 4)
        dt = np.float16 if info["dtype"] == "f16" else np.float32
        fd = os.open(f"{bank}/vecs.{info['dtype']}", os.O_RDONLY)
        dense = len(src_rows) > dense_frac * info["n"]           # dense selection: stream contiguous ranges; sparse: random preads (dense_frac=0 -> always stream: volume random reads crawled 2026-09-10)
        CH = 32768
        for c0 in range(0, len(src_rows), CH):
            rs = src_rows[c0:c0 + CH]; outs = order[c0:c0 + CH]
            if dense:
                lo, hi = int(rs[0]), int(rs[-1]) + 1
                blk = np.frombuffer(_pread_full(fd, (hi - lo) * row_b, lo * row_b), dt).reshape(hi - lo, D)[rs - lo]
            else:
                blk = np.empty((len(rs), D), dt)
                with ThreadPoolExecutor(threads) as ex:
                    list(ex.map(lambda j: blk.__setitem__(j, np.frombuffer(_pread_full(fd, row_b, int(rs[j]) * row_b), dt)), range(len(rs)), chunksize=256))
            x = blk.astype(np.float32)
            nn = np.linalg.norm(x, axis=1)
            norm_stats["min"] = min(norm_stats["min"], float(nn.min())); norm_stats["max"] = max(norm_stats["max"], float(nn.max()))
            assert 0.98 < nn.min() and nn.max() < 1.02, (bank, nn.min(), nn.max())
            x /= nn[:, None]                                     # re-normalize (f16 sources carry ~1e-4 norm error)
            vecs[outs] = x
            if (c0 // CH) % 16 == 15:                            # every 512k rows (10 GB of f32 scatter): msync so dirty page-cache stays bounded --
                vecs.flush()                                     # the 20M-row build (410 GB staging) was OOM-killed twice by dirty pages at 128 and 256 GiB
                try:
                    _fdv = os.open(f"{stage}/vecs.f32", os.O_RDONLY); os.posix_fadvise(_fdv, 0, 0, os.POSIX_FADV_DONTNEED); os.close(_fdv)
                except Exception:  # noqa
                    pass
        os.close(fd)
        _log(f"vectors <- {bank}: {len(src_rows)} rows ({'dense' if dense else 'random'} reads, {time.time() - t0:.0f}s)")
    vecs.flush()

    # ---- leak guard: every row vs every v2 eval direction family + every pool_heldout row ----
    t0 = time.time()
    es = torch.load(eval_cache, map_location="cpu", weights_only=False)   # every `*_dirs` family (v3 adds the held-out 2M-SAE enc/dec dirs)
    ref_names = sorted(k[:-5] for k in es if k.endswith("_dirs") and torch.is_tensor(es[k]))
    refs = [F.normalize(es[f"{n}_dirs"].float().reshape(-1, D), dim=-1) for n in ref_names]
    use_ho = _CFG.PROFILE == "qwen36-27b"                   # /data/pool_heldout is the 27B universal-bank pool (5120-d)
    ho_fam = np.array([json.loads(l)["family"] for l in open(f"{POOL_HELDOUT}/records.jsonl")]) if use_ho else np.array([], str)
    if use_ho:
        ho_n = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32") // (4 * D)
        ho = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_n, D))
    for hf in sorted(set(ho_fam.tolist())):
        refs.append(F.normalize(torch.from_numpy(np.asarray(ho[np.flatnonzero(ho_fam == hf)])).float(), dim=-1)); ref_names.append(f"pool_heldout/{hf}")
    ref = torch.cat(refs).to(dev)
    offs = np.cumsum([0] + [len(r) for r in refs])
    fam_names = sorted(set(pf.tolist()))
    fam_idx = np.array([fam_names.index(f) for f in pf], np.int8)
    leak = np.full((len(fam_names), len(ref_names)), -1.0)
    maxcos = np.zeros(N, np.float32)
    with torch.no_grad():
        LC = 16384                                              # [16384, ~98k] fp32 cos = 6.4 GB: fits every SMALL_GPUS type
        for c0 in range(0, N, LC):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + LC])).to(dev)
            cos = x @ ref.T
            maxcos[c0:c0 + LC] = cos.max(1).values.cpu().numpy()
            fa = fam_idx[c0:c0 + LC]
            for ri in range(len(ref_names)):
                cm = cos[:, offs[ri]:offs[ri + 1]].max(1).values.cpu().numpy()
                for fi in range(len(fam_names)):
                    m = fa == fi
                    if m.any():
                        leak[fi, ri] = max(leak[fi, ri], float(cm[m].max()))
    bad = maxcos > LEAK_COS
    n_bad = int(bad.sum())
    leak_tbl = {f: {n: round(float(leak[fi, ri]), 4) for ri, n in enumerate(ref_names)} for fi, f in enumerate(fam_names)}
    dropped = {f: int((bad & (pf == f)).sum()) for f in fam_names}
    _log(f"leak guard vs {ref.shape[0]} reference dirs ({len(ref_names)} sets): {n_bad} rows > {LEAK_COS} -> dropped {dropped} ({time.time() - t0:.0f}s)")
    for f in fam_names:
        _log(f"  max-cos {f:13s} " + " ".join(f"{n.split('/')[-1][:11]}={leak_tbl[f][n]:.3f}" for n in ref_names))
    del ref; torch.cuda.empty_cache()

    # ---- publish: compact (skip leakers) sequentially into the volume; records line i == vec_idx i ----
    t0 = time.time()
    keep = np.flatnonzero(~bad)
    N_out = len(keep)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/vecs.f32.tmp", "wb") as fout:
        for c0 in range(0, N_out, 32768):
            fout.write(np.ascontiguousarray(vecs[keep[c0:c0 + 32768]]).tobytes())
    os.replace(f"{out}/vecs.f32.tmp", f"{out}/vecs.f32")
    assert os.path.getsize(f"{out}/vecs.f32") == N_out * D * 4
    _log(f"vecs.f32 written: {N_out} rows ({N_out * D * 4 / 2**30:.1f} GB, {time.time() - t0:.0f}s)")
    # records: parts' selected lines are re-read by line number (their line order is shuffled vs vec_idx)
    t0 = time.time()
    part_lines = {}
    for bid, bank in enumerate(bank_ids):
        if bid in src_line_of_vec:
            want_rows = pr[keep][pb[keep] == bid]
            want_lines = {int(src_line_of_vec[bid][r]): int(r) for r in want_rows.tolist()}
            got = {}
            with open(f"{bank}/records.jsonl") as fh:
                for i, line in enumerate(fh):
                    if i in want_lines:
                        got[want_lines[i]] = line.rstrip("\n")
            assert len(got) == len(want_lines), (bank, len(got), len(want_lines))
            part_lines[bid] = got
            _log(f"records <- {bank}: {len(got)} lines ({time.time() - t0:.0f}s)")
    fam_counts, n_trim_written = {}, {}
    with open(f"{out}/records.jsonl.tmp", "w") as fh:
        for i, g in enumerate(keep.tolist()):
            bid, row = int(pb[g]), int(pr[g])
            line = src_lines[bid][row] if bid in src_lines else part_lines[bid][row]
            r = json.loads(line)
            assert r["family"] == pf[g]
            r["src_bank"] = bank_ids[bid]; r["src_vec_idx"] = int(r.get("vec_idx", row)); r["vec_idx"] = i
            tt = trim_text.get((bid, row))
            if tt is not None:                                   # peak anchoring: text cut to ids[:peak_idx+1]; peak_idx unchanged
                r["target_text"], r["n_tok"], r["trimmed_from_n_tok"] = tt
                n_trim_written[r["family"]] = n_trim_written.get(r["family"], 0) + 1
            fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    _log(f"records.jsonl written ({time.time() - t0:.0f}s): {fam_counts}; trimmed-to-peak rows written: {n_trim_written}")
    # ---- trim_to_peak summary: per family (summed over the family's source banks) ----
    trim_summary = {}
    for f, parts in trim_stats.items():
        agg = {k: int(sum(p_[k] for p_ in parts)) for k in ("n", "already_last", "trimmed", "len_mismatch", "roundtrip_fail", "too_short",
                                                            "no_peak_info", "peak_out_of_range", "kept", "dropped")}
        agg["mean_tokens_removed"] = (round(sum(p_["mean_tokens_removed"] * p_["trimmed"] for p_ in parts if p_["trimmed"]) / agg["trimmed"], 3)
                                      if agg["trimmed"] else None)
        agg["written_trimmed_rows"] = int(n_trim_written.get(f, 0))
        agg["per_bank"] = parts
        trim_summary[f] = agg

    # ---- summaries from the source banks' meta (windows per feature / neuron, fresh provenance) ----
    src_meta = {}
    for bank in bank_ids:
        mp = f"{bank}/meta.json"
        if os.path.exists(mp):
            m = json.load(open(mp))
            keep_keys = {"family_recipes", "neurons", "pairs", "triples", "distinct_windows", "leak_check", "leakage_check", "scan", "acts_store", "shortfalls"}
            src_meta[bank] = {k: v for k, v in m.items() if k in keep_keys}
    stats = {"kind": "5M-row 8-family SFT midtrain mix (bank-5m): fresh-store realact_long / sae / bsf / mlp / mlp_pair / mlp_triple + unused-part realact + fixed probes",
             "n_examples": N_out, "n_vecs": N_out, "families": fam_counts, "seed": seed, "spec": spec, "sources": per_source,
             "source_banks": bank_info, "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i; src_bank/src_vec_idx point back)",
             "leak_check": {"threshold": LEAK_COS, "reference_sets": ref_names, "n_reference_dirs": int(offs[-1]), "rows_dropped": n_bad,
                            "rows_dropped_by_family": dropped, "max_cos_table": leak_tbl, "eval_cache": eval_cache, "pool_heldout": POOL_HELDOUT},
             "source_norm_range": norm_stats, "pretrain_corpus_parts_excluded": PRETRAIN_PARTS, "unused_parts_used": UNUSED_PARTS,
             "row_exclusions": excl_info,
             "trim_to_peak": trim_summary,
             "trim_to_peak_protocol": (f"families {sorted(trim_fams)}: target cut to tok(text, add_special_tokens=False)[:peak_idx+1] (peak_idx = 0-based "
                                       f"argmax of the direction's per-token cosine, sink dropped; exact by causality); rows dropped at selection when "
                                       f"len(ids) != n_tok, decode->re-encode != cut, or cut < {trim_min_tok} tokens; records carry n_tok=peak_idx+1 and "
                                       f"trimmed_from_n_tok; tokenizer {MODEL}" if trim_fams else None),
             "source_meta": src_meta, "d_model": D, "model": MODEL, "layer": _CFG.READ_LAYER, "created": time.time(), "wall_s": time.time() - T0}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump({**stats, "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32"}}, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    shutil.rmtree(stage, ignore_errors=True)
    _log(f"DONE {out}: {N_out} rows {fam_counts} in {(time.time() - T0) / 60:.1f} min")
    return {"out": out, "n_examples": N_out, "families": fam_counts, "leak_dropped": dropped, "minutes": (time.time() - T0) / 60,
            "trim_to_peak": {f: {k: a[k] for k in ("n", "already_last", "trimmed", "dropped", "written_trimmed_rows", "mean_tokens_removed")} for f, a in trim_summary.items()}}


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=1800)
def peek(out_name: str = "mix_5m", n: int = 2):
    import numpy as np
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    print(json.dumps({k: st[k] for k in ("n_examples", "families", "sources")}, indent=1))
    N = st["n_examples"]
    assert os.path.getsize(f"{out}/vecs.f32") == N * D * 4
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D))
    seen = {}
    with open(f"{out}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line); f = r["family"]
            if seen.get(f, 0) >= n:
                continue
            seen[f] = seen.get(f, 0) + 1
            assert r["vec_idx"] == i
            print(f"[{f}] row {i} |v|={np.linalg.norm(vecs[i]):.4f} src={r['src_bank'].split('/')[-1]}:{r['src_vec_idx']} :: {r['target_text'][:100]!r}")
            if len(seen) == len(st["families"]) and all(c >= n for c in seen.values()):
                break
    return st["families"]


@app.function(image=image, cpu=4, memory=32768, volumes={"/data": vol}, timeout=3600)
def trim_check(bank: str = "/data/banks/everything_5m_fresh", family: str = "cluster", n: int = 2000, seed: int = 0, show: int = 5,
               min_tok: int = TRIM_MIN_TOK):
    """Dry run of the trim_to_peak rule on a seeded sample of `n` rows of `family` in `bank`: the same counters build() records, plus a
    reason x `truncated` (last5_rp flag) cross-tab and a few before/after examples. Writes NOTHING. Also proves the Qwen tokenizer loads
    offline from /data/hf_cache (local_files_only)."""
    import time
    import numpy as np
    T0 = time.time()
    vol.reload()
    tok = _load_tok()
    needle = f'"family": "{family}"'
    recs = []
    with open(f"{bank}/records.jsonl") as fh:
        for line in fh:
            if needle in line:                                   # cheap prefilter (json.dumps default spacing), verified after parsing
                r = json.loads(line)
                if r["family"] == family:
                    recs.append(r)
    if not recs:                                                  # spacing differs -> full parse
        with open(f"{bank}/records.jsonl") as fh:
            recs = [r for r in map(json.loads, fh) if r["family"] == family]
    _log(f"{bank}: {len(recs)} rows of family {family!r} ({time.time() - T0:.0f}s); record keys: {sorted(recs[0].keys()) if recs else None}")
    rng = np.random.default_rng(seed)
    pick = [recs[j] for j in np.sort(rng.permutation(len(recs))[:n])]
    keep, new_text, cnt, reason = _trim_plan(tok, [r["target_text"] for r in pick], [r.get("peak_idx") for r in pick], [r.get("n_tok") for r in pick],
                                             min_tok=min_tok)
    xtab = {}
    for r, rs in zip(pick, reason):
        key = f"{rs}|truncated={r.get('truncated')}"
        xtab[key] = xtab.get(key, 0) + 1
    _log("reason x last5_rp truncated flag: " + json.dumps(dict(sorted(xtab.items()))))
    shown = 0
    for j, r in enumerate(pick):
        if reason[j] in ("trimmed", "len_mismatch", "roundtrip_fail") and shown < show * 2 and (reason[j] != "trimmed" or shown < show):
            ids = tok(r["target_text"], add_special_tokens=False)["input_ids"]
            p, L = r.get("peak_idx"), r.get("n_tok")
            tail = tok.decode(ids[p + 1:]) if isinstance(p, int) and p + 1 <= len(ids) else None
            peak_tok = tok.decode(ids[p:p + 1]) if isinstance(p, int) and p < len(ids) else None
            _log(f"  [{reason[j]}] src_vec_idx={r.get('src_vec_idx')} n_tok={L} retok={len(ids)} peak_idx={p} truncated={r.get('truncated')} "
                 f"peak_tok={peak_tok!r} dropped_tail={tail!r}")
            if reason[j] == "trimmed":
                _log(f"      before: {r['target_text'][-120:]!r}")
                _log(f"      after : {new_text[j][-120:]!r}")
            shown += 1
    out = {"bank": bank, "family": family, "n_family_rows": len(recs), "sampled": len(pick), "seed": seed, **cnt,
           "reason_x_truncated": dict(sorted(xtab.items())), "seconds": round(time.time() - T0, 1)}
    print(json.dumps(out, indent=1), flush=True)
    return out
