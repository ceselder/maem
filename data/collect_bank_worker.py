"""Sample-and-emit SFT-bank collector (one GPU per process).

Forward [BOS]+L-token FineFineWeb windows through the base model, read the layer-42 residual (RAW resid_post, same
hook as collect_acts27b_worker), and for K sampled positions per window emit ONE SFT example each:

    direction   = unit(act[p] - mu)            mu = acts27b/whiten_mu.npy (the suite's realact convention)
    target_text = decode(toks[p-W+1 .. p])     W ~ U[w_lo, w_hi], capped at the context length; the firing token
                                                is the LAST target token by construction (end-anchored)
    ctx_len     = p + 1  in [p_lo, p_hi]        the model saw exactly ctx_len tokens (windows are forwarded alone)

No full-sequence activation store is written (10M examples of [256, 5120] fp16 would be 1.6 TB); only the picked
vectors (fp16, 10 KB each) + records. Shards per rank:
    r{rank}_c{c:04d}.vecs.f16      [n, 5120] fp16 unit directions
    r{rank}_c{c:04d}.records.jsonl {"vec_idx" (shard-local), "target_text", "family": "realact", "ctx_len", "W", "src"}
    manifest_r{rank}.json          chunk list + reader offsets (crash-resume, same protocol as the acts collector)

Eval hygiene: --exclude-hashes is a json list of 16-hex sha1 prefixes of 64-token spans; any doc whose 512-aligned
window starts hash into that set is skipped entirely (the realact eval hold-out = the last 5% of acts27b rows).
Norm filter: positions whose raw ||act|| exceeds --norm-cap x the median raw norm (median fixed after the first
batch) are dropped, matching the bank builders' 10x-median filter.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/pmx")
sys.path.insert(0, "/pmx/helpers")
from collect_acts27b_worker import FffwReader, StreamReader, log  # noqa: E402
from mxf.config import D_MODEL, MODEL, READ_LAYER  # noqa: E402
from mxf.inject import read_resid  # noqa: E402

ALIGN = 512   # the acts27b store cut docs into 512-token windows; hash at those offsets to catch its docs


def span_hash(ids, n=64):
    return hashlib.sha1(np.asarray(ids[:n], dtype=np.int32).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--n-examples", type=int, required=True, help="examples THIS rank must produce")
    ap.add_argument("--seq-len", type=int, default=256, help="forwarded window (content tokens after BOS)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--per-window", type=int, default=8, help="positions sampled per forwarded window")
    ap.add_argument("--p-lo", type=int, default=8, help="min context length (tokens incl. the firing token)")
    ap.add_argument("--p-hi", type=int, default=256, help="max context length, inclusive (<= --seq-len)")
    ap.add_argument("--w-lo", type=int, default=8)
    ap.add_argument("--w-hi", type=int, default=32)
    ap.add_argument("--max-wins", type=int, default=4, help="windows per doc cap (diversity)")
    ap.add_argument("--norm-cap", type=float, default=10.0)
    ap.add_argument("--chunk-examples", type=int, default=50_000, help="~examples per shard file")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mu", required=True, help="whiten_mu.npy")
    ap.add_argument("--exclude-hashes", default="", help="json file: list of span hashes to skip (eval docs)")
    ap.add_argument("--w-full", action="store_true", help="target = the FULL context [0..p] of the standalone window (W = ctx_len): "
                    "re-encoding the target reproduces exactly the tokens that produced the activation (precise inversion)")
    ap.add_argument("--out", required=True, help="shard dir (on the volume)")
    ap.add_argument("--assignment", required=True, help="json from the driver: mode + file slices")
    a = ap.parse_args()
    assert a.p_hi <= a.seq_len and a.p_lo >= 1 and a.w_lo >= 1 and a.w_lo <= a.w_hi
    r, L, K = a.rank, a.seq_len, a.per_window
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed * 1000 + r)

    man_path = f"{a.out}/manifest_r{r}.json"
    kept, chunks, reader_state, n_seen_docs, n_skipped_docs, n_norm_drop = 0, [], None, 0, 0, 0
    if os.path.exists(man_path):
        m = json.load(open(man_path))
        kept, chunks, reader_state = m["kept"], m["chunks"], m["reader_state"]
        n_seen_docs, n_skipped_docs, n_norm_drop = m.get("docs", 0), m.get("skipped_docs", 0), m.get("norm_drop", 0)
        rng = np.random.default_rng(a.seed * 1000 + r + 7919 * len(chunks))   # fresh stream per resume
        log(r, f"RESUME: {kept}/{a.n_examples} examples in {len(chunks)} chunks")
    assign = json.load(open(a.assignment))
    mu = torch.tensor(np.load(a.mu).astype(np.float32), device="cuda:0")
    assert mu.shape == (D_MODEL,), mu.shape
    excl = set(json.load(open(a.exclude_hashes))) if a.exclude_hashes else set()
    log(r, f"exclusion set: {len(excl)} span hashes")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 local_files_only=True, device_map={"": "cuda:0"}).eval()
    assert model.config.hidden_size == D_MODEL
    log(r, f"model up in {time.time() - t0:.0f}s (bos={bos}, L={L}, K={K}, ctx [{a.p_lo},{a.p_hi}], W [{a.w_lo},{a.w_hi}])")

    if assign["mode"] == "fffw":
        reader = FffwReader(assign["repo"], assign["ranks"][r], reader_state, r)
    else:
        reader = StreamReader(assign, r, a.world, a.seed, reader_state)

    def write_manifest(done):
        m = {"rank": r, "n_examples_target": a.n_examples, "kept": kept, "chunks": chunks, "reader_state": reader.state(),
             "done": done, "bos_id": int(bos), "mode": assign["mode"], "docs": n_seen_docs, "skipped_docs": n_skipped_docs,
             "norm_drop": n_norm_drop, "norm_median": norm_med, "seq_len": L, "per_window": K,
             "ctx_range": [a.p_lo, a.p_hi], "w_range": [a.w_lo, a.w_hi], "w_full": bool(a.w_full), "full_forward": False}
        with open(man_path + ".tmp", "w") as f:
            json.dump(m, f)
        os.replace(man_path + ".tmp", man_path)

    pending, vec_buf, rec_buf, buf_n, norm_med = [], [], [], 0, None
    win_doc = {}                                                       # id(window) -> absolute document index (range mode; -1 otherwise)
    n_win = 0
    t0 = time.time()

    @torch.no_grad()
    def forward(wins):
        nonlocal buf_n, kept, norm_med, n_norm_drop, n_win
        ids = torch.tensor([[bos] + w for w in wins], device="cuda:0")
        h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
        content = h[:, 1:, :]                                          # fp32 [B, Lw, d]; row t = content token t
        B, Lw = content.shape[0], content.shape[1]                     # Lw = this batch's window length (<= L for doc tails)
        cand = np.arange(a.p_lo - 1, min(a.p_hi, Lw))                 # 0-indexed positions with ctx_len in range
        Kb = min(K, len(cand))
        pos = np.stack([rng.choice(cand, size=Kb, replace=False) for _ in range(B)])   # [B, Kb]
        acts = content[torch.arange(B, device="cuda:0")[:, None], torch.tensor(pos, device="cuda:0")]   # [B, K, d]
        norms = acts.norm(dim=-1)                                      # raw norms
        if norm_med is None:
            norm_med = float(content.norm(dim=-1).median())
            log(r, f"raw-norm median {norm_med:.1f} -> cap {a.norm_cap * norm_med:.1f}")
        keep = (norms <= a.norm_cap * norm_med).cpu().numpy()          # [B, K]
        dirs = torch.nn.functional.normalize(acts - mu, dim=-1).to(torch.float16).cpu().numpy()
        for b in range(B):
            for k in range(Kb):
                if kept >= a.n_examples:
                    break
                p = int(pos[b, k])
                if not keep[b, k]:
                    n_norm_drop += 1
                    continue
                ctx_len = p + 1
                W = ctx_len if a.w_full else int(rng.integers(a.w_lo, min(a.w_hi, ctx_len) + 1))
                text = tok.decode(wins[b][p - W + 1 : p + 1], skip_special_tokens=True)
                if not text.strip():
                    continue
                vec_buf.append(dirs[b, k])
                rec_buf.append({"vec_idx": buf_n, "target_text": text, "family": "realact", "ctx_len": ctx_len, "W": W,
                                "full_ctx": bool(a.w_full), "doc_idx": int(win_doc.get(id(wins[b]), -1)), "src": f"r{r}_w{n_win + b}"})
                buf_n += 1
                kept += 1
        for w in wins:
            win_doc.pop(id(w), None)
        n_win += B

    def drain():
        nonlocal pending
        while pending and kept < a.n_examples:
            take = min(a.batch, len(pending))
            forward(pending[:take])
            pending = pending[take:]
        if kept >= a.n_examples:
            pending = []

    def flush():
        nonlocal vec_buf, rec_buf, buf_n
        assert not pending, "flush with pending windows (reader offsets would desync)"
        if buf_n == 0:
            return
        c = len(chunks)
        arr = np.stack(vec_buf).astype(np.float16)
        assert arr.shape == (buf_n, D_MODEL), arr.shape
        p = f"{a.out}/r{r}_c{c:04d}.vecs.f16"
        arr.tofile(p + ".tmp"); os.replace(p + ".tmp", p)
        p = f"{a.out}/r{r}_c{c:04d}.records.jsonl"
        with open(p + ".tmp", "w") as f:
            for rec in rec_buf:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        chunks.append({"c": c, "n": buf_n})
        write_manifest(done=False)
        el = time.time() - t0
        log(r, f"chunk {c} ({buf_n} ex) -> {kept}/{a.n_examples} ({kept / max(el, 1):.0f} ex/s, {n_win} windows, "
               f"{n_seen_docs} docs, {n_skipped_docs} excluded, {n_norm_drop} norm-dropped, {el / 60:.1f} min)")
        vec_buf, rec_buf, buf_n = [], [], 0

    for text in reader.docs():
        n_seen_docs += 1
        doc_idx = int(getattr(reader, "cur_doc_index", -1))
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=a.max_wins * L + 8)["input_ids"]
        if len(ids) < a.p_lo:
            continue
        if excl and any(span_hash(ids[s:]) in excl for s in range(0, len(ids), ALIGN)):
            n_skipped_docs += 1
            continue
        nw = 0
        for s in range(0, len(ids), L):
            w = ids[s : s + L]
            if len(w) < a.p_lo:
                break
            pending.append(w)                                            # short doc tails get their own equal-length batches
            win_doc[id(w)] = doc_idx
            nw += 1
            if nw >= a.max_wins:
                break
        # windows of different lengths cannot share a batch: group by length
        while len(pending) >= a.batch and kept < a.n_examples:
            full = [w for w in pending if len(w) == L][: a.batch]
            if len(full) < a.batch:
                break
            forward(full)
            taken = set(map(id, full))
            pending = [w for w in pending if id(w) not in taken]
        if kept >= a.n_examples:
            pending = []
            break
        if buf_n >= a.chunk_examples:
            _drain_mixed(pending, forward, a, L, lambda: kept >= a.n_examples)
            pending = []
            flush()
            if kept >= a.n_examples:
                break
    _drain_mixed(pending, forward, a, L, lambda: kept >= a.n_examples)
    pending = []
    flush()
    if kept < a.n_examples:
        write_manifest(done=False)
        raise RuntimeError(f"rank {r}: corpus exhausted at {kept}/{a.n_examples} examples")
    write_manifest(done=True)
    log(r, f"DONE {kept} examples in {len(chunks)} chunks ({(time.time() - t0) / 60:.1f} min)")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # all shards + the done manifest are on disk (os.replace); skip interpreter teardown, which has crashed finished ranks


def _drain_mixed(pending, forward, a, L, done):
    """Forward the leftover windows grouped by exact length (short doc tails get their own batches)."""
    by_len = {}
    for w in pending:
        by_len.setdefault(len(w), []).append(w)
    for ln in sorted(by_len, reverse=True):
        ws = by_len[ln]
        for s in range(0, len(ws), a.batch):
            if done():
                return
            forward(ws[s : s + a.batch])


if __name__ == "__main__":
    main()
