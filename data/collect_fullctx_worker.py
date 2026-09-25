"""One GPU worker of data/modal_collect_bank.py::collect_fullctx: FULL-DOCUMENT forwards -> long-context real-activation rows.

Each document (ordered Ultra-FineWeb stream, range mode; docs with index % world == rank) is forwarded ONCE as [BOS] + its first
ctx_hi tokens (no 256/512-token chunking: the activation at position p saw the document's real first p+1 tokens), `per_doc`
positions with ctx_len = p+1 in [ctx_lo, min(ctx_hi, doc_len)] are sampled without replacement, and each gives one row:
    direction    = unit(h_L42[p] - mu)            (10x-median raw-norm cap, the suite's realact convention)
    target_text  = decode(tokens[p-tail+1 .. p])   the last `tail` tokens ENDING at p (logging / transcripts only)
    record       {vec_idx, target_text, family, ctx_len, W (= tail or ctx_len), doc_len, full_forward: true, src}
Batches are formed by document length (sorted within a buffer of 8 x batch documents) with right padding + attention mask.
Shards, manifests and resume semantics == data/collect_bank_worker.py, so modal_collect_bank._finalize publishes the bank.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/pmx")
sys.path.insert(0, "/pmx/helpers")
from mxf.config import D_MODEL, MODEL, READ_LAYER  # noqa: E402
from mxf.inject import read_resid  # noqa: E402
from collect_acts27b_worker import StreamReader, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--n-examples", type=int, required=True, help="rows THIS rank must produce")
    ap.add_argument("--ctx-lo", type=int, default=64, help="min context length (tokens incl. the firing token)")
    ap.add_argument("--ctx-hi", type=int, default=2048, help="max context length = forwarded tokens per document")
    ap.add_argument("--per-doc", type=int, default=8, help="positions sampled per document")
    ap.add_argument("--batch", type=int, default=4, help="documents per forward (right-padded)")
    ap.add_argument("--tail", type=int, default=64, help="target_text = the last `tail` tokens ending at p")
    ap.add_argument("--norm-cap", type=float, default=10.0)
    ap.add_argument("--chunk-examples", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mu", required=True, help="whiten_mu.npy")
    ap.add_argument("--out", required=True, help="shard dir (on the volume)")
    ap.add_argument("--assignment", required=True, help="json from the driver (mode hfstream: dataset/config/split/skip)")
    ap.add_argument("--family", default="realact_fullctx")
    a = ap.parse_args()
    assert 1 <= a.ctx_lo <= a.ctx_hi and a.tail >= 1 and a.per_doc >= 1
    r, K = a.rank, a.per_doc
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed * 1000 + r)

    man_path = f"{a.out}/manifest_r{r}.json"
    kept, chunks, reader_state, n_seen_docs, n_short_docs, n_norm_drop = 0, [], None, 0, 0, 0
    if os.path.exists(man_path):
        m = json.load(open(man_path))
        kept, chunks, reader_state = m["kept"], m["chunks"], m["reader_state"]
        n_seen_docs, n_short_docs, n_norm_drop = m.get("docs", 0), m.get("skipped_docs", 0), m.get("norm_drop", 0)
        rng = np.random.default_rng(a.seed * 1000 + r + 7919 * len(chunks))
        log(r, f"RESUME: {kept}/{a.n_examples} rows in {len(chunks)} chunks")
    assign = json.load(open(a.assignment))
    assert assign["mode"] == "hfstream", "collect_fullctx needs the ordered document-range stream"
    mu = torch.tensor(np.load(a.mu).astype(np.float32), device="cuda:0")
    assert mu.shape == (D_MODEL,), mu.shape

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 local_files_only=True, device_map={"": "cuda:0"}).eval()
    assert model.config.hidden_size == D_MODEL
    log(r, f"model up in {time.time() - t0:.0f}s (bos={bos} pad={pad}, ctx [{a.ctx_lo},{a.ctx_hi}], {K}/doc, batch {a.batch}, tail {a.tail}, family {a.family})")
    reader = StreamReader(assign, r, a.world, a.seed, reader_state)

    def write_manifest(done):
        m = {"rank": r, "n_examples_target": a.n_examples, "kept": kept, "chunks": chunks, "reader_state": reader.state(),
             "done": done, "bos_id": int(bos), "mode": assign["mode"], "docs": n_seen_docs, "skipped_docs": n_short_docs,
             "skipped_docs_meaning": f"documents shorter than ctx_lo={a.ctx_lo} tokens", "norm_drop": n_norm_drop, "norm_median": norm_med,
             "seq_len": a.ctx_hi, "per_window": K, "ctx_range": [a.ctx_lo, a.ctx_hi], "w_range": [1, a.tail], "w_full": False,
             "full_forward": True, "tail_tokens": a.tail, "family": a.family}
        with open(man_path + ".tmp", "w") as f:
            json.dump(m, f)
        os.replace(man_path + ".tmp", man_path)

    pending, vec_buf, rec_buf, buf_n, norm_med = [], [], [], 0, None
    n_fwd_docs = 0
    t0 = time.time()

    @torch.no_grad()
    def forward(docs):
        nonlocal buf_n, kept, norm_med, n_norm_drop, n_fwd_docs
        B = len(docs)
        Lmax = max(len(d) for d, _ in docs)
        ids = torch.full((B, Lmax + 1), int(pad), dtype=torch.long, device="cuda:0")
        mask = torch.zeros((B, Lmax + 1), dtype=torch.long, device="cuda:0")
        for b, (d, _) in enumerate(docs):
            ids[b, 0] = bos
            ids[b, 1:len(d) + 1] = torch.tensor(d, dtype=torch.long, device="cuda:0")
            mask[b, :len(d) + 1] = 1
        h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": mask}, pool="all")   # fp32 [B, Lmax+1, d]
        for b, (d, doc_idx) in enumerate(docs):
            n = len(d)
            lo, hi = a.ctx_lo - 1, min(a.ctx_hi, n) - 1                     # 0-indexed content positions; ctx_len = p + 1
            cand = np.arange(lo, hi + 1)
            Kb = min(K, len(cand))
            if Kb <= 0:
                continue
            pos = np.sort(rng.choice(cand, size=Kb, replace=False))
            acts = h[b, torch.tensor(pos + 1, device="cuda:0")]           # content position p lives at index p+1 (BOS at 0)
            norms = acts.norm(dim=-1)
            if norm_med is None:
                norm_med = float(h[b, 1:n + 1].norm(dim=-1).median())
                log(r, f"raw-norm median {norm_med:.1f} -> cap {a.norm_cap * norm_med:.1f}")
            dirs = F.normalize(acts - mu, dim=-1).to(torch.float16).cpu().numpy()
            keep = (norms <= a.norm_cap * norm_med).cpu().numpy()
            for k, p in enumerate(pos.tolist()):
                if kept >= a.n_examples:
                    break
                if not keep[k]:
                    n_norm_drop += 1
                    continue
                ctx_len = p + 1
                W = min(a.tail, ctx_len)
                text = tok.decode(d[p - W + 1 : p + 1], skip_special_tokens=True)
                if not text.strip():
                    continue
                vec_buf.append(dirs[k])
                rec_buf.append({"vec_idx": buf_n, "target_text": text, "family": a.family, "ctx_len": ctx_len, "W": W, "doc_len": n,
                                "full_forward": True, "doc_idx": int(doc_idx), "src": f"r{r}_d{n_fwd_docs + b}"})
                buf_n += 1
                kept += 1
        n_fwd_docs += B

    def run_pending():
        nonlocal pending
        docs = sorted(pending, key=lambda t: len(t[0]))                     # length-sorted batches -> little padding
        pending = []
        for i in range(0, len(docs), a.batch):
            if kept >= a.n_examples:
                break
            forward(docs[i:i + a.batch])

    def flush():
        nonlocal vec_buf, rec_buf, buf_n
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
        log(r, f"chunk {c} ({buf_n} rows) -> {kept}/{a.n_examples} ({kept / max(el, 1):.0f} rows/s, {n_fwd_docs} docs forwarded, "
               f"{n_seen_docs} seen, {n_short_docs} too short, {n_norm_drop} norm-dropped, {el / 60:.1f} min)")
        vec_buf, rec_buf, buf_n = [], [], 0

    BUF = 8 * a.batch
    for text in reader.docs():
        if kept >= a.n_examples:
            break
        n_seen_docs += 1
        ids_ = tok(text, add_special_tokens=False, truncation=True, max_length=a.ctx_hi)["input_ids"]
        if len(ids_) < a.ctx_lo:
            n_short_docs += 1
            continue
        pending.append((ids_, int(getattr(reader, "cur_doc_index", -1))))
        if len(pending) >= BUF:
            run_pending()
            if buf_n >= a.chunk_examples:
                flush()
    if pending and kept < a.n_examples:
        run_pending()
    pending = []
    flush()
    if kept < a.n_examples:
        write_manifest(done=False)
        raise RuntimeError(f"rank {r}: corpus exhausted at {kept}/{a.n_examples} rows")
    write_manifest(done=True)
    log(r, f"DONE {kept} rows in {len(chunks)} chunks ({(time.time() - t0) / 60:.1f} min)")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # all shards + the done manifest are on disk (os.replace); skip interpreter teardown, which has crashed finished ranks


if __name__ == "__main__":
    main()
