"""Per-GPU worker: Qwen3.6-27B layer-42 per-token residuals + token ids over FineFineWeb.

Launched by modal_acts27b.py (one process per GPU; CUDA_VISIBLE_DEVICES pins the device;
mounted in-container at /pmx/collect_acts27b_worker.py). Each rank owns a DISJOINT set of
FineFineWeb domain files and round-robins over ~8 open files (16 docs per turn), so every
chunk mixes domains. Docs are packed into non-overlapping [BOS]+512-token windows, batch-
forwarded through the base model with the early-exit hook at layer 42
(mxf.inject.read_resid), the BOS/sink position is dropped, and RAW fp16 acts + i32 token
ids land in chunk shards under --out:

    r{rank}_c{c:04d}.acts.f16   [n, 512, 5120] fp16 raw resid_post (no norm filter)
    r{rank}_c{c:04d}.toks.i32   [n, 512] i32 content-token ids (positions 1..512 of fwd)
    musum_r{rank}.npy           float64 [5120] running activation sum (whitening mean)
    manifest_r{rank}.json       chunk list + reader offsets (crash-resume) + counters

Crash-resume: shards/manifest/musum are written atomically at chunk boundaries, with the
reader's per-file line offsets snapshotted only after the pending-window queue is fully
drained — a rerun skips exactly the consumed lines, never duplicating or dropping windows.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ALIGN = 512   # docs are hashed at 512-token offsets (== the store's window cut) so any doc that contributed a row is caught


def span_hash(ids, n=64):
    """sha1[:16] of the first n token ids (int32) — the suite's document-exclusion key (== collect_bank_worker.span_hash)."""
    import numpy as np
    return hashlib.sha1(np.asarray(ids[:n], dtype=np.int32).tobytes()).hexdigest()[:16]


def log(rank, *a):
    print(f"[r{rank}]", *a, flush=True)


class FffwReader:
    """Round-robin block reader over this rank's assigned FineFineWeb jsonl files.

    Files download on first open (hf_hub_download -> container-local /tmp, NOT the volume
    hf_cache). ~OPEN_TARGET files are open at once; each turn pulls BLOCK_DOCS qualifying
    docs from one file then rotates, so consecutive windows span ~8 domains.
    """
    OPEN_TARGET = 8
    BLOCK_DOCS = 16
    DL_RETRIES = 5

    def __init__(self, repo, files, state, rank):
        self.repo, self.files, self.rank = repo, files, rank
        st = state or {}
        self.consumed = dict(st.get("consumed", {}))   # filename -> lines read
        self.exhausted = set(st.get("exhausted", []))
        self.pool = []                                  # [filename, filehandle]
        self.next_idx = 0
        self.i = 0
        from concurrent.futures import ThreadPoolExecutor
        want = [f for f in self.files if f not in self.exhausted][: self.OPEN_TARGET]
        with ThreadPoolExecutor(4) as ex:               # prefetch initial pool in parallel
            list(ex.map(lambda f: self._download(f, soft=True), want))
        while len(self.pool) < self.OPEN_TARGET and self._open_next():
            pass
        if not self.pool:
            raise RuntimeError(f"rank {rank}: no readable corpus files")

    def _download(self, fn, soft=False):
        from huggingface_hub import hf_hub_download
        for att in range(self.DL_RETRIES):
            try:
                return hf_hub_download(repo_id=self.repo, filename=fn, repo_type="dataset",
                                       cache_dir="/tmp/fffw_cache")
            except Exception as e:
                log(self.rank, f"download {fn} attempt {att + 1}/{self.DL_RETRIES}: "
                               f"{type(e).__name__}: {str(e)[:120]}")
                time.sleep(min(60, 3 * 2 ** att))
        if soft:
            return None
        raise RuntimeError(f"download failed: {fn}")

    def _open_next(self):
        while self.next_idx < len(self.files):
            fn = self.files[self.next_idx]
            self.next_idx += 1
            if fn in self.exhausted:
                continue
            path = self._download(fn, soft=True)
            if path is None:                 # unreadable after retries: drop it, keep going
                self.exhausted.add(fn)
                continue
            fh = open(path, encoding="utf-8", errors="replace")
            for _ in range(self.consumed.get(fn, 0)):   # resume: skip already-consumed lines
                if not fh.readline():
                    break
            self.pool.append([fn, fh])
            log(self.rank, f"opened {fn} (skip {self.consumed.get(fn, 0)} lines)")
            return True
        return False

    def docs(self):
        while self.pool:
            if self.i >= len(self.pool):
                self.i = 0
            fn, fh = self.pool[self.i]
            got, dead = 0, False
            while got < self.BLOCK_DOCS:
                line = fh.readline()
                if not line:
                    dead = True
                    break
                self.consumed[fn] = self.consumed.get(fn, 0) + 1
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = row.get("text") or ""
                tc = row.get("token_count")             # corpus-side count: cheap prefilter
                if len(text) < 1500 or (tc is not None and tc < 450):
                    continue
                got += 1
                yield text
            if dead:
                fh.close()
                self.exhausted.add(fn)
                del self.pool[self.i]
                self._open_next()                       # replacement lands at the end
            else:
                self.i += 1

    def state(self):
        return {"consumed": self.consumed, "exhausted": sorted(self.exhausted)}


class StreamReader:
    """Fallback: HF streaming (fineweb sample) with modulo doc sharding + skip-resume."""

    def __init__(self, assign, rank, world, seed, state):
        from datasets import load_dataset
        self.rank, self.world = rank, world
        time.sleep(rank * 3)                            # stagger hub API hits across ranks
        ds = None
        for att in range(8):
            try:
                ds = load_dataset(assign["dataset"], name=assign.get("config") or None,
                                  split=assign.get("split", "train"), streaming=True)
                break
            except Exception as e:
                log(rank, f"load_dataset attempt {att + 1}/8: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(min(120, 5 * 2 ** att))
        if ds is None:
            raise RuntimeError("fallback load_dataset failed after 8 attempts")
        # Document-RANGE mode (assign["skip"] > 0 and/or assign["shuffle"] false): consume the stream IN ORDER from single-stream
        # document index `skip`, so the documents a collection touched are exactly [skip, skip + docs_seen) and collections with
        # non-overlapping ranges are document-disjoint by construction (the 2M SAE consumed Ultra-FineWeb docs [100k, ~1.8M) the
        # same way). Default (no skip, shuffle=True) keeps the old behaviour.
        self.skip = int(assign.get("skip", 0) or 0)
        self.shuffle = bool(assign.get("shuffle", True))
        self.cur_doc_index = -1                                 # exact only when shuffle is False (range mode)
        if self.shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        self.docs_seen = (state or {}).get("docs_seen", 0)
        if self.skip + self.docs_seen:
            ds = ds.skip(self.skip + self.docs_seen)
        self.ds = ds
        log(rank, f"stream {assign['dataset']} config={assign.get('config')} split={assign.get('split', 'train')} "
                  f"skip={self.skip} shuffle={self.shuffle} resume docs_seen={self.docs_seen}")

    def docs(self):
        for row in self.ds:
            idx = self.docs_seen
            self.docs_seen += 1
            if idx % self.world != self.rank:
                continue
            text = row.get("text") or row.get("content") or ""
            if len(text) < 1500:
                continue
            self.cur_doc_index = self.skip + idx            # ABSOLUTE single-stream document index (range mode) -> per-row doc_idx
            yield text

    def state(self):
        return {"docs_seen": self.docs_seen, "skip": self.skip, "doc_range": [self.skip, self.skip + self.docs_seen]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--n-seq", type=int, required=True, help="sequences THIS rank must produce")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk-seqs", type=int, default=512, help="~seqs per shard file")
    ap.add_argument("--max-wins", type=int, default=8, help="cap windows per doc (diversity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="shard dir (on the volume)")
    ap.add_argument("--assignment", required=True, help="json from the driver: mode + file slices")
    ap.add_argument("--exclude-hashes", default="", help="json list of 16-hex span hashes; docs whose 512-aligned window "
                    "starts hash into it are skipped (fresh stores must not re-use eval / existing-store documents)")
    a = ap.parse_args()
    r, L = a.rank, a.seq_len
    excl = set(json.load(open(a.exclude_hashes))) if a.exclude_hashes else set()

    sys.path.insert(0, "/pmx/helpers")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import D_MODEL, MODEL, READ_LAYER
    from mxf.inject import read_resid

    os.makedirs(a.out, exist_ok=True)
    man_path = f"{a.out}/manifest_r{r}.json"
    mu_path = f"{a.out}/musum_r{r}.npy"

    chunks, kept, mu_count, reader_state = [], 0, 0, None
    n_docs, n_skipped_docs = 0, 0
    musum = np.zeros(D_MODEL, dtype=np.float64)
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        if man.get("done"):
            log(r, f"already done ({man['kept']} seqs) — nothing to do")
            return
        chunks, kept, mu_count = man["chunks"], man["kept"], man["mu_count"]
        n_docs, n_skipped_docs = man.get("docs", 0), man.get("skipped_docs", 0)
        reader_state = man["reader_state"]
        musum = np.load(mu_path).astype(np.float64)
        log(r, f"RESUME: {kept}/{a.n_seq} seqs in {len(chunks)} chunks")
    assign = json.load(open(a.assignment))

    t0 = time.time()
    log(r, f"loading {MODEL} (read layer {READ_LAYER}, d={D_MODEL})")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id  # suite convention
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, device_map={"": "cuda:0"}).eval()
    assert model.config.hidden_size == D_MODEL, model.config.hidden_size
    log(r, f"model up in {time.time() - t0:.0f}s (bos={bos}); exclusion set: {len(excl)} span hashes")

    if assign["mode"] == "fffw":
        reader = FffwReader(assign["repo"], assign["ranks"][r], reader_state, r)
    else:
        reader = StreamReader(assign, r, a.world, a.seed, reader_state)

    def write_manifest(done):
        m = {"rank": r, "n_seq_target": a.n_seq, "kept": kept, "mu_count": mu_count,
             "chunks": chunks, "reader_state": reader.state(), "done": done,
             "bos_id": int(bos), "mode": assign["mode"], "docs": n_docs, "skipped_docs": n_skipped_docs}
        with open(man_path + ".tmp", "w") as f:
            json.dump(m, f)
        os.replace(man_path + ".tmp", man_path)

    pending, acts_buf, toks_buf, buf_n = [], [], [], 0
    t0 = time.time()

    def forward(wins):
        nonlocal buf_n, kept, mu_count, musum
        ids = torch.tensor([[bos] + w for w in wins], device="cuda:0")
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        h, _ = read_resid(model, READ_LAYER, batch, pool="all")   # fp32 [B, L+1, d] on GPU
        content = h[:, 1:, :]                                     # drop BOS/sink position 0
        musum += content.reshape(-1, D_MODEL).sum(0).double().cpu().numpy()
        mu_count += content.shape[0] * L
        acts_buf.append(content.to(torch.float16).cpu().numpy())
        toks_buf.append(ids[:, 1:].to(torch.int32).cpu().numpy())
        buf_n += len(wins)
        kept += len(wins)

    def drain():
        nonlocal pending
        while pending and kept < a.n_seq:
            take = min(a.batch, len(pending), a.n_seq - kept)
            forward(pending[:take])
            pending = pending[take:]
        if kept >= a.n_seq:
            pending = []

    def flush():
        nonlocal acts_buf, toks_buf, buf_n
        assert not pending, "flush with pending windows (reader offsets would desync)"
        if buf_n == 0:
            return
        c = len(chunks)
        arr_a = np.concatenate(acts_buf, axis=0)
        arr_t = np.concatenate(toks_buf, axis=0)
        assert arr_a.shape == (buf_n, L, D_MODEL) and arr_t.shape == (buf_n, L), \
            (arr_a.shape, arr_t.shape)
        for arr, ext in ((arr_a, "acts.f16"), (arr_t, "toks.i32")):
            p = f"{a.out}/r{r}_c{c:04d}.{ext}"
            arr.tofile(p + ".tmp")
            os.replace(p + ".tmp", p)
        np.save(mu_path + ".tmp.npy", musum)                      # sums match flushed chunks
        os.replace(mu_path + ".tmp.npy", mu_path)
        chunks.append({"c": c, "n": buf_n})
        write_manifest(done=False)
        el = time.time() - t0
        log(r, f"chunk {c} ({buf_n} seqs) -> {kept}/{a.n_seq} seqs "
               f"({kept * L / max(el, 1):.0f} tok/s, {el / 60:.1f} min)")
        acts_buf, toks_buf, buf_n = [], [], 0

    for text in reader.docs():
        ids = tok(text, add_special_tokens=False, truncation=True,
                  max_length=a.max_wins * L + 8)["input_ids"]
        n_docs += 1
        if excl and any(span_hash(ids[s:]) in excl for s in range(0, len(ids), ALIGN)):
            n_skipped_docs += 1
            continue
        nw = 0
        for s in range(0, len(ids) - L + 1, L):
            pending.append(ids[s:s + L])
            nw += 1
            if nw >= a.max_wins:
                break
        while len(pending) >= a.batch and kept < a.n_seq:
            take = min(a.batch, a.n_seq - kept)   # cap at target: no overshoot past n_seq
            forward(pending[:take])
            pending = pending[take:]
        if kept >= a.n_seq:
            pending = []
            break
        if buf_n >= a.chunk_seqs:
            drain()
            flush()
            if kept >= a.n_seq:
                break
    drain()
    flush()
    if kept < a.n_seq:
        write_manifest(done=False)
        raise RuntimeError(f"rank {r}: corpus exhausted at {kept}/{a.n_seq} seqs")
    write_manifest(done=True)
    log(r, f"DONE {kept} seqs ({kept * L:,} tokens) from {n_docs} docs ({n_skipped_docs} excluded by hash) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
