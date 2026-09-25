"""Modal app: collect a BIG real-activation SFT bank directly (sample-and-emit), no activation store.

    /data/banks/<out_name>/vecs.f16        [N, 5120] fp16 unit directions  = unit(act_L42[p] - mu)
    /data/banks/<out_name>/records.jsonl   {"vec_idx","target_text","family":"realact","ctx_len","W","src"}  (shuffled)
    /data/banks/<out_name>/build_stats.json

One container, `world` single-GPU worker processes (data/collect_bank_worker.py) over DISJOINT FineFineWeb file slices
(same assignment scheme as modal_acts27b.py, different seed => different files). Each worker forwards [BOS]+256-token
windows, samples K positions with context length in [p_lo, p_hi], target = W~U[w_lo,w_hi] tokens ENDING at the
position (the firing token is the last target token). Docs that hash into the acts27b / long-ctx eval stores are
skipped, so the realact eval hold-outs cannot leak in. Crash/24h safe: shards + manifests at chunk boundaries, periodic
volume commits, rerun = resume.

    COLLECT_GPU=H200:8 modal run data/modal_collect_bank.py::run_collect --n-examples 20000000 --out-name realact_short_20m
    COLLECT_SMOKE_GPU=H200:1 modal run data/modal_collect_bank.py::run_smoke
    modal run data/modal_collect_bank.py::run_peek --out-name realact_short_20m
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("COLLECT_APP", "maemm-collect-bank")   # COLLECT_APP=maemm-collect-bank-s2m = an independent deployment
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==5.15.0",
        "accelerate==1.14.0",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
        "datasets",
    )
    .pip_install("flash-linear-attention==0.5.2")   # GDN forward via fla's Triton chunk kernel (forward is fine on Hopper)
    .add_local_file(REPO / "data" / "collect_acts27b_worker.py", "/pmx/collect_acts27b_worker.py")
    .add_local_file(REPO / "data" / "collect_bank_worker.py", "/pmx/collect_bank_worker.py")
    .add_local_file(REPO / "data" / "collect_fullctx_worker.py", "/pmx/collect_fullctx_worker.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

GPU = os.environ.get("COLLECT_GPU", "H200:8")
SMOKE_GPU = os.environ.get("COLLECT_SMOKE_GPU", "H200:1")
DATASET = "m-a-p/FineFineWeb"
FALLBACK = ("HuggingFaceFW/fineweb", "sample-10BT")
def _mxf_config():
    """mxf/config.py loaded standalone (container: the mounted /pmx/helpers/mxf; deploy host: the repo) -> the model profile."""
    import importlib.util
    for p in ("/pmx/helpers/mxf/config.py", str(REPO / "mxf" / "config.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("mxf_config_standalone", p)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            return m
    raise FileNotFoundError("mxf/config.py")


MU_PATH = _mxf_config().WHITEN_MU          # 27B: /data/acts27b/whiten_mu.npy; qwen3-8b: /data/qwen3_8b/whiten_mu_l27.npy
EXCLUDE_STORES = ["/data/acts27b", "/data/acts27b_long", "/data/acts_longctx"]   # any that exist: hash their rows


def _build_assignment(world: int, seed: int, exclude_files=()):
    """Same scheme as modal_acts27b._build_assignment: 2 FineFineWeb jsonl files per domain (seeded), dealt round-robin.
    exclude_files: files already used by another bank (embarrassingly-parallel collections must not share documents)."""
    import random

    from huggingface_hub import HfApi

    try:
        files = [f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".jsonl")]
        files = [f for f in files if f not in set(exclude_files)]
        assert files, "no jsonl files listed"
    except Exception as e:  # noqa
        print(f"[modal] {DATASET} unavailable ({type(e).__name__}: {e}) -- FALLING BACK to {FALLBACK}", flush=True)
        return {"mode": "hfstream", "dataset": FALLBACK[0], "config": FALLBACK[1], "split": "train",
                "note": f"fallback: FineFineWeb failed with {type(e).__name__}: {e}"}
    by_dom = {}
    for f in files:
        by_dom.setdefault(f.split("/")[0], []).append(f)
    rng = random.Random(seed)
    doms = sorted(by_dom)
    for d in doms:
        by_dom[d].sort()
        rng.shuffle(by_dom[d])
    ordered = []
    for tier in range(2):
        tier_files = [by_dom[d][tier] for d in doms if len(by_dom[d]) > tier]
        rng.shuffle(tier_files)
        ordered += tier_files
    return {"mode": "fffw", "repo": DATASET, "dataset": DATASET, "n_domains": len(doms), "n_files": len(ordered),
            "seed": seed, "ranks": [ordered[r::world] for r in range(world)]}


def _exclusion_hashes():
    """sha1[:16] of the first 64 tokens of EVERY row of every existing activation store (acts27b = the realact eval
    hold-out's source; long-ctx stores = realact_long eval). Rows are windows cut at 512/8192-token offsets from doc
    starts; the worker hashes each doc at 512-aligned offsets, so any doc that contributed a row is caught."""
    import hashlib
    import json

    import numpy as np

    out, per_store = set(), {}
    for store in EXCLUDE_STORES:
        meta_p = f"{store}/meta.json"
        if not os.path.exists(meta_p) or not os.path.exists(f"{store}/toks.i32"):
            continue
        meta = json.load(open(meta_p))
        n, L = int(meta["n_seq"]), int(meta["seq_len"])
        tt = np.memmap(f"{store}/toks.i32", np.int32, "r", shape=(n, L))
        hs = {hashlib.sha1(np.ascontiguousarray(tt[i, :64]).astype(np.int32).tobytes()).hexdigest()[:16] for i in range(n)}
        per_store[store] = len(hs)
        out |= hs
    print(f"[modal] exclusion hashes: {len(out)} from {per_store}", flush=True)
    return sorted(out), per_store


def _other_bank_files(bank_name: str):
    """Files used by another (running or finalized) bank, for disjoint parallel collections."""
    import json
    import time
    # A bank that is FINALIZING moves shards/assignment.json -> <bank>/assignment.json (copy + rmtree of shards); a container
    # whose /data mount predates that copy can see neither path for a while -> reload the volume and retry before failing
    # (this exact race killed part p of the 200M collection while part m finalized).
    for attempt in range(12):
        for cand in (f"/data/banks/{bank_name}/shards/assignment.json", f"/data/banks/{bank_name}/assignment.json"):
            if os.path.exists(cand):
                a = json.load(open(cand))
                return [f for r in a.get("ranks", []) for f in r]
        try:
            vol.reload()
        except Exception as e:  # noqa
            print(f"[modal] vol.reload() failed ({e}); retrying", flush=True)
        time.sleep(15)
    raise FileNotFoundError(f"no assignment.json for bank {bank_name} after 3 min of retries")


def _run(out_name: str, n_examples: int, world: int, batch: int, per_window: int, p_lo: int, p_hi: int, w_lo: int,
         w_hi: int, max_wins: int, chunk_examples: int, seed: int, exclude_from: str = "", hf_stream: str = "", w_full: bool = False):
    import json
    import subprocess
    import sys
    import threading
    import time

    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json"):
        raise RuntimeError(f"{out}/build_stats.json exists -- bank already finalized; new --out-name?")
    shards = f"{out}/shards"
    os.makedirs(shards, exist_ok=True)
    os.environ["HF_HOME"] = "/data/hf_cache"
    assert os.path.exists(MU_PATH), f"{MU_PATH} missing (acts27b whiten_mu is the suite's realact convention)"

    assign_path = f"{shards}/assignment.json"
    if os.path.exists(assign_path):
        assign = json.load(open(assign_path))
        assert assign.get("world", world) == world, "resume with a different world size is not supported"
    elif hf_stream:
        # ordered document-RANGE streaming of an HF dataset (no file assignment): hf_stream = JSON {"dataset", "config", "split", "skip"};
        # every rank reads the same ordered stream from single-stream doc index `skip` and keeps the docs with index % world == rank
        # (collect_acts27b_worker.StreamReader), so the collection touches exactly docs [skip, skip + docs_seen): collections with
        # non-overlapping ranges are document-disjoint, and disjoint from the 2M SAE's Ultra-FineWeb span [100k, ~1.8M).
        spec = json.loads(hf_stream)
        assign = {"mode": "hfstream", "dataset": spec["dataset"], "config": spec.get("config"), "split": spec.get("split", "train"),
                  "skip": int(spec.get("skip", 0)), "shuffle": bool(spec.get("shuffle", False)), "world": world, "seed": seed,
                  "note": "document-range mode (ordered stream from `skip`, modulo-`world` sharding by document index)"}
        json.dump(assign, open(assign_path, "w"))
        print(f"[modal] assignment: ORDERED STREAM {assign['dataset']} config={assign['config']} split={assign['split']} "
              f"skip={assign['skip']} shuffle={assign['shuffle']} over {world} ranks", flush=True)
    else:
        excl_files = []
        for other in [x for x in exclude_from.split(",") if x.strip()]:
            excl_files += _other_bank_files(other.strip())
        assign = _build_assignment(world, seed, exclude_files=excl_files)
        assign["world"] = world
        assign["excluded_files_from"] = exclude_from
        assign["n_excluded_files"] = len(set(excl_files))
        json.dump(assign, open(assign_path, "w"))
        print(f"[modal] assignment: {assign.get('n_files')} files over {world} ranks (excluded {len(set(excl_files))} files of {exclude_from!r})", flush=True)
    excl_path = f"{shards}/exclude_hashes.json"
    if not os.path.exists(excl_path):
        hashes, per_store = _exclusion_hashes()
        json.dump(hashes, open(excl_path, "w"))
        json.dump(per_store, open(f"{shards}/exclude_sources.json", "w"))
    vol.commit()

    # make sure the base model is in the shared cache before 8 workers hit it
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    vol.commit()

    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:  # noqa
                print(f"[modal] periodic commit failed: {e}", flush=True)
    threading.Thread(target=committer, daemon=True).start()

    per = [n_examples // world + (1 if r < n_examples % world else 0) for r in range(world)]
    t0 = time.time()
    procs = []
    for r in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(r)
        env["PYTHONPATH"] = "/pmx:/pmx/helpers"
        env["TOKENIZERS_PARALLELISM"] = "false"
        # NOT offline: the reader downloads FineFineWeb jsonl files from the Hub (model loads use local_files_only)
        cmd = [sys.executable, "/pmx/collect_bank_worker.py", "--rank", str(r), "--world", str(world),
               "--n-examples", str(per[r]), "--seq-len", str(p_hi), "--batch", str(batch), "--per-window", str(per_window),
               "--p-lo", str(p_lo), "--p-hi", str(p_hi), "--w-lo", str(w_lo), "--w-hi", str(w_hi), "--max-wins", str(max_wins),
               "--chunk-examples", str(chunk_examples), "--seed", str(seed), "--mu", MU_PATH,
               "--exclude-hashes", excl_path, "--out", shards, "--assignment", assign_path] + (["--w-full"] if w_full else [])
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)
    rcs = [p.wait() for p in procs]
    stop.set()
    vol.commit()
    fails = [r for r in range(world) if not _manifest_done(shards, r)]      # success = a DONE manifest on disk, whatever the exit code
    bad_rc = [r for r, rc in enumerate(rcs) if rc != 0 and r not in fails]
    if bad_rc:
        print(f"[modal] rank(s) {bad_rc} exited non-zero AFTER writing a done manifest (teardown crash) -- ignored", flush=True)
    if fails:
        raise RuntimeError(f"worker rank(s) {fails} failed -- shards persisted; rerun run_collect (same out-name) to resume")
    print(f"[modal] all workers done in {(time.time() - t0) / 60:.1f} min -- finalizing", flush=True)
    _finalize(out, world, n_examples, seed, assign, time.time() - t0)
    vol.commit()


def _manifest_done(shards: str, r: int) -> bool:
    import json
    try:
        m = json.load(open(f"{shards}/manifest_r{r}.json"))
        return bool(m.get("done")) and int(m.get("kept", 0)) >= int(m.get("n_examples_target", 1))
    except Exception:  # noqa
        return False


@app.function(image=image, volumes={"/data": vol}, timeout=8 * 3600, cpu=16, memory=96 * 1024, ephemeral_disk=512 * 1024)
def finalize_only(out_name: str, n_examples: int, seed: int):
    """Publish a bank whose workers all wrote DONE manifests but whose collect call died afterwards (a rank's interpreter teardown
    crashed with 'PyGILState_Release ... runtime state: finalizing'): shards -> vecs.f16 + records.jsonl + build_stats.json +
    doc_ids_used.json (== the tail of _run)."""
    import json, time
    vol.reload()
    out = f"/data/banks/{out_name}"; shards = f"{out}/shards"
    assign = json.load(open(f"{shards}/assignment.json")); world = int(assign["world"])
    fails = [r for r in range(world) if not _manifest_done(shards, r)]
    if fails:
        raise RuntimeError(f"rank(s) {fails} have no done manifest -- resume the collect first")
    t0 = time.time()
    _finalize(out, world, n_examples, seed, assign, float("nan"))
    vol.commit()
    st = json.load(open(f"{out}/build_stats.json"))
    return {"out": out, "n_examples": st["n_examples"], "families": st["families"], "n_docs_used": st.get("n_docs_used"), "finalize_s": time.time() - t0}


def _finalize(out: str, world: int, n_examples: int, seed: int, assign: dict, wall_s: float):
    """Shards -> vecs.f16 (rank/chunk order) + records.jsonl (global vec_idx, shuffled) + build_stats.json."""
    import json
    import shutil
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL, READ_LAYER

    shards = f"{out}/shards"
    mans = []
    for r in range(world):
        m = json.load(open(f"{shards}/manifest_r{r}.json"))
        assert m["done"], f"rank {r} manifest not done -- resume the collect first"
        mans.append(m)
    total = sum(m["kept"] for m in mans)
    assert total >= n_examples, f"collected {total} < target {n_examples}"
    t0 = time.time()
    stage_dir = "/root/finalize_stage"; os.makedirs(stage_dir, exist_ok=True)
    local_vecs = f"{stage_dir}/vecs.f16"                                  # LOCAL disk, then ONE sequential copy to the volume
    vecs = np.memmap(local_vecs, np.float16, "w+", shape=(total, D_MODEL))
    recs, off = [], 0
    ctx_hist, w_hist = np.zeros(2049, np.int64), np.zeros(65, np.int64)
    for r, m in enumerate(mans):
        for ch in m["chunks"]:
            c, n = ch["c"], ch["n"]
            arr = np.fromfile(f"{shards}/r{r}_c{c:04d}.vecs.f16", np.float16).reshape(n, D_MODEL)
            vecs[off : off + n] = arr
            with open(f"{shards}/r{r}_c{c:04d}.records.jsonl") as f:
                for i, line in enumerate(f):
                    rec = json.loads(line)
                    assert rec["vec_idx"] == i
                    rec["vec_idx"] = off + i
                    recs.append(rec)
                    ctx_hist[min(rec["ctx_len"], 2048)] += 1
                    w_hist[min(rec["W"], 64)] += 1
            off += n
    assert off == total and len(recs) == total, (off, total, len(recs))
    vecs.flush(); del vecs
    print(f"[modal] shards assembled locally ({total} rows, {time.time() - t0:.0f}s); copying vecs.f16 to the volume", flush=True)
    shutil.copyfile(local_vecs, f"{out}/vecs.f16.tmp"); os.replace(f"{out}/vecs.f16.tmp", f"{out}/vecs.f16"); os.remove(local_vecs)
    print(f"[modal] vecs.f16 on the volume ({time.time() - t0:.0f}s)", flush=True)
    # spot-check unit norms
    mm = np.memmap(f"{out}/vecs.f16", np.float16, "r", shape=(total, D_MODEL))
    idx = np.random.default_rng(0).choice(total, size=min(4096, total), replace=False)
    norms = np.linalg.norm(mm[np.sort(idx)].astype(np.float32), axis=1)
    assert 0.98 < norms.min() and norms.max() < 1.02, (norms.min(), norms.max())
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    with open(f"{out}/records.jsonl.tmp", "w") as f:
        for i in order:
            f.write(json.dumps(recs[i], ensure_ascii=False) + "\n")
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    stats = {
        "kind": "realact sample-and-emit bank: dir = unit(act_L42[p] - mu_acts27b), target = W-token window ENDING at p "
                "(firing token last), ctx_len = p+1 = tokens the model saw; windows forwarded alone with BOS",
        "model": MODEL, "layer": READ_LAYER, "d": D_MODEL, "n_examples": total, "families": {(recs[0]["family"] if recs else "realact"): total},
        "recipe": ("full_document_forward" if mans[0].get("full_forward") else ("standalone_window_full_context_target" if mans[0].get("w_full") else "standalone_window_suffix_target")),
        "w_full": bool(mans[0].get("w_full", False)), "full_forward": bool(mans[0].get("full_forward", False)), "tail_tokens": mans[0].get("tail_tokens"),
        "ctx_range": mans[0]["ctx_range"], "w_range": mans[0]["w_range"], "seq_len": mans[0]["seq_len"],
        "per_window": mans[0]["per_window"], "world": world, "seed": seed, "dataset": assign.get("dataset"),
        "n_files": assign.get("n_files"), "n_domains": assign.get("n_domains"), "mode": assign.get("mode"),
        "hf_stream": ({"config": assign.get("config"), "split": assign.get("split"), "skip": assign.get("skip"), "shuffle": assign.get("shuffle"),
                       "doc_range": [assign.get("skip", 0), assign.get("skip", 0) + max(int((m.get("reader_state") or {}).get("docs_seen", 0)) for m in mans)],
                       "docs_iterated_per_rank": [int((m.get("reader_state") or {}).get("docs_seen", 0)) for m in mans]}
                      if assign.get("mode") == "hfstream" else None),
        "docs_seen": int(sum(m.get("docs", 0) for m in mans)), "docs_excluded_eval_hash": int(sum(m.get("skipped_docs", 0) for m in mans)),
        "positions_norm_dropped": int(sum(m.get("norm_drop", 0) for m in mans)),
        "norm_median_per_rank": [m.get("norm_median") for m in mans],
        "ctx_len_hist": {str(i): int(v) for i, v in enumerate(ctx_hist) if v}, "W_hist": {str(i): int(v) for i, v in enumerate(w_hist) if v},
        "vec_norm_check": {"min": float(norms.min()), "max": float(norms.max())},
        "wall_s": wall_s, "finalize_s": time.time() - t0, "created": time.time(),
        "files": {"vecs.f16": f"float16 [{total},{D_MODEL}]", "records.jsonl": "shuffled; vec_idx -> row"},
    }
    # ---- document registry: every Ultra-FineWeb document (absolute single-stream index) that contributed >= 1 row ----
    doc_ids = sorted({int(rec["doc_idx"]) for rec in recs if int(rec.get("doc_idx", -1)) >= 0})
    reg = {"bank": out, "dataset": assign.get("dataset"), "config": assign.get("config"), "split": assign.get("split"), "mode": assign.get("mode"),
           "skip": assign.get("skip"), "doc_range_iterated": (stats.get("hf_stream") or {}).get("doc_range"), "n_docs_used": len(doc_ids),
           "n_rows": total, "rows_with_doc_idx": int(sum(1 for rec in recs if int(rec.get("doc_idx", -1)) >= 0)),
           "doc_idx_meaning": "absolute document index in the ordered single stream of dataset/split (0-based); rows carry it as doc_idx",
           "doc_idx": doc_ids}
    json.dump(reg, open(f"{out}/doc_ids_used.json", "w"))
    stats["n_docs_used"] = len(doc_ids)
    stats["doc_ids_file"] = f"{out}/doc_ids_used.json"
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=2)
    shutil.copy(f"{shards}/assignment.json", f"{out}/assignment.json")   # keeps the file list for disjoint follow-up banks
    shutil.rmtree(shards, ignore_errors=True)
    print(f"[modal] FINALIZED {out}: {total} examples, {stats['docs_seen']} docs, {stats['docs_excluded_eval_hash']} excluded, "
          f"vecs.f16 {total * D_MODEL * 2 / 2**30:.1f} GB ({time.time() - t0:.0f}s)", flush=True)


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=86400,
              cpu=32, memory=192 * 1024)
def collect(n_examples: int = 20_000_000, out_name: str = "realact_short_20m", batch: int = 64, per_window: int = 8,
            p_lo: int = 8, p_hi: int = 256, w_lo: int = 8, w_hi: int = 32, max_wins: int = 4, chunk_examples: int = 50_000,
            seed: int = 7, exclude_from: str = "", hf_stream: str = "", w_full: bool = False):
    """hf_stream: JSON {"dataset": "openbmb/Ultra-FineWeb", "config": "default", "split": "en", "skip": 3000000} -> ordered
    document-range streaming instead of the FineFineWeb file assignment (see _run). w_full: target = the FULL context of the
    standalone window (W = ctx_len; precise inversion) instead of a W~U[w_lo,w_hi] window."""
    _collect_body(n_examples, out_name, batch, per_window, p_lo, p_hi, w_lo, w_hi, max_wins, chunk_examples, seed, exclude_from, hf_stream, w_full)


def _collect_body(n_examples, out_name, batch, per_window, p_lo, p_hi, w_lo, w_hi, max_wins, chunk_examples, seed, exclude_from, hf_stream="", w_full=False):
    import subprocess
    n = len([ln for ln in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if ln.strip()])
    _run(out_name, n_examples, world=n, batch=batch, per_window=per_window, p_lo=p_lo, p_hi=p_hi, w_lo=w_lo, w_hi=w_hi,
         max_wins=max_wins, chunk_examples=chunk_examples, seed=seed, exclude_from=exclude_from, hf_stream=hf_stream, w_full=w_full)


def _run_fullctx(out_name: str, n_examples: int, world: int, ctx_lo: int, ctx_hi: int, per_doc: int, batch: int, tail: int,
                 chunk_examples: int, seed: int, hf_stream: str, family: str):
    """FULL-DOCUMENT forwards (collect_fullctx_worker.py): each document is forwarded as [BOS] + its first ctx_hi tokens (no window
    chunking), per_doc positions p with ctx_len = p+1 in [ctx_lo, min(ctx_hi, doc_len)] are sampled, direction = unit(h[p] - mu);
    target_text = the last `tail` tokens ending at p (for logging / transcripts; the RL reward never reads it). Ordered
    document-range streaming only (hf_stream). Same shard / manifest / finalize path as _run."""
    import json
    import subprocess
    import sys
    import threading
    import time

    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json"):
        raise RuntimeError(f"{out}/build_stats.json exists -- bank already finalized; new out_name?")
    shards = f"{out}/shards"
    os.makedirs(shards, exist_ok=True)
    os.environ["HF_HOME"] = "/data/hf_cache"
    assert os.path.exists(MU_PATH), f"{MU_PATH} missing (acts27b whiten_mu is the suite's realact convention)"
    assert hf_stream, "collect_fullctx needs hf_stream (ordered document-range streaming)"
    assign_path = f"{shards}/assignment.json"
    if os.path.exists(assign_path):
        assign = json.load(open(assign_path))
        assert assign.get("world", world) == world, "resume with a different world size is not supported"
    else:
        spec = json.loads(hf_stream)
        assign = {"mode": "hfstream", "dataset": spec["dataset"], "config": spec.get("config"), "split": spec.get("split", "train"),
                  "skip": int(spec.get("skip", 0)), "shuffle": bool(spec.get("shuffle", False)), "world": world, "seed": seed,
                  "note": "document-range mode (ordered stream from `skip`, modulo-`world` sharding by document index); FULL-document forwards"}
        json.dump(assign, open(assign_path, "w"))
        print(f"[modal] assignment: ORDERED STREAM {assign['dataset']} split={assign['split']} skip={assign['skip']} over {world} ranks; "
              f"full forwards ctx [{ctx_lo},{ctx_hi}] x {per_doc}/doc, family {family}", flush=True)
    vol.commit()
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    vol.commit()
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:  # noqa
                print(f"[modal] periodic commit failed: {e}", flush=True)
    threading.Thread(target=committer, daemon=True).start()
    per = [n_examples // world + (1 if r < n_examples % world else 0) for r in range(world)]
    t0 = time.time()
    procs = []
    for r in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(r)
        env["PYTHONPATH"] = "/pmx:/pmx/helpers"
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [sys.executable, "/pmx/collect_fullctx_worker.py", "--rank", str(r), "--world", str(world), "--n-examples", str(per[r]),
               "--ctx-lo", str(ctx_lo), "--ctx-hi", str(ctx_hi), "--per-doc", str(per_doc), "--batch", str(batch), "--tail", str(tail),
               "--chunk-examples", str(chunk_examples), "--seed", str(seed), "--mu", MU_PATH, "--out", shards, "--assignment", assign_path,
               "--family", family]
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)
    rcs = [p.wait() for p in procs]
    stop.set()
    vol.commit()
    fails = [r for r in range(world) if not _manifest_done(shards, r)]      # success = a DONE manifest on disk, whatever the exit code
    bad_rc = [r for r, rc in enumerate(rcs) if rc != 0 and r not in fails]
    if bad_rc:
        print(f"[modal] rank(s) {bad_rc} exited non-zero AFTER writing a done manifest (teardown crash) -- ignored", flush=True)
    if fails:
        raise RuntimeError(f"worker rank(s) {fails} failed -- shards persisted; rerun collect_fullctx (same out_name) to resume")
    print(f"[modal] all workers done in {(time.time() - t0) / 60:.1f} min -- finalizing", flush=True)
    _finalize(out, world, n_examples, seed, assign, time.time() - t0)
    vol.commit()


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=86400,
              cpu=32, memory=192 * 1024)
def collect_fullctx(n_examples: int, out_name: str, ctx_lo: int = 64, ctx_hi: int = 2048, per_doc: int = 8, batch: int = 4, tail: int = 64,
                    chunk_examples: int = 50_000, seed: int = 7, hf_stream: str = "", family: str = "realact_fullctx"):
    """Long-context real-activation rows from FULL-document forwards (see _run_fullctx). e.g. ctx_lo=64 ctx_hi=512 (batch 8) and
    ctx_lo=513 ctx_hi=2048 (batch 4) for the RL pool's two context buckets."""
    import subprocess
    n = len([ln for ln in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if ln.strip()])
    _run_fullctx(out_name, n_examples, n, ctx_lo, ctx_hi, per_doc, batch, tail, chunk_examples, seed, hf_stream, family)


@app.function(image=image, gpu="B200:8", volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=86400,
              cpu=32, memory=192 * 1024)
def collect_b200(n_examples: int = 9_000_000, out_name: str = "realact_short_20m_b", batch: int = 64, per_window: int = 8,
                 p_lo: int = 8, p_hi: int = 256, w_lo: int = 8, w_hi: int = 32, max_wins: int = 4, chunk_examples: int = 50_000,
                 seed: int = 8, exclude_from: str = "realact_short_20m"):
    """Same collector on 8xB200 (embarrassingly parallel with `collect` on H200s): a different seed + the other bank's files
    excluded => disjoint documents. Merge the finished parts with `merge`."""
    _collect_body(n_examples, out_name, batch, per_window, p_lo, p_hi, w_lo, w_hi, max_wins, chunk_examples, seed, exclude_from)


@app.function(image=image, gpu="B200:4", volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=86400,
              cpu=16, memory=96 * 1024)
def collect_b200x4(n_examples: int = 3_000_000, out_name: str = "realact_short_20m_d", batch: int = 64, per_window: int = 8,
                   p_lo: int = 8, p_hi: int = 256, w_lo: int = 8, w_hi: int = 32, max_wins: int = 4, chunk_examples: int = 50_000,
                   seed: int = 10, exclude_from: str = "realact_short_20m,realact_short_20m_b,realact_short_20m_c"):
    """4xB200 part (4-GPU B200 requests schedule far more easily than 8-GPU ones)."""
    _collect_body(n_examples, out_name, batch, per_window, p_lo, p_hi, w_lo, w_hi, max_wins, chunk_examples, seed, exclude_from)


@app.function(image=image, volumes={"/data": vol}, timeout=8 * 3600, cpu=16, memory=64 * 1024,
              ephemeral_disk=int(os.environ.get("MERGE_DISK_GB", "1200")) * 1024)   # volume writes stage on local disk: 50M rows = 512 GB, 200M = 2.1 TB (max 3 TiB)
def merge(out_name: str, parts: str, seed: int = 0):
    """Concatenate finalized part banks (comma list) into /data/banks/<out_name>: vecs.f16 (parts in order), records with
    re-based vec_idx, globally shuffled; build_stats = sums + per-part stats."""
    import json
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL
    t0 = time.time()
    out = f"/data/banks/{out_name}"
    assert not os.path.exists(f"{out}/build_stats.json"), f"{out} already finalized"
    os.makedirs(out, exist_ok=True)
    names = [x.strip() for x in parts.split(",") if x.strip()]
    stats = [json.load(open(f"/data/banks/{n}/build_stats.json")) for n in names]
    total = sum(st["n_examples"] for st in stats)
    vecs = np.memmap(f"{out}/vecs.f16.tmp", np.float16, "w+", shape=(total, D_MODEL))
    # STREAMING: records are re-based and written part by part (no in-RAM list -- 200M dicts would need >150 GB); rows are NOT
    # globally shuffled (sft/pretrain.py length-sorts and shuffles micro-batches itself; rl samples rows uniformly), so
    # records.jsonl is in part order. `seed` is kept for signature compatibility.
    off, n_rec = 0, 0
    with open(f"{out}/records.jsonl.tmp", "w") as fout:
        for n, st in zip(names, stats):
            N = st["n_examples"]
            src = np.memmap(f"/data/banks/{n}/vecs.f16", np.float16, "r", shape=(N, D_MODEL))
            for s0 in range(0, N, 200_000):
                vecs[off + s0 : off + min(N, s0 + 200_000)] = src[s0 : min(N, s0 + 200_000)]
            del src
            with open(f"/data/banks/{n}/records.jsonl") as f:
                for line in f:
                    r = json.loads(line); r["vec_idx"] += off; r["part"] = n
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n"); n_rec += 1
            off += N
            print(f"[merge] {n}: {N} examples appended ({time.time() - t0:.0f}s)", flush=True)
    assert off == total and n_rec == total, (off, n_rec, total)
    vecs.flush(); del vecs
    os.replace(f"{out}/vecs.f16.tmp", f"{out}/vecs.f16")
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    merged = {"kind": "merge of " + ", ".join(names) + " (disjoint FineFineWeb files; see parts)", "n_examples": total,
              "families": {"realact": total}, "parts": {n: st for n, st in zip(names, stats)},
              "docs_seen": sum(st.get("docs_seen", 0) for st in stats),
              "docs_excluded_eval_hash": sum(st.get("docs_excluded_eval_hash", 0) for st in stats),
              "ctx_range": stats[0]["ctx_range"], "w_range": stats[0]["w_range"], "created": time.time(),
              "files": {"vecs.f16": f"float16 [{total},{D_MODEL}]", "records.jsonl": "PART ORDER (not shuffled; loaders shuffle); vec_idx -> row; part = source bank"}}
    json.dump(merged, open(f"{out}/build_stats.json", "w"), indent=2)
    vol.commit()
    print(f"[merge] FINALIZED {out}: {total} examples from {names} in {time.time() - t0:.0f}s", flush=True)


@app.function(image=image, gpu=SMOKE_GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=7200,
              cpu=8, memory=64 * 1024)
def smoke(n_examples: int = 3000, out_name: str = "realact_short_smoke", batch: int = 32, per_window: int = 8):
    """1-GPU end-to-end validation of the exact collect+finalize path on a tiny target."""
    _run(out_name, n_examples, world=1, batch=batch, per_window=per_window, p_lo=8, p_hi=256, w_lo=8, w_hi=32, max_wins=4,
         chunk_examples=1000, seed=7)


@app.function(image=image, timeout=300, cpu=2)
def code_check():
    """What worker code THIS deployment mounts (guards against stale mounts): doc_idx occurrences + sha1 of each worker file."""
    import hashlib
    out = {}
    for f in ("/pmx/collect_bank_worker.py", "/pmx/collect_fullctx_worker.py", "/pmx/collect_acts27b_worker.py"):
        src = open(f).read()
        out[f] = {"doc_idx": src.count("doc_idx"), "cur_doc_index": src.count("cur_doc_index"), "sha1_12": hashlib.sha1(src.encode()).hexdigest()[:12], "bytes": len(src)}
    return out


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=1800, cpu=4)
def peek(out_name: str = "realact_short_smoke", n: int = 6):
    import json
    import sys

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL
    out = f"/data/banks/{out_name}"
    stats = json.load(open(f"{out}/build_stats.json"))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("ctx_len_hist", "W_hist")}, indent=2), flush=True)
    ch = stats["ctx_len_hist"]; wh = stats["W_hist"]
    ctx = np.array([[int(k), v] for k, v in ch.items()]); w = np.array([[int(k), v] for k, v in wh.items()])
    print("ctx_len: min", ctx[:, 0].min(), "max", ctx[:, 0].max(), "mean", float((ctx[:, 0] * ctx[:, 1]).sum() / ctx[:, 1].sum()))
    print("W      : min", w[:, 0].min(), "max", w[:, 0].max(), "mean", float((w[:, 0] * w[:, 1]).sum() / w[:, 1].sum()))
    N = stats["n_examples"]
    mm = np.memmap(f"{out}/vecs.f16", np.float16, "r", shape=(N, D_MODEL))
    with open(f"{out}/records.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            v = mm[rec["vec_idx"]].astype(np.float32)
            print(f"--- {rec['src']} ctx {rec['ctx_len']} W {rec['W']} |v|={np.linalg.norm(v):.3f}: {rec['target_text']!r}", flush=True)


@app.local_entrypoint()
def run_collect(n_examples: int = 20_000_000, out_name: str = "realact_short_20m", batch: int = 64, per_window: int = 8,
                p_lo: int = 8, p_hi: int = 256, w_lo: int = 8, w_hi: int = 32, max_wins: int = 4, seed: int = 7,
                exclude_from: str = ""):
    collect.remote(n_examples=n_examples, out_name=out_name, batch=batch, per_window=per_window, p_lo=p_lo, p_hi=p_hi,
                   w_lo=w_lo, w_hi=w_hi, max_wins=max_wins, seed=seed, exclude_from=exclude_from)


@app.local_entrypoint()
def run_merge(out_name: str, parts: str, seed: int = 0):
    merge.remote(out_name=out_name, parts=parts, seed=seed)


@app.local_entrypoint()
def run_smoke(n_examples: int = 3000, out_name: str = "realact_short_smoke"):
    smoke.remote(n_examples=n_examples, out_name=out_name)


@app.local_entrypoint()
def run_peek(out_name: str = "realact_short_smoke", n: int = 6):
    peek.remote(out_name=out_name, n=n)
