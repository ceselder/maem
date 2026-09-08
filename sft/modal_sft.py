"""Modal app: universal-inverter SFT (train/pretrain.py) on 8xB200, one container per datamix.

The trainer lives at train/pretrain.py in this repo; it is mounted into the container at
/pmx/SL/pretrain.py (its original path on the training box), with mxf/ importable via
PYTHONPATH=/pmx/helpers. Same image/volume as modal_rl.py (no vllm — pretrain is pure HF).
Data lives on the `maemm-data` Volume:
    /data/<bank>            SFT bank: records.jsonl ({"vec_idx", "target_text"} per line)
                            + vecs.f32 or vecs.f16 (N x 5120 f32/f16 memmap, N >= max(vec_idx)+1)
                            --data-dir may be a COMMA-SEPARATED list of such PART banks: every part is staged and
                            pretrain.py consumes them as one virtual bank (100M+ rows no longer fit one merged file)
    /data/hf_cache          HF_HOME (Qwen/Qwen3.6-27B downloads once, persists)
    /data/sft_mix/<run>/    output: run_meta.json + heartbeat + step_* ckpts + final

Every run is an independent spawn — launch MANY datamixes in parallel:
    modal deploy modal_sft.py                     # registers train + the auto-resume supervisor
    modal run modal_sft.py::launch --run-name mix-a --data-dir /data/banks/mix_a
    modal run modal_sft.py::launch --run-name mix-b --data-dir /data/banks/mix_b
Each run saves --n-ckpts (default 14) evenly-spaced intermediate ckpts + final under
/data/sft_mix/<run-name>/ (the per-SFT-step curves; n-ckpts=0 = only `final` = the bug that
lost them last time — the harness refuses it). The supervisor respawns any run whose
heartbeat went stale before `final` exists (24h Modal cap / crash), resuming from the latest
step_N via --init-adapter + --skip-steps (batch order is deterministic) + --wandb-id.
Pause auto-resume: `touch /data/sft_mix/resume_paused` (global) or `<run>/resume_paused`.

Needs Modal secrets `maemm-hf` (HF_TOKEN) and `maemm-wandb` (WANDB_API_KEY).
"""

from pathlib import Path

import os

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)

APP_NAME = os.environ.get("SFT_APP_NAME", "maemm-sft-8xb200")   # SFT_APP_NAME=maemm-sft-fullft = a second, independent deployment
app = modal.App(APP_NAME)

# torch 2.10.0+cu128 == the box venv; cu128 wheels carry sm_100 (B200) kernels. Identical pins to
# modal_rl.py so both trainers see one environment; pretrain needs no vllm (pure HF fwd/bwd).
# --prefix-cache (sft/prefix_cache.py) needs the transformers fork with autograd-safe linear-attention cache writes:
#   SFT_TRANSFORMERS="transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3"
# Default stays the stock pin so existing deployments rebuild nothing.
_SFT_TRANSFORMERS = os.environ.get("SFT_TRANSFORMERS", "transformers==5.15.0")
image = modal.Image.debian_slim(python_version="3.11")
if "git+" in _SFT_TRANSFORMERS:
    image = image.apt_install("git")
image = (
    image
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        _SFT_TRANSFORMERS,
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    # fla: transformers' Qwen3.5/3.6 GatedDeltaNet uses flash-linear-attention's Triton chunk kernels when importable,
    # else a pure-torch fallback (the last5_rp/big_rp SFT runs used the fallback: 10.6% MFU). Same pin as the RL image.
    .pip_install("flash-linear-attention==0.5.2")
    # torchao: float8 training linears for --fp8-base (sft/fp8.py). 0.16.0 is the release built against torch 2.10.0
    # (pytorch/ao#2919 compat table; 0.17+ target 2.11). Pure-python path (casts + torch._scaled_mm), no torchao kernels.
    .pip_install("torchao==0.16.0")
)
# Hopper (H100/H200) opt-in at deploy/run time: SFT_TRITON=3.7.1. fla 0.5.2 refuses its gated chunk_bwd_dqkwg on Triton
# 3.4-3.7.0 on Hopper (fla #640). No vLLM in this image, so the global Triton can simply be upgraded.
_SFT_TRITON = os.environ.get("SFT_TRITON", "")
if _SFT_TRITON:
    image = image.pip_install(f"triton=={_SFT_TRITON}")
# train()'s ephemeral disk (GiB) -- a deploy-time knob. Baked into the image env too, because inside the container
# shutil.disk_usage reports a fictitious 2^33 GiB filesystem, so _preflight can only check the staging total against this.
SFT_DISK_GB = int(os.environ.get("SFT_DISK_GB", "600"))
image = (
    image
    .env({"SFT_DISK_GB": str(SFT_DISK_GB)})
    .add_local_file(REPO / "sft" / "pretrain.py", "/pmx/SL/pretrain.py")
    .add_local_file(REPO / "sft" / "prefix_cache.py", "/pmx/SL/prefix_cache.py")   # --prefix-cache sibling import
    .add_local_file(REPO / "sft" / "fullft.py", "/pmx/SL/fullft.py")               # --full-ft (FSDP2) sibling import
    .add_local_file(REPO / "sft" / "fullft_smoke.py", "/pmx/SL/fullft_smoke.py")   # 2-rank exactness smoke of the full-ft knobs
    .add_local_file(REPO / "sft" / "tp.py", "/pmx/SL/tp.py")                       # --tp (tensor parallel) sibling import
    .add_local_file(REPO / "sft" / "test_tp.py", "/pmx/SL/test_tp.py")             # TP unit test (tiny model, TP-2)
    .add_local_file(REPO / "sft" / "tp_exactness.py", "/pmx/SL/tp_exactness.py")   # TP-8 vs FSDP2 on the real 27B
    .add_local_file(REPO / "sft" / "fp8.py", "/pmx/SL/fp8.py")            # --fp8-base (imported by pretrain.py)
    .add_local_file(REPO / "sft" / "fp8_eval.py", "/pmx/SL/fp8_eval.py")  # fp8 speed/fidelity harness (fp8_eval fn)
    .add_local_file(REPO / "sft" / "fp8_microbench.py", "/pmx/SL/fp8_microbench.py")  # fp8 GEMM/layer/profile bench
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

SFT_ROOT = "/data/sft_mix"
STALE_HEARTBEAT_S = 30 * 60   # live legs touch+commit the heartbeat every <=5 min; 30 min = dead
SPAWN_COOLDOWN_S = 3600       # never double-spawn while a fresh leg pulls image/model (~15-30 min)


VEC_BANK_FILES = (("vecs.f32", 4), ("vecs.f16", 2))   # (filename, bytes per element); same as pretrain.py


def _vec_bank_file(data_dir: str):
    """(filename, itemsize) of the vector bank in data_dir -- vecs.f32 preferred, else vecs.f16."""
    import os

    for fname, itemsize in VEC_BANK_FILES:
        if os.path.exists(f"{data_dir}/{fname}"):
            return fname, itemsize
    raise AssertionError(f"bank incomplete: neither vecs.f32 nor vecs.f16 in {data_dir}")


def _preflight(run_name: str, data_dir: str, n_ckpts: int, resume_from: str, disk_cap_gb: int = 0):
    """Shared guardrails + bank staging for train/smoke. `data_dir` is one bank or a comma-separated list of PART banks
    (pretrain.py --data-dir takes the same list and consumes the parts as one virtual bank -- records and vector rows
    concatenated in the listed order). EVERY part is staged to container-local disk; the total is printed and, when
    disk_cap_gb > 0 (train passes its deploy-time SFT_DISK_GB), refused up front if it cannot fit.
    Returns (save_dir, staged comma list, d_model)."""
    import glob
    import json
    import math
    import os
    import shutil
    import sys
    import time

    # ---- ckpt-cadence guard: n_ckpts=0 means pretrain.py saves ONLY `final` — that is the exact
    # bug that lost the per-SFT-step curves. Refuse it for real runs. ----
    assert n_ckpts > 0, "n_ckpts must be > 0 (0 = only `final`, no per-SFT-step curve — the old bug)"
    assert run_name and "/" not in run_name, f"bad run_name {run_name!r} (path component, no slashes)"

    # ---- resume-support guard: the mounted trainer MUST carry the crash-resume flags the
    # supervisor relies on (--skip-steps fast-forward, --wandb-id, --n-ckpts). Refuse stale code. ----
    with open("/pmx/SL/pretrain.py") as f:
        _src = f.read()
    for _flag in ("--n-ckpts", "--skip-steps", "--wandb-id"):
        assert _flag in _src, f"mounted pretrain.py is stale: {_flag} missing (resume/ckpt support)"
    print("[modal] mounted-trainer check OK (--n-ckpts/--skip-steps/--wandb-id present)", flush=True)

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL  # single source of truth (5120, Qwen/Qwen3.6-27B)

    save_dir = f"{SFT_ROOT}/{run_name}"
    if os.path.exists(f"{save_dir}/final"):
        raise RuntimeError(f"{save_dir}/final exists — run complete; pick a new --run-name")
    prior = sorted(glob.glob(f"{save_dir}/step_*"))
    if prior and not resume_from:
        raise RuntimeError(f"{save_dir} already has {len(prior)} ckpts — a fresh leg would clobber "
                           "them; pick a new --run-name (the supervisor passes resume_from)")

    # single-flight base-model download into the persistent volume (avoids 8 ranks racing)
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    # ---- bank schema guard PER PART, then stage every part onto container-local NVMe: memmap over the volume
    # FUSE mount is the one thing we don't trust, and per-batch random row reads are faster locally. ----
    parts = [p.strip() for p in data_dir.split(",") if p.strip()]
    assert parts, f"empty data_dir {data_dir!r}"
    sizes, n_vecs_total, n_ex_total = [], 0, 0
    for d in parts:
        assert os.path.isdir(d), f"bank not found on volume: {d}"
        assert os.path.exists(f"{d}/records.jsonl"), f"bank incomplete: {d}/records.jsonl missing"
        vec_file, itemsize = _vec_bank_file(d)   # vecs.f32 or vecs.f16 (pretrain.py reads either)
        rec0 = json.loads(open(f"{d}/records.jsonl").readline())
        assert "vec_idx" in rec0 and "target_text" in rec0, f"records.jsonl schema: got {sorted(rec0)}"
        vsize = os.path.getsize(f"{d}/{vec_file}")
        assert vsize % (D_MODEL * itemsize) == 0, \
            f"{d}/{vec_file} = {vsize} B, not a multiple of one {D_MODEL}-x-{itemsize}B row"
        rsize = os.path.getsize(f"{d}/records.jsonl")
        n_ex = None
        if os.path.exists(f"{d}/build_stats.json"):
            n_ex = json.load(open(f"{d}/build_stats.json")).get("n_examples")
            n_ex_total += n_ex or 0
        n_vecs_total += vsize // (D_MODEL * itemsize)
        sizes.append(vsize + rsize)
        print(f"[modal] bank OK: {d}: {vsize // (D_MODEL * itemsize)} vecs x {D_MODEL} ({vec_file}, {vsize / 2**30:.1f} GiB) "
              f"+ records.jsonl ({rsize / 2**30:.2f} GiB)" + (f", n_examples={n_ex}" if n_ex is not None else ""), flush=True)
    total_gb = sum(sizes) / 2**30
    if len(parts) > 1:
        print(f"[modal] {len(parts)} bank parts -> one virtual bank: {n_vecs_total} vectors, build_stats n_examples sum "
              f"{n_ex_total}, {total_gb:.1f} GiB to stage", flush=True)

    # ---- stage: unique local dir per DISTINCT source (the same part listed twice is staged once and passed twice);
    # the total must fit the container's ephemeral disk (deploy-time SFT_DISK_GB) or copytree dies hours in. ----
    local_of = {}
    staged, taken = [], {}
    for d in parts:
        if d in local_of:
            staged.append(local_of[d])
            continue
        base = f"/root/bank_{os.path.basename(d.rstrip('/'))}"
        local = base
        k = 1
        while local in taken:   # a DIFFERENT source with the same basename
            local = f"{base}_{k}"; k += 1
        taken[local] = d
        local_of[d] = local
        staged.append(local)
    size_of = dict(zip(parts, sizes))                     # a source listed twice is copied (and counted) once
    need = sum(sz for d, sz in size_of.items() if not os.path.exists(local_of[d]))
    margin_gb = 20                                        # wandb dir, /tmp shards (--full-ft nontext shard), slack
    du = shutil.disk_usage("/root")
    du_real = du.total < 2**50                            # Modal's sandbox reports a 2^33 GiB filesystem: meaningless
    print(f"[modal] staging {len(local_of)} distinct part(s), {need / 2**30:.1f} GiB to copy (total corpus {total_gb:.1f} GiB, "
          f"+{margin_gb} GiB margin); local disk cap: "
          + (f"{disk_cap_gb} GiB (ephemeral_disk = SFT_DISK_GB at deploy)" if disk_cap_gb else "not enforced for this function")
          + (f"; statvfs says {du.free / 2**30:.0f} GiB free of {du.total / 2**30:.0f} GiB" if du_real
             else "; statvfs reports a fictitious filesystem size and is ignored"), flush=True)
    need_gb = need / 2**30 + margin_gb
    cap_gb = min(disk_cap_gb or float("inf"), du.free / 2**30 if du_real else float("inf"))
    if need_gb > cap_gb:
        raise RuntimeError(f"not enough local disk to stage the bank(s): need {need_gb:.0f} GiB (incl. margin) but the cap is "
                           f"{cap_gb:.0f} GiB -- redeploy with SFT_DISK_GB={int(math.ceil(need_gb / 100.0)) * 100 + 200} or larger")
    t_all = time.time()
    _stage_parallel(local_of, size_of)
    print(f"[modal] bank staged to {','.join(staged)} ({len(staged)} part(s), {total_gb:.1f} GiB, {time.time() - t_all:.0f}s)", flush=True)
    return save_dir, ",".join(staged), D_MODEL


def _copy_range(src: str, dst: str, start: int, end: int, chunk: int = 64 * 2**20, progress=None):
    """Copy bytes [start, end) of src into the SAME offsets of a pre-sized dst (pread/pwrite; parallel-safe)."""
    fi = os.open(src, os.O_RDONLY); fo = os.open(dst, os.O_WRONLY)
    try:
        pos = start
        while pos < end:
            b = os.pread(fi, min(chunk, end - pos), pos)
            if not b:
                raise IOError(f"short read at {pos} of {src}")
            os.pwrite(fo, b, pos); pos += len(b)
            if progress is not None:
                progress[0] += len(b)
    finally:
        os.close(fi); os.close(fo)


def _stage_parallel(local_of: dict, size_of: dict, streams_per_big_file: int = 4, big_file_bytes: int = 4 * 2**30,
                    stall_mib_s: float = 250.0, stall_minutes: float = 5.0):
    """Stage every part with MANY parallel streams: one worker per small file, `streams_per_big_file` byte-range workers per
    file > big_file_bytes (vecs.f16 is 85-512 GiB). Modal's volume read path throttles a single long-running stream to ~20 MiB/s
    after a burst (measured Sep 7: 1.2 GB/s -> 19 MiB/s), while fresh streams read at 400-700 MiB/s, so parallel ranges
    multiply throughput. A monitor prints the aggregate rate every 60 s and raises if it stays below `stall_mib_s` for
    `stall_minutes` (after a 3-min grace) so a throttled leg fails fast and gets respawned instead of crawling for hours."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = []  # (fn, args)
    progress = [0]; total = 0
    for d, local in local_of.items():
        if os.path.exists(local) and os.path.exists(f"{local}/.staged"):
            print(f"[modal] bank part already staged: {local}", flush=True); continue
        os.makedirs(local, exist_ok=True)
        for root, _, files in os.walk(d):
            rel = os.path.relpath(root, d); os.makedirs(os.path.join(local, rel), exist_ok=True)
            for fn in files:
                src = os.path.join(root, fn); dst = os.path.join(local, rel, fn); n = os.path.getsize(src); total += n
                with open(dst, "wb") as f:
                    if n:
                        f.truncate(n)
                if n > big_file_bytes:
                    step = -(-n // streams_per_big_file)
                    for k in range(streams_per_big_file):
                        a, b = k * step, min(n, (k + 1) * step)
                        if a < b:
                            jobs.append((_copy_range, (src, dst, a, b, 64 * 2**20, progress)))
                else:
                    jobs.append((_copy_range, (src, dst, 0, n, 64 * 2**20, progress)))
    if not jobs:
        return
    print(f"[modal] parallel staging: {len(jobs)} streams over {len(local_of)} part(s), {total / 2**30:.1f} GiB", flush=True)
    stop = threading.Event(); t0 = time.time(); hist = []

    def monitor():
        last = 0
        while not stop.wait(60):
            now = progress[0]; rate = (now - last) / 60 / 2**20; last = now; hist.append(rate)
            print(f"[modal] staging {now / 2**30:.0f}/{total / 2**30:.0f} GiB ({100 * now / max(total, 1):.0f}%) at {rate:.0f} MiB/s "
                  f"(avg {now / 2**20 / max(time.time() - t0, 1):.0f} MiB/s, {(time.time() - t0) / 60:.0f} min)", flush=True)
            if len(hist) >= 3 + stall_minutes and all(r < stall_mib_s for r in hist[-int(stall_minutes):]):
                print(f"[modal] STAGING THROTTLED: < {stall_mib_s} MiB/s for {stall_minutes:.0f} min -- aborting this leg (respawn lands on a fresh host)", flush=True)
                os._exit(75)
    th = threading.Thread(target=monitor, daemon=True); th.start()
    with ThreadPoolExecutor(max_workers=min(16, len(jobs))) as ex:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for f in as_completed(futs):
            f.result()
    stop.set()
    for local in local_of.values():
        open(f"{local}/.staged", "w").close()
    print(f"[modal] parallel staging done: {progress[0] / 2**30:.1f} GiB in {time.time() - t0:.0f}s ({progress[0] / 2**20 / max(time.time() - t0, 1):.0f} MiB/s)", flush=True)


def _train_env(backend: str):
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers"
    # pretrain.py is STANDARD cuda-tensor DDP (unlike rl.py's by-design CPU collectives that force
    # gloo) — nccl is the correct default here; gloo is the training-box fallback ("NCCL deadlocks
    # on this box" was a box bug, not a trainer requirement).
    env["DDP_BACKEND"] = backend
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"          # /pmx is a read-only mount; wandb writes to cwd otherwise
    # ranks must load PURELY from the validated cache: 8 concurrent hub re-resolutions returned
    # spurious missing-shard errors on the RL app even though the cached snapshot was complete.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # variable-length padded batches fragment the caching allocator on 178GB B200s (RL app OOM'd on
    # a 24MB alloc with 159GB allocated); expandable_segments is the canonical fix.
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    return env


def _stream(cmd, env, tag):
    import subprocess

    print(f"[{tag}] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


@app.function(
    image=image,
    gpu=os.environ.get("SFT_GPU", "B200:8"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=86400,
    # _preflight stages EVERY listed bank part locally (vecs.f16 = 10 KiB/row + ~190 B/row of records): 23M rows = 224 GiB,
    # 50M = 487 GiB (SFT_DISK_GB=1200), 104M (realact_short_50m_all + parts h..m) = 1012 GiB -> SFT_DISK_GB>=1400,
    # 203M (all 24 realact_short_20m* parts) = 1971 GiB -> SFT_DISK_GB=2600-3000 (Modal's ephemeral-disk ceiling is ~3 TiB)
    ephemeral_disk=SFT_DISK_GB * 1024,
    memory=256 * 1024,
)
def train(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
          batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
          backend: str = "nccl", extra_args: str = "",
          resume_from: str = "", skip_steps: int = 0, wandb_id: str = ""):
    # One SFT run (one datamix). run_name/data_dir/n_ckpts parameterize independent parallel
    # spawns. batch_size/lr/max_seq: 0 = TrainConfig defaults (64 / 3e-5 / 192). extra_args:
    # whitespace-split, appended last (argparse last-wins), e.g. "--compile --log-steps 50".
    # resume_from/skip_steps/wandb_id: supervisor crash-resume — --init-adapter <ckpt> +
    # --skip-steps fast-forward through the deterministic batch order + same wandb run.
    import glob
    import json
    import os
    import secrets as pysecrets
    import string
    import threading
    import time

    # one bank or a comma-separated list of parts (each may be volume-relative); the normalized list is what run_meta
    # records, so the supervisor's resume legs stage exactly the same parts in the same order
    data_dir = ",".join(p if p.startswith("/") else f"/data/{p}" for p in (q.strip() for q in data_dir.split(",")) if p)
    save_dir, local_bank, _ = _preflight(run_name, data_dir, n_ckpts, resume_from,
                                         disk_cap_gb=int(os.environ.get("SFT_DISK_GB", "0")))   # baked into the image at deploy
    os.makedirs(save_dir, exist_ok=True)

    # ---- run_meta.json: everything the supervisor needs to respawn this run faithfully. Written
    # once on the first leg (wandb id minted here so every leg logs to ONE wandb run). ----
    meta_path = f"{save_dir}/run_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        wandb_id = wandb_id or meta.get("wandb_id", "")
    else:
        wandb_id = wandb_id or "".join(
            pysecrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        meta = {"run_name": run_name, "data_dir": data_dir, "n_ckpts": n_ckpts, "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "max_seq": max_seq, "backend": backend,
                "extra_args": extra_args, "wandb_id": wandb_id, "created": time.time(), "legs": []}
    meta["legs"].append({"ts": time.time(), "resume_from": resume_from, "skip_steps": skip_steps})
    json.dump(meta, open(meta_path, "w"), indent=2)

    # ---- heartbeat + committer: touch <run>/heartbeat every 60s (supervisor aliveness signal,
    # started BEFORE the slow model download so fresh legs read as alive immediately) and
    # vol.commit right after every new ckpt lands (else every 300s) so saves are durable even if
    # the container dies mid-run. ----
    def _committer():
        last_commit, last_ckpts = 0.0, -1
        while True:
            try:
                with open(f"{save_dir}/heartbeat", "w") as f:
                    f.write(str(time.time()))
                n = len(glob.glob(f"{save_dir}/step_*")) + len(glob.glob(f"{save_dir}/final"))
                if n != last_ckpts or time.time() - last_commit > 300:
                    vol.commit()
                    last_commit, last_ckpts = time.time(), n
            except Exception as e:
                print(f"[modal] heartbeat/commit failed: {e}", flush=True)
            time.sleep(60)

    threading.Thread(target=_committer, daemon=True).start()
    vol.commit()  # publish run_meta + heartbeat now: the supervisor must see this leg exists

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=8", "SL/pretrain.py",
        "--data-dir", local_bank,
        "--save-dir", save_dir,
        "--run-name", run_name,
        "--n-ckpts", str(n_ckpts),
        "--epochs", str(epochs),
        "--wandb-id", wandb_id,
    ]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if lr:
        cmd += ["--lr", str(lr)]
    if max_seq:
        cmd += ["--max-seq", str(max_seq)]
    if resume_from:
        cmd += ["--init-adapter", resume_from, "--skip-steps", str(skip_steps)]
    if extra_args:
        cmd += extra_args.split()

    rc = _stream(cmd, _train_env(backend), "modal")
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"torchrun exited rc={rc} (backend={backend}) — supervisor will resume")
    assert os.path.exists(f"{save_dir}/final"), "rc=0 but no final ckpt — trainer exited early?"
    print(f"[modal] SFT run {run_name} COMPLETE -> {save_dir}/final", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("SFT_SMOKE_GPU", "B200:1"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=7200,
    memory=int(os.environ.get("SFT_SMOKE_MEM_GB", "64")) * 1024,
)
def smoke(data_dir: str, n_records: int = 256, batch_size: int = 8, extra_args: str = "", nproc: int = 1,
          save_dir: str = "/tmp/smoke_ckpt", reload_check: bool = False):
    """1xB200 pipeline validation: pre-warms /data/hf_cache, carves a tiny bank (first n_records
    of records.jsonl + the full vecs.f32 / vecs.f16) and runs world=1 SFT over it (no wandb, ckpts
    to /tmp, including the --n-ckpts intermediate-save path). extra_args reach pretrain.py verbatim,
    e.g. "--compile --grad-ckpt 0 --autocast-bf16 --head-on-labels --parity-check --log-steps 10"."""
    import os
    import shutil

    data_dir = ",".join(p if p.startswith("/") else f"/data/{p}" for p in (q.strip() for q in data_dir.split(",")) if p)
    _, local_bank, _ = _preflight("smoke", data_dir, n_ckpts=2, resume_from="")
    # one tiny bank per staged part (first n_records of its records.jsonl + its full vecs file) -> a comma list exercises
    # pretrain.py's multi-part path end to end (a single part keeps the legacy /root/bank_smoke name)
    parts = local_bank.split(",")
    tinies = []
    for k, part in enumerate(parts):
        vec_file, _ = _vec_bank_file(part)
        tiny = "/root/bank_smoke" if len(parts) == 1 else f"/root/bank_smoke_{k}"
        if not os.path.exists(tiny):
            os.makedirs(tiny)
            with open(f"{part}/records.jsonl") as fin, open(f"{tiny}/records.jsonl", "w") as fout:
                for i, line in enumerate(fin):
                    if i >= n_records:
                        break
                    fout.write(line)
            shutil.copy(f"{part}/{vec_file}", f"{tiny}/{vec_file}")
        tinies.append(tiny)
        print(f"[modal-smoke] tiny bank {k}: first {n_records} records of {part} -> {tiny}", flush=True)
    tiny = ",".join(tinies)

    launcher = ["torchrun", "--standalone", f"--nproc_per_node={nproc}"] if nproc > 1 else ["python"]
    cmd = launcher + [
        "SL/pretrain.py",
        "--data-dir", tiny,
        "--save-dir", save_dir,
        "--run-name", "smoke",
        "--n-ckpts", "2",
        "--epochs", "1",
        "--batch-size", str(batch_size),
        "--no-wandb",
    ]
    if extra_args:
        cmd += extra_args.split()   # e.g. "--compile --grad-ckpt 0 --autocast-bf16 --log-steps 10" (speed test)
    rc = _stream(cmd, _train_env("nccl"), "modal-smoke")
    if rc != 0:
        raise RuntimeError(f"smoke exited rc={rc}")
    saved = sorted(os.listdir(save_dir))
    assert "final" in saved and any(s.startswith("step_") for s in saved), f"ckpt cadence broken: {saved}"
    print(f"[modal-smoke] OK — saved {saved}", flush=True)
    if save_dir.startswith("/data"):
        vol.commit()
    if reload_check:   # --full-ft: the checkpoint must load as a plain HF model and give a sane loss on the tiny bank
        chk = ["python", "-c", f"""
import json, os, sys, time, torch
sys.path.insert(0, '/pmx/helpers'); os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer
from mxf.config import MODEL
from mxf.prompts import build_sft_ids
ck = '{save_dir}/final'
assert os.path.exists(ck + '/SAVE_DONE'), 'SAVE_DONE missing'
print('[reload] SAVE_DONE', json.load(open(ck + '/SAVE_DONE')), flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={{'': 'cuda:0'}})
print(f'[reload] loaded {{type(m).__name__}} from {{ck}} in {{time.time() - t0:.0f}}s', flush=True)
recs = [json.loads(l) for _, l in zip(range(8), open('{tinies[0]}/records.jsonl'))]
losses = []
for r in recs:
    ids, labs, pos = build_sft_ids(tok, r['target_text'])
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device='cuda:0'), labels=torch.tensor([labs], device='cuda:0'))
    losses.append(out.loss.item())
print(f'[reload] CE on 8 tiny-bank examples (no injection): {{sum(losses) / len(losses):.3f}} (each {{[round(x, 2) for x in losses]}})', flush=True)
assert all(torch.isfinite(torch.tensor(losses))), 'non-finite loss after reload'
print('RELOAD_OK', flush=True)
"""]
        rc = _stream(chk, _train_env("nccl"), "modal-smoke-reload")
        if rc != 0:
            raise RuntimeError(f"reload check exited rc={rc}")


@app.function(
    image=image,
    gpu=os.environ.get("SFT_SMOKE_GPU", "B200:1"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=3 * 3600,
)
def fp8_eval(data_dir: str, n_records: int = 6400, extra_args: str = "", script: str = "SL/fp8_eval.py"):
    """1 GPU: sft/fp8_eval.py -- bf16 vs --fp8-base speed (meter TFLOP/s, examples/s, peak mem), same-weights logit
    KL / top-1, and N-step loss curves from one LoRA init on one batch order. Tiny bank = first n_records (the harness
    needs train-steps*batch + kl batches). extra_args are appended (e.g. "--train-steps 300 --bench-batch-sizes 16,32").
    script=SL/fp8_microbench.py runs the GEMM/layer/profiler microbenchmark instead (no bank; synthetic tokens).
    Result JSON is printed and copied to /data/sft_mix/_fp8_eval/<utc>.json."""
    import os
    import shutil
    import time

    out = "/tmp/fp8_eval.json"
    if script.endswith("fp8_microbench.py"):
        os.environ["HF_HOME"] = "/data/hf_cache"  # --profile-model loads the (already cached) base model
        cmd = ["python", script, "--out", out]
    else:
        if not data_dir.startswith("/"):
            data_dir = f"/data/{data_dir}"
        _, local_bank, _ = _preflight("smoke", data_dir, n_ckpts=2, resume_from="")
        local_bank = local_bank.split(",")[0]   # the harness carves its tiny bank from the first part
        tiny = "/root/bank_fp8eval"
        if not os.path.exists(tiny):
            os.makedirs(tiny)
            with open(f"{local_bank}/records.jsonl") as fin, open(f"{tiny}/records.jsonl", "w") as fout:
                for i, line in enumerate(fin):
                    if i >= n_records:
                        break
                    fout.write(line)
            shutil.copy(f"{local_bank}/vecs.f32", f"{tiny}/vecs.f32")
        print(f"[modal-fp8-eval] tiny bank: first {n_records} records", flush=True)
        cmd = ["python", script, "--data-dir", tiny, "--out", out]
    if extra_args:
        cmd += extra_args.split()
    rc = _stream(cmd, _train_env("nccl"), "modal-fp8-eval")
    if os.path.exists(out):
        dst = f"{SFT_ROOT}/_fp8_eval/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(out, dst)
        vol.commit()
        print(f"[modal-fp8-eval] result JSON -> {dst}", flush=True)
        print("FP8_EVAL_JSON_BEGIN"); print(open(out).read()); print("FP8_EVAL_JSON_END", flush=True)
    if rc != 0:
        raise RuntimeError(f"fp8_eval exited rc={rc}")


@app.function(
    image=image,
    gpu=os.environ.get("SFT_GPU", "B200:8"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=int(os.environ.get("SFT_BENCH_TIMEOUT_S", str(55 * 60))),   # hard cap: the container MUST exit
    memory=256 * 1024,
)
def fullft_bench(data_dir: str, configs: str = "32x64;64x32;128x16", n_records: int = 132000,
                 stop_after_step: int = 6, max_seq: int = 160, lr: float = 5e-5,
                 common: str = "--full-ft --prefix-cache --grad-ckpt 0 --pad-multiple 1 --length-bucket --log-steps 1",
                 oom_fallback: str = "--prefix-accum 2", deadline_min: float = 40.0) -> dict:
    """SPEED-ONLY bench of --full-ft micro-batch configs on ONE 8-GPU container (deploy as a separate app, e.g.
    SFT_APP_NAME=maemm-sft-fullft-bench SFT_SUPERVISOR=0). Stages the (tiny) bank, replicates records.jsonl to
    n_records rows (same vec_idx -> the vecs file stays tiny; speed depends only on the target-length distribution),
    then runs each `BxGA[:extra flags]` config (';'-separated) as its own world-8 torchrun (effective batch = 8*B*GA)
    with --no-wandb and ckpts under /tmp, and SIGTERMs it after `stop_after_step` logged optimizer steps so no 54 GB
    final save happens. A config that OOMs is re-run once with `oom_fallback` appended (--grad-ckpt 1 is not
    combinable with --prefix-cache). Nothing is written under /data/sft_mix. Returns the parsed per-step metrics and
    prints a JSON summary between FULLFT_BENCH_JSON_BEGIN / FULLFT_BENCH_JSON_END."""
    import json
    import os
    import re
    import shutil
    import signal
    import subprocess
    import time

    t_start = time.time()
    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    _, local_bank, _ = _preflight("_bench_fullft", data_dir, n_ckpts=1, resume_from="")   # stages only; no save_dir is created
    local_bank = local_bank.split(",")[0]   # the bench replicates the first part's records
    vec_file, _ = _vec_bank_file(local_bank)
    bank = f"/root/bank_bench_{n_records}"
    lines = open(f"{local_bank}/records.jsonl").read().splitlines()
    if not os.path.exists(bank):
        os.makedirs(bank)
        with open(f"{bank}/records.jsonl", "w") as f:
            for i in range(n_records):
                f.write(lines[i % len(lines)] + "\n")
        shutil.copy(f"{local_bank}/{vec_file}", f"{bank}/{vec_file}")
    print(f"[bench] {len(lines)} source records replicated to {n_records} rows at {bank}", flush=True)

    step_re = re.compile(r"ep(\d+) step (\d+)/(\d+) loss ([\d.]+) \| (\d+) TFLOP/s MFU (\d+)% \| ([\d.]+) ex/s (\d+) tok/s "
                         r"pad ([\d.]+)% \(([\d.]+) s/step, (\d+) micro\) \| peak ([\d.]+) GB")
    env = _train_env("nccl")

    def gpu_mem_used():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            return [int(x) for x in out.split()] if out.strip() else []
        except Exception as e:  # noqa -- nvidia-smi missing: memory check is best-effort
            print(f"[bench] nvidia-smi unavailable ({e!r})", flush=True)
            return []

    def kill_stragglers():
        """SIGKILL any leftover torchrun / pretrain.py worker (pure /proc scan: debian_slim has no pkill)."""
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            except OSError:
                continue
            if b"SL/pretrain.py" in cmd or b"torchrun" in cmd:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass

    def wait_gpus_free(timeout_s=120):
        t0 = time.time()
        used = gpu_mem_used()
        while used and max(used) > 2048 and time.time() - t0 < timeout_s:
            time.sleep(5)
            used = gpu_mem_used()
        return used

    def run_one(tag, B, GA, extra):
        save_dir = f"/tmp/bench_{tag}"
        cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "SL/pretrain.py",
               "--data-dir", bank, "--save-dir", save_dir, "--run-name", f"_bench_{tag}",
               "--n-ckpts", "1", "--epochs", "1", "--batch-size", str(B), "--grad-accum", str(GA),
               "--lr", str(lr), "--max-seq", str(max_seq), "--no-wandb"] + common.split() + extra.split()
        print(f"[bench {tag}] launching:", " ".join(cmd), flush=True)
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
        steps, notes, oom, stopped = [], [], False, False
        for line in p.stdout:
            print(line, end="", flush=True)
            if "CUDA out of memory" in line or "OutOfMemoryError" in line:
                oom = True
            if (line.startswith("[fullft]") or line.startswith("[pretrain]") or line.startswith("steps_total")
                    or line.startswith("[prof]") or line.startswith("[mem]") or line.startswith("[optim]")
                    or "records," in line[:40]):
                notes.append(line.rstrip())
            m = step_re.search(line)
            if m:
                steps.append(dict(step=int(m[2]), steps_total=int(m[3]), loss=float(m[4]), tflops=float(m[5]),
                                  mfu=int(m[6]), ex_s=float(m[7]), tok_s=float(m[8]), pad_frac=float(m[9]) / 100,
                                  dt=float(m[10]), micro=int(m[11]), peak_gb=float(m[12]), profiled="PROFILED" in line))
                if int(m[2]) >= stop_after_step:
                    stopped = True
                    print(f"[bench {tag}] {len(steps)} steps logged -> SIGTERM torchrun (skip the final save)", flush=True)
                    os.killpg(p.pid, signal.SIGTERM)
                    break
        try:
            rest, _ = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            rest, _ = p.communicate()
        if rest:
            oom = oom or ("CUDA out of memory" in rest or "OutOfMemoryError" in rest)
            print(rest[-3000:], flush=True)
        kill_stragglers()
        used = wait_gpus_free()
        shutil.rmtree(save_dir, ignore_errors=True)
        meas = [s for s in steps if s["step"] >= 2 and not s.get("profiled")] or steps[1:] or steps
        summ = {}
        if meas:
            # --tp N: every rank runs the same examples (no data parallelism) -> examples/step = B * GA, not 8 * B * GA
            tp_all = re.findall(r"--tp\s+(\d+)", common + " " + extra)   # argparse last-wins: extra may override common
            dp = 1 if (tp_all and int(tp_all[-1]) > 0) else 8
            ex_per_step = dp * B * GA
            mean_dt = sum(s["dt"] for s in meas) / len(meas)
            summ = dict(n_measured=len(meas), mean_dt=mean_dt, median_dt=sorted(s["dt"] for s in meas)[len(meas) // 2],
                        ex_s=ex_per_step / mean_dt, mean_ex_s_printed=sum(s["ex_s"] for s in meas) / len(meas),
                        mean_tok_s=sum(s["tok_s"] for s in meas) / len(meas),
                        mean_mfu=sum(s["mfu"] for s in meas) / len(meas), mean_pad=sum(s["pad_frac"] for s in meas) / len(meas),
                        peak_gb=max(s["peak_gb"] for s in steps))
        tp_all = re.findall(r"--tp\s+(\d+)", common + " " + extra)
        tp_deg = int(tp_all[-1]) if tp_all else 0
        res = dict(tag=tag, batch_size=B, grad_accum=GA, eff_batch=(1 if tp_deg > 0 else 8) * B * GA,
                   tp=tp_deg, extra=extra, cmd=" ".join(cmd),
                   rc=p.returncode, stopped_by_bench=stopped, oom=oom, wall_s=time.time() - t0,
                   gpu_mem_used_after_mib=used, steps=steps, notes=notes, summary=summ)
        print(f"[bench {tag}] done: rc={p.returncode} stopped={stopped} oom={oom} wall={res['wall_s']:.0f}s "
              f"summary={json.dumps(summ)} gpu_mem_after={used}", flush=True)
        return res

    results = {"data_dir": data_dir, "n_records": n_records, "common": common, "max_seq": max_seq, "lr": lr,
               "stop_after_step": stop_after_step, "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "?",
               "runs": []}
    for spec in [s.strip() for s in configs.split(";") if s.strip()]:
        bga, _, extra = spec.partition(":")
        B, GA = (int(x) for x in bga.lower().split("x"))
        elapsed_min = (time.time() - t_start) / 60
        if elapsed_min > deadline_min:
            print(f"[bench] {elapsed_min:.0f} min elapsed > deadline {deadline_min} min -> skipping {spec}", flush=True)
            results["runs"].append(dict(tag=f"mb{B}_ga{GA}", batch_size=B, grad_accum=GA, skipped="deadline"))
            continue
        r = run_one(f"mb{B}_ga{GA}", B, GA, extra.strip())
        results["runs"].append(r)
        if r["oom"] and oom_fallback and oom_fallback not in extra and (time.time() - t_start) / 60 < deadline_min:
            print(f"[bench] {spec} OOM -> retrying with {oom_fallback!r}", flush=True)
            results["runs"].append(run_one(f"mb{B}_ga{GA}_fb", B, GA, (extra + " " + oom_fallback).strip()))
    results["wall_min"] = (time.time() - t_start) / 60
    print("FULLFT_BENCH_JSON_BEGIN"); print(json.dumps(results)); print("FULLFT_BENCH_JSON_END", flush=True)
    return results


@app.function(
    image=image,
    gpu=os.environ.get("SFT_FFT_SMOKE_GPU", "L4:2"),
    volumes={"/data": vol},
    timeout=int(os.environ.get("SFT_FFT_SMOKE_TIMEOUT_S", str(30 * 60))),
    memory=32 * 1024,
)
def fullft_smoke(args: str = "") -> int:
    """2-rank FSDP2 exactness smoke of the --full-ft speed knobs on a TINY random Qwen3.5 (sft/fullft_smoke.py): shared
    prefix accumulator, suffix checkpointing, label head, persistent unshard, prefetch, low-bit optimizers -- gradients
    and multi-step params compared against the per-micro-batch reference. Cheap GPUs (L4:2 default); no model download."""
    import subprocess

    env = _train_env("nccl")
    cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "SL/fullft_smoke.py"] + args.split()
    rc = _stream(cmd, env, "fft-smoke")
    print(f"[fft-smoke] rc={rc}", flush=True)
    return rc


@app.local_entrypoint()
def run_fullft_smoke(args: str = ""):
    print(fullft_smoke.remote(args))


@app.function(
    image=image,
    gpu=os.environ.get("SFT_GPU", "B200:8"),
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb")],
    timeout=int(os.environ.get("SFT_TPX_TIMEOUT_S", str(75 * 60))),   # hard cap: the container MUST exit
    memory=256 * 1024,
    ephemeral_disk=600 * 1024,   # the fp32 reference gradient dump is 108 GB (Modal minimum for the knob is 512 GiB)
)
def tp_exactness(data_dir: str = "/data/banks/realact_short_smoke3", n_records: int = 8192, batch_size: int = 32,
                 steps: int = 20, lr: float = 1e-5, max_seq: int = 160, unit_test: bool = True,
                 legs: str = "fsdp,fsdp-rerun,tp,tp-nofused") -> dict:
    """TP-8 vs FSDP2 exactness on the real 27B (sft/tp_exactness.py): two world-8 torchruns in ONE container over the same
    DP-dealt micro-batch window of the (tiny) bank -- FSDP2 first (dumps fp32 reference grads + a `steps`-step loss
    trace), then TP-8 (gathers its grads, compares, runs the same trace). Optionally runs the TP-2 unit test first
    (tiny random model, nccl). Nothing is written under /data. Prints the summary JSON between TP_EXACTNESS_JSON_BEGIN/END."""
    import json
    import os
    import shutil
    import subprocess
    import time

    t_start = time.time()
    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    _, local_bank, _ = _preflight("_bench_tp_exactness", data_dir, n_ckpts=1, resume_from="")   # stages only
    local_bank = local_bank.split(",")[0]
    vec_file, _ = _vec_bank_file(local_bank)
    bank = f"/root/bank_tpx_{n_records}"
    if not os.path.exists(bank):
        os.makedirs(bank)
        lines = open(f"{local_bank}/records.jsonl").read().splitlines()
        with open(f"{bank}/records.jsonl", "w") as f:
            for i in range(n_records):
                f.write(lines[i % len(lines)] + "\n")
        shutil.copy(f"{local_bank}/{vec_file}", f"{bank}/{vec_file}")
    env = _train_env("nccl")
    out = "/tmp/tp_exact"
    results = {"data_dir": data_dir, "n_records": n_records, "batch_size": batch_size, "steps": steps, "lr": lr, "rc": {}}
    if unit_test:
        # GPU unit tests: fp32 tiny model; on GPU fla's chunk backward accumulates dg with atomics, so the g-path params
        # (A_log, dt_bias, in_proj_a) land at rel ~3e-3 rather than the CPU/gloo 1e-4 -- read the cos / loss lines
        for tag_, args_ in (("kv2", ["--kv-heads", "2"]), ("kv1", ["--kv-heads", "1"]),
                            ("kv2-fused-d1024", ["--kv-heads", "2", "--fused-norm", "1", "--hidden", "1024"])):
            rc = _stream(["torchrun", "--standalone", "--nproc_per_node=2", "SL/test_tp.py"] + args_, env, f"tp-unit-{tag_}")
            results["rc"][f"unit_{tag_}"] = rc
    # four world-8 legs over the SAME window: fsdp (reference dump + trace), fsdp again (noise floor: compare vs the dump),
    # tp with the fused norm (production), tp with HF's eager norm (isolates the fused-norm contribution)
    all_legs = {"fsdp": [], "fsdp-rerun": ["--compare-only", "1", "--tag", "fsdp-rerun"],
                # same objective, different GEMM shapes (each rank's micro-batch as 2 halves): the bf16 re-rounding noise floor
                "fsdp-split2": ["--compare-only", "1", "--split", "2", "--tag", "fsdp-split2"],
                "tp": ["--fused-norm", "1"], "tp-nofused": ["--fused-norm", "0", "--tag", "tp-nofused"],
                "tp-fp32reduce": ["--fused-norm", "1", "--reduce-fp32", "1", "--tag", "tp-fp32reduce"],
                "tp-nofused-fp32reduce": ["--fused-norm", "0", "--reduce-fp32", "1", "--tag", "tp-nofused-fp32reduce"]}
    leg_list = [(t.strip(), all_legs[t.strip()]) for t in legs.split(",") if t.strip()]
    for tag_, extra_ in leg_list:
        mode = "fsdp" if tag_.startswith("fsdp") else "tp"
        cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "SL/tp_exactness.py", "--mode", mode, "--data-dir", bank,
               "--out", out, "--batch-size", str(batch_size), "--steps", str(steps), "--lr", str(lr), "--max-seq", str(max_seq)] + extra_
        rc = _stream(cmd, env, f"tpx-{tag_}")
        results["rc"][tag_] = rc
        if rc != 0:
            print(f"[tpx] {tag_} exited rc={rc}; aborting", flush=True)
            break
        if (time.time() - t_start) / 60 > 60:
            print("[tpx] over 60 min; stopping after this leg", flush=True)
            break
    results["summaries"] = {}
    for tag_ in [t for t, _ in leg_list if t != "fsdp"]:
        fn = f"{out}/{tag_}_exactness.json"
        if os.path.exists(fn):
            r = json.load(open(fn))
            results["summaries"][tag_] = r["summary"]
            results.setdefault("result", {})[tag_] = r
    results["wall_min"] = (time.time() - t_start) / 60
    print("TP_EXACTNESS_JSON_BEGIN"); print(json.dumps({k: v for k, v in results.items() if k != "result"})); print("TP_EXACTNESS_JSON_END", flush=True)
    return {k: v for k, v in results.items() if k != "result"}


@app.local_entrypoint()
def run_tp_exactness(data_dir: str = "/data/banks/realact_short_smoke3", n_records: int = 8192, batch_size: int = 32,
                     steps: int = 20, lr: float = 1e-5, max_seq: int = 160, unit_test: bool = True):
    """Spawn tp_exactness on the DEPLOYED app (detached; prints the call id)."""
    call = modal.Function.from_name(APP_NAME, "tp_exactness").spawn(
        data_dir=data_dir, n_records=n_records, batch_size=batch_size, steps=steps, lr=lr, max_seq=max_seq, unit_test=unit_test)
    print(f"spawned tp_exactness on {APP_NAME}: {call.object_id}")


@app.function(image=image, timeout=600, cpu=2)
def env_check():
    """CPU-only image check: pins + the torchao float8 imports pretrain.py --fp8-base relies on."""
    import importlib
    import sys

    import torch
    print(f"python {sys.version.split()[0]} torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
    for pkg in ("torchao", "transformers", "peft", "accelerate", "fla", "triton"):
        try:
            print(f"{pkg} {getattr(importlib.import_module(pkg), '__version__', '?')}")
        except Exception as e:  # noqa
            print(f"{pkg} IMPORT FAILED: {e!r}")
    from torchao.float8 import Float8LinearConfig
    from torchao.float8.float8_linear import Float8Linear
    from torchao.float8.float8_linear_utils import swap_linear_layers
    sys.path.insert(0, "/pmx/SL")
    import fp8
    print(f"torchao float8 API OK: {Float8Linear.__name__}, {swap_linear_layers.__name__}")
    print(f"fp8.py OK: recipe default={fp8.DEFAULT_RECIPE} MIN_DIM={fp8.MIN_DIM} "
          f"config={Float8LinearConfig.from_recipe_name(fp8.DEFAULT_RECIPE)}")
    print("ENV_CHECK_OK", flush=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=7200,
    cpu=8,
)
def prewarm():
    """CPU-only: download the base model into the persistent HF cache on the volume."""
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal-prewarm] base model cached ({time.time() - t0:.0f}s)", flush=True)


# ---- auto-resume supervisor: Modal caps functions at 24h; big banks can outrun that, and crashed
# legs must not silently stall a datamix sweep. Every 20 min: for each /data/sft_mix/<run>/ with a
# run_meta.json and no `final`, if the heartbeat is stale (live legs touch+commit it every <=5 min)
# and no spawn is cooling down, respawn `train` in RESUME mode: --init-adapter <latest step_N> +
# --skip-steps N+1 (step_N is saved after batch N of the deterministic order) + the same wandb id.
# No step ckpts yet -> clean fresh respawn. Unlike the RL supervisor this is fully generic across
# N parallel runs: per-run heartbeat/state files, nothing hardcoded. ----
@app.function(schedule=modal.Period(minutes=20) if os.environ.get("SFT_SUPERVISOR", "1") == "1" else None,
              volumes={"/data": vol}, timeout=600)   # SFT_SUPERVISOR=0 at deploy: a second deployment must not double-resume runs
def supervisor():
    import glob
    import json
    import os
    import time

    vol.reload()
    if os.path.exists(f"{SFT_ROOT}/resume_paused"):
        print(f"[supervisor] auto-resume PAUSED ({SFT_ROOT}/resume_paused present) — no spawns", flush=True)
        return
    for meta_path in sorted(glob.glob(f"{SFT_ROOT}/*/run_meta.json")):
        run_dir = os.path.dirname(meta_path)
        run = os.path.basename(run_dir)
        if os.path.exists(f"{run_dir}/final"):
            print(f"[supervisor] {run}: COMPLETE", flush=True)
            continue
        if os.path.exists(f"{run_dir}/resume_paused"):
            print(f"[supervisor] {run}: paused ({run}/resume_paused present)", flush=True)
            continue
        hb = f"{run_dir}/heartbeat"
        age = time.time() - (os.path.getmtime(hb) if os.path.exists(hb) else os.path.getmtime(meta_path))
        if age < STALE_HEARTBEAT_S:
            print(f"[supervisor] {run}: alive (heartbeat {age / 60:.0f} min old)", flush=True)
            continue
        st_path = f"{run_dir}/resume_state.json"
        st = json.load(open(st_path)) if os.path.exists(st_path) else {}
        if time.time() - st.get("last_spawn_ts", 0) < SPAWN_COOLDOWN_S:
            print(f"[supervisor] {run}: stale but in post-spawn cooldown ({st.get('call')})", flush=True)
            continue
        meta = json.load(open(meta_path))
        steps = sorted(int(p.rsplit("_", 1)[-1]) for p in glob.glob(f"{run_dir}/step_*")
                       if p.rsplit("_", 1)[-1].isdigit())
        resume_from = f"{run_dir}/step_{steps[-1]}" if steps else ""
        skip = steps[-1] + 1 if steps else 0  # step_N lands after batch N -> skip batches 0..N
        print(f"[supervisor] {run}: DEAD (heartbeat {age / 60:.0f} min old) — respawning "
              f"{'from ' + resume_from + f' skip={skip}' if steps else 'FRESH (no ckpts yet)'}", flush=True)
        call = modal.Function.from_name(APP_NAME, "train").spawn(
            run_name=meta["run_name"], data_dir=meta["data_dir"], n_ckpts=meta["n_ckpts"],
            epochs=meta.get("epochs", 1), batch_size=meta.get("batch_size", 0),
            lr=meta.get("lr", 0.0), max_seq=meta.get("max_seq", 0),
            backend=meta.get("backend", "nccl"), extra_args=meta.get("extra_args", ""),
            resume_from=resume_from, skip_steps=skip, wandb_id=meta.get("wandb_id", ""))
        json.dump({"last_spawn_ts": time.time(), "call": call.object_id,
                   "from_step": steps[-1] if steps else -1}, open(st_path, "w"))
        vol.commit()
        print(f"[supervisor] {run}: resume leg spawned {call.object_id}", flush=True)


@app.local_entrypoint()
def launch(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
           batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
           backend: str = "nccl", extra_args: str = ""):
    """Spawn one SFT run on the DEPLOYED app (run `modal deploy modal_sft.py` first). Returns
    immediately — invoke once per datamix to train many mixes in parallel."""
    call = modal.Function.from_name(APP_NAME, "train").spawn(
        run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
        batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend, extra_args=extra_args)
    print(f"spawned SFT run {run_name!r} on bank {data_dir}: {call.object_id}")
    print(f"logs:  modal app logs {APP_NAME}   |   ckpts: /data/sft_mix/{run_name}/step_*")


@app.local_entrypoint()
def run_train(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
              batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
              backend: str = "nccl", extra_args: str = ""):
    """Attached single run (live logs; use `modal run --detach` to survive disconnect)."""
    train.remote(run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
                 batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend,
                 extra_args=extra_args)


@app.local_entrypoint()
def run_smoke(data_dir: str, n_records: int = 256, batch_size: int = 8, extra_args: str = "", nproc: int = 1,
              save_dir: str = "/tmp/smoke_ckpt", reload_check: bool = False):
    smoke.remote(data_dir=data_dir, n_records=n_records, batch_size=batch_size, extra_args=extra_args, nproc=nproc,
                 save_dir=save_dir, reload_check=reload_check)


@app.local_entrypoint()
def run_prewarm():
    prewarm.remote()


@app.local_entrypoint()
def run_fp8_eval(data_dir: str = "banks/last5_rp", n_records: int = 6400, extra_args: str = "",
                 script: str = "SL/fp8_eval.py"):
    fp8_eval.remote(data_dir=data_dir, n_records=n_records, extra_args=extra_args, script=script)


@app.local_entrypoint()
def run_env_check():
    env_check.remote()


@app.local_entrypoint()
def run_fullft_bench(data_dir: str = "/data/banks/realact_short_smoke3", configs: str = "32x64;64x32;128x16",
                     n_records: int = 132000, stop_after_step: int = 6, max_seq: int = 160, lr: float = 5e-5,
                     common: str = "--full-ft --prefix-cache --grad-ckpt 0 --pad-multiple 1 --length-bucket --log-steps 1",
                     oom_fallback: str = "--prefix-accum 2", deadline_min: float = 40.0):
    """Spawn fullft_bench on the DEPLOYED app (detached; prints the call id — poll with modal.FunctionCall.from_id)."""
    call = modal.Function.from_name(APP_NAME, "fullft_bench").spawn(
        data_dir=data_dir, configs=configs, n_records=n_records, stop_after_step=stop_after_step, max_seq=max_seq,
        lr=lr, common=common, oom_fallback=oom_fallback, deadline_min=deadline_min)
    print(f"spawned fullft_bench on {APP_NAME}: {call.object_id}")
