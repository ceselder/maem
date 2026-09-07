"""Modal app: universal-inverter SFT (train/pretrain.py) on 8xB200, one container per datamix.

The trainer lives at train/pretrain.py in this repo; it is mounted into the container at
/pmx/SL/pretrain.py (its original path on the training box), with mxf/ importable via
PYTHONPATH=/pmx/helpers. Same image/volume as modal_rl.py (no vllm — pretrain is pure HF).
Data lives on the `maemm-data` Volume:
    /data/<bank>            SFT bank: records.jsonl ({"vec_idx", "target_text"} per line)
                            + vecs.f32 or vecs.f16 (N x 5120 f32/f16 memmap, N >= max(vec_idx)+1)
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

MULTI-NODE (`train_multinode`, modal.experimental.clustered): the SAME run on N containers x 8 GPUs = one torchrun
job of world 8N, ~N x faster at the same effective batch (per-GPU micro-batch unchanged, --grad-accum 32/N).
Cluster size / GPU / RDMA are decorator constants = deploy-time env (SFT_N_NODES=4, SFT_MN_GPU=B200:8, SFT_MN_RDMA=0:
RDMA is allowlisted and this workspace lacks it -> NCCL over Modal's 50 Gbps TCP network); Modal requires the full GPU
count of a node per container (":8"). ALWAYS deploy/run it through
sft/modal_sft_multinode.py, which pins SFT_APP_NAME=maemm-sft-multinode (never the production single-node app):
    SFT_N_NODES=4 modal deploy sft/modal_sft_multinode.py
    modal run sft/modal_sft_multinode.py::run_cluster_check          # NCCL all-reduce across all nodes + volume read speed
    modal run sft/modal_sft_multinode.py::launch_multinode --run-name R --data-dir /data/banks/B --extra-args "... --grad-accum 8"
Rank-0 container owns run_meta/heartbeat/commits; torchrun global rank 0 writes ckpts + wandb (pretrain.py is_main).
Auto-resume: `supervisor_multinode` (scheduled iff SFT_MULTINODE_SUPERVISOR=1) handles runs whose run_meta has
n_nodes > 1; pause one with `<run>/multinode_resume_paused` (a multi-node run always carries `<run>/resume_paused`
so the legacy single-node supervisor never resumes it as ONE node at a quarter of the effective batch).

Needs Modal secrets `maemm-hf` (HF_TOKEN) and `maemm-wandb` (WANDB_API_KEY).
"""

from pathlib import Path

import os

import modal
import modal.experimental  # clustered / get_cluster_info are NOT re-exported by `import modal`

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
image = (
    image
    .add_local_file(REPO / "sft" / "pretrain.py", "/pmx/SL/pretrain.py")
    .add_local_file(REPO / "sft" / "prefix_cache.py", "/pmx/SL/prefix_cache.py")   # --prefix-cache sibling import
    .add_local_file(REPO / "sft" / "fullft.py", "/pmx/SL/fullft.py")               # --full-ft (FSDP2) sibling import
    .add_local_file(REPO / "sft" / "fp8.py", "/pmx/SL/fp8.py")            # --fp8-base (imported by pretrain.py)
    .add_local_file(REPO / "sft" / "fp8_eval.py", "/pmx/SL/fp8_eval.py")  # fp8 speed/fidelity harness (fp8_eval fn)
    .add_local_file(REPO / "sft" / "fp8_microbench.py", "/pmx/SL/fp8_microbench.py")  # fp8 GEMM/layer/profile bench
    .add_local_file(REPO / "sft" / "ddp_check.py", "/pmx/SL/ddp_check.py")  # cluster_check torchrun target (multi-node NCCL probe)
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

SFT_ROOT = "/data/sft_mix"
STALE_HEARTBEAT_S = 30 * 60   # live legs touch+commit the heartbeat every <=5 min; 30 min = dead
SPAWN_COOLDOWN_S = 3600       # never double-spawn while a fresh leg pulls image/model (~15-30 min)

# ---- multi-node (modal.experimental.clustered): N containers x 8 GPUs = ONE torchrun job. Cluster size, GPU type and
# RDMA are decorator constants -> deploy-time env. Modal: "clustered functions must use the full number of GPU devices
# per node" (":8"), up to 64 devices per cluster; rdma=True = the 3,200 Gbps RoCE scale-out network (else 50 Gbps TCP).
# rdma is allowlisted per workspace: safety-sahan gets PermissionDeniedError('RDMA is not supported on this workspace.')
# (probed 2026-09-07), so the default is TCP; SFT_MN_RDMA=1 once Modal enables it. ----
N_NODES = int(os.environ.get("SFT_N_NODES", "4"))
MN_GPU = os.environ.get("SFT_MN_GPU", "B200:8")
MN_RDMA = os.environ.get("SFT_MN_RDMA", "0") == "1"
MN_PORT = 29500                                       # torchrun (static c10d) rendezvous port on the rank-0 container
MN_RDZV_TIMEOUT_S = 3600                              # nodes reach torchrun at different times (per-node bank staging)
# The container re-imports this module WITHOUT the deploy-time env, so the constants above would silently fall back to
# their defaults there (first smoke: cluster of 2 vs N_NODES=4). This secret carries them into the container env.
_MN_CONFIG = modal.Secret.from_dict({"SFT_N_NODES": str(N_NODES), "SFT_MN_GPU": MN_GPU, "SFT_MN_RDMA": "1" if MN_RDMA else "0"})


VEC_BANK_FILES = (("vecs.f32", 4), ("vecs.f16", 2))   # (filename, bytes per element); same as pretrain.py


def _vec_bank_file(data_dir: str):
    """(filename, itemsize) of the vector bank in data_dir -- vecs.f32 preferred, else vecs.f16."""
    import os

    for fname, itemsize in VEC_BANK_FILES:
        if os.path.exists(f"{data_dir}/{fname}"):
            return fname, itemsize
    raise AssertionError(f"bank incomplete: neither vecs.f32 nor vecs.f16 in {data_dir}")


def _preflight(run_name: str, data_dir: str, n_ckpts: int, resume_from: str, save_root: str = SFT_ROOT):
    """Shared guardrails + bank staging for train/smoke. Returns (save_dir, local_bank, d_model)."""
    import glob
    import json
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

    save_dir = f"{save_root}/{run_name}"
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

    # ---- bank schema guard, then stage onto container-local NVMe: memmap over the volume FUSE
    # mount is the one thing we don't trust, and per-batch random row reads are faster locally. ----
    assert os.path.isdir(data_dir), f"bank not found on volume: {data_dir}"
    assert os.path.exists(f"{data_dir}/records.jsonl"), f"bank incomplete: {data_dir}/records.jsonl missing"
    vec_file, itemsize = _vec_bank_file(data_dir)   # vecs.f32 or vecs.f16 (pretrain.py reads either)
    rec0 = json.loads(open(f"{data_dir}/records.jsonl").readline())
    assert "vec_idx" in rec0 and "target_text" in rec0, f"records.jsonl schema: got {sorted(rec0)}"
    vsize = os.path.getsize(f"{data_dir}/{vec_file}")
    assert vsize % (D_MODEL * itemsize) == 0, \
        f"{vec_file} = {vsize} B, not a multiple of one {D_MODEL}-x-{itemsize}B row"
    print(f"[modal] bank OK: {vsize // (D_MODEL * itemsize)} vecs x {D_MODEL} ({vec_file}) + records.jsonl", flush=True)

    t0 = time.time()
    local_bank = f"/root/bank_{os.path.basename(data_dir.rstrip('/'))}"
    if not os.path.exists(local_bank):
        shutil.copytree(data_dir, local_bank)
    print(f"[modal] bank staged to {local_bank} ({time.time() - t0:.0f}s)", flush=True)
    return save_dir, local_bank, D_MODEL


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


def _start_committer(save_dir: str):
    """Heartbeat + committer thread: touch <run>/heartbeat every 60s (supervisor aliveness signal, started BEFORE the
    slow model load so fresh legs read as alive immediately) and vol.commit right after every new ckpt lands (else
    every 300s) so saves are durable even if the container dies mid-run."""
    import glob
    import threading
    import time

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


def _run_meta(save_dir: str, wandb_id: str, fresh: dict, leg: dict):
    """run_meta.json = everything the supervisor needs to respawn this run faithfully. Written once on the first leg
    (wandb id minted here so every leg logs to ONE wandb run); later legs only append to `legs`. Returns wandb_id."""
    import json
    import os
    import secrets as pysecrets
    import string
    import time

    meta_path = f"{save_dir}/run_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        wandb_id = wandb_id or meta.get("wandb_id", "")
    else:
        wandb_id = wandb_id or "".join(
            pysecrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        meta = {**fresh, "wandb_id": wandb_id, "created": time.time(), "legs": []}
    meta["legs"].append({"ts": time.time(), **leg})
    json.dump(meta, open(meta_path, "w"), indent=2)
    return wandb_id


def _pretrain_argv(local_bank, save_dir, run_name, n_ckpts, epochs, wandb_id, batch_size, lr, max_seq,
                   resume_from, skip_steps, extra_args):
    """pretrain.py argv shared by train / train_multinode (extra_args whitespace-split, appended last: argparse last-wins)."""
    argv = ["SL/pretrain.py", "--data-dir", local_bank, "--save-dir", save_dir, "--run-name", run_name,
            "--n-ckpts", str(n_ckpts), "--epochs", str(epochs), "--wandb-id", wandb_id]
    if batch_size:
        argv += ["--batch-size", str(batch_size)]
    if lr:
        argv += ["--lr", str(lr)]
    if max_seq:
        argv += ["--max-seq", str(max_seq)]
    if resume_from:
        argv += ["--init-adapter", resume_from, "--skip-steps", str(skip_steps)]
    if extra_args:
        argv += extra_args.split()
    return argv


def _cluster():
    """(rank, size, master_addr, cluster_id) of this container's modal.experimental cluster. container_ips are the
    IPv6 addresses sorted by rank; the rank-0 container hosts the torchrun rendezvous."""
    import modal.experimental

    ci = modal.experimental.get_cluster_info()
    return ci.rank, len(ci.container_ips), ci.container_ips[0], ci.cluster_id


def _gpus_per_node() -> int:
    """GPUs visible in this container (authoritative), falling back to the deploy-time gpu spec."""
    try:
        import torch
        if torch.cuda.device_count():
            return torch.cuda.device_count()
    except Exception:  # noqa
        pass
    return int(MN_GPU.split(":")[1]) if ":" in MN_GPU else 1


def _torchrun_multinode(rank: int, size: int, master: str, nproc: int):
    """torchrun prefix for one node of a static (c10d) multi-node rendezvous on the rank-0 container. IPv6 master addr
    is passed bare, as in Modal's own example (torchrun rsplits host:port on the last colon)."""
    return ["torchrun", f"--nnodes={size}", f"--node_rank={rank}", f"--nproc_per_node={nproc}",
            f"--master_addr={master}", f"--master_port={MN_PORT}", f"--rdzv_conf=timeout={MN_RDZV_TIMEOUT_S}"]


@app.function(
    image=image,
    gpu=os.environ.get("SFT_GPU", "B200:8"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=86400,
    ephemeral_disk=int(os.environ.get("SFT_DISK_GB", "600")) * 1024,   # _preflight stages the bank locally: 23M examples = 235 GB, 50M = 512 GB (deploy with SFT_DISK_GB=1200)
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
    import os

    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    save_dir, local_bank, _ = _preflight(run_name, data_dir, n_ckpts, resume_from)
    os.makedirs(save_dir, exist_ok=True)

    wandb_id = _run_meta(save_dir, wandb_id,
                         fresh={"run_name": run_name, "data_dir": data_dir, "n_ckpts": n_ckpts, "epochs": epochs,
                                "batch_size": batch_size, "lr": lr, "max_seq": max_seq, "backend": backend,
                                "extra_args": extra_args},
                         leg={"resume_from": resume_from, "skip_steps": skip_steps})
    _start_committer(save_dir)
    vol.commit()  # publish run_meta + heartbeat now: the supervisor must see this leg exists

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8"] + _pretrain_argv(
        local_bank, save_dir, run_name, n_ckpts, epochs, wandb_id, batch_size, lr, max_seq,
        resume_from, skip_steps, extra_args)
    rc = _stream(cmd, _train_env(backend), "modal")
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"torchrun exited rc={rc} (backend={backend}) — supervisor will resume")
    assert os.path.exists(f"{save_dir}/final"), "rc=0 but no final ckpt — trainer exited early?"
    print(f"[modal] SFT run {run_name} COMPLETE -> {save_dir}/final", flush=True)


@app.function(
    image=image,
    gpu=MN_GPU,
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
        _MN_CONFIG,
    ],
    timeout=86400,
    ephemeral_disk=int(os.environ.get("SFT_DISK_GB", "600")) * 1024,   # EVERY node stages its own bank copy (_preflight)
    memory=256 * 1024,
)
@modal.experimental.clustered(size=N_NODES, rdma=MN_RDMA)
def train_multinode(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
                    batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
                    backend: str = "nccl", extra_args: str = "",
                    resume_from: str = "", skip_steps: int = 0, wandb_id: str = "",
                    n_nodes: int = N_NODES, save_root: str = SFT_ROOT):
    """`train` across N_NODES containers x 8 GPUs = world 8*N_NODES, ONE torchrun DDP job (static c10d rendezvous on
    the rank-0 container). Same args as `train` plus n_nodes (runtime guard: must equal the deployed cluster size,
    SFT_N_NODES) and save_root (default /data/sft_mix; smoke tests point it at /tmp so nothing production-facing
    sees them). Every container stages the bank + HF cache itself; ONLY the rank-0 container owns run_meta /
    heartbeat / volume commits, and only its torchrun rank 0 writes ckpts + wandb (pretrain.py is_main = RANK 0).
    Hold the effective batch by keeping the per-GPU micro-batch and passing --grad-accum 32/n_nodes in extra_args:
    64 x 8 GPUs x 4 nodes x 8 accum = 16,384 = the 1-node 64 x 8 x 32. Same optimizer-step count + LR schedule;
    pretrain.py shards records as i % world, so the example composition of each step differs from the 1-node run
    (statistically equivalent, not bit-identical)."""
    import os

    rank, size, master, cluster_id = _cluster()
    nproc = _gpus_per_node()
    world = size * nproc
    tag = f"modal-mn{rank}/{size}"
    # get_cluster_info() is authoritative; n_nodes is the caller's intent (effective-batch arithmetic depends on it)
    assert size == n_nodes, f"cluster size {size} != n_nodes {n_nodes} (deployed SFT_N_NODES={N_NODES})"
    is_leader = rank == 0
    toks = extra_args.split()
    accum = int(toks[toks.index("--grad-accum") + 1]) if "--grad-accum" in toks else 1
    bs = batch_size or 64
    print(f"[{tag}] cluster {cluster_id}: rank {rank}/{size}, rendezvous {master}:{MN_PORT}, {nproc} GPU/node -> "
          f"world {world}; effective batch = {bs} x {world} x grad_accum {accum} = {bs * world * accum}", flush=True)

    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    # every node: HF cache check + bank -> container-local NVMe (same as the 1-node path, N copies)
    save_dir, local_bank, _ = _preflight(run_name, data_dir, n_ckpts, resume_from, save_root=save_root)
    os.makedirs(save_dir, exist_ok=True)

    if is_leader:
        wandb_id = _run_meta(save_dir, wandb_id,
                             fresh={"run_name": run_name, "data_dir": data_dir, "n_ckpts": n_ckpts, "epochs": epochs,
                                    "batch_size": batch_size, "lr": lr, "max_seq": max_seq, "backend": backend,
                                    "extra_args": extra_args, "n_nodes": n_nodes, "gpu": MN_GPU, "rdma": MN_RDMA,
                                    "save_root": save_root, "app_name": APP_NAME},
                             leg={"resume_from": resume_from, "skip_steps": skip_steps, "cluster_id": cluster_id})
        # The single-node supervisor (production deployment, older code) would respawn a stale multi-node run as ONE
        # node carrying this run's --grad-accum -> 1/n_nodes of the effective batch. Its only per-run off switch is
        # <run>/resume_paused, so every multi-node run carries it; supervisor_multinode pauses on
        # <run>/multinode_resume_paused instead (see _supervise).
        with open(f"{save_dir}/resume_paused", "w") as f:
            f.write("multi-node run (modal_sft.train_multinode): the single-node supervisor must not resume it; "
                    "pause the multi-node supervisor with <run>/multinode_resume_paused\n")
        _start_committer(save_dir)
        vol.commit()  # publish run_meta + heartbeat + marker now

    cmd = _torchrun_multinode(rank, size, master, nproc) + _pretrain_argv(
        local_bank, save_dir, run_name, n_ckpts, epochs, wandb_id, batch_size, lr, max_seq,
        resume_from, skip_steps, extra_args)
    env = _train_env(backend)
    env.setdefault("NCCL_DEBUG", os.environ.get("SFT_NCCL_DEBUG", "WARN"))
    rc = _stream(cmd, env, tag)
    if is_leader:
        vol.commit()
    if rc != 0:
        raise RuntimeError(f"[{tag}] torchrun exited rc={rc} (backend={backend}) — supervisor_multinode will resume")
    if is_leader:
        assert os.path.exists(f"{save_dir}/final"), "rc=0 but no final ckpt — trainer exited early?"
        print(f"[{tag}] multi-node SFT run {run_name} COMPLETE -> {save_dir}/final", flush=True)
    return {"rank": rank, "cluster_id": cluster_id, "rc": rc}


@app.function(
    image=image,
    gpu=MN_GPU,
    volumes={"/data": vol},
    secrets=[_MN_CONFIG],
    timeout=1800,
    memory=64 * 1024,
)
@modal.experimental.clustered(size=N_NODES, rdma=MN_RDMA)
def cluster_check(allreduce_mb: int = 1800, read_gb: float = 0.0,
                  read_path: str = "/data/banks/realact_short_20m_all/vecs.f16"):
    """Cheap multi-node probe (no model, no bank): prints the cluster info Modal hands each container, runs
    sft/ddp_check.py under the SAME torchrun rendezvous train_multinode uses (one NCCL group of 8*N ranks; all-reduce
    time at LoRA-grad size = the per-step cross-node comm cost) with NCCL INIT/NET logging (shows IB/RoCE vs Socket),
    and optionally reads read_gb GB of read_path on EVERY node concurrently = the bank-staging throughput a real run
    will see. Returns rank 0's dict."""
    import os
    import socket
    import subprocess
    import time

    import modal.experimental

    ci = modal.experimental.get_cluster_info()
    rank, size, master, cluster_id = _cluster()
    nproc = _gpus_per_node()
    tag = f"cluster-check r{rank}/{size}"
    print(f"[{tag}] cluster {cluster_id} host={socket.gethostname()} ips={ci.container_ips} "
          f"ipv4={ci.container_ipv4_ips} fabric_ids={ci.fabric_ids} rdma_requested={MN_RDMA}", flush=True)
    print(f"[{tag}] " + subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
          .replace("\n", f"\n[{tag}] "), flush=True)
    ib = sorted(os.listdir("/sys/class/infiniband")) if os.path.isdir("/sys/class/infiniband") else []
    print(f"[{tag}] /sys/class/infiniband: {ib or 'none'}; NCCL env: "
          f"{ {k: v for k, v in os.environ.items() if k.startswith('NCCL')} }", flush=True)

    env = _train_env("nccl")
    env["NCCL_DEBUG"], env["NCCL_DEBUG_SUBSYS"] = "INFO", "INIT,NET"
    cmd = _torchrun_multinode(rank, size, master, nproc) + ["SL/ddp_check.py", "--allreduce-mb", str(allreduce_mb)]
    t0 = time.time()
    rc = _stream(cmd, env, tag)
    print(f"[{tag}] torchrun rc={rc} ({time.time() - t0:.0f}s incl. rendezvous)", flush=True)
    if rc != 0:
        raise RuntimeError(f"[{tag}] ddp_check exited rc={rc}")

    out = {"rank": rank, "cluster_id": cluster_id, "nodes": size, "gpus_per_node": nproc, "rc": rc}
    if read_gb > 0:
        n, chunk, done = int(read_gb * 2**30), 64 * 2**20, 0
        t0 = time.time()
        with open(read_path, "rb") as f:
            while done < n:
                b = f.read(min(chunk, n - done))
                if not b:
                    break
                done += len(b)
        dt = time.time() - t0
        out["volume_read_gbps"] = done / dt / 1e9
        print(f"[{tag}] volume read {done / 2**30:.1f} GiB of {read_path} in {dt:.0f}s = {done / dt / 1e9:.2f} GB/s "
              f"(all {size} nodes reading concurrently)", flush=True)
    return out


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

    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    _, local_bank, _ = _preflight("smoke", data_dir, n_ckpts=2, resume_from="")
    vec_file, _ = _vec_bank_file(local_bank)

    tiny = "/root/bank_smoke"
    if not os.path.exists(tiny):
        os.makedirs(tiny)
        with open(f"{local_bank}/records.jsonl") as fin, open(f"{tiny}/records.jsonl", "w") as fout:
            for i, line in enumerate(fin):
                if i >= n_records:
                    break
                fout.write(line)
        shutil.copy(f"{local_bank}/{vec_file}", f"{tiny}/{vec_file}")
    print(f"[modal-smoke] tiny bank: first {n_records} records", flush=True)

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
recs = [json.loads(l) for _, l in zip(range(8), open('{tiny}/records.jsonl'))]
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
def _supervise(multinode: bool):
    """One supervisor pass. multinode=False: single-node runs (run_meta without n_nodes > 1) -> `train`, pause marker
    <run>/resume_paused. multinode=True: runs with run_meta n_nodes > 1 -> `train_multinode` (same cluster size as
    deployed), pause marker <run>/multinode_resume_paused (<run>/resume_paused is the legacy-supervisor blocker every
    multi-node run carries -- see train_multinode). The global /data/sft_mix/resume_paused pauses both."""
    import glob
    import json
    import os
    import time

    who, fn = ("supervisor-mn", "train_multinode") if multinode else ("supervisor", "train")
    vol.reload()
    if os.path.exists(f"{SFT_ROOT}/resume_paused"):
        print(f"[{who}] auto-resume PAUSED ({SFT_ROOT}/resume_paused present) — no spawns", flush=True)
        return
    for meta_path in sorted(glob.glob(f"{SFT_ROOT}/*/run_meta.json")):
        run_dir = os.path.dirname(meta_path)
        run = os.path.basename(run_dir)
        meta = json.load(open(meta_path))
        if (meta.get("n_nodes", 1) > 1) != multinode:
            continue   # the other supervisor's run
        if os.path.exists(f"{run_dir}/final"):
            print(f"[{who}] {run}: COMPLETE", flush=True)
            continue
        pause = "multinode_resume_paused" if multinode else "resume_paused"
        if os.path.exists(f"{run_dir}/{pause}"):
            print(f"[{who}] {run}: paused ({run}/{pause} present)", flush=True)
            continue
        hb = f"{run_dir}/heartbeat"
        age = time.time() - (os.path.getmtime(hb) if os.path.exists(hb) else os.path.getmtime(meta_path))
        if age < STALE_HEARTBEAT_S:
            print(f"[{who}] {run}: alive (heartbeat {age / 60:.0f} min old)", flush=True)
            continue
        st_path = f"{run_dir}/resume_state.json"
        st = json.load(open(st_path)) if os.path.exists(st_path) else {}
        if time.time() - st.get("last_spawn_ts", 0) < SPAWN_COOLDOWN_S:
            print(f"[{who}] {run}: stale but in post-spawn cooldown ({st.get('call')})", flush=True)
            continue
        steps = sorted(int(p.rsplit("_", 1)[-1]) for p in glob.glob(f"{run_dir}/step_*")
                       if p.rsplit("_", 1)[-1].isdigit())
        resume_from = f"{run_dir}/step_{steps[-1]}" if steps else ""
        skip = steps[-1] + 1 if steps else 0  # step_N lands after batch N -> skip batches 0..N
        print(f"[{who}] {run}: DEAD (heartbeat {age / 60:.0f} min old) — respawning "
              f"{'from ' + resume_from + f' skip={skip}' if steps else 'FRESH (no ckpts yet)'}", flush=True)
        kwargs = dict(
            run_name=meta["run_name"], data_dir=meta["data_dir"], n_ckpts=meta["n_ckpts"],
            epochs=meta.get("epochs", 1), batch_size=meta.get("batch_size", 0),
            lr=meta.get("lr", 0.0), max_seq=meta.get("max_seq", 0),
            backend=meta.get("backend", "nccl"), extra_args=meta.get("extra_args", ""),
            resume_from=resume_from, skip_steps=skip, wandb_id=meta.get("wandb_id", ""))
        if multinode:
            kwargs.update(n_nodes=meta["n_nodes"], save_root=meta.get("save_root", SFT_ROOT))
        call = modal.Function.from_name(APP_NAME, fn).spawn(**kwargs)
        json.dump({"last_spawn_ts": time.time(), "call": call.object_id,
                   "from_step": steps[-1] if steps else -1}, open(st_path, "w"))
        vol.commit()
        print(f"[{who}] {run}: resume leg spawned {call.object_id} ({fn})", flush=True)


@app.function(schedule=modal.Period(minutes=20) if os.environ.get("SFT_SUPERVISOR", "1") == "1" else None,
              volumes={"/data": vol}, timeout=600)   # SFT_SUPERVISOR=0 at deploy: a second deployment must not double-resume runs
def supervisor():
    _supervise(multinode=False)


@app.function(schedule=modal.Period(minutes=20) if os.environ.get("SFT_MULTINODE_SUPERVISOR", "0") == "1" else None,
              volumes={"/data": vol}, timeout=600)   # scheduled only in the multi-node deployment (sft/modal_sft_multinode.py)
def supervisor_multinode():
    _supervise(multinode=True)


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
def launch_multinode(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
                     batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
                     backend: str = "nccl", extra_args: str = "", n_nodes: int = N_NODES, save_root: str = SFT_ROOT):
    """Spawn one multi-node SFT run on the DEPLOYED cluster app (`SFT_N_NODES=<n> modal deploy
    sft/modal_sft_multinode.py` first). Pass --grad-accum <32 / n_nodes> in extra_args to keep the effective batch."""
    assert n_nodes == N_NODES, f"n_nodes {n_nodes} != SFT_N_NODES {N_NODES} this file was loaded with"
    assert APP_NAME != "maemm-sft-8xb200", "multi-node launches go through sft/modal_sft_multinode.py, never the production app"
    call = modal.Function.from_name(APP_NAME, "train_multinode").spawn(
        run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
        batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend, extra_args=extra_args,
        n_nodes=n_nodes, save_root=save_root)
    print(f"spawned {n_nodes}-node SFT run {run_name!r} on bank {data_dir}: {call.object_id}")
    print(f"logs:  modal app logs {APP_NAME}   |   ckpts: {save_root}/{run_name}/step_*")


@app.local_entrypoint()
def run_train_multinode(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
                        batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
                        backend: str = "nccl", extra_args: str = "", n_nodes: int = N_NODES, save_root: str = SFT_ROOT):
    """Attached multi-node run (ephemeral app, live logs of every node; containers exit with the call). Smoke recipe:
    --data-dir /data/banks/realact_short_smoke3 --n-ckpts 2 --batch-size 8 --save-root /tmp/mn_smoke
    --extra-args "--no-wandb --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 1"."""
    assert n_nodes == N_NODES, f"n_nodes {n_nodes} != SFT_N_NODES {N_NODES} this file was loaded with"
    assert APP_NAME != "maemm-sft-8xb200", "multi-node launches go through sft/modal_sft_multinode.py, never the production app"
    print(train_multinode.remote(run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
                                 batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend,
                                 extra_args=extra_args, n_nodes=n_nodes, save_root=save_root))


@app.local_entrypoint()
def run_cluster_check(allreduce_mb: int = 1800, read_gb: float = 0.0,
                      read_path: str = "/data/banks/realact_short_20m_all/vecs.f16"):
    """Attached multi-node probe (ephemeral app; no deploy needed): cluster info + NCCL all-reduce across all nodes."""
    assert APP_NAME != "maemm-sft-8xb200", "run this through sft/modal_sft_multinode.py"
    print(cluster_check.remote(allreduce_mb=allreduce_mb, read_gb=read_gb, read_path=read_path))


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
