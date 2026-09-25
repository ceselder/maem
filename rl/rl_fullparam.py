"""--full-param for rl/rl_disagg.py: the RL POLICY is the WHOLE Qwen3.6-27B (every weight trainable), not a LoRA.

Trainer ranks (Y of them, one GPU each) hold the policy FSDP2-sharded exactly like sft/pretrain.py --full-ft does
(sft/fullft.py shard_full_model: fp32 sharded masters + fp32 sharded grads + fp32 AdamW moments, bf16 compute through
MixedPrecisionPolicy, one decoder layer all-gathered at a time). The reward scorer -- the CLEAN original base -- can no
longer be "the actor with its LoRA off", so every trainer rank also holds a frozen bf16 copy of MODEL, FSDP2-sharded as
well (no grad; layers [0, READ_LAYER] only when the gates are off) and read WITHOUT the read_resid _Stop exception (an
exception inside an FSDP2 forward would leave its unshard/reshard state machine half-way). The vLLM rollout engines serve
the policy base WITHOUT LoRA slots and receive the new bf16 weights after every optimizer step:

  publish-mode nccl (default)   trainer rank 0 + every vLLM worker process form ONE extra NCCL communicator (vLLM's
      StatelessProcessGroup + PyNcclCommunicator -- the RLHF weight-sync pattern; torch.distributed's default group is
      untouched). A publish = `lora/pending` -> every rollout rank finishes its in-flight block, writes `lora/ready_<r>_<k>`
      and enters worker.wu_recv (fast_lens_ext.FastSteerExtension) -> the trainer ranks all-gather each param in bf16
      (DTensor.to(bf16).full_tensor(), one at a time, identical order everywhere) and rank 0 broadcasts it on the weight
      group; the workers feed the stream lazily into vLLM's own model.load_weights (in place: CUDA graphs, the steering
      hook and the KV cache stay valid) -> rank 0 writes lora/step_<k>/meta.json (hnorm + checksums) and flips `latest`.
      Nothing is transferred while an engine generates (no NCCL kernels next to vLLM's forward, none next to FSDP's
      collectives either: the trainer's main thread is idle during the broadcast), so the wait for the slowest engine's
      block boundary (<= one block, ~6 s here) is the price of the exactness; every engine serves exactly step k after it.
  publish-mode fs               every trainer rank writes its own bf16 shards to lora/step_<k>/shard_<t>.safetensors
      (RAM-backed --work-dir), the engines pick the step up at their next block boundary like the LoRA path and stream
      the concatenated shards through the same load_weights generator. Needs ~54 GB of --work-dir per kept step.

Both modes keep rl_disagg's step/lag protocol (lora/latest, blk_<adapter_step>_*, --max-lag) and its metrics; the
sampler-vs-trainer |dlogp| at step >= 1 is the end-to-end proof that the served weights == the trainer's policy.

Checkpoints are FULL HF model dirs (sft/fullft.py save_full_ckpt layout + SAVE_DONE) at --save-steps / final, evaluated
by eval/eval_ckpt_daemon.py --full-model (eval/modal_eval_ckpt.py fullmodel_daemon). Optimizer state: --save-optim writes
the sharded AdamW state with torch.distributed.checkpoint next to the checkpoint (fp32, ~2x the model); --load-optim reads it.

Pure pieces (manifest / shard geometry / loss scaling / read_resid / fs round trip) are unit-tested on CPU in
rl/test_rl_disagg_fullparam.py.
"""
import contextlib
import glob
import json
import os
import shutil
import time

import torch
import torch.distributed as dist

try:                                   # model profile: Qwen3.6 multimodal key layout vs plain CausalLM ("model.")
    from mxf.config import LM_PREFIX as _LM_PREFIX
except ImportError:  # pragma: no cover
    _LM_PREFIX = "model.language_model."


def fullft_module():
    """sft/fullft.py (repo checkout) or fullft.py (mounted next to mxf/ in the Modal image)."""
    try:
        from sft import fullft as FT
    except ImportError:
        import fullft as FT
    return FT


def map_name(name):
    """Qwen3_5ForCausalLM param name -> base repo (checkpoint) tensor name == fullft.map_name."""
    if name.startswith("model."):
        return _LM_PREFIX + name[len("model."):]
    return name


def is_dtensor(t):
    try:
        from torch.distributed.tensor import DTensor
        return isinstance(t, DTensor)
    except ImportError:  # pragma: no cover
        return False


# ----------------------------------------------------------------------------------------------
# manifest: the exact tensor stream every publish sends, in named_parameters() order
# ----------------------------------------------------------------------------------------------
def shard_rows(n_rows, world):
    """Rows of dim 0 each rank's FSDP2 shard holds (torch.chunk semantics: ceil(n/world) per rank, the tail possibly short
    or empty) -- what DTensor.to_local() has on rank t."""
    chunk = -(-n_rows // world)
    return [max(0, min(chunk, n_rows - t * chunk)) for t in range(world)]


def check_names(names):
    """A handful of NON-fused tensors whose |sum| the trainer records and every engine verifies after a load (a wrong name
    mapping or shard order shows up here immediately, before the sampler |dlogp| would)."""
    want = ["embed_tokens.weight", "norm.weight", "lm_head.weight"]
    layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in names if ".layers." in n})
    if layers:
        want += [f"layers.{layers[0]}.mlp.down_proj.weight", f"layers.{layers[-1]}.mlp.down_proj.weight",
                 f"layers.{layers[len(layers) // 2]}.self_attn.o_proj.weight"]
    out = []
    for w in want:
        hits = [n for n in names if n.endswith(w) and (w != "norm.weight" or n == f"{_LM_PREFIX}norm.weight")]
        if len(hits) == 1:
            out.append(hits[0])
    return out


def build_manifest(model, world):
    """Rank-independent description of the publish stream: [ckpt name, global shape, rows per rank] per parameter, in
    model.named_parameters() order (== the order every rank gathers / broadcasts in). Local row counts are gathered from
    every rank (dist.all_gather_object) rather than trusted from shard_rows(); the two are asserted equal."""
    names, shapes, local_rows = [], [], []
    for n, p in model.named_parameters():
        names.append(map_name(n))
        shapes.append(list(p.shape))
        local_rows.append(int(p.to_local().shape[0]) if is_dtensor(p) else int(p.shape[0]))
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered = [local_rows]
    rows = [[g[i] for g in gathered] for i in range(len(names))]
    for n, s, r in zip(names, shapes, rows):
        assert sum(r) == s[0], f"{n}: shards {r} do not cover dim0 {s[0]}"
        assert r == shard_rows(s[0], len(gathered)), f"{n}: shard rows {r} != torch.chunk geometry {shard_rows(s[0], len(gathered))}"
    return {"format": "rl_fullparam_manifest_v1", "n_trainer": len(gathered), "dtype": "bfloat16", "names": names, "shapes": shapes,
            "rows": rows, "checks": check_names(names),
            "n_params": int(sum(int(torch.tensor(s).prod()) for s in shapes)),
            "bytes_bf16": int(sum(int(torch.tensor(s).prod()) * 2 for s in shapes))}


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, path)


def read_manifest(path):
    with open(path) as f:
        return json.load(f)


def iter_full_bf16(model, manifest=None):
    """(ckpt name, full bf16 tensor) for every parameter, one at a time -- COLLECTIVE over the FSDP mesh (every rank must
    iterate to the end): the cast happens on the local shard (half the all-gather traffic of gathering fp32)."""
    for i, (n, p) in enumerate(model.named_parameters()):
        name = map_name(n)
        if manifest is not None:
            assert manifest["names"][i] == name, f"parameter order changed: manifest[{i}] = {manifest['names'][i]} vs {name}"
        with torch.no_grad():
            full = p.to(torch.bfloat16).full_tensor() if is_dtensor(p) else p.detach().to(torch.bfloat16)
        yield name, full.contiguous()
        del full


def abs_sum(t):
    return float(t.float().abs().sum().item())


# ----------------------------------------------------------------------------------------------
# publish, trainer side
# ----------------------------------------------------------------------------------------------
class TrainerWeightComm:
    """Rank 0 of the weight-update NCCL group = trainer rank 0; ranks 1..X = the rollout workers (fast_lens_ext.wu_init).
    vLLM's StatelessProcessGroup (a TCPStore that does not touch torch.distributed's default group) carries the NCCL unique
    id; PyNcclCommunicator wraps the raw NCCL calls. Imported lazily: `import vllm` must come AFTER the HF models are
    loaded (its config registration clobbers transformers' AutoConfig for this model, see eval_ckpt_daemon)."""

    def __init__(self, host, port, n_rollout, device, store_timeout=1800):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
        t0 = time.time()
        self.world = 1 + int(n_rollout)
        self.pg = StatelessProcessGroup.create(host, int(port), rank=0, world_size=self.world, store_timeout=store_timeout)
        self.comm = PyNcclCommunicator(self.pg, device=device)
        assert not self.comm.disabled, "PyNcclCommunicator is disabled (NCCL library missing?)"
        self.stream = torch.cuda.current_stream(device)
        self.init_s = time.time() - t0

    def broadcast(self, t):
        self.comm.broadcast(t, src=0, stream=self.stream)


def wait_for_files(paths, timeout_s, poll_s=0.02, stop=None):
    """Block until every path exists (or `stop()` is true). Returns True when all exist."""
    t0 = time.time()
    while True:
        if all(os.path.exists(p) for p in paths):
            return True
        if stop is not None and stop():
            return False
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"waited {timeout_s:.0f}s for {[p for p in paths if not os.path.exists(p)]}")
        time.sleep(poll_s)


def publish_nccl(model, manifest, comm_getter, is_main, work, step, n_rollout, ready_timeout_s, log, stop=None):
    """COLLECTIVE over the trainer ranks. rank 0: flag `pending`, wait until every rollout rank is at a block boundary
    (ready files), obtain the communicator (comm_getter(): creates it on the first publish -- the workers join in wu_init
    right after writing their ready file), then everyone all-gathers param by param (bf16) and rank 0 broadcasts each to
    the engines. Returns (timings dict, checks dict) on every rank (checks only on rank 0)."""
    t0 = time.time()
    pend = f"{work}/lora/pending"
    ready = [f"{work}/lora/ready_{r}_{step}" for r in range(n_rollout)]
    go = [None]
    if is_main:
        tmp = f"{pend}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(str(step))
        os.replace(tmp, pend)
        ok = wait_for_files(ready, ready_timeout_s, stop=stop)
        t_ready = time.time() - t0
        comm = comm_getter() if ok else None
        go[0] = {"ok": ok, "t_ready": t_ready, "t_comm": time.time() - t0 - t_ready}
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast_object_list(go, src=0)
    if not go[0]["ok"]:
        return {"aborted": True}, {}
    t_ready = go[0]["t_ready"]
    t1 = time.time()
    checks, n_bytes = {}, 0
    want = set(manifest["checks"])
    for name, full in iter_full_bf16(model, manifest):
        if is_main:
            comm.broadcast(full)
            if name in want:
                checks[name] = abs_sum(full)
        n_bytes += full.numel() * full.element_size()
    if is_main:
        comm.stream.synchronize()
        for p in ready:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        try:
            os.remove(pend)
        except FileNotFoundError:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_xfer = time.time() - t1
    return {"t_wait_ready": t_ready, "t_comm_init": go[0]["t_comm"], "t_transfer": t_xfer, "gb": n_bytes / 2**30,
            "gbps": n_bytes / 2**30 / max(t_xfer, 1e-6), "total": time.time() - t0}, checks


def local_check_abs_sums(model, manifest):
    """publish-mode fs: |sum| of THIS rank's bf16 shards of the check tensors; the per-rank values add up to the full tensor's
    |sum| (all_reduce_dict_sum), which is what the engine computes after loading -- no read-back of 54 GB needed."""
    want = set(manifest["checks"])
    out = {n: 0.0 for n in want}
    for n, p in model.named_parameters():
        name = map_name(n)
        if name in want:
            loc = (p.to_local() if is_dtensor(p) else p.detach()).to(torch.bfloat16)
            out[name] = abs_sum(loc) if loc.numel() else 0.0
    return out


def all_reduce_dict_sum(d, device):
    keys = sorted(d)
    t = torch.tensor([d[k] for k in keys], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: float(v) for k, v in zip(keys, t.tolist())}


def write_shard_file(model, manifest, path, rank):
    """publish-mode fs: this rank's bf16 local shards of every parameter (ckpt names) -> ONE safetensors file. Empty
    shards (tail ranks of tiny tensors) are skipped; the manifest's `rows` tells the reader which ranks hold what."""
    from safetensors.torch import save_file
    t0 = time.time()
    out, n_bytes = {}, 0
    for i, (n, p) in enumerate(model.named_parameters()):
        name = map_name(n)
        assert manifest["names"][i] == name
        if manifest["rows"][i][rank] == 0:
            continue
        with torch.no_grad():
            loc = (p.to_local() if is_dtensor(p) else p.detach()).to(torch.bfloat16).contiguous()
        out[name] = loc.to("cpu", copy=True)
        n_bytes += out[name].numel() * 2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    tmp = f"{path}.tmp"
    save_file(out, tmp, metadata={"format": "pt", "rank": str(rank)})
    os.replace(tmp, path)
    return {"to_cpu_s": t1 - t0, "write_s": time.time() - t1, "gb": n_bytes / 2**30}


def fs_shard_paths(step_dir, n_trainer):
    return [f"{step_dir}/shard_{t}.safetensors" for t in range(n_trainer)]


def iter_fs_weights(step_dir, manifest, device):
    """(ckpt name, full bf16 tensor on `device`) from the trainer ranks' shard files, manifest order -- the engines' fs-mode
    load_weights stream. Shards are moved to the device one by one and concatenated there (host cat of 2.5 GB is slow)."""
    from safetensors import safe_open
    files = [safe_open(p, framework="pt", device="cpu") for p in fs_shard_paths(step_dir, manifest["n_trainer"])]
    try:
        for name, shape, rows in zip(manifest["names"], manifest["shapes"], manifest["rows"]):
            parts = [files[t].get_tensor(name).to(device, non_blocking=True) for t in range(len(rows)) if rows[t] > 0]
            full = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
            assert list(full.shape) == list(shape), f"{name}: {tuple(full.shape)} != {tuple(shape)}"
            yield name, full.to(torch.bfloat16)
            del full, parts
    finally:
        del files


def fs_checks(step_dir, manifest, device="cpu"):
    """|sum| of the check tensors as the reader will see them (rank 0, after every shard file is complete)."""
    want = set(manifest["checks"])
    out = {}
    for name, full in iter_fs_weights(step_dir, manifest, device):
        if name in want:
            out[name] = abs_sum(full)
    return out


def prune_fs_steps(work, keep):
    """publish-mode fs: every kept step is ~54 GB of RAM-backed work dir -- keep only the newest `keep` shard sets
    (meta.json dirs of older steps stay, the shard files go)."""
    dirs = sorted(glob.glob(f"{work}/lora/step_*"), key=lambda p: int(p.rsplit("_", 1)[-1]))
    for d in dirs[:-keep] if keep > 0 else []:
        for f in glob.glob(f"{d}/shard_*.safetensors"):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


def verify_checks(mine, theirs, rtol=1e-3):
    """{name: |sum|} from the trainer vs from the engine after the load; fp32 accumulation order differs -> relative tolerance."""
    bad = {}
    for k, v in theirs.items():
        if k not in mine:
            continue
        if abs(mine[k] - v) > rtol * max(abs(mine[k]), 1e-6):
            bad[k] = (mine[k], v)
    return bad


# ----------------------------------------------------------------------------------------------
# engine-side: which vLLM parameter is checkpoint tensor X
# ----------------------------------------------------------------------------------------------
def find_vllm_param(params, ckpt_name):
    """params: dict(model.named_parameters()) of the vLLM model. Tries the Qwen3_5ForConditionalGeneration mapper first
    (model.language_model.X -> language_model.model.X, lm_head.X -> language_model.lm_head.X), then the longest unique
    suffix match. Returns (vllm name, param) or (None, None) for fused tensors (qkv, gate_up, in_proj_qkvz...)."""
    cands = []
    if ckpt_name.startswith(_LM_PREFIX):
        cands.append("language_model.model." + ckpt_name[len(_LM_PREFIX):])
    if ckpt_name.startswith("lm_head."):
        cands.append("language_model." + ckpt_name)
    cands.append(ckpt_name)
    for c in cands:
        if c in params:
            return c, params[c]
    parts = ckpt_name.split(".")
    for k in range(len(parts) - 1, 0, -1):          # longest suffix first; shorter suffixes only match MORE names
        suf = ".".join(parts[-k:])
        hits = [n for n in params if n == suf or n.endswith("." + suf)]   # whole components only ('norm.weight' must not match 'layernorm.weight')
        if len(hits) == 1:
            return hits[0], params[hits[0]]
        if len(hits) > 1:
            break                                   # ambiguous (fused / repeated leaf name) -> not checkable
    return None, None


def engine_checks(vllm_model, names):
    params = dict(vllm_model.named_parameters())
    out = {}
    for n in names:
        vn, p = find_vllm_param(params, n)
        if p is not None:
            out[n] = abs_sum(p.data)
    return out


# ----------------------------------------------------------------------------------------------
# the frozen scorer / KL reference under FSDP2 (no exceptions inside forwards)
# ----------------------------------------------------------------------------------------------
class TinyHead(torch.nn.Module):
    """Stand-in lm_head of a truncated scorer: a [B, T, 1] zero tensor so the CausalLM forward completes normally (the
    reward reads the layer-READ_LAYER residual from a hook; logits are never used)."""

    def forward(self, h, *args, **kw):
        return h[..., :1].detach() * 0


def truncate_scorer_fsdp_safe(base, n_keep):
    """Keep decoder layers [0, n_keep) and replace lm_head by TinyHead (rl_disagg._truncate_scorer's raising head would abort
    an FSDP2 forward). Returns (n_layers_before, head_dropped)."""
    n_layers = len(base.model.layers)
    if not (0 < n_keep < n_layers):
        return n_layers, False
    base.model.layers = base.model.layers[:n_keep]
    tied = base.lm_head.weight.data_ptr() == base.model.embed_tokens.weight.data_ptr()
    if not tied:
        base.lm_head = TinyHead()
    return n_layers, not tied


@torch.no_grad()
def read_resid_noraise(model, layer, batch, pool="mean"):
    """mxf.inject.read_resid WITHOUT the _Stop exception: the forward runs to its (truncated) end so FSDP2's pre/post-forward
    hooks all fire. Same return values. Installed as rl_hf.read_resid in --full-param mode only."""
    from mxf.inject import get_layer
    captured = {}

    def cap(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).float()
    handle = get_layer(model, layer).register_forward_hook(cap)
    try:
        model(**batch)
    finally:
        handle.remove()
    h = captured["h"]
    mask = batch["attention_mask"].bool()
    if pool == "all":
        return h, mask
    if pool == "last":
        idx = mask.sum(1) - 1
        return h[torch.arange(h.shape[0]), idx]
    summed = (h * mask.unsqueeze(-1)).sum(1)
    return summed / mask.sum(1, keepdim=True).clamp(min=1)


def shard_frozen_bf16(model, world, device_type="cuda"):
    """FSDP2-shard a FROZEN bf16 model (scorer / KL reference): per decoder layer + root, params stay bf16 (no fp32
    masters), requires_grad False everywhere, reshard after forward (one layer's 0.8 GB gathered at a time)."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mesh = init_device_mesh(device_type, (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------
# fp32 lm_head for a TRAINABLE (FSDP2) head
# ----------------------------------------------------------------------------------------------
def install_fp32_head_trainable(model):
    """rl_disagg.install_fp32_head asserts a frozen head. Here lm_head is a trainable FSDP2 param: inside the forward hook the
    module's `weight` is the all-gathered bf16 param, so F.linear(x.float(), weight.float()) gives fp32 logits AND a
    gradient path into the (bf16 -> reduce-scattered fp32) weight. The fp32 copy of the head (5 GB for 248k x 5120) is
    a per-micro-batch transient saved for backward, freed with the graph."""
    import torch.nn.functional as F
    head = model.lm_head
    assert isinstance(head, torch.nn.Linear), f"lm_head must be nn.Linear, got {type(head).__name__}"

    def _fp32_out(mod, inp, out):
        w = mod.weight
        b = mod.bias
        with torch.autocast(inp[0].device.type, enabled=False):
            return F.linear(inp[0].float(), w.float(), None if b is None else b.float())
    return head.register_forward_hook(_fp32_out)


# ----------------------------------------------------------------------------------------------
# loss scaling: FSDP2 averages the reduce-scattered grads over ranks; the LoRA path's _sync_grads computes the
# sync_w-weighted mean (global grad = sum_r w_r g_r / sum_r w_r). Scaling rank r's loss by w_r * world / sum_r w_r
# before backward makes FSDP's mean exactly that weighted mean (unit-tested).
# ----------------------------------------------------------------------------------------------
def all_reduce_scalar(x, device):
    t = torch.tensor([float(x)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def all_reduce_max(x, device):
    t = torch.tensor([float(x)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def reshard_root(model):
    """After a no-grad forward the ROOT group's params (embed_tokens / norm / lm_head) stay unsharded until the next backward
    (FSDP2 keeps them for the backward that never comes): model.parameters() then yields plain bf16 tensors instead of the
    fp32 DTensor shards, which would break the shard-file publish, the checksums and the optimizer-state estimate. Reshard."""
    if hasattr(model, "reshard"):
        model.reshard()


@torch.no_grad()
def dummy_scorer_forward(scorer, tok, device):
    """One 2-token forward of the sharded scorer: every FSDP2 forward is a collective, so ranks whose reward shard needed fewer
    score() batches (uneven shards / empty texts) run this until every rank has issued the same number of forwards."""
    from mxf.config import READ_LAYER
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[sink, sink]], dtype=torch.long, device=device)
    read_resid_noraise(scorer, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")


def loss_scale(sync_w, tot_w, world):
    return (float(sync_w) * world / tot_w) if tot_w > 0 else 0.0


def weighted_mean_reference(grads, weights):
    """sum_r w_r g_r / sum_r w_r (the LoRA path's _sync_grads result), for the unit test."""
    tot = sum(weights)
    return sum(w * g for w, g in zip(weights, grads)) / tot


# ----------------------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------------------
def save_full_checkpoint(model, path, tok, model_id, is_main, world, nontext_shard, log, extra_meta):
    FT = fullft_module()
    FT.save_full_ckpt(model, path, tok, model_id, is_main, world, nontext_shard=nontext_shard, log=log, extra_meta=extra_meta)


def save_optim_dcp(model, opt, path, log=print):
    """Sharded AdamW state (fp32, ~2x the model) via torch.distributed.checkpoint; COLLECTIVE. Reload with load_optim_dcp on
    the same world size (DCP resharding across sizes works in principle; not exercised here)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    t0 = time.time()
    sd = get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=False))
    dcp.save({"optim": sd}, checkpoint_id=path)
    log(f"[fullparam] optimizer state -> {path} in {time.time() - t0:.0f}s")


def load_optim_dcp(model, opt, path, log=print):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
    t0 = time.time()
    sd = {"optim": get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=False))}
    dcp.load(sd, checkpoint_id=path)
    set_optimizer_state_dict(model, opt, optim_state_dict=sd["optim"], options=StateDictOptions(full_state_dict=False))
    log(f"[fullparam] optimizer state <- {path} in {time.time() - t0:.0f}s")


def peak_gb_all_ranks(device):
    """(this rank's peak, max over ranks) of torch.cuda.max_memory_allocated since the last reset."""
    mine = torch.cuda.max_memory_allocated(device) / 2**30
    t = torch.tensor([mine], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return mine, float(t.item())


@contextlib.contextmanager
def timed(d, key):
    t0 = time.time()
    try:
        yield
    finally:
        d[key] = d.get(key, 0.0) + time.time() - t0

# ----------------------------------------------------------------------------------------------
# step-time knobs for the FSDP2 policy: exact per-layer activation checkpointing of the suffix forward (sft/prefix_cache
# SuffixCheckpointer -- FSDP2-tested in sft/fullft_smoke.py) and a chunked, recomputed lm_head so no micro-batch ever
# materializes [tokens x 248k] logits. Both let the micro-batch grow from 8 to 32-64, i.e. 4-8x fewer per-micro-batch
# FSDP all-gathers / reduce-scatters (the whole cost of the update at mb 8: ~22 TB of NVLink traffic per step).
# ----------------------------------------------------------------------------------------------
def prefix_cache_module():
    try:
        from sft import prefix_cache as pc
    except ImportError:
        import prefix_cache as pc
    return pc


def suffix_checkpointer(model):
    """sft/prefix_cache.SuffixCheckpointer on the (FSDP2) policy: every decoder layer's forward is recomputed in the backward
    when `enabled` and a past_key_values cache is passed (the suffix forward); the prefix / marker-norm / no-grad forwards are
    untouched. Toggle .enabled around the suffix passes (FullParamCtx.suffix_ctx)."""
    return prefix_cache_module().SuffixCheckpointer(model)


class ChunkedHead:
    """Replaces lm_head.forward while `active`: the head input (final normed hidden states, [B, S, d]) is stashed and a [B, S, 1]
    dummy that still depends on it is returned, so HF's forward never builds [B, S, 248k] logits. `logp()` then computes, per
    chunk of `vocab_chunk` positions and under torch.utils.checkpoint (recomputed in the backward, nothing saved but the
    hidden chunk): fp32 logits = F.linear(h.float(), W.float()) -> log_softmax -> gather(target) and the entropy. The head
    weight is the FSDP2 root group's gathered parameter (the root is not resharded after forward), so gradients reach it
    exactly as through the original forward. Same math as rl_disagg._chunked_logp with --fp32-head."""

    def __init__(self, model):
        self.head = model.lm_head
        self._orig = self.head.forward
        self.active = False
        self.hidden = None
        self.head.forward = self._forward

    def undo(self):
        self.head.forward = self._orig

    def _forward(self, x, *args, **kw):
        if not self.active:
            return self._orig(x, *args, **kw)
        self.hidden = x
        return x[..., :1]

    def logp(self, hidden, targets, vocab_chunk, need_entropy_grad=False, fp32=True):
        """hidden [B, T, d] (positions predicting targets [B, T]) -> (new_lp [B, T] with grad, entropy [B, T])."""
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint
        W, b = self.head.weight, self.head.bias

        def fn(h, tgt, W, b):
            with torch.autocast(h.device.type, enabled=False):
                if fp32:
                    logits = F.linear(h.float(), W.float(), None if b is None else b.float())
                else:
                    logits = F.linear(h, W, b).float()
                lpf = torch.log_softmax(logits, -1)
                lp = lpf.gather(-1, tgt[..., None]).squeeze(-1)
                ent = -(lpf.exp() * lpf).sum(-1)
            return lp, ent
        T = hidden.shape[1]
        lps, ents = [], []
        for c0 in range(0, T, vocab_chunk):
            c1 = min(c0 + vocab_chunk, T)
            lp, ent = checkpoint(fn, hidden[:, c0:c1], targets[:, c0:c1], W, b, use_reentrant=False, preserve_rng_state=False)
            lps.append(lp); ents.append(ent if need_entropy_grad else ent.detach())
        return torch.cat(lps, 1), torch.cat(ents, 1)
