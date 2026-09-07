"""torchrun target for modal_sft.cluster_check: proves every rank of a multi-node launch joined ONE NCCL
process group (world size + distinct hosts) and measures all-reduce bandwidth at the LoRA-gradient size, so
the cross-node comm cost of a DDP step (one all-reduce of the trainable grads per optimizer step) is a
measured number, not a guess. Prints DDP_CHECK_OK ... on rank 0."""
import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allreduce-mb", type=int, default=1800,
                    help="fp32 all-reduce payload in MiB (~LoRA r64 all-linear grads of a 27B)")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()

    dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl"))
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    # 1) membership: sum of ranks == world*(world-1)/2 iff everyone is in the same group; gather hosts
    t = torch.tensor([float(rank)], device=dev)
    dist.all_reduce(t)
    assert int(t.item()) == world * (world - 1) // 2, f"rank sum {t.item()} != {world * (world - 1) // 2}"
    hosts = [None] * world
    dist.all_gather_object(hosts, f"{socket.gethostname()}:node{os.environ.get('GROUP_RANK', '?')}")
    if rank == 0:
        print(f"[ddp_check] world={world} joined one group; nodes={sorted(set(hosts))}", flush=True)

    # 2) bandwidth at gradient size (ring all-reduce bus bw = alg bw * 2(n-1)/n)
    n = a.allreduce_mb * 2**20 // 4
    x = torch.ones(n, device=dev)
    for _ in range(2):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(a.iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / a.iters
    nbytes = n * 4
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world
    assert torch.isfinite(x).all()
    if rank == 0:
        print(f"DDP_CHECK_OK world={world} nodes={len(set(hosts))} allreduce {a.allreduce_mb} MiB fp32: "
              f"{dt * 1e3:.0f} ms/iter  algbw {algbw:.1f} GB/s  busbw {busbw:.1f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
