"""CPU correctness tests for sharing prefix gradients across optimizer micro-batches.

Run: OMP_NUM_THREADS=1 python -m pytest -q sft/test_prefix_sharing.py
The tiny Qwen integration test additionally uses the pinned transformers prefix-cache fork.
"""
import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from sft.prefix_cache import PrefixGradientAccumulator, expand_cache_copy, sync_accumulated_gradients


class ToyCache:
    def __init__(self, h):
        self.layers = [SimpleNamespace(keys=h, values=h.cos(), conv_states={0: h.sin(), 1: None},
                                       recurrent_states={0: h}, has_previous_state={0: True})]

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.keys = layer.keys.repeat_interleave(repeats, dim=0)
            layer.values = layer.values.repeat_interleave(repeats, dim=0)
            layer.conv_states[0] = layer.conv_states[0].repeat_interleave(repeats, dim=0)
            layer.recurrent_states[0] = layer.recurrent_states[0].repeat_interleave(repeats, dim=0)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(3, 4).double()
        self.head = torch.nn.Linear(4, 2).double()
        self.unused = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        self.rank_only = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))

    def prefix(self):
        return ToyCache(self.shared(torch.tensor([[0.3, 0.1, -0.2]], dtype=torch.float64)))

    def forward(self, x, cache, rank_extra=False):
        layer = cache.layers[0]
        h = self.shared(x) + layer.keys + layer.values + layer.conv_states[0] + layer.recurrent_states[0]
        out = self.head(h.tanh())
        return out + self.rank_only if rank_extra else out


def batches(rank=0):
    g = torch.Generator().manual_seed(42 + rank)
    return [(torch.randn(n, 3, generator=g, dtype=torch.float64),
             torch.randn(n, 2, generator=g, dtype=torch.float64)) for n in (2, 3, 1)]


def backward_repeated(model, data, rank_extra=False):
    loss = 0.0
    for x, target in data:
        cache = expand_cache_copy(model.prefix(), len(x))
        term = (model(x, cache, rank_extra) - target).square().mean() / len(data)
        loss += float(term.detach())
        term.backward()
    return loss


def backward_shared(model, data, rank_extra=False):
    bare = model.module if isinstance(model, DDP) else model
    original = bare.prefix()
    original_h = original.layers[0].keys
    visits = []
    original_h.register_hook(lambda g: visits.append(1))
    accum = PrefixGradientAccumulator(original)
    # Aliases must stay aliases; otherwise the prefix VJP can double-count gradients.
    assert accum.cache.layers[0].keys is accum.cache.layers[0].recurrent_states[0]
    loss = 0.0
    with model.no_sync() if isinstance(model, DDP) else contextlib.nullcontext():
        for x, target in data:
            cache = expand_cache_copy(accum.cache, len(x))
            term = (model(x, cache, rank_extra) - target).square().mean() / len(data)
            loss += float(term.detach())
            term.backward()
            accum.accumulate()
            assert not visits, "suffix backward unexpectedly traversed the prefix"
        accum.backward()
    assert visits == [1], "prefix must run backward exactly once"
    assert original.layers[0].keys is original_h and original_h.shape == (1, 4)
    with pytest.raises(RuntimeError, match="already been consumed"):
        accum.backward()
    return loss


def assert_grads_match(actual, expected):
    for (name, p), (ref_name, ref) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == ref_name
        assert (p.grad is None) == (ref.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, ref.grad, atol=1e-12, rtol=1e-11, msg=name)


def test_shared_prefix_loss_gradients_and_optimizer_steps():
    torch.manual_seed(7)
    ref = ToyModel()
    actual = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    opt = torch.optim.AdamW(actual.parameters(), lr=1e-3)
    for step in range(3):
        assert backward_shared(actual, batches(step)) == pytest.approx(backward_repeated(ref, batches(step)))
        assert_grads_match(actual, ref)
        sync_accumulated_gradients(actual.parameters())  # world=1 no-op
        opt.step()
        opt_ref.step()
        for p, r in zip(actual.parameters(), ref.parameters()):
            torch.testing.assert_close(p, r, atol=1e-12, rtol=1e-11)
        opt.zero_grad(set_to_none=True)
        opt_ref.zero_grad(set_to_none=True)


def test_cache_gradients_accumulate_in_fp32():
    h = torch.ones(1, 4, dtype=torch.bfloat16, requires_grad=True)
    accum = PrefixGradientAccumulator(ToyCache(h * 2))
    for _ in range(512):
        accum.cache.layers[0].keys.sum().backward()
        accum.accumulate()
    accum.backward()
    torch.testing.assert_close(h.grad, torch.full_like(h, 1024))


def _distributed_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        model = ToyModel()
        ddp = DDP(model)
        for _ in range(2):
            reference = copy.deepcopy(model)
            # Match DDP's mean of local means, including a parameter used on rank 0 only.
            for r in range(2):
                backward_repeated(reference, batches(r), rank_extra=r == 0)
            for p in reference.parameters():
                if p.grad is not None:
                    p.grad.div_(2)
            backward_shared(ddp, batches(rank), rank_extra=rank == 0)
            sync_accumulated_gradients(model.parameters(), bucket_bytes=32)
            assert_grads_match(model, reference)
            assert model.unused.grad is None
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-0.01)
            model.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_reduces_suffix_and_prefix_gradients(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("inject_mode", ["add", "replace"])
@pytest.mark.parametrize("marker", [2, 5])
def test_tiny_qwen_prefix_sharing(inject_mode, marker):
    import sys
    if not torch.cuda.is_available():
        sys.modules.setdefault("fla", None)
    from transformers import cache_utils
    if not hasattr(cache_utils, "_write_cached_state"):
        pytest.skip("integration requires the pinned transformers prefix-cache fork")
    from sft.test_head_on_labels import tiny_model
    from sft.prefix_cache import PrefixCache
    from mxf.inject import get_layer, hooked, make_inject_hook

    model = tiny_model()
    # Exercise gradients through both LoRA factors, including the shared prefix.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.01)
    reference = copy.deepcopy(model)
    full_reference = copy.deepcopy(model)
    prompt = [11, 12, 13, 14, 15, 16]
    pc = PrefixCache(model, prompt, marker, 0, get_layer(model, 1), 1.0, "cpu",
                     inject_dtype=torch.float32, inject_mode=inject_mode)
    ref_pc = PrefixCache(reference, prompt, marker, 0, get_layer(reference, 1), 1.0, "cpu",
                         inject_dtype=torch.float32, inject_mode=inject_mode)
    assert pc.prefix_ids == prompt[:marker]
    assert pc.suffix_prompt == prompt[marker:]
    dirs = torch.randn(3, 64)
    targets = [[20, 21, 22, 2], [30, 31, 2], [40, 41, 42, 43, 2]]
    shared = PrefixGradientAccumulator(pc.run_prefix())
    for start, end in ((0, 2), (2, 3)):
        args = (dirs[start:end], targets[start:end])
        # Independent reference: no prefix cache at all, and inject at the absolute
        # marker position with a different direction in each row. Also test a marker
        # inside the prompt: any prompt tokens AFTER injection must be recomputed.
        rows = [prompt + target for target in targets[start:end]]
        ids = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
            labels[i, len(prompt):len(row)] = torch.tensor(row[len(prompt):])
            mask[i, :len(row)] = True
        hook = make_inject_hook([v[None] for v in dirs[start:end]], [[marker]] * len(rows),
                                1.0, "cpu", torch.float32, mode=inject_mode)
        with hooked(get_layer(full_reference, 1), hook):
            full_out = full_reference(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        if len(rows) > 1:
            # Causality: changing the injected direction leaves earlier tokens
            # unchanged, but the injected marker itself is NOT reusable.
            torch.testing.assert_close(full_out.logits[0, :marker], full_out.logits[1, :marker])
            assert not torch.allclose(full_out.logits[0, marker], full_out.logits[1, marker])
        (full_out.loss / 2).backward()
        ref_out = ref_pc.forward(*args)
        (ref_out.loss / 2).backward()
        out = pc.forward(*args, prefix_cache=shared.cache)
        torch.testing.assert_close(out.loss, ref_out.loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out.loss, full_out.loss, atol=1e-6, rtol=1e-6)
        for i, row in enumerate(rows):
            torch.testing.assert_close(out.logits[i, :len(row) - marker],
                                       full_out.logits[i, marker:len(row)], atol=2e-6, rtol=1e-4)
        (out.loss / 2).backward()
        shared.accumulate()
    shared.backward()
    for expected in (reference, full_reference):
        for (name, p), (_, ref) in zip(model.named_parameters(), expected.named_parameters()):
            assert (p.grad is None) == (ref.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, ref.grad, atol=2e-6, rtol=1e-4, msg=name)
