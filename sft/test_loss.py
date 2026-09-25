"""Loss/gradient/update parity for the opt-in compiled cross-entropy."""
import copy

import pytest
import torch

from sft.loss import cross_entropy_sum
from sft.test_head_on_labels import batch, tiny_model
from sft.pretrain import LabelHeadLM
from sft.prefix_cache import install_prefix_label_head


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_ce_ragged_ignored_and_strided(dtype):
    g = torch.Generator().manual_seed(45)
    for n, vocab in ((7, 101), (13, 257), (1, 101)):
        x = torch.randn(vocab, n, generator=g, dtype=dtype).T.requires_grad_()
        labels = torch.randint(vocab, (n,), generator=g)
        labels[::3] = -100
        ref = cross_entropy_sum(x, labels)
        actual = cross_entropy_sum(x, labels, fused=True)
        torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-6)
        gr, ga = [torch.autograd.grad(loss, x)[0] for loss in (ref, actual)]
        torch.testing.assert_close(ga, gr, atol=1e-6, rtol=0.01 if dtype == torch.bfloat16 else 1e-5)


@pytest.mark.parametrize("prefix_head", [False, True])
@pytest.mark.parametrize("ce_chunk", [0, 5])
def test_compiled_head_gradients_and_updates(prefix_head, ce_chunk):
    ref = tiny_model()
    # Fresh LoRA B=0 masks A gradients; exercise trained adapters too.
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.01)
    model = copy.deepcopy(ref)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if prefix_head:
        undo = install_prefix_label_head(model, ce_chunk, fused_ce=True)
        forward = lambda kw: model(**kw).loss
    else:
        head = LabelHeadLM(model, ce_chunk, fused_ce=True)
        forward = lambda kw: head(**kw)
    for step in range(2):
        kw = batch(seed=step + 1)
        expected, actual = ref(**kw).loss, forward(kw)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        expected.backward(); actual.backward()
        for (name, p), (_, q) in zip(ref.named_parameters(), model.named_parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(q.grad, p.grad, atol=2e-6, rtol=2e-4, msg=name)
        opt_ref.step(); opt.step()
        opt_ref.zero_grad(); opt.zero_grad()
        for p, q in zip(ref.parameters(), model.parameters()):
            torch.testing.assert_close(q, p, atol=2e-6, rtol=2e-4)
    if prefix_head:
        undo()
