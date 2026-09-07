"""CPU tests for sft/pretrain.py::step_groups -- the micro-batch composition of the per-example SFT path.

    python -m pytest -q sft/test_length_bucket.py

Checks: (1) the default reproduces the legacy inline order byte-for-byte (length-sorted micro-batches, shuffled
whole); (2) --length-bucket keeps every optimizer step's example multiset equal to its i.i.d. window of the seeded
example shuffle (only the grouping into micro-batches changes), partitions the epoch, and leaves the micro-batch /
step counts (=> steps_total, checkpoint cadence, --save-examples) unchanged; (3) the composition is a pure function
of (lengths, epoch, flags), so --skip-steps resume replays exactly the same micro-batches; (4) padding: bucketed
micro-batches pad a few percent where randomly composed ones pad >30% (8-32-token targets)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sft.pretrain import step_groups  # noqa: E402

PROMPT = 103   # prompt+marker tokens of the current prompt; rows are prompt + 8..32 target + eos


def _lengths(n, seed):
    return PROMPT + np.random.default_rng(seed).integers(9, 34, size=n)


def _legacy(lengths, bs, ga, ep):
    """The pre-step_groups inline code: toks_cache.sort(key=len) -> chunks -> rng(ep).shuffle -> groups."""
    toks = [([0] * int(L), i) for i, L in enumerate(lengths)]
    toks.sort(key=lambda t: len(t[0]))
    micro = [toks[s : s + bs] for s in range(0, len(toks), bs)]
    np.random.default_rng(ep).shuffle(micro)
    groups = [micro[s : s + ga] for s in range(0, len(micro), ga)]
    return [[[t[1] for t in mb] for mb in g] for g in groups]


def _as_lists(groups):
    return [[list(map(int, mb)) for mb in g] for g in groups]


def _step_ids(group):
    return sorted(int(i) for mb in group for i in mb)


@pytest.mark.parametrize("n,bs,ga", [(1000, 16, 4), (2048, 64, 8), (777, 8, 3)])
def test_default_is_the_legacy_inline_order(n, bs, ga):
    for ep in (0, 1):
        L = _lengths(n, ep)
        assert _as_lists(step_groups(L, bs, ga, ep, length_bucket=False)) == _legacy(L, bs, ga, ep)


@pytest.mark.parametrize("n,bs,ga", [(1000, 16, 4), (2048, 64, 8), (777, 8, 3), (2050, 64, 32)])
def test_length_bucket_step_multiset_is_the_iid_window(n, bs, ga):
    L = _lengths(n, 3)
    groups = step_groups(L, bs, ga, 0, length_bucket=True)
    perm = np.random.default_rng(0).permutation(n)
    W = bs * ga
    assert len(groups) == math.ceil(n / W) == math.ceil(math.ceil(n / bs) / ga)   # steps_total as pretrain.py computes it
    assert sum(len(g) for g in groups) == math.ceil(n / bs)                        # micro-batches/epoch unchanged
    for k, g in enumerate(groups):
        window = perm[k * W : (k + 1) * W]
        assert _step_ids(g) == sorted(window.tolist())          # same SET of examples as the un-sorted random window
        sizes = [len(mb) for mb in g]
        assert sum(sizes) == len(window) and all(s == bs for s in sizes[:-1]) and 0 < sizes[-1] <= bs
        assert np.all(np.diff(L[np.concatenate(g)]) >= 0)      # length-sorted within the step => minimal padding
    assert sorted(i for g in groups for i in _step_ids(g)) == list(range(n))   # every example exactly once


@pytest.mark.parametrize("length_bucket", [False, True])
def test_pure_function_so_skip_steps_resume_is_exact(length_bucket):
    L = _lengths(3000, 5)
    full = _as_lists(step_groups(L, 16, 4, 0, length_bucket))
    again = _as_lists(step_groups(L, 16, 4, 0, length_bucket))
    assert full == again
    skip = 17                                                   # --skip-steps 17: the trainer iterates the same list
    assert full[skip:] == again[skip:] and len(full) == math.ceil(3000 / 64)
    other_epoch = _as_lists(step_groups(L, 16, 4, 1, length_bucket))
    assert other_epoch != full                                  # the epoch seed is actually used


def _pad_frac(groups, suffix, pad_multiple=1):
    real = slots = 0
    for g in groups:
        for mb in g:
            Lp = math.ceil(int(suffix[mb].max()) / pad_multiple) * pad_multiple
            real += int(suffix[mb].sum())
            slots += Lp * len(mb)
    return 1 - real / slots


def test_bucketed_padding_is_small_random_is_not():
    n, bs, ga = 64 * 32 * 20, 64, 32
    suffix = np.random.default_rng(0).integers(10, 35, size=n)   # marker + 8..32 target + eos
    lengths = 102 + suffix
    perm = np.random.default_rng(0).permutation(n)
    random_windows = [[perm[s + k * bs : s + (k + 1) * bs] for k in range(ga)] for s in range(0, n, bs * ga)]
    pf_random = _pad_frac(random_windows, suffix)
    pf_bucket = _pad_frac(step_groups(lengths, bs, ga, 0, True), suffix)
    pf_default = _pad_frac(step_groups(lengths, bs, ga, 0, False), suffix)
    assert pf_random > 0.30, pf_random
    assert pf_bucket < 0.05 and pf_default < 0.02, (pf_bucket, pf_default)
    # the legacy pad_multiple=8 rounding alone costs ~15% on these lengths -- that is why --pad-multiple exists
    assert 0.10 < _pad_frac(step_groups(lengths, bs, ga, 0, False), suffix, pad_multiple=8) < 0.20
