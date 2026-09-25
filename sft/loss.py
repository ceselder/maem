"""Cross-entropy with fp32 reduction and an optional compiled implementation."""
from functools import lru_cache

import torch
import torch.nn.functional as F


def _cross_entropy_sum(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction="sum")


@lru_cache(maxsize=1)
def _compiled_cross_entropy():
    # Fuse the cast with CE so Inductor can avoid full fp32 logits/log-softmax
    # intermediates. The vocabulary GEMM stays in PyTorch.
    return torch.compile(_cross_entropy_sum, fullgraph=True, dynamic=True)


def cross_entropy_sum(logits, targets, fused=False):
    """Same sum-over-labels objective; compiled reductions may round differently."""
    return (_compiled_cross_entropy() if fused else _cross_entropy_sum)(logits, targets)
