"""CPU unit test for sft/pretrain.py's exact-speed paths (no GPU, tiny random Qwen3.5-arch model):

  * LabelHeadLM (--head-on-labels) loss == HF ForCausalLM loss, with and without --ce-chunk, and the
    LoRA gradients match.
  * open_vec_bank / gather_rows read vecs.f32 and vecs.f16 banks to identical float32 rows.

    PYTHONPATH=.:sft python sft/test_head_on_labels.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

if not torch.cuda.is_available():
    # transformers binds fla's Triton GatedDeltaNet kernel at import time when `fla` is importable; on CPU
    # there is no Triton driver, so hide fla and take the pure-torch fallback (what the test needs anyway).
    sys.modules.setdefault("fla", None)

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain  # noqa: E402  (sft/pretrain.py)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        max_position_embeddings=256, pad_token_id=0,
    )
    model = Qwen3_5ForCausalLM(cfg).float()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, use_rslora=True,
                                             target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.train()
    return model


def batch(seed=1, B=3, L=24, prompt_len=10, vocab=512):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab, (B, L), generator=g)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100                       # prompt masked
    attn = torch.ones(B, L, dtype=torch.bool)
    for b, tail in enumerate([0, 3, 7][:B]):            # ragged right padding
        if tail:
            labels[b, L - tail:] = -100
            attn[b, L - tail:] = False
            input_ids[b, L - tail:] = 0
    return dict(input_ids=input_ids, attention_mask=attn, labels=labels)


def lora_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def test_loss_and_grad_parity():
    kw = batch()
    for ce_chunk in (0, 5):
        model = tiny_model()
        loss_hf = model(**kw).loss
        loss_hf.backward()
        g_hf = lora_grads(model)
        model.zero_grad()
        head = pretrain.LabelHeadLM(model, ce_chunk=ce_chunk)
        loss_lh = head(**kw)
        loss_lh.backward()
        g_lh = lora_grads(model)
        diff = (loss_hf - loss_lh).abs().item()
        n_lab = int((kw["labels"][:, 1:] != -100).sum())
        print(f"ce_chunk={ce_chunk}: hf {loss_hf.item():.6f} head_on_labels {loss_lh.item():.6f} "
              f"|diff| {diff:.2e} ({n_lab} label tokens)")
        assert diff < 1e-5, diff
        assert g_hf.keys() == g_lh.keys() and g_hf, "LoRA grads missing"
        worst = max((g_hf[n] - g_lh[n]).abs().max().item() / (g_hf[n].abs().max().item() + 1e-12) for n in g_hf)
        print(f"  worst relative grad diff over {len(g_hf)} LoRA tensors: {worst:.2e}")
        assert worst < 1e-4, worst
    # parity_check() itself (the --parity-check flag) on the same batch
    pretrain.parity_check(model, pretrain.LabelHeadLM(model), kw)


def test_vec_bank_f16_and_f32():
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((7, pretrain.D_MODEL)).astype(np.float16)   # f16-representable values
    with tempfile.TemporaryDirectory() as d32, tempfile.TemporaryDirectory() as d16:
        rows.astype(np.float32).tofile(f"{d32}/vecs.f32")
        rows.tofile(f"{d16}/vecs.f16")
        v32, f32 = pretrain.open_vec_bank(d32, 7)
        v16, f16 = pretrain.open_vec_bank(d16, 7)
        assert (f32, f16) == ("vecs.f32", "vecs.f16"), (f32, f16)
        for idx in ([3, 0, 6], 5):
            a, b = pretrain.gather_rows(v32, idx), pretrain.gather_rows(v16, idx)
            assert a.dtype == b.dtype == torch.float32, (a.dtype, b.dtype)
            assert torch.equal(a, b) and torch.equal(a, torch.from_numpy(rows[idx].astype(np.float32)))
        try:
            pretrain.open_vec_bank(tempfile.gettempdir() + "/definitely_missing_bank", 1)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing bank should raise")
    print("vec bank f32/f16: OK")


def test_batched_tokenization_matches_per_record_chat_template():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from mxf.prompts import MARKER, build_sft_ids

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2,
                                         "hello": 3, "world": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]",
                                 unk_token="[UNK]", additional_special_tokens=[MARKER])
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %} assistant"
    records = [{"vec_idx": i * 3, "target_text": text} for i, text in enumerate(
        ["hello world", "", "héllo 世界\nworld", "hello " * 200, "[EOS] hello", "world", "   "])]
    for max_seq in (64, 192, 512):
        expected = []
        for record in records:
            ids, labels, positions = build_sft_ids(tok, record["target_text"])
            expected.append((ids[:max_seq], labels[:max_seq], positions, record["vec_idx"]))
        for chunk in (1, 3, 4096):
            assert pretrain.tokenize_records(records, tok, max_seq, chunk_size=chunk) == expected


if __name__ == "__main__":
    test_loss_and_grad_parity()
    test_vec_bank_f16_and_f32()
    test_batched_tokenization_matches_per_record_chat_template()
    print("ALL_TESTS_PASSED")
