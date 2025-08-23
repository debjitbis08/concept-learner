import torch
import pytest

from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.encoder import TinyEncoder, TinyEncoderConfig


def make_batch(texts, max_len=24):
    tok = HFTokenizerWrapper()
    ids = []
    mask = []
    for t in texts:
        e = tok.encode(t)
        ids.append(e.ids)
        mask.append(e.mask)
    return (
        tok,
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


def test_encoder_output_shapes_and_grad():
    tok, ids, mask = make_batch(["2 : 3 :: 5 : ?", "color of banana ?"], max_len=24)
    cfg = TinyEncoderConfig(
        vocab_size=tok.tok.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_len=128,
        dropout=0.0,
    )
    model = TinyEncoder(cfg)
    h, H = model(ids, mask)
    B, T = ids.shape
    assert h.shape == (B, cfg.d_model)
    assert H.shape == (B, T, cfg.d_model)

    (h.sum() + H.sum()).backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_cls_pooling_is_padding_invariant():
    tok, ids1, mask1 = make_batch(["2 : 3 :: 5 : ?"], max_len=24)
    _, ids2, mask2 = make_batch(["2 : 3 :: 5 : ?"], max_len=32)
    cfg = TinyEncoderConfig(vocab_size=tok.tok.vocab_size)
    model = TinyEncoder(cfg)
    model.eval()
    h1, _ = model(ids1, mask1)
    h2, _ = model(ids2, mask2)
    assert torch.allclose(h1, h2, atol=1e-5, rtol=1e-5)
