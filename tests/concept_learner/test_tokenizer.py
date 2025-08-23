import pytest
from concept_learner.tokenizer import HFTokenizerWrapper, EncodeOutput


def test_encode_returns_correct_lengths():
    tok = HFTokenizerWrapper("bert-base-cased")
    out = tok.encode("2 : 3 :: 5 : ?", max_len=16)
    assert isinstance(out, EncodeOutput)
    assert len(out.ids) == 16
    assert len(out.mask) == 16


def test_mask_matches_pad_ids():
    tok = HFTokenizerWrapper("bert-base-cased")
    out = tok.encode("banana ?", max_len=12)
    # check: whenever there’s a PAD id, mask should be 0
    for i, m in zip(out.ids, out.mask):
        if i == tok.pad_id:
            assert m == 0
        else:
            assert m == 1


def test_cls_and_sep_present():
    tok = HFTokenizerWrapper("bert-base-cased")
    out = tok.encode("color of banana ?", max_len=12)
    assert tok.cls_id in out.ids
    assert tok.sep_id in out.ids


def test_decode_roundtrip():
    tok = HFTokenizerWrapper("bert-base-cased")
    out = tok.encode("color of banana ?", max_len=12)
    text = tok.decode(out.ids, skip_special_tokens=True)
    assert "banana" in text.lower()
