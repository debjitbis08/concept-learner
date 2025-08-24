import torch
from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.model import CLModel

torch.manual_seed(0)


def make_batch(texts, max_len=24):
    tok = HFTokenizerWrapper("bert-base-cased")
    ids, mask = [], []
    for t in texts:
        e = tok.encode(t, max_len=max_len)
        ids.append(e.ids)
        mask.append(e.mask)
    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    return tok, ids, mask


def test_e2e_shapes_and_backward():
    tok, ids, mask = make_batch(["2 : 3 :: 5 : ?", "color of banana ?"], 24)
    model = CLModel(vocab_size=tok.tok.vocab_size, d_model=128, num_classes=3)

    logits_tok, logits_seq, vq_loss, _, _, _ = model(ids, mask)
    B, T = ids.shape
    assert logits_tok.shape == (B, T, 3)
    assert logits_seq.shape == (B, 3)
    assert vq_loss.shape == ()

    # fake labels
    y_tok = torch.zeros(B, T, dtype=torch.long)  # all class 0
    y_seq = torch.zeros(B, dtype=torch.long)  # all class 0

    # token loss, mask-aware
    C = logits_tok.size(-1)
    loss_tok_per_pos = torch.nn.functional.cross_entropy(
        logits_tok.view(-1, C), y_tok.view(-1), reduction="none"
    ).view(B, T)
    mask_f = mask.float()
    loss_tok = (loss_tok_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    # sequence loss
    loss_seq = torch.nn.functional.cross_entropy(logits_seq, y_seq)

    loss = loss_tok + loss_seq + 0.1 * vq_loss
    loss.backward()

    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_end_to_end_shapes_and_backward():
    tok, ids, mask = make_batch(["2 : 3 :: 5 : ?", "color of banana ?"], 16)
    model = CLModel(vocab_size=tok.vocab_size, d_model=64, num_classes=3)
    logits_tok, logits_seq, vq_loss, indices, stop_logits, _ = model(ids, mask)
    print("logits_tok:", logits_tok)
    print("logits_seq:", logits_seq)
    print("vq_loss:", vq_loss)
    B, T = ids.shape
    assert logits_tok.shape == (B, T, 3)
    assert logits_seq.shape == (B, 3)
    assert vq_loss.shape == ()
    # quick backward
    y_tok = torch.zeros(B, T, dtype=torch.long)
    y_seq = torch.zeros(B, dtype=torch.long)
    loss_tok = torch.nn.functional.cross_entropy(logits_tok.view(-1, 3), y_tok.view(-1))
    loss_seq = torch.nn.functional.cross_entropy(logits_seq, y_seq)
    loss = loss_tok + loss_seq + 0.1 * vq_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_pooled_head_shapes():
    tok, ids, mask = make_batch(["banana ?", "color of banana ?"], 24)
    model = CLModel(vocab_size=tok.tok.vocab_size, d_model=128, num_classes=3)
    logits_tok, logits_seq, vq_loss, _, _, _ = model(ids, mask)
    B, T = ids.shape
    assert logits_tok.shape == (B, T, 3)
    assert logits_seq.shape == (B, 3)
    assert vq_loss.shape == ()
