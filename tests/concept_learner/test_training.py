import torch
from concept_learner.trainer import setup, CLConfig, make_synth_batch, train_step
from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.model import CLModel


def _make_model_and_tok(cfg: CLConfig):
    tok = HFTokenizerWrapper(cfg.pretrained_tok)
    model = CLModel(
        vocab_size=tok.tok.vocab_size, d_model=cfg.d_model, num_classes=cfg.num_classes
    ).to(cfg.device)
    return tok, model


def test_backward_no_nans():
    cfg = CLConfig()
    cfg.device = setup(seed=0)
    tok, model = _make_model_and_tok(cfg)
    ids, mask, y_tok, y_seq, y_stop = make_synth_batch(
        tok, B=3, T=cfg.max_len, C=cfg.num_classes, device=cfg.device
    )
    logits_tok, logits_seq, vq_loss, _, _, _ = model(ids, mask)

    # simple combined loss
    C = logits_tok.size(-1)
    loss_tok = torch.nn.functional.cross_entropy(logits_tok.view(-1, C), y_tok.view(-1))
    loss_seq = torch.nn.functional.cross_entropy(logits_seq, y_seq)
    loss = loss_tok + loss_seq + 0.1 * vq_loss
    loss.backward()

    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_eval_determinism():
    cfg = CLConfig()
    cfg.device = setup(seed=123)
    tok, model = _make_model_and_tok(cfg)
    model.eval()
    ids, mask, _, _, _ = make_synth_batch(
        tok, B=2, T=cfg.max_len, C=cfg.num_classes, device=cfg.device
    )
    with torch.no_grad():
        out1 = model(ids, mask)
        out2 = model(ids, mask)
    # compare logits
    assert torch.allclose(out1[0], out2[0]) and torch.allclose(out1[1], out2[1])


def test_train_step_runs_and_updates():
    cfg = CLConfig()
    cfg.device = setup(seed=0)
    tok, model = _make_model_and_tok(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    batch = make_synth_batch(
        tok, B=4, T=cfg.max_len, C=cfg.num_classes, device=cfg.device
    )
    stats1 = train_step(model, batch, opt, lambda_vq=cfg.lambda_vq)
    stats2 = train_step(model, batch, opt, lambda_vq=cfg.lambda_vq)
    # loss should be finite and (often) change across steps
    assert all(_all_finite(v) for v in stats1.values())
    assert all(_all_finite(v) for v in stats2.values())


def _all_finite(x) -> bool:
    t = torch.as_tensor(x, dtype=torch.float32)
    return torch.isfinite(t).all().item()
