from concept_learner.trainer import setup, CLConfig, make_synth_batch, train_step
from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.model import CLModel
import torch


def main():
    cfg = CLConfig()
    cfg.device = setup(seed=0)

    tok = HFTokenizerWrapper(cfg.pretrained_tok)
    model = CLModel(
        vocab_size=tok.vocab_size,
        d_model=cfg.d_model,
        num_classes=cfg.num_classes,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(5):
        batch = make_synth_batch(
            tok, B=4, T=cfg.max_len, C=cfg.num_classes, max_steps=4, device=cfg.device
        )
        stats = train_step(
            model, batch, opt, lambda_vq=cfg.lambda_vq, lambda_stop=cfg.lambda_stop
        )
        print(
            f"step {step}: loss={stats['loss']:.3f} tok={stats['tok']:.3f} seq={stats['seq']:.3f} "
            f"vq={stats['vq']:.3f} stop={stats['stop']:.3f} stop_probs={stats['stop_probs']}"
        )


if __name__ == "__main__":
    main()
