from dataclasses import dataclass
import random, numpy as np, torch
import torch.nn.functional as F
from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.model import CLModel


@dataclass
class CLConfig:
    d_model: int = 128
    rvq_dim: int = 64
    codebook_size: int = 24
    num_quantizers: int = 3
    num_classes: int = 3
    lambda_vq: float = 0.1
    lambda_stop: float = 0.1
    lr: float = 3e-4
    max_len: int = 24
    pretrained_tok: str = "bert-base-cased"
    device: str = "cpu"  # or set by setup()


def setup(seed: int = 0, device: str | None = None) -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_synth_batch(tok, B=4, T=16, C=3, max_steps=4, device="cpu"):
    texts = ["2 : 3 :: 5 : ?", "color of banana ?", "banana banana ?", "is 3 > 2 ?"]
    ids, mask = [], []
    for i in range(B):
        e = tok.encode(texts[i % len(texts)], max_len=T)
        ids.append(e.ids)
        mask.append(e.mask)

    ids = torch.tensor(ids, dtype=torch.long, device=device)
    mask = torch.tensor(mask, dtype=torch.long, device=device)

    # random labels (for smoke testing)
    y_tok = torch.randint(0, C, (B, T), device=device)
    y_seq = torch.randint(0, C, (B,), device=device)

    # STOP: pick a random step for each example
    y_stop = torch.zeros(B, max_steps, dtype=torch.long, device=device)
    for i in range(B):
        stop_idx = torch.randint(0, max_steps, (1,)).item()
        y_stop[i, stop_idx] = 1

    return ids, mask, y_tok, y_seq, y_stop


def train_step(
    model: CLModel,
    batch,
    optimizer: torch.optim.Optimizer,
    lambda_vq: float = 0.1,
    lambda_stop: float = 0.1,
):
    ids, mask, y_tok, y_seq, y_stop = batch
    model.train()
    logits_tok, logits_seq, vq_loss, _, stop_logits, action_logits = model(ids, mask)

    # token loss (mask-aware)
    B, T, C = logits_tok.shape
    loss_tok_per_pos = F.cross_entropy(
        logits_tok.view(-1, C), y_tok.view(-1), reduction="none"
    ).view(B, T)
    mask_f = mask.float()
    loss_tok = (loss_tok_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    # sequence loss
    loss_seq = F.cross_entropy(logits_seq, y_seq)

    # STOP loss
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, y_stop.float())

    # total
    loss = loss_tok + loss_seq + lambda_vq * vq_loss + lambda_stop * stop_loss

    # action loss
    num_ops = action_logits.size(-1)
    y_act = torch.randint(
        0, num_ops, (ids.size(0), action_logits.size(1)), device=ids.device
    )
    act_loss = F.cross_entropy(action_logits.reshape(-1, num_ops), y_act.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # convert STOP to probs for logging
    stop_probs = torch.sigmoid(stop_logits).detach().cpu().numpy().mean(axis=0)

    return {
        "loss": float(loss.item()),
        "tok": float(loss_tok.item()),
        "seq": float(loss_seq.item()),
        "vq": float(vq_loss.item()),
        "stop": float(stop_loss.item()),
        "stop_probs": stop_probs.tolist(),  # avg prob per step
    }
