import argparse

import torch

from concept_learner.data.episode_gen import EpisodeConfig, EpisodeGenerator
from concept_learner.train import ConceptLearner, TrainConfig
from utils.ema import EMA


@torch.no_grad()
def eval_analogy(model: ConceptLearner, gen: EpisodeGenerator, device: str = "cpu", batches: int = 20, batch: int = 128):
    model.eval()
    correct = 0
    total = 0
    for _ in range(batches):
        analog = gen.sample_analogies(batch)
        A = model.encode(analog["A_desc"].to(device), analog["A_mask"].to(device))["z_q"]
        B = model.encode(analog["B_desc"].to(device), analog["B_mask"].to(device))["z_q"]
        C = model.encode(analog["C_desc"].to(device), analog["C_mask"].to(device))["z_q"]
        D = model.encode(analog["D_desc"].to(device), analog["D_mask"].to(device))["z_q"]
        # Predict D in-batch by nearest relation vector
        r_ab = model.analogy.rel_vec(A, B)  # (B, P)
        # Candidate relations r_cd over all Ds in batch for each C
        r_cd_all = model.analogy.rel_vec(C.unsqueeze(1), D.unsqueeze(0))  # (B, B, P)
        sim = torch.einsum(
            "bp,bnp->bn",
            torch.nn.functional.normalize(r_ab, dim=-1),
            torch.nn.functional.normalize(r_cd_all, dim=-1),
        )
        pred = sim.argmax(dim=-1)
        labels = torch.arange(batch, device=device)
        correct += (pred == labels).sum().item()
        total += batch
    acc = correct / total
    print(f"Analogy accuracy (in-batch): {acc:.3f}")


def _resolve_device(arg_device: str | None) -> str:
    dev = (arg_device or "auto").lower()
    if dev in ("auto", "", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="cuda|cpu|auto (default auto)")
    parser.add_argument("--ckpt", default="", help="optional checkpoint path to load (uses ema if present)")
    args = parser.parse_args()
    device = _resolve_device(args.device)
    print(f"[eval] Using device: {device} (cuda_available={torch.cuda.is_available()})")

    tcfg = TrainConfig(device=device)
    ecfg = EpisodeConfig(device=device)
    gen = EpisodeGenerator(ecfg)
    model = ConceptLearner(tcfg).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        # If EMA present, apply EMA weights for eval
        ema_state = state.get("ema", None)
        if ema_state is not None:
            ema = EMA(model)
            ema.load_state_dict(ema_state)
            ema.apply_to(model)
    eval_analogy(model, gen, device=device)


if __name__ == "__main__":
    main()
