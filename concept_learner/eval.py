import argparse

import torch

from concept_learner.data.episode_gen import EpisodeConfig, EpisodeGenerator
from concept_learner.train import ConceptLearner, TrainConfig


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tcfg = TrainConfig(device=args.device)
    ecfg = EpisodeConfig(device=args.device)
    gen = EpisodeGenerator(ecfg)
    model = ConceptLearner(tcfg).to(args.device)
    eval_analogy(model, gen, device=args.device)


if __name__ == "__main__":
    main()
