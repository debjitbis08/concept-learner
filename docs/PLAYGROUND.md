Playground (Gradio) — Usage Guide

Overview
The playground is a small Gradio UI that lets you query analogies against a trained Concept Learner model. It’s intended as a lightweight demo and scaffold; the API is minimal and can be extended.

Prerequisites
- Train and save a checkpoint (see README): latest.pt or ckpt_*.pt
- Install Gradio: pip install gradio

Run the app
python apps/playground.py --ckpt checkpoints/latest.pt --device auto

Flags
- --ckpt: path to a checkpoint (latest.pt or a specific ckpt_*.pt)
- --device: cuda|cpu|auto (auto uses CUDA if available)

Colab usage
1) In a new Colab notebook, install dependencies and clone your repo (or upload files):
   !pip install gradio torch
   !git clone <your-repo-url>
   %cd concept-learner  # adjust to your repo root

2) Ensure you have a checkpoint (train in Colab or upload one to /content). Then run the playground with a public share link:
   !python apps/playground.py --ckpt checkpoints/latest.pt --device auto --share

3) Click the public Gradio link (starts with https://) to open the UI. Colab’s local URL won’t be reachable from your browser; the --share flag is required.

Notes
- If you’re using a GPU runtime in Colab, set Runtime → Change runtime type → GPU. The app will auto-detect CUDA.
- If your checkpoint is in Google Drive, mount it (from the left sidebar) and pass the path via --ckpt.

What it does now
- Loads the model checkpoint via ConceptAPI and displays a simple UI with inputs A, B, C for an analogy A:B :: C:?.
- ConceptAPI.complete_analogy is implemented for the toy numbers domain: it searches D in 0..N-1 that maximizes alignment of relation vectors r(A,B) and r(C,D) under the model’s analogy projector.

Extending the playground (recommended)
Implement ConceptAPI.complete_analogy in concept_learner/api.py to run the same in-batch analogy retrieval used in training/eval:
1) Render A, B, C, D candidates (D from a candidate pool) to tokens/masks using EpisodeGenerator.
2) Encode with model.encode(tokens, mask) → z_q.
3) Compute relation vectors r_ab = proj(A-B) and r_cd = proj(C-Dc) for each candidate Dc.
4) Score by normalized dot product and return the argmax candidate and scores.

Template pseudo-code
from concept_learner.data.episode_gen import EpisodeConfig, EpisodeGenerator
from concept_learner.train import TrainConfig
import torch, torch.nn.functional as F

class ConceptAPI:
    ...
    def complete_analogy(self, A, B, C):
        # Map raw inputs to integer indices or directly to rendered tokens/masks.
        # For the toy numbers domain, A/B/C can be integers.
        ecfg = EpisodeConfig(device=self.device)
        gen = EpisodeGenerator(ecfg)
        # Build candidate set D (e.g., all 0..N-1 or a subset)
        D_idx = torch.arange(0, gen.n_items, device=self.device)
        A_desc, A_mask, _ = gen._render_batch(torch.tensor([A], device=self.device))
        B_desc, B_mask, _ = gen._render_batch(torch.tensor([B], device=self.device))
        C_desc, C_mask, _ = gen._render_batch(torch.tensor([C], device=self.device))
        D_desc, D_mask, _ = gen._render_batch(D_idx)
        A_z = self.model.encode(A_desc, A_mask)["z_q"]
        B_z = self.model.encode(B_desc, B_mask)["z_q"]
        C_z = self.model.encode(C_desc, C_mask)["z_q"]
        D_z = self.model.encode(D_desc, D_mask)["z_q"]
        r_ab = self.model.analogy.rel_vec(A_z, B_z)              # (1,P)
        r_cd = self.model.analogy.rel_vec(C_z, D_z)              # (N,P)
        sim = torch.einsum("ip,np->in", F.normalize(r_ab, -1), F.normalize(r_cd, -1)).squeeze(0)
        pred_idx = int(sim.argmax().item())
        return {"prediction": int(D_idx[pred_idx].item()), "scores": sim.tolist()}

Notes
- You can cache an EpisodeGenerator inside ConceptAPI for speed.
- For multi-domain inputs (words/shapes/rhythm), map inputs to the appropriate descriptor space before rendering tokens/masks.
- Consider adding an explain() endpoint (nearest codes, prototypes) to display interpretability info in the UI.
