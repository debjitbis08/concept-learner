from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Reuse the building blocks from the repo without importing train.py (which
# depends on optional data modules that may not be present at import time).
from .model.backbone import TinyBackbone
from .model.domain import DomainAdapter, DomainToken
from .model.relation_head import DistMultHead, AnalogyProjector
from .model.vq_layer import EmaVectorQuantizer


class VisualConceptModel(nn.Module):
    """
    Minimal end-to-end model assembled from the building blocks in
    concept_learner/model to mirror ConceptLearner's architecture for graph
    visualization.

    The forward path touches all major components so that torchview includes
    them in the rendered graph.
    """

    def __init__(
        self,
        vocab_size: int = 12,
        d_model: int = 128,
        code_dim: int = 64,
        num_codes: int = 32,
        num_relations: int = 8,
        max_len: int = 8,
        num_domains: int = 1,
        use_domain_token: bool = True,
        use_domain_adapter: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = TinyBackbone(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        self.domain_token = DomainToken(num_domains, d_model) if use_domain_token else None
        self.domain_adapter = DomainAdapter(num_domains, d_model) if use_domain_adapter else None

        self.enc = nn.Linear(d_model, code_dim)
        self.vq_global = EmaVectorQuantizer(num_codes=num_codes, code_dim=code_dim, commitment_cost=0.25)

        self.proj1 = nn.Linear(code_dim, code_dim)
        self.proj2 = nn.Linear(code_dim, code_dim)

        self.rel = DistMultHead(concept_dim=code_dim, num_relations=num_relations)
        self.analogy = AnalogyProjector(dim=code_dim, proj_dim=min(32, code_dim))

        self.same_head = nn.Sequential(
            nn.Linear(2 * code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, 2),
        )
        self.instance_head = nn.Sequential(
            nn.Linear(2 * code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, 2),
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Use a fixed domain=0 for visualization; DomainAdapter/Token support ints
        h = self.backbone(tokens, mask)
        if self.domain_token is not None:
            h = self.domain_token(h, 0)
        if self.domain_adapter is not None:
            h = self.domain_adapter(h, 0)
        z = self.enc(h)
        z_q, indices, vq_loss = self.vq_global(z)

        # Touch heads so they appear in the graph. Use simple self-analogies.
        _ = self.proj1(z_q)
        _ = self.proj2(z_q)

        # DistMultHead expects subject, relation ids, object
        B = tokens.size(0)
        r = torch.zeros(B, dtype=torch.long, device=tokens.device)
        _ = self.rel(z_q, r, z_q)

        # Analogy projector: r(a,b)
        _ = self.analogy.rel_vec(z_q, z_q)

        # Same/instance heads over concatenated codes
        concat = torch.cat([z_q, z_q], dim=-1)
        _ = self.same_head(concat)
        _ = self.instance_head(concat)

        # Return the main quantized code as the output node
        return z_q


def _fallback_svg(out_path: Path, width: int = 960, height: int = 540) -> None:
    # Lightweight, dependency-free fallback so CI can still produce an artifact
    # even if torchview/graphviz are not available in the environment.
    msg = (
        "torchview is not available in this environment.\n"
        "Install it and run the update command to regenerate this image.\n\n"
        "This placeholder represents the ConceptLearner architecture composed of:\n"
        "TinyBackbone -> [DomainToken, DomainAdapter] -> Linear(enc) -> EmaVectorQuantizer\n"
        "-> {proj1, proj2, DistMultHead, AnalogyProjector, same_head, instance_head}."
    )
    # Simple SVG with mono text
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#0b1021'/>",
        "<g font-family='monospace' font-size='14' fill='#d6e2ff'>",
    ]
    x, y = 24, 40
    for line in msg.splitlines():
        svg.append(f"<text x='{x}' y='{y}'>{line}</text>")
        y += 22
    svg.append("</g>")
    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:  # pragma: no cover - small utility
    parser = argparse.ArgumentParser(description="Render the model architecture graph using torchview")
    parser.add_argument("--out", default="docs/model_architecture.svg", help="output image path (.svg or .png)")
    parser.add_argument("--batch", type=int, default=2, help="dummy batch size for the forward pass")
    parser.add_argument("--len", type=int, default=8, help="sequence length for dummy tokens/mask")
    parser.add_argument("--vocab", type=int, default=12, help="vocab size for dummy tokens")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = VisualConceptModel(vocab_size=args.vocab, max_len=args.len)
    model.eval()

    # Dummy integer tokens and mask
    tokens = torch.randint(low=0, high=args.vocab, size=(args.batch, args.len), dtype=torch.long)
    mask = torch.ones(args.batch, args.len, dtype=torch.bool)

    try:
        # Import torchview lazily so this utility can still run without it
        from torchview import draw_graph  # type: ignore

        graph = draw_graph(
            model,
            input_data=(tokens, mask),
            expand_nested=True,
            graph_name="ConceptLearner",
        )
        # Choose format from extension
        fmt = out.suffix.lstrip(".") or "svg"
        # torchview API has changed across versions; try a few safe options
        try:
            # Older torchview accepted a size tuple
            graph.resize_graph(scale=1.0, size=(18, 12))
        except TypeError:
            # Newer torchview exposes scale/size_per_element/min_size only
            graph.resize_graph(scale=1.0)

        if hasattr(graph, "save_graph"):
            graph.save_graph(str(out), format=fmt)
        else:
            # Older torchview exposes the underlying graphviz object
            vg = graph.visual_graph
            vg.render(outfile=str(out), format=fmt)
        print(f"Saved model architecture graph to {out}")
    except Exception as ex:
        # Fallback: emit a placeholder SVG so docs don't break
        print(f"torchview unavailable or failed ({ex}); writing a placeholder SVG to {out}")
        if out.suffix.lower() != ".svg":
            # force .svg on fallback
            out = out.with_suffix(".svg")
        _fallback_svg(out)


if __name__ == "__main__":  # pragma: no cover
    main()
