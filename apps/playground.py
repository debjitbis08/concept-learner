from __future__ import annotations

import json
import sys

try:
    import gradio as gr
except Exception as e:  # pragma: no cover
    print("Gradio is not installed. Install with `pip install gradio` to run the playground.")
    sys.exit(0)

from concept_learner.api import ConceptAPI
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt", help="path to checkpoint (latest.pt or a specific ckpt_*.pt)")
    parser.add_argument("--device", default="auto", help="cuda|cpu|auto")
    parser.add_argument("--share", action="store_true", help="launch Gradio with public share link (useful in Colab)")
    args = parser.parse_args()

    dev = args.device.lower()
    if dev in ("auto", "gpu"):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = dev

    api = ConceptAPI.load(args.ckpt, device=device)

    def analogy(a: str, b: str, c: str):
        res = api.complete_analogy(a, b, c)
        return {"prediction": res.prediction, "scores": res.scores, "concepts": res.concepts}

    def feedback(a: str, b: str, c: str, d: str, correct: bool):
        api.record_feedback(a, b, c, d, correct)
        return "Thanks!"

    demo = gr.Interface(
        fn=analogy,
        inputs=[gr.Text(label="A"), gr.Text(label="B"), gr.Text(label="C")],
        outputs=gr.JSON(label="Result"),
        allow_flagging="never",
        title="Concept Learner Playground",
        description=f"Query analogies and inspect model concepts. Using device={device} ckpt={args.ckpt}",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
