from __future__ import annotations

import json
import sys

try:
    import gradio as gr
except Exception as e:  # pragma: no cover
    print("Gradio is not installed. Install with `pip install gradio` to run the playground.")
    sys.exit(0)

from concept_learner.api import ConceptAPI


def main():
    api = ConceptAPI.load("checkpoints/latest.pt")  # adjust path as needed

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
        description="Query analogies and inspect model concepts.",
    )
    demo.launch()


if __name__ == "__main__":
    main()

