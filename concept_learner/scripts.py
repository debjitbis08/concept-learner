from __future__ import annotations

import sys


def train_numbers_llm() -> None:  # pragma: no cover
    """
    Poetry script: run a long numbers + LLM-teacher training with our recommended knobs.
    Mirrors the manual CLI the user runs frequently.
    """
    from concept_learner.train import train_main

    argv = [
        "train_numbers_llm",
        "--device",
        "auto",
        "--steps",
        "10000",
        "--batch",
        "128",
        "--eval_analogy_batches",
        "1",
        "--sched",
        "cosine",
        "--warmup",
        "1000",
        "--lr",
        "1e-3",
        "--lr_min",
        "3e-5",
        "--vq_weight",
        "0.05",
        "--vq_final_weight",
        "0.2",
        "--ce_temp",
        "0.1",
        "--ce_mode",
        "pair",
        "--use_mlp_scorer",
        "--vq_warmup",
        "4000",
        "--rank_weight",
        "1.0",
        "--rank_margin",
        "0.2",
        "--relcls_weight",
        "0.2",
        "--analogy_weight",
        "0.1",
        "--analogy_warmup",
        "5000",
        "--use_llm_teacher",
        "--llm_model",
        "gpt-5",
        "--ckpt_dir",
        "checkpoints",
        "--save_every",
        "1000",
        "--resume_path",
        "checkpoints/best_an.pt",
        "--llm_data_json",
        "data/llm_numbers.json",
    ]
    sys.argv = argv
    train_main()
