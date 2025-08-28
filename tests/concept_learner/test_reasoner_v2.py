import torch
from concept_learner.reasoning_v2 import ReasonerV2


torch.manual_seed(0)


def test_std_lib_installs_and_forward_shapes_macro_and_micro():
    B, T, d = 2, 5, 32
    model = ReasonerV2(d_model=d, max_steps=3, use_functions=True, num_fn_slots=8)
    # At least some function slots should be non-empty after std lib install
    nonempty = sum(1 for s in model.op_function.slots if len(s.steps) > 0)
    assert nonempty > 0

    H = torch.randn(B, T, d)
    z = torch.randn(B, d)
    mask = torch.ones(B, T, dtype=torch.long)

    # Macro mode
    model.exec_mode = "macro"
    out = model(H, z, mask)
    H_reasoned, s, stop_logits, action_logits = out
    assert H_reasoned.shape == (B, T, d)
    assert s.shape == (B, d)
    assert stop_logits.shape == (B, model.max_steps)
    assert action_logits.shape[0] == B and action_logits.shape[1] == model.max_steps
    # num actions = prims + function slots
    assert action_logits.shape[2] == (len(model.prims) + model.num_fn_slots)

    # Micro mode (stacks present in telemetry)
    model.exec_mode = "micro"
    out2 = model(H, z, mask)
    H2, s2, stop2, act2 = out2
    assert H2.shape == (B, T, d) and s2.shape == (B, d)
    tel = getattr(model, "_telemetry", {})
    assert "stack_depths" in tel and isinstance(tel["stack_depths"], list)


def test_mine_candidates_picks_frequent_patterns():
    d = 16
    model = ReasonerV2(d_model=d, max_steps=3, use_functions=True, num_fn_slots=8)
    # Create traces with repeated primitive pairs (0,1)
    traces = [
        [("prim", 0), ("prim", 1), ("prim", 2)],
        [("prim", 0), ("prim", 1)],
        [("prim", 3), ("prim", 0), ("prim", 1)],
        [("prim", 0), ("prim", 1), ("prim", 0)],
        [("prim", 0), ("prim", 1)],
    ]
    cands = model.mine_candidates(traces, min_support=2, max_len=3, topM=8, mdl_alpha=0.0, mdl_gain_threshold=0.0)
    assert len(cands) > 0
    # Expect first candidate to contain at least 2 primitive steps
    assert len(cands[0].steps) >= 2

