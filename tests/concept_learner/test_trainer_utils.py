import torch

from concept_learner.trainer import (
    extract_primitive_patterns_from_library,
    compress_trace_with_patterns,
    make_stop_targets_from_traces,
    ReplayBuffer,
)
from concept_learner.reasoning_v2 import ReasonerV2


torch.manual_seed(0)


def test_extract_patterns_and_compress_trace():
    d = 24
    r = ReasonerV2(d_model=d, use_functions=True, num_fn_slots=8)
    pats = extract_primitive_patterns_from_library(r)
    # Should return tuples of (sid, [prim_idxs...]) with length >= 2 for some entries
    assert isinstance(pats, list)
    has_len2 = any(len(seq) >= 2 for _, seq in pats)
    assert has_len2

    # Build a simple pattern list and compress a primitive trace
    if pats:
        sid, seq = pats[0]
        # Global action ids: primitives are [0..num_prims-1], functions start at offset
        num_prims = len(r.prims)
        trace_prim = list(seq) + list(seq)
        trace_global = trace_prim  # already primitive ids
        out = compress_trace_with_patterns(trace_global, [(sid, seq)], num_prims=num_prims)
        # Expect at least one replacement -> global id for function = num_prims + sid
        assert (num_prims + sid) in out


def test_make_stop_targets_from_traces_and_replay_buffer():
    traces = [[1, 2, 3], [], [4]]
    y = make_stop_targets_from_traces(traces, max_steps=4, device="cpu")
    assert y.shape == (3, 4)
    # stop index is len(trace)-1 (clamped)
    assert y[0].argmax().item() == 2
    assert y[1].argmax().item() in (0,)
    assert y[2].argmax().item() == 0

    # ReplayBuffer add/sample
    rb = ReplayBuffer(max_items=2)
    ids = torch.arange(6)
    mask = torch.tensor([1, 1, 1, 0, 0, 0])
    rb.add(ids, mask, [1, 2])
    rb.add(ids, mask, [3])
    rb.add(ids, mask, [4])  # triggers drop of oldest
    assert len(rb) == 2
    sample = rb.sample(1)
    assert len(sample) == 1
    it = sample[0]
    assert hasattr(it, "ids") and hasattr(it, "mask") and hasattr(it, "trace")

