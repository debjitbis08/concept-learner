import torch
from concept_learner.vq_layer import ResidualVQLayer

torch.manual_seed(0)


def _flatten_last_if_needed(t: torch.Tensor) -> torch.Size:
    # helper for index shapes (some impls return (B,1) for vectors)
    if t.ndim == 2 and t.shape[1] == 1:
        return torch.Size([t.shape[0]])
    return t.shape


def test_rvq_vector_shapes_and_grads():
    B, D = 8, 128
    layer = ResidualVQLayer(
        in_dim=D,
        rvq_dim=64,
        codebook_size=24,
        num_quantizers=3,
        decay=0.99,
        commitment_weight=0.25,
        kmeans_init=True,
        kmeans_iters=4,
        use_cosine_sim=True,
        threshold_ema_dead_code=2,
    )
    x = torch.randn(B, D, requires_grad=True)
    z_q, indices, loss = layer(x)
    assert z_q.shape == (B, D)
    assert isinstance(indices, (list, tuple)) and len(indices) == 3
    for idx in indices:
        s = _flatten_last_if_needed(idx)
        assert s == torch.Size([B])
        assert idx.dtype == torch.long
    assert loss.shape == () and loss.requires_grad
    (z_q.sum() + loss).backward()
    assert x.grad is not None


def test_rvq_sequence_shapes():
    B, L, D = 4, 5, 128
    layer = ResidualVQLayer(
        in_dim=D,
        rvq_dim=64,
        codebook_size=16,
        num_quantizers=2,
        decay=0.99,
        commitment_weight=0.25,
        use_cosine_sim=True,
    )
    x = torch.randn(B, L, D, requires_grad=True)
    z_q, indices, loss = layer(x)
    assert z_q.shape == (B, L, D)
    assert len(indices) == 2
    for idx in indices:
        assert idx.shape == (B, L)
    (z_q.sum() + loss).backward()
    assert x.grad is not None


def test_rvq_eval_is_deterministic():
    B, D = 2, 64
    layer = ResidualVQLayer(
        in_dim=D,
        rvq_dim=64,
        codebook_size=8,
        num_quantizers=2,
        decay=0.99,
        commitment_weight=0.25,
        use_cosine_sim=True,
    )
    layer.eval()
    x = torch.randn(B, D)
    with torch.no_grad():
        z1, i1, _ = layer(x)
        z2, i2, _ = layer(x)
    assert torch.allclose(z1, z2)
    # compare per-level
    for a, b in zip(i1, i2):
        assert torch.equal(a, b)
