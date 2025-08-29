import torch
import torch.nn as nn
from concept_learner.encoder import TinyEncoder, TinyEncoderConfig
from concept_learner.vq_layer import ResidualVQLayer
from concept_learner.reasoning_v2 import ReasonerV2
from concept_learner.decoder import UnifiedDecoder


class CLModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_classes: int = 4,
        pad_id: int = 0,
        cls_id: int = 1,
        max_len: int = 128,
        # VQ configuration (optional overrides)
        vq_dim: int | None = None,
        vq_codebook_size: int | None = None,
        vq_num_quantizers: int | None = None,
        vq_num_parallel_heads: int | None = None,
        vq_serial_codebook_size: int | None = None,
        vq_commitment_weight: float | None = None,
        vq_pre_vq_noise_std: float | None = None,
        vq_orth_weight: float | None = None,
        vq_entropy_weight: float | None = None,
    ):
        super().__init__()
        self.enc = TinyEncoder(
            TinyEncoderConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                max_len=max_len,
                pad_id=pad_id,
                cls_id=cls_id,
            )
        )
        # Defaults
        _rvq_dim = 48 if vq_dim is None else int(vq_dim)
        _codebook = 16 if vq_codebook_size is None else int(vq_codebook_size)
        _num_q = 3 if vq_num_quantizers is None else int(vq_num_quantizers)
        _num_ph = 2 if vq_num_parallel_heads is None else int(vq_num_parallel_heads)
        _serial_codebook = (
            8 if vq_serial_codebook_size is None else int(vq_serial_codebook_size)
        )
        _commit_w = (
            0.45 if vq_commitment_weight is None else float(vq_commitment_weight)
        )
        _noise_std = 0.07 if vq_pre_vq_noise_std is None else float(vq_pre_vq_noise_std)
        _orth_w = 1e-4 if vq_orth_weight is None else float(vq_orth_weight)
        _ent_w = 0.0 if vq_entropy_weight is None else float(vq_entropy_weight)

        self.rvq = ResidualVQLayer(
            in_dim=d_model,
            rvq_dim=_rvq_dim,
            codebook_size=_codebook,
            num_quantizers=_num_q,
            num_parallel_heads=_num_ph,
            serial_codebook_size=_serial_codebook,
            commitment_weight=_commit_w,
            pre_vq_noise_std=_noise_std,
            orth_weight=_orth_w,
            entropy_weight=_ent_w,
            use_cosine_sim=True,  # pass via allowed kwargs
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )
        self.head = nn.Linear(
            d_model, num_classes
        )  # token-level classifier (just for smoke tests)
        # Use value passed to out_dim instead of d_model if specified
        # in ResidualVQLayer
        self.film = FiLM(d_model)
        # Projection to condense rich RVQ features (concat of parallel + serial) to d_model
        z_all_dim = _rvq_dim * (1 + max(0, _num_q - _num_ph))
        self.z_all_proj = nn.Linear(z_all_dim, d_model)

        self.reasoner = ReasonerV2(
            d_model=d_model,
            max_steps=4,
            temperature=1.0,
            lambda_sparse=1e-3,
            lambda_halt=1e-3,
        )
        self.decoder = UnifiedDecoder(d_model, num_classes)  # NEW
        # Auxiliary numeric head (scalar) for ranking/regression losses
        self.num_head = nn.Sequential(
            nn.Linear(d_model, max(32, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(32, d_model // 2), 1),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        # ids: (B,T) long, mask: (B,T) long (1=real, 0=pad)
        h, H = self.enc(ids, mask)  # h: (B,d), H: (B,T,d)
        z_q, indices, vq_loss = self.rvq(h)  # z_q: (B,d)
        z_all = getattr(self.rvq, "_last_all", None)
        if isinstance(z_all, torch.Tensor) and z_all.dim() == 2:
            z_for_reason = self.z_all_proj(z_all)
        else:
            z_for_reason = z_q
        # H_cond = H + z_q.unsqueeze(1)  # (B,T,d)
        H_cond = self.film(H, z_for_reason)
        H_reasoned, s_final, stop_logits, action_logits = self.reasoner(
            H_cond, z_for_reason, mask
        )
        # expose typed state for downstream consumers (e.g., proto routing)
        try:
            self.reasoner._last_s = s_final
        except Exception:
            pass
        val_final = getattr(self.reasoner, "_last_val", None)
        logits_tok, logits_seq = self.decoder(H_reasoned, mask, val=val_final)
        return logits_tok, logits_seq, vq_loss, indices, stop_logits, action_logits

    def predict_scalar(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict a scalar numeric value from a token sequence.

        Uses encoder pooled state passed through a small MLP. This head is
        intentionally separate from the main classifier so it can be used for
        ranking/regression auxiliary losses without affecting decoder shapes.
        Returns tensor of shape (B,1).
        """
        h, _ = self.enc(ids, mask)
        v = self.num_head(h)
        return v


class FiLM(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        h = hidden or max(64, d // 2)
        self.gamma_net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        self.beta_net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
        self.post_ln = nn.LayerNorm(d)
        # global strength (logit) → starts tiny
        self.strength = nn.Parameter(torch.tensor(-3.0))  # σ(-3)≈0.047

    def forward(self, H, z):
        s = torch.sigmoid(self.strength)
        gamma = self.gamma_net(z).unsqueeze(1) * s
        beta = self.beta_net(z).unsqueeze(1) * s
        return self.post_ln(H * (1 + gamma) + beta)


# class FiLM(nn.Module):
#     def __init__(self, d: int):
#         super().__init__()
#         self.to_gamma = nn.Sequential(
#             nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
#         )
#         self.to_beta = nn.Sequential(
#             nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
#         )
#
#     def forward(self, H, z):
#         gamma = self.to_gamma(z).unsqueeze(1)
#         beta  = self.to_beta(z).unsqueeze(1)
#         return H * (1 + gamma) + beta
