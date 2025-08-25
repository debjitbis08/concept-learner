import torch
import torch.nn as nn
from concept_learner.encoder import TinyEncoder, TinyEncoderConfig
from concept_learner.vq_layer import ResidualVQLayer
from concept_learner.reasoning import Reasoner
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
        self.rvq = ResidualVQLayer(
            in_dim=d_model,
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
        self.head = nn.Linear(
            d_model, num_classes
        )  # token-level classifier (just for smoke tests)
        # Use value passed to out_dim instead of d_model if specified
        # in ResidualVQLayer
        self.film = FiLM(d_model)

        self.reasoner = ReasonerV2(
            d_model=d_model,
            max_steps=4,
            temperature=1.0,
        )
        self.decoder = UnifiedDecoder(d_model, num_classes)  # NEW

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        # ids: (B,T) long, mask: (B,T) long (1=real, 0=pad)
        h, H = self.enc(ids, mask)  # h: (B,d), H: (B,T,d)
        z_q, indices, vq_loss = self.rvq(h)  # z_q: (B,d)
        # H_cond = H + z_q.unsqueeze(1)  # (B,T,d)
        H_cond = self.film(H, z_q)
        H_reasoned, s_final, stop_logits, action_logits = self.reasoner(
            H_cond, z_q, mask
        )
        logits_tok, logits_seq = self.decoder(H_reasoned, mask)
        return logits_tok, logits_seq, vq_loss, indices, stop_logits, action_logits


class FiLM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_gamma = nn.Linear(d, d)
        self.to_beta = nn.Linear(d, d)

    def forward(self, H, z):
        gamma = self.to_gamma(z).unsqueeze(1)
        beta = self.to_beta(z).unsqueeze(1)
        return H * (1 + gamma) + beta


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
