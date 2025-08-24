from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyEncoderConfig:
    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    max_len: int = 128
    dropout: float = 0.0  # keep 0.0 for deterministic tests
    cls_id: int = 1  # reserve an explicit [CLS]
    pad_id: int = 0  # reserve PAD=0 by convention


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        assert T <= self.pe.size(0), f"Sequence length {T} > max_len {self.pe.size(0)}"
        return x + self.pe[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0)


class TinyEncoder(nn.Module):
    """
    Returns:
      h: (B, d) pooled ([CLS] vector or masked-mean fallback)
      H: (B, T, d) sequence states
    """

    def __init__(self, cfg: TinyEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.out_ln = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        token_ids: torch.Tensor,  # (B,T) int64
        attn_mask: Optional[torch.Tensor] = None,  # (B,T) 1 for real tokens, 0 for pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        # build mask if not provided
        if attn_mask is None:
            attn_mask = (token_ids != cfg.pad_id).to(torch.uint8)  # 1/0

        x = self.tok_emb(token_ids) * math.sqrt(cfg.d_model)  # scale embeddings
        x = self.pos(x)  # add sin/cos PE

        key_padding_mask = attn_mask == 0  # True where pad
        H = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,T,d)
        H = self.out_ln(H)

        # pooled h: prefer [CLS] position if present, else masked mean
        if (token_ids[:, 0] == cfg.cls_id).all():
            h = H[:, 0, :]
        else:
            # masked mean over non-pad tokens
            lengths = attn_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # (B,1)
            h = (H * attn_mask.unsqueeze(-1)).sum(dim=1) / lengths

        return h, H
