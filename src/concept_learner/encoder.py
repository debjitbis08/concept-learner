from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn as nn


@dataclass
class TinyEncoderConfig:
    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    max_len: int = 128
    dropout: float = 0.0  # keep 0.0 for deterministic tests


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
        # x: (B,T,d)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TinyEncoder(nn.Module):
    """
    Returns:
      h: (B, d) pooled (we use [CLS] vector)
      H: (B, T, d) sequence states
    Matches the spec: 2 layers, d≈128, 4 heads; pooled h and sequence H. :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, cfg: TinyEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
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
        self, token_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: (B,T) int64
        attn_mask: (B,T) 1 for real tokens, 0 for pad
        """
        x = self.tok_emb(token_ids)  # (B,T,d)
        x = self.pos(x)  # (B,T,d)
        key_padding_mask = attn_mask == 0  # True where pad
        H = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,T,d)
        H = self.out_ln(H)

        # pooled h = [CLS] position (assume [CLS] is at index 0)
        h = H[:, 0, :]
        return h, H
