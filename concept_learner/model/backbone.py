from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L]


class TinyBackbone(nn.Module):
    """
    Tiny transformer encoder that maps a sequence of discrete tokens (digits)
    into a fixed-size representation.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_len: int = 8):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token(tokens)
        x = self.pos(x)
        key_padding = None
        if mask is not None:
            key_padding = ~mask
        x = self.encoder(x, src_key_padding_mask=key_padding)
        if mask is not None:
            denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            x = (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        return self.proj(x)
