"""JEPA model components for DNA sequence representation learning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EncoderConfig:
    vocab_size: int
    max_len: int
    embed_dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1


class SequenceEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.mlp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must be (batch, seq_len)")
        batch, seq_len = tokens.shape
        if seq_len > self.cfg.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.cfg.max_len}"
            )
        if pad_mask is not None and pad_mask.shape != tokens.shape:
            raise ValueError("pad_mask must match tokens shape")
        positions = torch.arange(seq_len, device=tokens.device)
        positions = positions.unsqueeze(0).expand(batch, seq_len)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(x)


class JEPA(nn.Module):
    def __init__(
        self,
        encoder: SequenceEncoder,
        mask_token_id: int,
        ema_decay: float = 0.996,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()
        self.predictor = nn.Sequential(
            nn.Linear(encoder.cfg.embed_dim, encoder.cfg.embed_dim * 2),
            nn.GELU(),
            nn.Linear(encoder.cfg.embed_dim * 2, encoder.cfg.embed_dim),
        )
        self.mask_token_id = mask_token_id
        self.ema_decay = ema_decay
        self.l2_normalize = l2_normalize

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the target encoder in eval mode to avoid dropout noise.
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self) -> None:
        for target_param, context_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target_param.data.mul_(self.ema_decay).add_(
                context_param.data, alpha=1.0 - self.ema_decay
            )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        return_stats: bool = False,
    ):
        if mask.shape != tokens.shape:
            raise ValueError("mask must have the same shape as tokens")
        context_tokens = tokens.clone()
        context_tokens[mask] = self.mask_token_id

        context_out = self.context_encoder(context_tokens)
        pred = self.predictor(context_out)

        with torch.no_grad():
            target_out = self.target_encoder(tokens)

        pred_masked = pred[mask]
        target_masked = target_out[mask]
        if pred_masked.numel() == 0:
            raise ValueError("Mask selects zero positions; check mask config.")
        if self.l2_normalize:
            pred_masked = F.normalize(pred_masked, dim=-1)
            target_masked = F.normalize(target_masked, dim=-1)
        loss = F.mse_loss(pred_masked, target_masked)

        if not return_stats:
            return loss

        with torch.no_grad():
            pred_norm = pred_masked.norm(dim=-1).mean().item()
            target_norm = target_masked.norm(dim=-1).mean().item()
            cosine = F.cosine_similarity(pred_masked, target_masked, dim=-1).mean().item()
        return loss, {"pred_norm": pred_norm, "target_norm": target_norm, "cosine": cosine}


class RegressionModel(nn.Module):
    def __init__(
        self, encoder: SequenceEncoder, dropout: float = 0.1, linear_only: bool = False
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if linear_only:
            self.head = nn.Linear(encoder.cfg.embed_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(encoder.cfg.embed_dim),
                nn.Linear(encoder.cfg.embed_dim, encoder.cfg.embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(encoder.cfg.embed_dim, 1),
            )

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(tokens, pad_mask=pad_mask)
        if pad_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            keep = (~pad_mask).float().unsqueeze(-1)
            summed = (encoded * keep).sum(dim=1)
            denom = keep.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return self.head(pooled).squeeze(-1)
