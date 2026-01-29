"""Multiplex guide scoring utilities for CRISPR assay design."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .tokens import DNATokenizer


@dataclass(frozen=True)
class ScoredGuide:
    """Guide candidate with cached score/embedding."""

    sequence: str
    gene: str
    score: float
    embedding: torch.Tensor


class MultiplexGuideScorer(nn.Module):
    """Score guide sets for multiplex compatibility.

    This module wraps a sequence encoder and three heads:
    - efficiency_head: predicts individual guide efficiency from pooled embeddings.
    - interaction_head: scores pairwise compatibility between guides.
    - multiplex_head: scores overall multiplex quality from pooled embeddings.

    Note: interaction_head/multiplex_head are untrained by default. For production
    usage, train them on multiplex assay outcomes or proxy objectives. The
    optimize_guide_set helper returns both a combined score (used for selection)
    and the raw multiplex_head score.
    """

    def __init__(
        self,
        encoder: nn.Module,
        efficiency_head: Optional[nn.Module] = None,
        interaction_head: Optional[nn.Module] = None,
        multiplex_head: Optional[nn.Module] = None,
        pool: str = "mean",
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.l2_normalize = l2_normalize

        embed_dim = getattr(getattr(encoder, "cfg", None), "embed_dim", None)
        if embed_dim is None and efficiency_head is None:
            raise ValueError("encoder.cfg.embed_dim required when efficiency_head is None")

        if efficiency_head is None:
            efficiency_head = nn.Linear(embed_dim, 1)
        if interaction_head is None:
            interaction_head = nn.Sequential(
                nn.Linear(embed_dim * 2, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        if multiplex_head is None:
            multiplex_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            nn.init.constant_(multiplex_head[-1].bias, 0.5)

        self.efficiency_head = efficiency_head
        self.interaction_head = interaction_head
        self.multiplex_head = multiplex_head

    def _pool_embeddings(
        self, encoded: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.pool != "mean":
            raise ValueError(f"Unsupported pool mode: {self.pool}")
        if pad_mask is None:
            return encoded.mean(dim=1)
        keep = (~pad_mask).float().unsqueeze(-1)
        summed = (encoded * keep).sum(dim=1)
        denom = keep.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def encode_tokens(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        encoded = self.encoder(tokens, pad_mask=pad_mask)
        pooled = self._pool_embeddings(encoded, pad_mask)
        if self.l2_normalize:
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    def score_tokens(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode_tokens(tokens, pad_mask)
        scores = self.efficiency_head(embeddings).squeeze(-1)
        return scores, embeddings

    def pairwise_compatibility(
        self, embeddings: torch.Tensor, mode: str = "cosine"
    ) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be 2D (N, D)")
        n = embeddings.shape[0]
        if n == 0:
            return torch.empty((0, 0), device=embeddings.device)

        if mode == "cosine":
            if self.l2_normalize:
                sim = embeddings @ embeddings.t()
            else:
                normed = F.normalize(embeddings, dim=-1)
                sim = normed @ normed.t()
            compat = (sim + 1.0) * 0.5
            compat.fill_diagonal_(1.0)
            return compat

        if mode != "interaction":
            raise ValueError(f"Unknown compatibility mode: {mode}")

        compat = torch.ones((n, n), device=embeddings.device)
        for i in range(n):
            for j in range(i + 1, n):
                pair_emb = torch.cat([embeddings[i], embeddings[j]], dim=-1)
                score = self.interaction_head(pair_emb.unsqueeze(0)).squeeze()
                compat[i, j] = score
                compat[j, i] = score
        return compat

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        compat_mode: str = "cosine",
    ) -> Dict[str, torch.Tensor]:
        scores, embeddings = self.score_tokens(tokens, pad_mask)
        pairwise = self.pairwise_compatibility(embeddings, mode=compat_mode)
        multiplex = self.multiplex_head(embeddings.mean(dim=0, keepdim=True)).squeeze()
        return {
            "individual_scores": scores,
            "pairwise_compat": pairwise,
            "multiplex_score": multiplex,
            "embeddings": embeddings,
        }

    def _batch_score_sequences(
        self,
        sequences: Sequence[str],
        tokenizer: DNATokenizer,
        guide_length: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            return torch.empty(0), torch.empty((0, 0))

        all_scores: List[torch.Tensor] = []
        all_embeddings: List[torch.Tensor] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            tokens = torch.tensor(
                [tokenizer.encode(seq, target_length=guide_length) for seq in chunk],
                dtype=torch.long,
                device=device,
            )
            pad_mask = tokens.eq(tokenizer.pad_id)
            scores, embeddings = self.score_tokens(tokens, pad_mask)
            all_scores.append(scores.detach().cpu())
            all_embeddings.append(embeddings.detach().cpu())
        return torch.cat(all_scores), torch.cat(all_embeddings)

    @torch.no_grad()
    def optimize_guide_set(
        self,
        candidate_guides: Sequence[str],
        target_genes: Sequence[str],
        tokenizer: Optional[DNATokenizer] = None,
        guide_length: int = 23,
        per_gene_top_k: int = 5,
        max_combinations: int = 1000,
        compat_mode: str = "cosine",
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        multiplex_threshold: float = 0.3,
        fallback_weights: Tuple[float, float] = (0.6, 0.4),
        batch_size: int = 256,
        seed: int = 13,
    ) -> Dict[str, object]:
        if len(candidate_guides) != len(target_genes):
            raise ValueError("candidate_guides and target_genes must have same length")
        if not candidate_guides:
            raise ValueError("No candidate guides provided")

        tokenizer = tokenizer or DNATokenizer()
        device = next(self.parameters()).device

        gene_order: List[str] = []
        gene_to_guides: Dict[str, List[str]] = {}
        for guide, gene in zip(candidate_guides, target_genes):
            gene = str(gene)
            if gene not in gene_to_guides:
                gene_order.append(gene)
                gene_to_guides[gene] = []
            gene_to_guides[gene].append(guide)

        scored_by_gene: Dict[str, List[ScoredGuide]] = {}
        for gene in gene_order:
            guides = gene_to_guides[gene]
            scores, embeddings = self._batch_score_sequences(
                guides, tokenizer, guide_length, batch_size, device
            )
            if scores.numel() == 0:
                continue
            k = min(per_gene_top_k, scores.numel())
            top_idx = torch.topk(scores, k=k).indices.tolist()
            scored_by_gene[gene] = [
                ScoredGuide(
                    sequence=guides[idx],
                    gene=gene,
                    score=float(scores[idx].item()),
                    embedding=embeddings[idx],
                )
                for idx in top_idx
            ]

        candidate_lists = [scored_by_gene[g] for g in gene_order if g in scored_by_gene]
        if not candidate_lists:
            raise ValueError("No guide candidates available after scoring")

        total_combinations = 1
        for candidates in candidate_lists:
            total_combinations *= max(1, len(candidates))

        rng = random.Random(seed)
        if total_combinations > max_combinations:
            combinations: Iterable[Tuple[ScoredGuide, ...]] = (
                tuple(rng.choice(candidates) for candidates in candidate_lists)
                for _ in range(max_combinations)
            )
        else:
            combinations = itertools.product(*candidate_lists)

        best_score = -float("inf")
        best_combo: Optional[Tuple[ScoredGuide, ...]] = None
        best_pairwise = None
        best_multiplex = None

        weight_ind, weight_pair, weight_mux = weights
        fallback_ind, fallback_pair = fallback_weights
        used_multiplex_head = False
        for combo in combinations:
            embeddings = torch.stack([item.embedding for item in combo]).to(device)
            scores = torch.tensor([item.score for item in combo], device=device)
            pairwise = self.pairwise_compatibility(embeddings, mode=compat_mode)
            if pairwise.numel() > 0:
                mask = ~torch.eye(pairwise.shape[0], dtype=torch.bool, device=device)
                pairwise_mean = pairwise[mask].mean() if mask.any() else torch.tensor(1.0, device=device)
            else:
                pairwise_mean = torch.tensor(1.0, device=device)
            multiplex = self.multiplex_head(embeddings.mean(dim=0, keepdim=True)).squeeze()
            if multiplex is not None and float(multiplex.item()) >= multiplex_threshold:
                combined = (
                    weight_ind * scores.mean()
                    + weight_pair * pairwise_mean
                    + weight_mux * multiplex
                )
                used_multiplex_head = True
            else:
                combined = fallback_ind * scores.mean() + fallback_pair * pairwise_mean
            score_val = float(combined.item())
            if score_val > best_score:
                best_score = score_val
                best_combo = combo
                best_pairwise = pairwise.detach().cpu()
                best_multiplex = multiplex.detach().cpu()

        if best_combo is None:
            raise ValueError("Failed to score any guide combinations")

        return {
            "selected_guides": [item.sequence for item in best_combo],
            "genes": [item.gene for item in best_combo],
            "individual_scores": [item.score for item in best_combo],
            "compatibility_matrix": best_pairwise,
            "combined_score": float(best_score),
            "multiplex_score": float(best_multiplex.item()) if best_multiplex is not None else None,
            "used_multiplex_head": used_multiplex_head,
            "multiplex_threshold": multiplex_threshold,
            "compat_mode": compat_mode,
            "weights": weights,
            "fallback_weights": fallback_weights,
        }
