"""Datasets and utilities for JEPA pretraining and CRISPR fine-tuning."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .fasta_index import FastaIndex
from .tokens import DNATokenizer


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    normalized = {_normalize_column(col): col for col in columns}
    for cand in candidates:
        key = _normalize_column(cand)
        if key in normalized:
            return normalized[key]
    # fuzzy: candidate substring match
    for cand in candidates:
        cand_norm = _normalize_column(cand)
        for col in columns:
            if cand_norm in _normalize_column(col):
                return col
    raise KeyError(f"Could not find column among candidates: {candidates}")


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "pandas is required to read the Excel dataset. "
            "Install with: pip install pandas openpyxl"
        ) from exc
    return pd


@dataclass
class MaskConfig:
    num_blocks: int = 2
    block_len: Optional[int] = None
    mask_ratio: float = 0.25


class GenomeWindowDataset(Dataset):
    """Sample fixed-length windows from a FASTA file for JEPA pretraining."""

    def __init__(
        self,
        fasta_path: str | Path,
        window_len: int,
        num_samples: int,
        tokenizer: Optional[DNATokenizer] = None,
        mask_config: Optional[MaskConfig] = None,
        seed: int = 13,
        contig_allowlist: Optional[Sequence[str]] = None,
    ) -> None:
        self.index = FastaIndex(fasta_path)
        self.window_len = window_len
        self.num_samples = num_samples
        self.tokenizer = tokenizer or DNATokenizer()
        self.mask_config = mask_config or MaskConfig()
        self.seed = seed

        contigs = [rec for rec in self.index.records.values() if rec.length >= window_len]
        if contig_allowlist:
            allow = set(contig_allowlist)
            contigs = [rec for rec in contigs if rec.name in allow]
        if not contigs:
            raise ValueError("No contigs long enough for the requested window length.")
        self.contigs = contigs

    def __len__(self) -> int:
        return self.num_samples

    def _sample_mask(self, seq_len: int, rng: random.Random) -> torch.Tensor:
        cfg = self.mask_config
        block_len = cfg.block_len
        if block_len is None:
            block_len = max(1, int(seq_len * cfg.mask_ratio / cfg.num_blocks))
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for _ in range(cfg.num_blocks):
            start = None
            for _ in range(20):
                candidate = rng.randint(0, seq_len - block_len)
                if not mask[candidate : candidate + block_len].any().item():
                    start = candidate
                    break
            if start is None:
                start = rng.randint(0, seq_len - block_len)
            mask[start : start + block_len] = True
        return mask

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx)
        record = self.contigs[rng.randrange(len(self.contigs))]
        max_start = record.length - self.window_len
        start = rng.randint(0, max_start)
        seq = self.index.fetch(record.name, start, start + self.window_len)
        tokens = torch.tensor(
            self.tokenizer.encode(seq, target_length=self.window_len),
            dtype=torch.long,
        )
        mask = self._sample_mask(self.window_len, rng)
        return {"tokens": tokens, "mask": mask}


class DeepSpCas9Dataset(Dataset):
    """DeepSpCas9 HT_Cas9_Train/Test dataset from Table S1."""

    TRAIN_SHEET = "HT_Cas9_Train"
    TEST_SHEET = "HT_Cas9_Test"

    SEQ_COL_CANDIDATES = [
        "Target context sequence",
        "Target context sequence (4+20+3+3)",
        "Target context sequence 4+20+3+3",
    ]

    TRAIN_LABEL_CANDIDATES = [
        "Background subtracted indel (%)",
        "Background-subtracted indel (%)",
    ]

    TEST_LABEL_CANDIDATES = [
        "Background subtracted indel frequencies (average, %)",
        "Background subtracted indel frequency (average, %)",
        "Background-subtracted indel frequencies (average, %)",
    ]

    def __init__(
        self,
        xlsx_path: str | Path,
        split: str = "train",
        tokenizer: Optional[DNATokenizer] = None,
        seq_col: Optional[str] = None,
        label_col: Optional[str] = None,
        label_scale: float = 100.0,
        expected_len: Optional[int] = None,
    ) -> None:
        pd = _require_pandas()
        self.xlsx_path = Path(xlsx_path)
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.xlsx_path}")
        self.split = split.lower()
        sheet = self.TRAIN_SHEET if self.split == "train" else self.TEST_SHEET
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet)
        columns = list(df.columns)

        seq_col = seq_col or _find_column(columns, self.SEQ_COL_CANDIDATES)
        if label_col is None:
            label_candidates = (
                self.TRAIN_LABEL_CANDIDATES
                if self.split == "train"
                else self.TEST_LABEL_CANDIDATES
            )
            label_col = _find_column(columns, label_candidates)

        seqs = df[seq_col].astype(str).str.upper()
        labels = df[label_col].astype(float) / label_scale
        mask_valid = seqs.str.len() > 0
        mask_labels = labels.notna()
        mask_seq_chars = seqs.str.fullmatch(r"[ACGTN]+").fillna(False)
        mask_all = mask_valid & mask_labels & mask_seq_chars
        seqs = seqs[mask_all]
        labels = labels[mask_all]

        lengths = seqs.str.len().tolist()
        if expected_len is None:
            if not lengths:
                raise ValueError("No sequences found in dataset.")
            length_counts = {}
            for length in lengths:
                length_counts[length] = length_counts.get(length, 0) + 1
            expected_len = max(length_counts, key=length_counts.get)
        self.seq_len = expected_len

        self.seqs = seqs.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer or DNATokenizer()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq = self.seqs[idx]
        tokens = torch.tensor(
            self.tokenizer.encode(seq, target_length=self.seq_len), dtype=torch.long
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"tokens": tokens, "label": label}


def pad_collate(batch: List[dict], pad_id: int) -> dict:
    tokens = [item["tokens"] for item in batch]
    max_len = max(t.shape[0] for t in tokens)
    padded = torch.full((len(tokens), max_len), pad_id, dtype=torch.long)
    for idx, token in enumerate(tokens):
        padded[idx, : token.shape[0]] = token
    output = {"tokens": padded}
    output["pad_mask"] = padded.eq(pad_id)
    if "mask" in batch[0]:
        masks = [item["mask"] for item in batch]
        mask_pad = torch.zeros((len(tokens), max_len), dtype=torch.bool)
        for idx, mask in enumerate(masks):
            mask_pad[idx, : mask.shape[0]] = mask
        output["mask"] = mask_pad
    if "label" in batch[0]:
        output["label"] = torch.stack([item["label"] for item in batch])
    return output
