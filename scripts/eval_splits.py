#!/usr/bin/env python
"""Evaluate a finetuned model on train/val/test splits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

if "--allow-omp-duplicate" in sys.argv:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jepa.data import DeepSpCas9Dataset, pad_collate
from src.jepa.model import EncoderConfig, RegressionModel, SequenceEncoder
from src.jepa.tokens import DNATokenizer
from src.jepa.train_finetune import _pearson, _spearman, _r2


def _detect_linear_head(state_dict: dict) -> bool:
    has_mlp = any(key.startswith("head.0.") for key in state_dict.keys())
    has_linear = "head.weight" in state_dict and "head.bias" in state_dict
    return has_linear and not has_mlp


def _build_model(encoder_cfg: EncoderConfig, linear_only: bool) -> RegressionModel:
    encoder = SequenceEncoder(encoder_cfg)
    model = RegressionModel(encoder, dropout=encoder_cfg.dropout, linear_only=linear_only)
    return model


def _eval_loader(model: RegressionModel, loader: DataLoader, device: str):
    model.eval()
    preds: List[float] = []
    labels: List[float] = []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            pad_mask = batch.get("pad_mask")
            if pad_mask is not None:
                pad_mask = pad_mask.to(device)
            label = batch["label"].to(device)
            pred = model(tokens, pad_mask=pad_mask)
            preds.extend(pred.cpu().tolist())
            labels.extend(label.cpu().tolist())
    return preds, labels


def _make_loader(ds, tokenizer: DNATokenizer, batch_size: int, num_workers: int):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JEPA finetune checkpoint.")
    parser.add_argument("--xlsx", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pad-to-max-len", action="store_true")
    parser.add_argument("--allow-omp-duplicate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    encoder_cfg_dict = checkpoint.get("encoder_cfg")
    if not encoder_cfg_dict:
        raise SystemExit("Checkpoint missing encoder_cfg. Re-run training with updated code.")

    encoder_cfg = EncoderConfig(**encoder_cfg_dict)
    linear_only = _detect_linear_head(state)

    tokenizer = DNATokenizer()
    expected_len = encoder_cfg.max_len if args.pad_to_max_len else None
    train_ds = DeepSpCas9Dataset(args.xlsx, split="train", tokenizer=tokenizer, expected_len=expected_len)
    test_ds = DeepSpCas9Dataset(args.xlsx, split="test", tokenizer=tokenizer, expected_len=expected_len)

    indices = list(range(len(train_ds)))
    if args.val_fraction > 0:
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(indices), generator=rng).tolist()
        val_size = max(1, int(len(indices) * args.val_fraction))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(train_ds, val_idx)
    else:
        train_subset = train_ds
        val_subset = None

    model = _build_model(encoder_cfg, linear_only=linear_only)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing keys: {missing}")
    if unexpected:
        print(f"warning: unexpected keys: {unexpected}")
    model.to(args.device)

    train_loader = _make_loader(train_subset, tokenizer, args.batch_size, args.num_workers)
    train_preds, train_labels = _eval_loader(model, train_loader, args.device)
    print(
        f"train pearson={_pearson(train_preds, train_labels):.4f} "
        f"spearman={_spearman(train_preds, train_labels):.4f} "
        f"r2={_r2(train_preds, train_labels):.4f}"
    )

    if val_subset is not None:
        val_loader = _make_loader(val_subset, tokenizer, args.batch_size, args.num_workers)
        val_preds, val_labels = _eval_loader(model, val_loader, args.device)
        print(
            f"val pearson={_pearson(val_preds, val_labels):.4f} "
            f"spearman={_spearman(val_preds, val_labels):.4f} "
            f"r2={_r2(val_preds, val_labels):.4f}"
        )

    test_loader = _make_loader(test_ds, tokenizer, args.batch_size, args.num_workers)
    test_preds, test_labels = _eval_loader(model, test_loader, args.device)
    print(
        f"test pearson={_pearson(test_preds, test_labels):.4f} "
        f"spearman={_spearman(test_preds, test_labels):.4f} "
        f"r2={_r2(test_preds, test_labels):.4f}"
    )


if __name__ == "__main__":
    main()
