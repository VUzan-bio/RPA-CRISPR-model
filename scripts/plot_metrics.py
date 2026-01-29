#!/usr/bin/env python
"""Plot training metrics from CSV logs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _to_float(value: str):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def plot_pretrain(rows, out_path: Path | None) -> None:
    import matplotlib.pyplot as plt

    steps = []
    loss = []
    cosine = []
    lr = []
    ema = []
    for row in rows:
        steps.append(int(float(row.get("step", 0))))
        loss.append(_to_float(row.get("loss")))
        cosine.append(_to_float(row.get("cosine")))
        lr.append(_to_float(row.get("lr")))
        ema.append(_to_float(row.get("ema")))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, loss, label="loss")
    if any(c is not None for c in cosine):
        axes[0].plot(steps, cosine, label="cosine")
    axes[0].set_ylabel("loss / cosine")
    axes[0].legend()

    axes[1].plot(steps, lr, label="lr")
    axes[1].plot(steps, ema, label="ema")
    axes[1].set_ylabel("lr / ema")
    axes[1].set_xlabel("step")
    axes[1].legend()

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()


def plot_finetune(rows, out_path: Path | None) -> None:
    import matplotlib.pyplot as plt

    by_fold = defaultdict(list)
    for row in rows:
        fold = row.get("fold", "")
        by_fold[fold].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for fold, fold_rows in by_fold.items():
        epochs = [int(float(r["epoch"])) for r in fold_rows]
        train_loss = [_to_float(r.get("train_loss")) for r in fold_rows]
        val_loss = [_to_float(r.get("val_loss")) for r in fold_rows]
        pearson = [_to_float(r["pearson"]) for r in fold_rows]
        label = f"fold {fold}" if fold != "" else "test"
        axes[0].plot(epochs, train_loss, label=f"{label} train")
        if any(v is not None for v in val_loss):
            axes[0].plot(epochs, val_loss, label=f"{label} val")
        axes[1].plot(epochs, pearson, label=label)

    axes[0].set_ylabel("train/val loss")
    axes[1].set_ylabel("pearson")
    axes[1].set_xlabel("epoch")
    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot JEPA metrics from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics CSV.")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG).")
    parser.add_argument("--kind", type=str, choices=["pretrain", "finetune"], default=None)
    args = parser.parse_args()

    path = Path(args.csv)
    rows = _load_rows(path)
    if not rows:
        raise SystemExit("No rows found in metrics CSV.")

    kind = args.kind
    if kind is None:
        kind = "pretrain" if "step" in rows[0] else "finetune"

    out_path = Path(args.out) if args.out else None
    if kind == "pretrain":
        plot_pretrain(rows, out_path)
    else:
        plot_finetune(rows, out_path)


if __name__ == "__main__":
    main()
