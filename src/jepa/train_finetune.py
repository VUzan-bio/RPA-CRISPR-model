"""Fine-tune a JEPA encoder on the DeepSpCas9 dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List
import csv
import random

import torch
from torch.utils.data import DataLoader

from .data import DeepSpCas9Dataset, pad_collate
from .model import EncoderConfig, RegressionModel, SequenceEncoder
from .tokens import DNATokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune JEPA on DeepSpCas9.")
    parser.add_argument(
        "--xlsx",
        type=str,
        default="data/sequences/aax9249_Table_S1.xlsx",
        help="Path to DeepSpCas9 Table S1 Excel file.",
    )
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained JEPA checkpoint.")
    parser.add_argument("--output", type=str, default="checkpoints/jepa_finetune.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--inspect-samples", type=int, default=0)
    parser.add_argument("--overfit-n", type=int, default=0)
    parser.add_argument("--ignore-position-mismatch", action="store_true")
    parser.add_argument("--pad-to-max-len", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", choices=["pearson", "val_loss"], default="pearson")
    parser.add_argument("--encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--head-lr-mult", type=float, default=1.0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _pearson(x, y) -> float:
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt()).clamp_min(1e-8)
    return float((vx * vy).sum() / denom)


def _rankdata(values):
    values = torch.tensor(values)
    sorted_idx = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[sorted_idx] = torch.arange(len(values), dtype=torch.float32)
    return ranks.tolist()


def _spearman(x, y) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _r2(x, y) -> float:
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    ss_res = (y - x).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum().clamp_min(1e-8)
    return float(1.0 - ss_res / ss_tot)


def _resize_pos_embed(weight: torch.Tensor, target_len: int) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError("pos_embed weight must be 2D")
    src_len, dim = weight.shape
    if target_len == src_len:
        return weight
    if target_len < src_len:
        return weight[:target_len]
    # interpolate for longer length
    weight_t = weight.transpose(0, 1).unsqueeze(0)  # (1, dim, src_len)
    resized = torch.nn.functional.interpolate(
        weight_t, size=target_len, mode="linear", align_corners=False
    )
    return resized.squeeze(0).transpose(0, 1)


def _load_pretrained_encoder(
    model: RegressionModel,
    ckpt_path: Path,
    ignore_position_mismatch: bool = False,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("context_encoder."):
            encoder_state[key.replace("context_encoder.", "")] = value
        elif key.startswith("encoder."):
            encoder_state[key.replace("encoder.", "")] = value
    if not encoder_state:
        raise ValueError("No encoder weights found in checkpoint.")

    model_state = model.encoder.state_dict()
    filtered_state = {}
    for key, value in encoder_state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            if key.endswith("pos_embed.weight") and value.dim() == 2:
                if ignore_position_mismatch:
                    print(
                        f"warning: ignored {key} due to shape mismatch "
                        f"{tuple(value.shape)} != {tuple(model_state[key].shape)}"
                    )
                    continue
                resized = _resize_pos_embed(value, model_state[key].shape[0])
                filtered_state[key] = resized
                print(
                    f"warning: resized {key} from {tuple(value.shape)} "
                    f"to {tuple(resized.shape)}"
                )
            else:
                print(
                    f"warning: skip {key} due to shape mismatch "
                    f"{tuple(value.shape)} != {tuple(model_state[key].shape)}"
                )
            continue
        filtered_state[key] = value

    missing, unexpected = model.encoder.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"warning: missing encoder keys: {missing}")
    if unexpected:
        print(f"warning: unexpected encoder keys: {unexpected}")


def _build_model(
    args: argparse.Namespace,
    tokenizer: DNATokenizer,
    max_len: int,
) -> RegressionModel:
    encoder_cfg = EncoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=max_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    encoder = SequenceEncoder(encoder_cfg)
    model = RegressionModel(
        encoder, dropout=args.dropout, linear_only=args.linear_probe
    )
    if args.pretrained:
        _load_pretrained_encoder(
            model,
            Path(args.pretrained),
            ignore_position_mismatch=args.ignore_position_mismatch,
        )
    if args.linear_probe:
        args.freeze_encoder = True
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    return model


def _run_training(
    model: RegressionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    fold: int | None = None,
    metrics_writer: csv.DictWriter | None = None,
) -> None:
    model.to(args.device)
    param_groups = []
    if any(p.requires_grad for p in model.encoder.parameters()):
        param_groups.append(
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr_mult": args.encoder_lr_mult,
            }
        )
    param_groups.append(
        {
            "params": [p for p in model.head.parameters() if p.requires_grad],
            "lr_mult": args.head_lr_mult,
        }
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = min(args.warmup_steps, total_steps)

    def _lr_for_step(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return args.lr * float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return args.min_lr
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.min_lr + (args.lr - args.min_lr) * cosine)

    best_metric = -float("inf")
    best_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            lr = _lr_for_step(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_mult", 1.0)
            tokens = batch["tokens"].to(args.device)
            pad_mask = batch.get("pad_mask")
            if pad_mask is not None:
                pad_mask = pad_mask.to(args.device)
            labels = batch["label"].to(args.device)
            preds = model(tokens, pad_mask=pad_mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()
            global_step += 1

        model.eval()
        preds = []
        labels = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(args.device)
                pad_mask = batch.get("pad_mask")
                if pad_mask is not None:
                    pad_mask = pad_mask.to(args.device)
                label = batch["label"].to(args.device)
                pred = model(tokens, pad_mask=pad_mask)
                val_loss += loss_fn(pred, label).item()
                preds.extend(pred.cpu().tolist())
                labels.extend(label.cpu().tolist())

        avg_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        pearson = _pearson(preds, labels)
        spearman = _spearman(preds, labels)
        r2 = _r2(preds, labels)
        fold_tag = f" fold={fold}" if fold is not None else ""
        print(
            f"epoch {epoch}{fold_tag} train_loss={avg_loss:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"pearson={pearson:.4f} spearman={spearman:.4f} r2={r2:.4f}"
        )
        if metrics_writer:
            metrics_writer.writerow(
                {
                    "fold": fold if fold is not None else "",
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "pearson": pearson,
                    "spearman": spearman,
                    "r2": r2,
                }
            )

        if args.patience and args.patience > 0:
            if args.early_stop_metric == "pearson":
                improved = pearson > best_metric
            else:
                improved = avg_val_loss < best_loss

            if improved:
                best_metric = max(best_metric, pearson)
                best_loss = min(best_loss, avg_val_loss)
                patience_counter = 0
                if args.checkpoint:
                    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "encoder_cfg": model.encoder.cfg.__dict__,
                        },
                        args.checkpoint,
                    )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break


def _inspect_samples(dataset: DeepSpCas9Dataset, name: str, n: int) -> None:
    n = min(n, len(dataset))
    print(f"inspect {name} samples (n={n}):")
    for idx in range(n):
        seq = dataset.seqs[idx]
        label = dataset.labels[idx]
        print(f"{name}[{idx}] seq={seq} label={label:.4f}")


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    tokenizer = DNATokenizer()

    expected_len = None
    if args.pad_to_max_len:
        if args.max_len is None:
            raise ValueError("--pad-to-max-len requires --max-len.")
        expected_len = args.max_len

    train_ds = DeepSpCas9Dataset(
        args.xlsx, split="train", tokenizer=tokenizer, expected_len=expected_len
    )
    test_ds = DeepSpCas9Dataset(
        args.xlsx, split="test", tokenizer=tokenizer, expected_len=expected_len
    )

    if args.inspect_samples and args.inspect_samples > 0:
        _inspect_samples(train_ds, "train", args.inspect_samples)
        _inspect_samples(test_ds, "test", args.inspect_samples)

    max_len = max(train_ds.seq_len, test_ds.seq_len)
    if args.max_len is not None:
        if args.max_len < max_len:
            raise ValueError(
                f"--max-len ({args.max_len}) must be >= sequence length ({max_len})."
            )
        max_len = args.max_len

    metrics_writer = None
    metrics_handle = None
    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_handle = metrics_path.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(
            metrics_handle,
            fieldnames=["fold", "epoch", "train_loss", "val_loss", "pearson", "spearman", "r2"],
        )
        metrics_writer.writeheader()

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint.")
        model = _build_model(args, tokenizer, max_len)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=False)
        model.to(args.device)
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            eval_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
            )
            for batch in eval_loader:
                tokens = batch["tokens"].to(args.device)
                pad_mask = batch.get("pad_mask")
                if pad_mask is not None:
                    pad_mask = pad_mask.to(args.device)
                label = batch["label"].to(args.device)
                pred = model(tokens, pad_mask=pad_mask)
                preds.extend(pred.cpu().tolist())
                labels.extend(label.cpu().tolist())
        pearson = _pearson(preds, labels)
        spearman = _spearman(preds, labels)
        r2 = _r2(preds, labels)
        print(f"eval-only pearson={pearson:.4f} spearman={spearman:.4f} r2={r2:.4f}")
    elif args.cv_folds and args.cv_folds > 1:
        if args.overfit_n and args.overfit_n > 0:
            raise ValueError("--overfit-n cannot be used with --cv-folds.")
        indices = list(range(len(train_ds)))
        random.Random(args.seed).shuffle(indices)
        folds = [indices[i:: args.cv_folds] for i in range(args.cv_folds)]

        for fold_idx in range(args.cv_folds):
            val_indices = folds[fold_idx]
            val_set = set(val_indices)
            train_idx = [i for i in indices if i not in val_set]
            train_subset = torch.utils.data.Subset(train_ds, train_idx)
            val_subset = torch.utils.data.Subset(train_ds, val_indices)

            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
            )

            model = _build_model(args, tokenizer, max_len)
            _run_training(model, train_loader, val_loader, args, fold=fold_idx, metrics_writer=metrics_writer)
    else:
        if args.overfit_n and args.overfit_n > 0:
            n = min(args.overfit_n, len(train_ds))
            subset_idx = list(range(n))
            train_subset = torch.utils.data.Subset(train_ds, subset_idx)
            val_subset = torch.utils.data.Subset(train_ds, subset_idx)
        else:
            train_subset = train_ds
            val_subset = test_ds

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
        )

        model = _build_model(args, tokenizer, max_len)
        _run_training(model, train_loader, val_loader, args, fold=None, metrics_writer=metrics_writer)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "encoder_cfg": model.encoder.cfg.__dict__,
                "tokenizer": {
                    "mask_id": tokenizer.mask_id,
                    "pad_id": tokenizer.pad_id,
                },
            },
            args.output,
        )

    if metrics_handle:
        metrics_handle.close()


if __name__ == "__main__":
    main()
