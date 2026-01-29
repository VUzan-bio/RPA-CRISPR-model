"""JEPA pretraining on genomic windows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import math
import csv

import torch
from torch.utils.data import DataLoader

from .data import GenomeWindowDataset, MaskConfig, pad_collate
from .model import EncoderConfig, JEPA, SequenceEncoder
from .tokens import DNATokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JEPA pretraining on genome windows.")
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file.")
    parser.add_argument("--output", type=str, default="checkpoints/jepa_pretrain.pt")
    parser.add_argument("--window-len", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--block-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--ema-start", type=float, default=None)
    parser.add_argument("--ema-end", type=float, default=0.9999)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-teacher-epochs", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--l2-normalize", dest="l2_normalize", action="store_true")
    parser.add_argument("--no-l2-normalize", dest="l2_normalize", action="store_false")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(l2_normalize=True)
    return parser


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    tokenizer = DNATokenizer()
    mask_cfg = MaskConfig(
        num_blocks=args.num_blocks,
        block_len=args.block_len,
        mask_ratio=args.mask_ratio,
    )
    num_samples = args.steps_per_epoch * args.batch_size
    dataset = GenomeWindowDataset(
        fasta_path=args.fasta,
        window_len=args.window_len,
        num_samples=num_samples,
        tokenizer=tokenizer,
        mask_config=mask_cfg,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
    )

    encoder_cfg = EncoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.window_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    encoder = SequenceEncoder(encoder_cfg)
    model = JEPA(
        encoder=encoder,
        mask_token_id=tokenizer.mask_id,
        ema_decay=args.ema_decay,
        l2_normalize=args.l2_normalize,
    )
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = min(args.warmup_steps, total_steps)
    freeze_steps = max(0, args.freeze_teacher_epochs * args.steps_per_epoch)
    ema_warmup_steps = max(0, args.ema_warmup_steps)

    ema_start = args.ema_start if args.ema_start is not None else args.ema_decay

    def _ema_for_step(step: int) -> float:
        if step < freeze_steps:
            return 1.0
        if ema_warmup_steps > 0 and step < freeze_steps + ema_warmup_steps:
            t = float(step - freeze_steps) / float(ema_warmup_steps)
            return float(1.0 - t * (1.0 - ema_start))
        if total_steps <= freeze_steps:
            return args.ema_end
        progress = float(step - (freeze_steps + ema_warmup_steps)) / float(
            max(1, total_steps - (freeze_steps + ema_warmup_steps))
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.ema_end - (args.ema_end - ema_start) * cosine)

    def _lr_for_step(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return args.lr * float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return args.min_lr
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.min_lr + (args.lr - args.min_lr) * cosine)

    metrics_writer = None
    metrics_handle = None
    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_handle = metrics_path.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(
            metrics_handle,
            fieldnames=[
                "step",
                "epoch",
                "loss",
                "cosine",
                "pred_norm",
                "target_norm",
                "lr",
                "ema",
            ],
        )
        metrics_writer.writeheader()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        dataset.seed = args.seed + epoch * 100_000
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            lr = _lr_for_step(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.ema_decay = _ema_for_step(global_step)
            tokens = batch["tokens"].to(args.device)
            mask = batch["mask"].to(args.device)
            stats = None
            if args.log_interval > 0 and step % args.log_interval == 0:
                loss, stats = model(tokens, mask, return_stats=True)
                print(
                    f"step {global_step} loss={loss.item():.6f} "
                    f"pred_norm={stats['pred_norm']:.3f} "
                    f"target_norm={stats['target_norm']:.3f} "
                    f"cos={stats['cosine']:.3f} "
                    f"lr={lr:.2e} ema={model.ema_decay:.5f}"
                )
            else:
                loss = model(tokens, mask)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.context_encoder.parameters())
                    + list(model.predictor.parameters()),
                    args.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if global_step >= freeze_steps:
                model.update_target()
            running_loss += loss.item()
            if metrics_writer:
                metrics_writer.writerow(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "cosine": stats["cosine"] if stats else "",
                        "pred_norm": stats["pred_norm"] if stats else "",
                        "target_norm": stats["target_norm"] if stats else "",
                        "lr": lr,
                        "ema": model.ema_decay,
                    }
                )
            global_step += 1

        avg_loss = running_loss / step
        print(f"epoch {epoch} loss={avg_loss:.6f}")
        save_checkpoint(
            Path(args.output),
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "encoder_cfg": encoder_cfg.__dict__,
                "tokenizer": {
                    "mask_id": tokenizer.mask_id,
                    "pad_id": tokenizer.pad_id,
                },
                "optimizer": optimizer.state_dict(),
            },
        )

    if metrics_handle:
        metrics_handle.close()


if __name__ == "__main__":
    main()
