#!/usr/bin/env python
"""Generate POC visualizations for JEPA -> DeepSpCas9."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Must be set before importing torch/numpy to avoid OMP runtime init conflicts.
if "--allow-omp-duplicate" in sys.argv:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jepa.data import DeepSpCas9Dataset, pad_collate
from src.jepa.model import EncoderConfig, RegressionModel, SequenceEncoder
from src.jepa.tokens import DNATokenizer


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx ** 2).sum()) * np.sqrt((vy ** 2).sum())
    if denom < 1e-8:
        return float("nan")
    return float((vx * vy).sum() / denom)


def _detect_linear_head(state_dict: dict) -> bool:
    has_mlp = any(key.startswith("head.0.") for key in state_dict.keys())
    has_linear = "head.weight" in state_dict and "head.bias" in state_dict
    return has_linear and not has_mlp


def _build_model(encoder_cfg: EncoderConfig, linear_only: bool) -> RegressionModel:
    encoder = SequenceEncoder(encoder_cfg)
    return RegressionModel(encoder, dropout=encoder_cfg.dropout, linear_only=linear_only)


def _mean_pool(encoded: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    if pad_mask is None:
        return encoded.mean(dim=1)
    keep = (~pad_mask).float().unsqueeze(-1)
    summed = (encoded * keep).sum(dim=1)
    denom = keep.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _subsample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_points: int | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points is None or embeddings.shape[0] <= max_points:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
    return embeddings[idx], labels[idx]


def _eval_split(
    model: RegressionModel,
    loader: DataLoader,
    device: str,
    return_embeddings: bool = False,
):
    model.eval()
    preds: List[float] = []
    labels: List[float] = []
    embeddings: List[np.ndarray] = []
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
            if return_embeddings:
                encoded = model.encoder(tokens, pad_mask=pad_mask)
                pooled = _mean_pool(encoded, pad_mask)
                embeddings.append(pooled.cpu().numpy())
    if return_embeddings:
        emb = np.vstack(embeddings) if embeddings else np.zeros((0, 0))
        return np.asarray(preds), np.asarray(labels), emb
    return np.asarray(preds), np.asarray(labels)


def _scatter_pred_actual(output_dir: Path, splits):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (split_name, preds, labels) in enumerate(splits):
        ax = axes[idx]
        ax.scatter(labels, preds, alpha=0.4, s=20, edgecolors="none")
        lims = [0.0, 1.0]
        ax.plot(lims, lims, "r--", alpha=0.75, linewidth=2, label="Perfect prediction")
        if labels.size >= 2 and np.std(labels) > 0:
            z = np.polyfit(labels, preds, 1)
            p = np.poly1d(z)
            ax.plot(labels, p(labels), "b-", alpha=0.5, linewidth=1.5, label="Fit")
        pearson = _pearson(labels, preds)
        ax.set_xlabel("Actual Efficiency", fontsize=12)
        ax.set_ylabel("Predicted Efficiency", fontsize=12)
        ax.set_title(f"{split_name}: r={pearson:.3f}", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_scatter.png", dpi=300)


def _residual_plot(output_dir: Path, labels: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    residuals = labels - preds
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(labels, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Actual Efficiency", fontsize=12)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_title("Residual Plot (Test Set)", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=300)


def _umap_plot(output_dir: Path, embeddings: np.ndarray, labels: np.ndarray):
    import matplotlib.pyplot as plt

    try:
        from sklearn.manifold import UMAP  # type: ignore
    except Exception:
        try:
            from umap import UMAP  # type: ignore
        except Exception as exc:
            raise SystemExit("UMAP not available. Install umap-learn.") from exc

    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="RdYlGn",
        s=30,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CRISPR Efficiency", fontsize=12)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Test Set: Learned Embeddings Colored by Efficiency", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "umap_efficiency.png", dpi=300)


def _pca_plot(output_dir: Path, embeddings: np.ndarray, labels: np.ndarray):
    import matplotlib.pyplot as plt

    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as exc:
        raise SystemExit("PCA not available. Install scikit-learn.") from exc

    if embeddings.shape[0] < 2:
        return

    pca = PCA(n_components=2, random_state=42)
    embedding_2d = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum() * 100.0

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="RdYlGn",
        s=30,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CRISPR Efficiency", fontsize=12)
    ax.set_xlabel("PCA 1", fontsize=12)
    ax.set_ylabel("PCA 2", fontsize=12)
    ax.set_title(f"PCA Embeddings (Explained Variance {explained:.1f}%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_efficiency.png", dpi=300)


def _tsne_plot(
    output_dir: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    perplexity: float,
    n_iter: int,
    seed: int,
):
    import matplotlib.pyplot as plt
    import inspect

    try:
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as exc:
        raise SystemExit("t-SNE not available. Install scikit-learn.") from exc

    if embeddings.shape[0] < 2:
        return

    emb, lab = _subsample(embeddings, labels, max_points, seed)
    if emb.shape[1] > 50:
        emb = PCA(n_components=50, random_state=seed).fit_transform(emb)

    max_perplexity = max(5.0, min(perplexity, (emb.shape[0] - 1) / 3.0))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": max_perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
        "metric": "cosine",
    }
    sig = inspect.signature(TSNE)
    if "n_iter" in sig.parameters:
        tsne_kwargs["n_iter"] = n_iter
    elif "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = n_iter
    else:
        tsne_kwargs["n_iter"] = n_iter
    tsne = TSNE(**tsne_kwargs)
    embedding_2d = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=lab,
        cmap="RdYlGn",
        s=30,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CRISPR Efficiency", fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"t-SNE Embeddings (n={emb.shape[0]})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_efficiency.png", dpi=300)


def _embedding_norms(output_dir: Path, embeddings: np.ndarray, labels: np.ndarray):
    import matplotlib.pyplot as plt

    if embeddings.size == 0:
        return
    norms = np.linalg.norm(embeddings, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(norms, bins=40, alpha=0.75, edgecolor="black")
    axes[0].set_xlabel("Embedding L2 Norm", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Embedding Norm Distribution", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(norms, labels, alpha=0.5, s=20, edgecolors="none")
    axes[1].set_xlabel("Embedding L2 Norm", fontsize=12)
    axes[1].set_ylabel("CRISPR Efficiency", fontsize=12)
    axes[1].set_title("Norm vs Efficiency", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "embedding_norms.png", dpi=300)


def _cosine_similarity_heatmap(
    output_dir: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    seed: int,
):
    import matplotlib.pyplot as plt

    if embeddings.shape[0] < 2:
        return

    emb, lab = _subsample(embeddings, labels, max_points, seed)
    order = np.argsort(lab)
    emb = emb[order]
    lab = lab[order]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    emb_norm = emb / norms
    sim = emb_norm @ emb_norm.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=12)
    ax.set_title(f"Embedding Cosine Similarity (n={emb.shape[0]})", fontsize=13)
    ax.set_xlabel("Samples (sorted by efficiency)")
    ax.set_ylabel("Samples (sorted by efficiency)")
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_similarity_heatmap.png", dpi=300)


def _stratified_performance(output_dir: Path, labels: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    bins = [0, 0.25, 0.5, 0.75, 1.0]
    bin_labels = ["Low", "Medium-Low", "Medium-High", "High"]
    test_bins = np.digitize(labels, bins) - 1

    bin_pearsons = []
    for i in range(len(bin_labels)):
        mask = test_bins == i
        if mask.sum() > 5:
            r = np.corrcoef(labels[mask], preds[mask])[0, 1]
            bin_pearsons.append(r)
        else:
            bin_pearsons.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bin_labels))
    ax.bar(x, bin_pearsons, color=["red", "orange", "yellow", "green"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("Model Performance by Efficiency Range", fontsize=13)
    overall = _pearson(labels, preds)
    ax.axhline(y=overall, color="black", linestyle="--", label=f"Overall ({overall:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "stratified_performance.png", dpi=300)


def _topk_precision(output_dir: Path, labels: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    sorted_idx = np.argsort(preds)[::-1]
    threshold = np.percentile(labels, 80)
    top_k_range = [10, 20, 50, 100, 200, 500]
    precision_at_k = []

    for k in top_k_range:
        if k > len(labels):
            precision_at_k.append(np.nan)
            continue
        top_k_idx = sorted_idx[:k]
        true_positives = (labels[top_k_idx] >= threshold).sum()
        precision = true_positives / k
        precision_at_k.append(precision)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(top_k_range, precision_at_k, marker="o", linewidth=2, markersize=8)
    random_baseline = (labels >= threshold).mean()
    ax.axhline(
        y=random_baseline,
        color="red",
        linestyle="--",
        label=f"Random baseline ({random_baseline:.2f})",
    )
    ax.set_xlabel("Top K Predictions", fontsize=12)
    ax.set_ylabel("Precision (Fraction High-Efficiency)", fontsize=12)
    ax.set_title("Precision at K: Selecting High-Efficiency Guides", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "topk_precision.png", dpi=300)


def _error_hist(output_dir: Path, labels: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    errors = preds - labels
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(
        x=errors.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean error: {errors.mean():.3f}",
    )
    ax.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Error Distribution (Test Set)", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_distribution.png", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate POC visualizations.")
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="logs/figures")
    parser.add_argument("--pad-to-max-len", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-omp-duplicate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne-max-samples", type=int, default=2000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iter", type=int, default=1000)
    parser.add_argument("--similarity-max-samples", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    encoder_cfg_dict = checkpoint.get("encoder_cfg")
    if not encoder_cfg_dict:
        raise SystemExit("Checkpoint missing encoder_cfg.")
    encoder_cfg = EncoderConfig(**encoder_cfg_dict)
    linear_only = _detect_linear_head(state)

    tokenizer = DNATokenizer()
    expected_len = encoder_cfg.max_len if args.pad_to_max_len else None
    train_ds = DeepSpCas9Dataset(args.xlsx, split="train", tokenizer=tokenizer, expected_len=expected_len)
    test_ds = DeepSpCas9Dataset(args.xlsx, split="test", tokenizer=tokenizer, expected_len=expected_len)

    indices = list(range(len(train_ds)))
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(indices), generator=rng).tolist()
    val_size = max(1, int(len(indices) * args.val_fraction))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(train_ds, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_id),
    )

    model = _build_model(encoder_cfg, linear_only=linear_only)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing keys: {missing}")
    if unexpected:
        print(f"warning: unexpected keys: {unexpected}")
    model.to(args.device)

    train_preds, train_labels = _eval_split(model, train_loader, args.device)
    val_preds, val_labels = _eval_split(model, val_loader, args.device)
    test_preds, test_labels, test_emb = _eval_split(
        model, test_loader, args.device, return_embeddings=True
    )

    _scatter_pred_actual(
        output_dir,
        [
            ("Train", train_preds, train_labels),
            ("Val", val_preds, val_labels),
            ("Test", test_preds, test_labels),
        ],
    )
    _residual_plot(output_dir, test_labels, test_preds)
    _umap_plot(output_dir, test_emb, test_labels)
    _pca_plot(output_dir, test_emb, test_labels)
    _tsne_plot(
        output_dir,
        test_emb,
        test_labels,
        max_points=args.tsne_max_samples,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_iter,
        seed=args.seed,
    )
    _embedding_norms(output_dir, test_emb, test_labels)
    _cosine_similarity_heatmap(
        output_dir,
        test_emb,
        test_labels,
        max_points=args.similarity_max_samples,
        seed=args.seed,
    )
    _stratified_performance(output_dir, test_labels, test_preds)
    _topk_precision(output_dir, test_labels, test_preds)
    _error_hist(output_dir, test_labels, test_preds)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
