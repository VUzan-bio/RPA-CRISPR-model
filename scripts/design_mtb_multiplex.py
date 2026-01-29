#!/usr/bin/env python
"""Design an MTB multiplex guide set using a fine-tuned JEPA encoder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.design.guides import find_cas12a_guides
from src.design.primers import read_single_fasta
from src.jepa.model import EncoderConfig, RegressionModel, SequenceEncoder
from src.jepa.multiplex_scorer import MultiplexGuideScorer


def _detect_linear_head(state_dict: dict) -> bool:
    has_mlp = any(key.startswith("head.0.") for key in state_dict.keys())
    has_linear = "head.weight" in state_dict and "head.bias" in state_dict
    return has_linear and not has_mlp


def _load_markers(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _get_marker_gene(row: dict) -> Optional[str]:
    return row.get("gene") or row.get("Gene") or row.get("GENE")


def _load_regions(regions_dir: Path) -> Dict[str, str]:
    regions: Dict[str, str] = {}
    for fasta_path in regions_dir.glob("*.fasta"):
        _, seq = read_single_fasta(fasta_path)
        regions[fasta_path.stem] = seq
    return regions


def _match_region(gene: str, regions: Dict[str, str]) -> Optional[str]:
    if gene in regions:
        return gene
    gene_lower = gene.lower()
    candidates = [name for name in regions if gene_lower == name.lower()]
    if not candidates:
        candidates = [name for name in regions if gene_lower in name.lower()]
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def _generate_guides(seq: str, guide_length: int, snp_index: Optional[int] = None) -> List[str]:
    if not seq:
        return []
    if snp_index is None:
        snp_index = max(1, len(seq) // 2)
    guides = find_cas12a_guides(seq, snp_index=snp_index, guide_length=guide_length)
    seen = set()
    unique = []
    for guide in guides:
        if guide.protospacer in seen:
            continue
        seen.add(guide.protospacer)
        unique.append(guide.protospacer)
    return unique


def _window_around(pos: int, window_size: int, seq_len: int) -> tuple[int, int]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    half = window_size // 2
    start = max(1, pos - half)
    end = min(seq_len, start + window_size - 1)
    if end - start + 1 < window_size:
        start = max(1, end - window_size + 1)
    return start, end


def _save_matrix(path: Path, matrix: torch.Tensor) -> None:
    try:
        import numpy as np

        np.save(path, matrix.numpy())
    except Exception:
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in matrix.tolist():
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Design MTB multiplex guide set.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to finetuned JEPA checkpoint (from train_finetune.py).",
    )
    parser.add_argument(
        "--markers",
        default="data/mutations/tb_mdr_markers.csv",
        help="CSV with MDR markers (expects a 'gene' column).",
    )
    parser.add_argument(
        "--genome-fasta",
        default=None,
        help="Full H37Rv genome FASTA (single record) for coordinate-based guides.",
    )
    parser.add_argument(
        "--regions-dir",
        default="data/sequences",
        help="Directory with gene region FASTA files (one per gene).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/designed_assays/mtb_14plex",
        help="Output directory for selected guides.",
    )
    parser.add_argument("--guide-length", type=int, default=23)
    parser.add_argument("--per-gene-top-k", type=int, default=5)
    parser.add_argument("--max-combinations", type=int, default=1000)
    parser.add_argument("--compat-mode", choices=["cosine", "interaction"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--window-size",
        type=int,
        default=400,
        help="Window size around marker position when using --genome-fasta.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for scoring.",
    )
    args = parser.parse_args()

    markers_path = Path(args.markers)
    if not markers_path.exists():
        raise SystemExit(f"Markers CSV not found: {markers_path}")

    genome_seq = None
    if args.genome_fasta:
        genome_path = Path(args.genome_fasta)
        if not genome_path.exists():
            raise SystemExit(f"Genome FASTA not found: {genome_path}")
        _, genome_seq = read_single_fasta(genome_path)

    regions: Dict[str, str] = {}
    regions_dir = Path(args.regions_dir)
    if regions_dir.exists():
        regions = _load_regions(regions_dir)
    elif genome_seq is None:
        raise SystemExit(f"Regions directory not found: {regions_dir}")

    markers = _load_markers(markers_path)
    if not markers:
        raise SystemExit("No markers found in markers CSV.")

    candidate_guides: List[str] = []
    candidate_genes: List[str] = []
    missing_genes: List[str] = []
    missing_positions: List[str] = []
    for row in markers:
        gene = _get_marker_gene(row)
        if not gene:
            continue

        guides: List[str] = []
        if genome_seq is not None:
            pos = _parse_int(row.get("genomic_position") or row.get("position"))
            start = _parse_int(row.get("start"))
            end = _parse_int(row.get("end"))

            if pos is not None:
                start_pos, end_pos = _window_around(pos, args.window_size, len(genome_seq))
                seq = genome_seq[start_pos - 1 : end_pos]
                snp_index = pos - start_pos + 1
                guides = _generate_guides(seq, args.guide_length, snp_index=snp_index)
            elif start is not None and end is not None:
                if start > end:
                    start, end = end, start
                start = max(1, start)
                end = min(len(genome_seq), end)
                seq = genome_seq[start - 1 : end]
                guides = _generate_guides(seq, args.guide_length)
            else:
                missing_positions.append(gene)

        if not guides and regions:
            region_key = _match_region(gene, regions)
            if region_key:
                guides = _generate_guides(regions[region_key], args.guide_length)

        if not guides:
            missing_genes.append(gene)
            continue

        candidate_guides.extend(guides)
        candidate_genes.extend([gene] * len(guides))

    if not candidate_guides:
        raise SystemExit("No candidate guides generated from genome/regions.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder_cfg_dict = checkpoint.get("encoder_cfg")
    if not encoder_cfg_dict:
        raise SystemExit("Checkpoint missing encoder_cfg; use a finetune checkpoint.")

    encoder_cfg = EncoderConfig(**encoder_cfg_dict)
    encoder = SequenceEncoder(encoder_cfg)
    state = checkpoint.get("model", checkpoint)
    linear_only = _detect_linear_head(state)
    model = RegressionModel(encoder, dropout=encoder_cfg.dropout, linear_only=linear_only)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"warning: unexpected keys when loading checkpoint: {unexpected}")

    model.to(args.device)
    scorer = MultiplexGuideScorer(model.encoder, efficiency_head=model.head).to(args.device)
    scorer.eval()

    with torch.no_grad():
        result = scorer.optimize_guide_set(
            candidate_guides=candidate_guides,
            target_genes=candidate_genes,
            guide_length=args.guide_length,
            per_gene_top_k=args.per_gene_top_k,
            max_combinations=args.max_combinations,
            compat_mode=args.compat_mode,
            batch_size=args.batch_size,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guides_path = output_dir / "mtb_multiplex_guides.csv"
    with guides_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gene",
                "guide_sequence",
                "predicted_efficiency",
                "combined_score",
                "multiplex_score",
            ],
        )
        writer.writeheader()
        for gene, guide, score in zip(
            result["genes"],
            result["selected_guides"],
            result["individual_scores"],
        ):
            writer.writerow(
                {
                    "gene": gene,
                    "guide_sequence": guide,
                    "predicted_efficiency": f"{score:.6f}",
                    "combined_score": f"{result['combined_score']:.6f}",
                    "multiplex_score": (
                        f"{result['multiplex_score']:.6f}"
                        if result["multiplex_score"] is not None
                        else ""
                    ),
                }
            )

    compat_matrix = result["compatibility_matrix"]
    if isinstance(compat_matrix, torch.Tensor) and compat_matrix.numel() > 0:
        _save_matrix(output_dir / "compatibility_matrix.npy", compat_matrix)

    if missing_genes:
        print(f"warning: skipped genes with no matching region/guides: {missing_genes}")
    if missing_positions and genome_seq is not None:
        print(f"warning: markers missing positions; used region fallback where possible: {missing_positions}")

    print(f"Selected guides saved to: {guides_path}")
    print(f"Combined score: {result['combined_score']:.4f}")
    if result["multiplex_score"] is not None:
        print(f"Multiplex head score: {result['multiplex_score']:.4f}")
    if not result.get("used_multiplex_head", True):
        print(
            "note: multiplex head below threshold "
            f"({result.get('multiplex_threshold')}); "
            f"using fallback weights {result.get('fallback_weights')}"
        )


if __name__ == "__main__":
    main()
