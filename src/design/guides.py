from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .primers import reverse_complement


PAM_BASES = {"A", "C", "G", "T"}


@dataclass(frozen=True)
class GuideCandidate:
    """A simple Cas12a guide candidate tied to a PAM site."""

    pam_start: int
    pam_end: int
    protospacer_start: int
    protospacer_end: int
    pam: str
    protospacer: str
    strand: str
    snp_distance: int
    notes: str = ""


def _is_ttva(pam: str) -> bool:
    """Return True if PAM matches TTTV (V = A/C/G)."""

    if len(pam) != 4:
        return False
    pam_upper = pam.upper()
    if pam_upper[:3] != "TTT":
        return False
    return pam_upper[3] in {"A", "C", "G"}


def _distance_to_interval(position: int, start: int, end: int) -> int:
    """Distance from a 1-based position to a 1-based inclusive interval."""

    if start <= position <= end:
        return 0
    if position < start:
        return start - position
    return position - end


def _scan_forward_strand(
    amplicon: str,
    snp_index: int,
    guide_length: int,
) -> Iterable[GuideCandidate]:
    """Scan the forward strand for TTTV PAMs and downstream protospacers."""

    seq = amplicon.upper()
    seq_len = len(seq)

    for pam_start in range(1, seq_len - 4 + 2):
        pam_end = pam_start + 3
        pam = seq[pam_start - 1 : pam_end]
        if not _is_ttva(pam):
            continue

        protospacer_start = pam_end + 1
        protospacer_end = protospacer_start + guide_length - 1
        if protospacer_end > seq_len:
            continue

        protospacer = seq[protospacer_start - 1 : protospacer_end]
        snp_distance = _distance_to_interval(snp_index, protospacer_start, protospacer_end)

        notes = "Forward strand PAM TTTV with downstream protospacer."
        yield GuideCandidate(
            pam_start=pam_start,
            pam_end=pam_end,
            protospacer_start=protospacer_start,
            protospacer_end=protospacer_end,
            pam=pam,
            protospacer=protospacer,
            strand="+",
            snp_distance=snp_distance,
            notes=notes,
        )


def _scan_reverse_strand(
    amplicon: str,
    snp_index: int,
    guide_length: int,
) -> Iterable[GuideCandidate]:
    """Scan the reverse strand by scanning the reverse complement."""

    seq = amplicon.upper()
    seq_len = len(seq)
    rc = reverse_complement(seq)

    for rc_pam_start in range(1, seq_len - 4 + 2):
        rc_pam_end = rc_pam_start + 3
        rc_pam = rc[rc_pam_start - 1 : rc_pam_end]
        if not _is_ttva(rc_pam):
            continue

        rc_protospacer_start = rc_pam_end + 1
        rc_protospacer_end = rc_protospacer_start + guide_length - 1
        if rc_protospacer_end > seq_len:
            continue

        rc_protospacer = rc[rc_protospacer_start - 1 : rc_protospacer_end]

        # Map reverse-complement coordinates back to the forward strand.
        pam_end = seq_len - rc_pam_start + 1
        pam_start = pam_end - 3
        protospacer_end = seq_len - rc_protospacer_start + 1
        protospacer_start = protospacer_end - guide_length + 1

        protospacer = reverse_complement(rc_protospacer)
        snp_distance = _distance_to_interval(snp_index, protospacer_start, protospacer_end)

        notes = "Reverse strand PAM TTTV with downstream protospacer (mapped)."
        yield GuideCandidate(
            pam_start=pam_start,
            pam_end=pam_end,
            protospacer_start=protospacer_start,
            protospacer_end=protospacer_end,
            pam=reverse_complement(rc_pam),
            protospacer=protospacer,
            strand="-",
            snp_distance=snp_distance,
            notes=notes,
        )


def find_cas12a_guides(
    amplicon: str,
    snp_index: int,
    guide_length: int = 23,
) -> List[GuideCandidate]:
    """Find TTTV PAM sites and associated Cas12a protospacers.

    This routine is deliberately simple and intended for notebooks:
    it does not attempt off-target analysis or advanced guide scoring.
    """

    if guide_length < 18 or guide_length > 28:
        raise ValueError("Guide length should typically be between 18 and 28 nt")
    if snp_index < 1 or snp_index > len(amplicon):
        raise ValueError("snp_index must be a 1-based position within the amplicon")

    guides: list[GuideCandidate] = []
    guides.extend(_scan_forward_strand(amplicon, snp_index, guide_length))
    guides.extend(_scan_reverse_strand(amplicon, snp_index, guide_length))
    return guides


def rank_guides_by_distance(guides: Iterable[GuideCandidate]) -> List[GuideCandidate]:
    """Rank guides by SNP proximity, then by PAM position."""

    return sorted(guides, key=lambda g: (g.snp_distance, g.pam_start))
