from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NUCLEOTIDES = {"A", "C", "G", "T", "N"}


@dataclass(frozen=True)
class AmpliconCandidate:
    """A minimal, explicit representation of an RPA amplicon proposal."""

    start: int
    end: int
    mutation_index: int
    amplicon_seq: str
    forward_primer: str
    reverse_primer: str
    amplicon_gc: float
    forward_tm: float
    reverse_tm: float
    notes: str = ""


def read_single_fasta(path: str | Path) -> tuple[str, str]:
    """Read a single-record FASTA file.

    This intentionally avoids external dependencies for the mini-project scaffold.
    It will concatenate all non-header lines into a single uppercase sequence.
    """

    fasta_path = Path(path)
    header = ""
    seq_parts: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    raise ValueError(f"Expected a single FASTA record in {fasta_path}")
                header = line[1:]
                continue
            seq_parts.append(line)

    if not header:
        raise ValueError(f"No FASTA header found in {fasta_path}")

    sequence = "".join(seq_parts).upper()
    invalid = set(sequence) - NUCLEOTIDES
    if invalid:
        raise ValueError(f"Unexpected nucleotide characters in {fasta_path}: {sorted(invalid)}")

    return header, sequence


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    complement_table = str.maketrans({
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
    })
    return seq.upper().translate(complement_table)[::-1]


def gc_fraction(seq: str) -> float:
    """Compute GC fraction for a DNA sequence."""

    if not seq:
        raise ValueError("Sequence must be non-empty")

    seq_upper = seq.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    return gc_count / len(seq_upper)


def wallace_tm(seq: str) -> float:
    """Estimate melting temperature using the Wallace rule.

    Tm = 2 * (A + T) + 4 * (G + C)
    This is crude but adequate for quick, didactic checks.
    """

    if not seq:
        raise ValueError("Sequence must be non-empty")

    seq_upper = seq.upper()
    a_count = seq_upper.count("A")
    t_count = seq_upper.count("T")
    g_count = seq_upper.count("G")
    c_count = seq_upper.count("C")
    return 2 * (a_count + t_count) + 4 * (g_count + c_count)


def extract_region(sequence: str, start: int, end: int) -> str:
    """Extract a 1-based inclusive region from a sequence."""

    if start < 1 or end < 1:
        raise ValueError("Start and end must be 1-based positive indices")
    if end < start:
        raise ValueError("End must be greater than or equal to start")
    if end > len(sequence):
        raise ValueError("End index exceeds sequence length")

    # Python slices are 0-based and end-exclusive.
    return sequence[start - 1 : end]


def _bounded_window(mutation_index: int, amplicon_length: int, seq_len: int) -> tuple[int, int]:
    """Compute a centered window while respecting sequence boundaries."""

    half = amplicon_length // 2
    start = mutation_index - half
    end = start + amplicon_length - 1

    if start < 1:
        start = 1
        end = min(seq_len, amplicon_length)
    if end > seq_len:
        end = seq_len
        start = max(1, end - amplicon_length + 1)

    return start, end


def design_amplicon_window(
    reference_sequence: str,
    mutation_index: int,
    amplicon_length: int = 160,
    primer_length: int = 30,
) -> AmpliconCandidate:
    """Propose a single RPA amplicon around a mutation position.

    The algorithm is intentionally simple:
    1) pick a roughly centered amplicon window,
    2) take fixed-length primers at the ends,
    3) report basic GC and Tm metrics.
    """

    if amplicon_length <= primer_length * 2:
        raise ValueError("Amplicon length must exceed twice the primer length")
    if mutation_index < 1 or mutation_index > len(reference_sequence):
        raise ValueError("Mutation index must be within the reference sequence")

    seq_upper = reference_sequence.upper()
    start, end = _bounded_window(mutation_index, amplicon_length, len(seq_upper))
    amplicon_seq = extract_region(seq_upper, start, end)

    forward_primer = amplicon_seq[:primer_length]
    reverse_primer = reverse_complement(amplicon_seq[-primer_length:])

    amplicon_gc = gc_fraction(amplicon_seq)
    forward_tm = wallace_tm(forward_primer)
    reverse_tm = wallace_tm(reverse_primer)

    notes = (
        "Heuristic scaffold only. In notebooks, add strand awareness, "
        "SNP placement checks, and off-target screening."
    )

    return AmpliconCandidate(
        start=start,
        end=end,
        mutation_index=mutation_index,
        amplicon_seq=amplicon_seq,
        forward_primer=forward_primer,
        reverse_primer=reverse_primer,
        amplicon_gc=amplicon_gc,
        forward_tm=forward_tm,
        reverse_tm=reverse_tm,
        notes=notes,
    )
