"""FASTA indexing and random access utilities (faidx-style)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class FastaRecord:
    name: str
    length: int
    offset: int
    line_bases: int
    line_bytes: int


class FastaIndex:
    """Minimal FASTA index with random-access fetch support."""

    def __init__(self, fasta_path: str | Path) -> None:
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
        self.index_path = Path(f"{self.fasta_path}.fai")
        if self.index_path.exists():
            self.records = self._load_index(self.index_path)
        else:
            self.records = self._build_index()

    def _load_index(self, path: Path) -> Dict[str, FastaRecord]:
        records: Dict[str, FastaRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                name, length, offset, line_bases, line_bytes = line.rstrip("\n").split("\t")[:5]
                records[name] = FastaRecord(
                    name=name,
                    length=int(length),
                    offset=int(offset),
                    line_bases=int(line_bases),
                    line_bytes=int(line_bytes),
                )
        return records

    def _build_index(self) -> Dict[str, FastaRecord]:
        records: Dict[str, FastaRecord] = {}
        name = None
        length = 0
        offset = 0
        line_bases = 0
        line_bytes = 0

        with self.fasta_path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if name is not None:
                        records[name] = FastaRecord(
                            name=name,
                            length=length,
                            offset=offset,
                            line_bases=line_bases,
                            line_bytes=line_bytes,
                        )
                    header = line[1:].strip().split()
                    name = header[0].decode("utf-8")
                    length = 0
                    offset = handle.tell()
                    line_bases = 0
                    line_bytes = 0
                    continue

                seq_line = line.rstrip(b"\r\n")
                if line_bases == 0:
                    line_bases = len(seq_line)
                    line_bytes = len(line)
                length += len(seq_line)

        if name is not None:
            records[name] = FastaRecord(
                name=name,
                length=length,
                offset=offset,
                line_bases=line_bases,
                line_bytes=line_bytes,
            )

        with self.index_path.open("w", encoding="utf-8") as handle:
            for record in records.values():
                handle.write(
                    f"{record.name}\t{record.length}\t{record.offset}\t"
                    f"{record.line_bases}\t{record.line_bytes}\n"
                )
        return records

    def contigs(self) -> List[str]:
        return list(self.records.keys())

    def fetch(self, contig: str, start: int, end: int) -> str:
        if contig not in self.records:
            raise KeyError(f"Contig not found in index: {contig}")
        record = self.records[contig]
        if start < 0 or end > record.length or start >= end:
            raise ValueError(
                f"Invalid interval {contig}:{start}-{end} (len={record.length})"
            )
        start_offset = (
            record.offset
            + (start // record.line_bases) * record.line_bytes
            + (start % record.line_bases)
        )
        end_offset = (
            record.offset
            + (end // record.line_bases) * record.line_bytes
            + (end % record.line_bases)
        )
        read_len = end_offset - start_offset
        with self.fasta_path.open("rb") as handle:
            handle.seek(start_offset)
            data = handle.read(read_len)
        seq = data.replace(b"\n", b"").replace(b"\r", b"")
        if len(seq) != (end - start):
            seq = seq[: (end - start)]
        return seq.decode("utf-8").upper()
