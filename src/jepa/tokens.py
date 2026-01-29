"""DNA tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


BASES = "ACGTN"
TOKEN_TO_ID: Dict[str, int] = {base: idx for idx, base in enumerate(BASES)}
ID_TO_TOKEN: Dict[int, str] = {idx: base for base, idx in TOKEN_TO_ID.items()}

MASK_ID = len(TOKEN_TO_ID)
PAD_ID = MASK_ID + 1
VOCAB_SIZE = PAD_ID + 1


@dataclass(frozen=True)
class DNATokenizer:
    """Simple tokenizer for DNA sequences."""

    mask_id: int = MASK_ID
    pad_id: int = PAD_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def encode(self, seq: str, target_length: int | None = None) -> List[int]:
        seq = (seq or "").upper()
        ids = [TOKEN_TO_ID.get(ch, TOKEN_TO_ID["N"]) for ch in seq]
        if target_length is None:
            return ids
        if len(ids) < target_length:
            ids = ids + [self.pad_id] * (target_length - len(ids))
        elif len(ids) > target_length:
            ids = ids[:target_length]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            if idx == self.mask_id:
                tokens.append("N")
            elif idx == self.pad_id:
                continue
            else:
                tokens.append(ID_TO_TOKEN.get(idx, "N"))
        return "".join(tokens)
