# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""SHA3-256 receipt chain for partition boundaries and attention output."""
from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional, Tuple

GENESIS = "0" * 64


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha3_hex(body: bytes) -> str:
    return hashlib.sha3_256(body).hexdigest()


class ReceiptChain:
    """Linked SHA3-256 receipts. Genesis prev is 64 zero hex digits."""

    def __init__(self) -> None:
        self._rows: List[dict] = []

    def emit(self, payload: dict) -> str:
        prev = self._rows[-1]["digest"] if self._rows else GENESIS
        body = dict(payload)
        body["seq"] = len(self._rows)
        body["prev"] = prev
        digest = sha3_hex(_canon(body))
        row = {**body, "digest": digest}
        self._rows.append(row)
        return digest

    def verify(self) -> Tuple[bool, int, int]:
        prev = GENESIS
        for i, row in enumerate(self._rows):
            body = {k: v for k, v in row.items() if k != "digest"}
            if row.get("prev") != prev or sha3_hex(_canon(body)) != row.get("digest"):
                return False, i, i
            prev = row["digest"]
        return True, len(self._rows), -1

    def head(self) -> Optional[str]:
        return self._rows[-1]["digest"] if self._rows else None

    def __len__(self) -> int:
        return len(self._rows)
