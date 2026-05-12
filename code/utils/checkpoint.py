"""Lightweight checkpointing for resumable chunk execution.

Each chunk script wraps its work in `with checkpoint(chunk_id) as ck:`.
The checkpoint file lives at `outputs/results/checkpoint.json` and stores,
per chunk:
    {
        "chunk_id": "C3_extract_hcrl",
        "status":   "completed" | "in_progress",
        "started_at":  ISO8601,
        "completed_at": ISO8601 | null,
        "runtime_s":    float | null,
        "output_path":  str,
        "n_rows_done":  int,        # for batch-resumable chunks
    }
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        # utf-8-sig tolerates the BOM that PowerShell's Out-File writes.
        with self.path.open("r", encoding="utf-8-sig") as fh:
            return json.load(fh)

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self._data.get(chunk_id)

    def is_completed(self, chunk_id: str) -> bool:
        rec = self._data.get(chunk_id)
        return bool(rec and rec.get("status") == "completed")

    def mark_started(self, chunk_id: str, output_path: str | None = None) -> None:
        self._data[chunk_id] = {
            **self._data.get(chunk_id, {}),
            "chunk_id": chunk_id,
            "status": "in_progress",
            "started_at": _utcnow(),
            "completed_at": None,
            "output_path": output_path,
        }
        self._flush()

    def update_progress(self, chunk_id: str, **fields: Any) -> None:
        rec = self._data.setdefault(chunk_id, {"chunk_id": chunk_id})
        rec.update(fields)
        self._flush()

    def mark_completed(self, chunk_id: str, runtime_s: float, **extra: Any) -> None:
        rec = self._data.setdefault(chunk_id, {"chunk_id": chunk_id})
        rec.update({
            "status": "completed",
            "completed_at": _utcnow(),
            "runtime_s": round(runtime_s, 2),
            **extra,
        })
        self._flush()


@contextmanager
def checkpoint(store: CheckpointStore, chunk_id: str, output_path: str | None = None,
               force: bool = False):
    """Context manager — skips the body if the chunk is already completed.

    Usage::

        store = CheckpointStore(cfg["paths"]["checkpoint"])
        with checkpoint(store, "C3_extract_hcrl", out_path) as ck:
            if ck.skipped:
                return
            # ... do work, periodically call ck.progress(n_rows_done=...)
    """
    class _Token:
        def __init__(self) -> None:
            self.skipped = False

        def progress(self, **fields: Any) -> None:
            store.update_progress(chunk_id, **fields)

    token = _Token()

    if not force and store.is_completed(chunk_id):
        token.skipped = True
        print(f"[checkpoint] {chunk_id} already completed — skipping. "
              f"Pass --force to recompute.")
        yield token
        return

    store.mark_started(chunk_id, output_path=output_path)
    t0 = time.perf_counter()
    try:
        yield token
    except BaseException:
        store.update_progress(chunk_id, status="failed", failed_at=_utcnow())
        raise
    else:
        store.mark_completed(chunk_id, runtime_s=time.perf_counter() - t0,
                             output_path=output_path)
        print(f"[checkpoint] {chunk_id} done in "
              f"{time.perf_counter() - t0:.1f}s.")
