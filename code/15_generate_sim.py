"""Chunk 15 — generate the simulator-companion corpus.

Inputs   : (none; pure synthesis)
Output   : data/processed/sim.parquet

Schema matches data/processed/bench.parquet so the downstream feature pipeline
(07_features.py) ingests `sim` without modification. The capture grid is
defined by `utils.sim.default_corpus()`: four attack classes (dos, idsweep,
replay, spoofing) over five intensity levels per class, with two random seeds
per (attack, level), plus five pure-benign captures. Total 45 captures of 30 s
each. See `code/utils/sim.py` docstring for the design rationale, including
the bench-saturation memo it addresses.

Run::

    python code/15_generate_sim.py            # uses existing checkpoint
    python code/15_generate_sim.py --force    # recompute from scratch
    python code/15_generate_sim.py --smoke    # 4-capture smoke test, not committed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.cfg import load_config, project_root                       # noqa: E402
from utils.checkpoint import CheckpointStore, checkpoint               # noqa: E402
from utils.sim import (                                                # noqa: E402
    CaptureSpec, INTENSITY_LADDERS, build_capture, default_corpus,
)

CHUNK_ID = "C15_generate_sim"


def _smoke_corpus() -> list[CaptureSpec]:
    """A 4-capture sanity grid: one mid-intensity per attack class."""
    seed = 20260512
    return [
        CaptureSpec(capture_id=f"sim_smoke_dos_seed{seed}",
                    attack_type="dos", intensity_level=2, seed=seed),
        CaptureSpec(capture_id=f"sim_smoke_idsweep_seed{seed}",
                    attack_type="idsweep", intensity_level=2, seed=seed),
        CaptureSpec(capture_id=f"sim_smoke_replay_seed{seed}",
                    attack_type="replay", intensity_level=2, seed=seed),
        CaptureSpec(capture_id=f"sim_smoke_spoofing_seed{seed}",
                    attack_type="spoofing", intensity_level=2, seed=seed),
    ]


def _ladder_summary() -> str:
    pieces = ["intensity ladders:"]
    for attack, levels in INTENSITY_LADDERS.items():
        pieces.append(f"  {attack:<10s}")
        for i, p in enumerate(levels):
            kv = ", ".join(f"{k}={v}" for k, v in p.items())
            pieces.append(f"    lvl{i}: {kv}")
    return "\n".join(pieces)


def _per_capture_summary(df: pd.DataFrame) -> str:
    by_cap = (df.groupby(["capture_id", "attack_type", "source_role"])
                .size().rename("n_frames")
                .reset_index())
    return by_cap.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="recompute even if checkpoint says completed")
    p.add_argument("--smoke", action="store_true",
                   help="generate a 4-capture smoke test instead of the full grid; "
                        "writes to data/processed/sim_smoke.parquet and does not "
                        "touch the main checkpoint or output")
    args = p.parse_args(argv)

    cfg = load_config()
    root = project_root(cfg)

    if args.smoke:
        specs = _smoke_corpus()
        out_path = root / "data" / "processed" / "sim_smoke.parquet"
    else:
        specs = default_corpus()
        out_path = root / "data" / "processed" / "sim.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(_ladder_summary())
    print()
    print(f"generating {len(specs)} captures -> {out_path.name}")
    print()

    def _do_generate() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        t0 = time.perf_counter()
        for i, spec in enumerate(specs, start=1):
            t_cap = time.perf_counter()
            df = build_capture(spec)
            tx = int((df["source_role"] == "tx").sum())
            rx = int((df["source_role"] == "rx").sum())
            loss = 1.0 - rx / tx if tx else 0.0
            print(f"  {i:>2d}/{len(specs)}  {spec.capture_id:<40s} "
                  f"tx={tx:>6,} rx={rx:>6,} loss={loss*100:>5.2f}%  "
                  f"{time.perf_counter()-t_cap:.2f}s")
            frames.append(df)
        df_all = pd.concat(frames, ignore_index=True)
        print()
        print(f"total rows : {len(df_all):,}")
        print(f"runtime    : {time.perf_counter()-t0:.1f}s")
        return df_all

    if args.smoke:
        df = _do_generate()
        df.to_parquet(out_path, compression="snappy", index=False)
        print()
        print(f"wrote {out_path} ({out_path.stat().st_size/1024:.1f} KiB)")
        print()
        print(_per_capture_summary(df))
        return 0

    store = CheckpointStore(root / cfg["outputs"]["checkpoint_file"])
    with checkpoint(store, CHUNK_ID, str(out_path), force=args.force) as ck:
        if ck.skipped:
            return 0
        df = _do_generate()
        df.to_parquet(out_path, compression="snappy", index=False)
        print()
        print(f"wrote {out_path} ({out_path.stat().st_size/1024/1024:.2f} MiB)")
        print()
        print(_per_capture_summary(df))
        ck.progress(
            n_captures=int(df["capture_id"].nunique()),
            n_rows=int(len(df)),
            n_attack_rows=int((df["is_injected"] == 1).sum()),
            n_benign_rows=int((df["is_injected"] == 0).sum()),
            tx_rows=int((df["source_role"] == "tx").sum()),
            rx_rows=int((df["source_role"] == "rx").sum()),
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
