"""Chunk 07 — sliding-window feature extraction across all five datasets.

Inputs   : data/processed/{bench,hcrl,road,canmirgu,ctat}.parquet
Output   : data/processed/features.parquet

Per-window schema (one row per sliding window):
    source_dataset       str
    capture_id           str   (group key for StratifiedGroupKFold)
    window_idx           int64 (0-based within the capture)
    window_start_us      int64 (absolute, relative to capture start)
    window_end_us        int64
    n_frames             int32
    n_unique_ids         int32
    iat_mean_us          float32
    iat_std_us           float32
    iat_p50_us           float32
    iat_p95_us           float32
    id_entropy_bits      float32   Shannon entropy over the within-window ID histogram
    id_coverage          float32   unique IDs / total frames
    payload_byte_entropy_mean  float32
    payload_byte_entropy_max   float32
    payload_diff_rate    float32   mean Hamming distance between consecutive payloads
    loss_rate_burst      float32   (tx - rx) / tx; bench-only, NaN elsewhere
    id_coverage_shrink   float32   1 - rx_unique_ids / tx_unique_ids; bench-only
    label                int8     window label = max(is_injected in window)
    attack_class         str      dominant attack_type label within window

Window = 1.0 s, stride = 0.5 s (50% overlap). Group-aware splits over
`capture_id` prevent the overlap from leaking information across folds.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.cfg import load_config, project_root                       # noqa: E402
from utils.checkpoint import CheckpointStore, checkpoint               # noqa: E402

CHUNK_ID = "C07_features"
DATASETS = ["bench", "hcrl", "road", "canmirgu", "ctat", "sim"]


def _shannon_entropy(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _payload_stats(data_bytes_concat: bytes, n_frames: int, max_dlc: int = 8
                   ) -> tuple[float, float, float]:
    """Per-byte entropy summary + consecutive-payload diff rate.

    Bytes from a window are concatenated into one big bytes object and viewed
    as an (n_frames, max_dlc) array. Frames with shorter payloads are
    zero-padded (rare in CAN classical; required for vectorisation).
    """
    if n_frames == 0:
        return 0.0, 0.0, 0.0
    arr = np.frombuffer(data_bytes_concat, dtype=np.uint8)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size % max_dlc:
        # pad up to multiple of max_dlc
        pad = max_dlc - (arr.size % max_dlc)
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    mat = arr.reshape(-1, max_dlc)[:n_frames]
    # per-column entropy
    col_entropies = np.zeros(max_dlc, dtype=np.float32)
    for c in range(max_dlc):
        vals, cnts = np.unique(mat[:, c], return_counts=True)
        col_entropies[c] = _shannon_entropy(cnts)
    diff_rate = (np.mean(np.sum(mat[1:] != mat[:-1], axis=1)) / max_dlc
                 if mat.shape[0] > 1 else 0.0)
    return (float(col_entropies.mean()),
            float(col_entropies.max()),
            float(diff_rate))


def _features_for_capture(g: pd.DataFrame, capture_id: str,
                          window_s: float, stride_s: float,
                          has_tx: bool) -> list[dict]:
    """Slide a (window, stride) window across one capture and emit features."""
    if g.empty:
        return []
    g = g.sort_values("t_us", kind="mergesort").reset_index(drop=True)
    t_us = g["t_us"].to_numpy()
    ids = g["arbitration_id"].to_numpy()
    is_inj = g["is_injected"].to_numpy()
    attack_types = g["attack_type"].to_numpy()
    roles = g["source_role"].to_numpy()
    data_col = g["data"].to_numpy()

    window_us = int(window_s * 1e6)
    stride_us = int(stride_s * 1e6)
    t_min = int(t_us.min())
    t_max = int(t_us.max())

    rows: list[dict] = []
    starts = np.arange(t_min, t_max - window_us + 1, stride_us, dtype=np.int64)
    if starts.size == 0:                       # capture shorter than 1 window
        return []

    # vectorised binary search bounds for each window
    lo = np.searchsorted(t_us, starts,                side="left")
    hi = np.searchsorted(t_us, starts + window_us,   side="left")

    for w_idx, (start, lo_i, hi_i) in enumerate(zip(starts.tolist(),
                                                    lo.tolist(),
                                                    hi.tolist())):
        n = hi_i - lo_i
        if n < 2:                              # skip empty / single-frame windows
            continue
        ts_win = t_us[lo_i:hi_i]
        ids_win = ids[lo_i:hi_i]

        # IAT (already sorted by capture sort)
        iats = np.diff(ts_win.astype(np.int64))
        iat_mean = float(iats.mean())
        iat_std  = float(iats.std())
        iat_p50  = float(np.percentile(iats, 50))
        iat_p95  = float(np.percentile(iats, 95))

        # ID histogram entropy
        _, id_counts = np.unique(ids_win, return_counts=True)
        id_entropy = _shannon_entropy(id_counts)
        id_coverage = float(id_counts.size / n)

        # payload stats
        data_bytes = b"".join(data_col[lo_i:hi_i])
        pb_mean, pb_max, pd_rate = _payload_stats(data_bytes, n)

        # asymmetry (bench-only)
        if has_tx:
            roles_win = roles[lo_i:hi_i]
            tx_mask = (roles_win == "tx")
            rx_mask = (roles_win == "rx")
            tx_count = int(tx_mask.sum())
            rx_count = int(rx_mask.sum())
            loss_burst = ((tx_count - rx_count) / tx_count
                          if tx_count > 0 else np.nan)
            tx_uids = int(np.unique(ids_win[tx_mask]).size) if tx_count else 0
            rx_uids = int(np.unique(ids_win[rx_mask]).size) if rx_count else 0
            shrink = ((1.0 - rx_uids / tx_uids) if tx_uids > 0 else np.nan)
        else:
            loss_burst = np.nan
            shrink = np.nan

        # label = max is_injected in window
        is_inj_win = is_inj[lo_i:hi_i]
        if (is_inj_win == -1).any() and not (is_inj_win == 1).any():
            label = -1                          # unknown if any unknown but no positive
        else:
            label = 1 if (is_inj_win == 1).any() else 0

        # dominant non-benign attack_type in this window if attack window,
        # else "benign"
        if label == 1:
            mask = (is_inj_win == 1)
            attack_classes = attack_types[lo_i:hi_i][mask]
            vals, cnts = np.unique(attack_classes, return_counts=True)
            attack_class = str(vals[cnts.argmax()])
        elif label == 0:
            attack_class = "benign"
        else:
            attack_class = "unknown"

        rows.append({
            "capture_id":              capture_id,
            "window_idx":              w_idx,
            "window_start_us":         int(start),
            "window_end_us":           int(start + window_us),
            "n_frames":                int(n),
            "n_unique_ids":            int(id_counts.size),
            "iat_mean_us":             iat_mean,
            "iat_std_us":              iat_std,
            "iat_p50_us":              iat_p50,
            "iat_p95_us":              iat_p95,
            "id_entropy_bits":         float(id_entropy),
            "id_coverage":             float(id_coverage),
            "payload_byte_entropy_mean": float(pb_mean),
            "payload_byte_entropy_max":  float(pb_max),
            "payload_diff_rate":         float(pd_rate),
            "loss_rate_burst":         float(loss_burst) if not np.isnan(loss_burst) else np.nan,
            "id_coverage_shrink":      float(shrink) if not np.isnan(shrink) else np.nan,
            "label":                   int(label),
            "attack_class":            attack_class,
        })
    return rows


def _process_dataset(name: str, root: Path, window_s: float, stride_s: float
                     ) -> pd.DataFrame:
    pq = root / "data" / "processed" / f"{name}.parquet"
    print(f"\n[{name}] loading {pq.name}", flush=True)
    df = pd.read_parquet(pq)
    print(f"[{name}] {len(df):,} rows, {df['capture_id'].nunique()} captures",
          flush=True)
    has_tx = (df["source_role"] == "tx").any()
    all_rows: list[dict] = []
    n_cap = df["capture_id"].nunique()
    for i, (cap, g) in enumerate(df.groupby("capture_id", sort=False), start=1):
        rows = _features_for_capture(g, cap, window_s, stride_s, has_tx)
        if rows:
            all_rows.extend(rows)
        if i % max(1, n_cap // 10) == 0 or i == n_cap:
            print(f"[{name}]   {i}/{n_cap} captures, "
                  f"{len(all_rows):,} windows so far", flush=True)
    del df
    gc.collect()
    if not all_rows:
        return pd.DataFrame()
    out = pd.DataFrame(all_rows)
    out.insert(0, "source_dataset", name)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    p.add_argument("--datasets", nargs="*", default=DATASETS,
                   help="subset of datasets to process (default: all)")
    args = p.parse_args(argv)

    cfg = load_config()
    root = project_root(cfg)
    window_s = float(cfg["ids"]["window_seconds"])
    stride_s = float(cfg["ids"]["window_stride_seconds"])
    out_path = root / "data" / "processed" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    store = CheckpointStore(root / cfg["outputs"]["checkpoint_file"])
    with checkpoint(store, CHUNK_ID, str(out_path), force=args.force) as ck:
        if ck.skipped:
            return 0
        print(f"window={window_s}s stride={stride_s}s")
        dfs: list[pd.DataFrame] = []
        for name in args.datasets:
            d = _process_dataset(name, root, window_s, stride_s)
            if not d.empty:
                dfs.append(d)
        if not dfs:
            raise SystemExit("no features produced; did the parsers run?")
        feats = pd.concat(dfs, ignore_index=True)
        feats.to_parquet(out_path, compression="snappy", index=False)
        size_mib = out_path.stat().st_size / 1024 / 1024
        print(f"\nwrote {out_path} ({size_mib:.2f} MiB, {len(feats):,} windows)")
        print()
        print("per-dataset window counts:")
        print(feats["source_dataset"].value_counts().to_string())
        print()
        print("per-class window counts:")
        print(feats["attack_class"].value_counts().to_string())
        ck.progress(
            n_windows=int(len(feats)),
            per_dataset={k: int(v) for k, v in
                         feats["source_dataset"].value_counts().items()},
            per_class={k: int(v) for k, v in
                       feats["attack_class"].value_counts().items()},
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
