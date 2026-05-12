"""Synthetic CAN-bus traffic simulator for makale3 (companion to bench corpus).

Generates baseline ECU traffic and four attacker profiles (dos, idsweep, replay,
spoofing) with continuous subtlety knobs. Output schema matches `bench.parquet`
exactly, so the produced parquet drops into the existing pipeline (07_features
through 10_calibration) with no modification.

The motivation, recorded in memory note `bench-saturation-root-cause`, is that
the recorded bench attacks are physically extreme by design (DoS 4 kHz vs
baseline 263 Hz, idsweep 2048 IDs / 5 s, replay 70 percent within-capture
rate). Within-bench AUC therefore saturates at 1.000 regardless of split, and
the observation-asymmetry channel has no headroom to demonstrate gain. A
parameterised simulator closes both gaps: an intensity ladder per attack class
produces a difficulty curve that crosses 0.5 AUC at the subtle end and matches
recorded extremity at the loud end, while a load-dependent bus loss model
populates the tx/rx asymmetry that the spliced bench parquet lost.

Design points:

  * Pure numpy + Python. No python-can dependency; the bench corpus is not
    encoded through python-can either, and we want a single re-runnable script
    rather than a real driver.

  * Capture structure mirrors HCRL / CAN-MIRGU: pre-amble pure-benign, then
    an attack window, then post-amble pure-benign. This matches the structural
    pattern enforced by `code/01b_splice_bench.py`.

  * Bus loss is sampled per tx frame as a function of recent bus rate. A frame
    that survives loss emits an rx row 50 microseconds later (bus latency).
    A frame that does not survive emits only a tx row. The defender-side
    feature pipeline sees rx rows only; the ground truth is recoverable from
    tx rows for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Bench ECU set — mirrors outputs/tables/t1_bench_ecu_map.csv exactly so the
# simulated baseline is the same as the recorded baseline at the ID level.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ECU:
    can_id: int
    period_ms: float
    payload: bytes
    signal: str


BENCH_ECUS: tuple[ECU, ...] = (
    ECU(0x100, 20.0,  bytes([0x50, 0x00, 0x00, 0x00]), "vehicle_speed"),
    ECU(0x110, 10.0,  bytes([0x07, 0xD0, 0x00, 0x00]), "engine_rpm"),
    ECU(0x120, 50.0,  bytes([0x00]),                   "brake_state"),
    ECU(0x130, 50.0,  bytes([0x00, 0x00]),             "steering_angle"),
    ECU(0x140, 20.0,  bytes([0x0F]),                   "throttle_position"),
    ECU(0x150, 100.0, bytes([0x00]),                   "door_state"),
    ECU(0x160, 100.0, bytes([0x01]),                   "headlight_state"),
    ECU(0x170, 500.0, bytes([0x01]),                   "climate_control"),
)

SPOOF_TARGETS: tuple[int, ...] = (0x100, 0x110, 0x140)   # high-rate ECUs


# --------------------------------------------------------------------------
# Intensity ladders — subtlety knobs per attack class. Level 0 is hard to
# detect (overlaps with baseline); level 4 reproduces recorded-bench extremity.
# --------------------------------------------------------------------------

INTENSITY_LADDERS: dict[str, list[dict]] = {
    "dos": [
        # rate_hz at target_id. Levels 0-1 are *stealth-flood* on an existing
        # baseline ID (0x100 = vehicle speed, 20 ms nominal period) — no new ID
        # appears, only the IAT distribution shifts. Levels 2-4 escalate to a
        # priority-zero flood (id 0x000), which is the recorded-bench scenario.
        dict(rate_hz=10.0,   target_id=0x100),   # +20% rate on a 50 Hz ECU
        dict(rate_hz=50.0,   target_id=0x100),   # 2x its native rate
        dict(rate_hz=200.0,  target_id=0x000),   # priority-zero flood begins
        dict(rate_hz=1000.0, target_id=0x000),
        dict(rate_hz=4000.0, target_id=0x000),
    ],
    "idsweep": [
        # id_count = number of distinct IDs swept. start_id anchors the sweep:
        # at lvl0-1 the swept range overlaps baseline IDs (0x100-0x170) so the
        # detector cannot rely on n_unique_ids; lvl2+ sweep into territory the
        # baseline never occupies.
        dict(id_count=4,    start_id=0x140),
        dict(id_count=16,   start_id=0x100),
        dict(id_count=64,   start_id=0x080),
        dict(id_count=256,  start_id=0x000),
        dict(id_count=1024, start_id=0x000),
    ],
    "replay": [
        # replay_rate = fraction of recent-5s baseline that is re-played;
        # jitter_us = std-dev of timing perturbation around the chosen slot
        dict(replay_rate=0.05, jitter_us=5000),
        dict(replay_rate=0.15, jitter_us=2000),
        dict(replay_rate=0.30, jitter_us=1000),
        dict(replay_rate=0.60, jitter_us=500),
        dict(replay_rate=1.00, jitter_us=100),
    ],
    "spoofing": [
        # rate_ratio = attacker rate as a multiple of the target ECU's rate;
        # payload_drift = probability the attacker payload differs from legit
        dict(rate_ratio=0.5, payload_drift=0.05),
        dict(rate_ratio=1.0, payload_drift=0.10),
        dict(rate_ratio=2.0, payload_drift=0.30),
        dict(rate_ratio=3.0, payload_drift=0.70),
        dict(rate_ratio=5.0, payload_drift=1.00),
    ],
}


# --------------------------------------------------------------------------
# Baseline generation
# --------------------------------------------------------------------------

def _periodic_times(period_us: float, duration_us: int, jitter_frac: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Periodic emission with Gaussian jitter (`jitter_frac * period` stddev)."""
    n = int(duration_us // period_us)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    base = np.arange(n) * period_us
    jitter = rng.normal(0.0, period_us * jitter_frac, size=n)
    times = (base + jitter).clip(0, duration_us - 1).astype(np.int64)
    return np.sort(times)


def _drift_payload(nominal: bytes, drift_prob: float,
                   rng: np.random.Generator) -> bytes:
    """Per-byte 1-bit flip with probability drift_prob, applied independently."""
    if drift_prob <= 0.0:
        return nominal
    arr = np.frombuffer(nominal, dtype=np.uint8).copy()
    mask = rng.random(len(arr)) < drift_prob
    if mask.any():
        flips = rng.integers(0, 8, size=mask.sum())
        arr[mask] ^= (np.uint8(1) << flips.astype(np.uint8))
    return bytes(arr.tolist())


def gen_baseline(duration_us: int, rng: np.random.Generator,
                 payload_drift: float = 0.02) -> list[dict]:
    """Generate baseline traffic from the eight bench ECUs."""
    frames: list[dict] = []
    for ecu in BENCH_ECUS:
        period_us = ecu.period_ms * 1000.0
        times = _periodic_times(period_us, duration_us, jitter_frac=0.05, rng=rng)
        for t in times:
            frames.append({
                "t_us":           int(t),
                "arbitration_id": ecu.can_id,
                "dlc":            len(ecu.payload),
                "data":           _drift_payload(ecu.payload, payload_drift, rng),
                "is_injected":    0,
                "attack_type":    "benign",
            })
    return frames


# --------------------------------------------------------------------------
# Attack generators
# --------------------------------------------------------------------------

def attack_dos(start_us: int, duration_us: int, rate_hz: float,
               rng: np.random.Generator,
               target_id: int = 0x000) -> list[dict]:
    n = max(1, int(rate_hz * duration_us / 1e6))
    times = rng.uniform(start_us, start_us + duration_us, n).astype(np.int64)
    times.sort()
    out: list[dict] = []
    for t in times:
        out.append({
            "t_us":           int(t),
            "arbitration_id": target_id,
            "dlc":            8,
            "data":           bytes(rng.integers(0, 256, 8).tolist()),
            "is_injected":    1,
            "attack_type":    "dos",
        })
    return out


def attack_idsweep(start_us: int, duration_us: int, id_count: int,
                   rng: np.random.Generator,
                   start_id: int | None = None) -> list[dict]:
    """Sequential enumeration over `id_count` distinct IDs starting at start_id.

    When `start_id` is None a random anchor in [0, 0x800-id_count) is chosen.
    The intensity ladder pins `start_id` so the subtle-end sweeps overlap
    baseline IDs.
    """
    sweeps_per_second = 1.0
    n_frames = max(id_count, int(id_count * sweeps_per_second
                                  * (duration_us / 1e6)))
    if start_id is None:
        start_id = int(rng.integers(0, 0x800 - id_count))
    ids = np.tile(np.arange(id_count) + start_id, n_frames // id_count + 1)[:n_frames]
    times = np.linspace(start_us, start_us + duration_us - 1, n_frames
                        ).astype(np.int64)
    out: list[dict] = []
    for t, can_id in zip(times.tolist(), ids.tolist()):
        out.append({
            "t_us":           int(t),
            "arbitration_id": int(can_id),
            "dlc":            8,
            "data":           bytes(rng.integers(0, 256, 8).tolist()),
            "is_injected":    1,
            "attack_type":    "idsweep",
        })
    return out


def attack_replay(start_us: int, duration_us: int,
                  baseline: list[dict], replay_rate: float, jitter_us: int,
                  rng: np.random.Generator,
                  lookback_us: int = 5_000_000) -> list[dict]:
    """Replay a fraction of frames seen in the last `lookback_us` window."""
    src_lo = max(0, start_us - lookback_us)
    candidates = [f for f in baseline if src_lo <= f["t_us"] < start_us]
    n_replays = int(replay_rate * len(candidates))
    if n_replays == 0 or not candidates:
        return []
    picks = rng.integers(0, len(candidates), size=n_replays)
    slot_times = rng.uniform(start_us, start_us + duration_us, n_replays)
    jit = rng.normal(0.0, jitter_us, n_replays)
    times = (slot_times + jit).clip(start_us, start_us + duration_us - 1
                                    ).astype(np.int64)
    out: list[dict] = []
    for pick, t in zip(picks.tolist(), times.tolist()):
        src = candidates[pick]
        out.append({
            "t_us":           int(t),
            "arbitration_id": src["arbitration_id"],
            "dlc":            src["dlc"],
            "data":           src["data"],
            "is_injected":    1,
            "attack_type":    "replay",
        })
    return out


def attack_spoofing(start_us: int, duration_us: int,
                    target_ecu: ECU, rate_ratio: float, payload_drift: float,
                    rng: np.random.Generator) -> list[dict]:
    target_period_us = target_ecu.period_ms * 1000.0
    legit_rate_hz = 1e6 / target_period_us
    attacker_rate_hz = legit_rate_hz * rate_ratio
    n = max(1, int(attacker_rate_hz * duration_us / 1e6))
    times = rng.uniform(start_us, start_us + duration_us, n).astype(np.int64)
    times.sort()
    out: list[dict] = []
    for t in times:
        out.append({
            "t_us":           int(t),
            "arbitration_id": target_ecu.can_id,
            "dlc":            len(target_ecu.payload),
            "data":           _drift_payload(target_ecu.payload, payload_drift, rng),
            "is_injected":    1,
            "attack_type":    "spoofing",
        })
    return out


# --------------------------------------------------------------------------
# Bus loss model — RX observation probability as a function of recent load
# --------------------------------------------------------------------------

def apply_bus_loss(times_us: np.ndarray, rng: np.random.Generator,
                   base_loss: float = 0.02,
                   load_scale: float = 0.00025,
                   load_threshold_hz: float = 300.0,
                   max_loss: float = 0.50,
                   window_us: int = 100_000) -> np.ndarray:
    """Boolean array: True if the rx logger observes the frame."""
    if times_us.size == 0:
        return np.zeros(0, dtype=bool)
    # frames in the past `window_us` (inclusive of self)
    lo = np.searchsorted(times_us, times_us - window_us, side="left")
    rate_window = np.arange(times_us.size) - lo + 1
    rate_hz = rate_window / (window_us / 1e6)
    excess = np.maximum(rate_hz - load_threshold_hz, 0.0)
    loss_prob = np.clip(base_loss + excess * load_scale, 0.0, max_loss)
    return rng.random(times_us.size) > loss_prob


# --------------------------------------------------------------------------
# Capture assembly
# --------------------------------------------------------------------------

@dataclass
class CaptureSpec:
    capture_id: str
    attack_type: str                # 'benign' | 'dos' | 'idsweep' | 'replay' | 'spoofing'
    intensity_level: int            # 0..4 or -1 for benign-only
    seed: int
    duration_s: float = 30.0
    pre_s: float = 10.0
    attack_s: float = 10.0          # post_s = duration - pre - attack
    extra_params: dict = field(default_factory=dict)


def _build_attack(spec: CaptureSpec, baseline: list[dict],
                  rng: np.random.Generator) -> list[dict]:
    if spec.attack_type == "benign":
        return []
    params = INTENSITY_LADDERS[spec.attack_type][spec.intensity_level].copy()
    params.update(spec.extra_params)
    start_us = int(spec.pre_s * 1e6)
    dur_us = int(spec.attack_s * 1e6)
    if spec.attack_type == "dos":
        return attack_dos(start_us, dur_us, rng=rng, **params)
    if spec.attack_type == "idsweep":
        return attack_idsweep(start_us, dur_us, rng=rng, **params)
    if spec.attack_type == "replay":
        return attack_replay(start_us, dur_us, baseline=baseline, rng=rng, **params)
    if spec.attack_type == "spoofing":
        target_id = params.pop("target_id", SPOOF_TARGETS[spec.seed % len(SPOOF_TARGETS)])
        target_ecu = next(e for e in BENCH_ECUS if e.can_id == target_id)
        return attack_spoofing(start_us, dur_us, target_ecu=target_ecu, rng=rng,
                               **params)
    raise ValueError(f"unknown attack_type: {spec.attack_type}")


def build_capture(spec: CaptureSpec) -> pd.DataFrame:
    """Generate one capture and return the canonical-schema rows as a DataFrame."""
    rng = np.random.default_rng(spec.seed)
    duration_us = int(spec.duration_s * 1e6)

    baseline = gen_baseline(duration_us, rng)
    attack = _build_attack(spec, baseline, rng)

    all_frames = baseline + attack
    all_frames.sort(key=lambda f: f["t_us"])

    times = np.array([f["t_us"] for f in all_frames], dtype=np.int64)
    keep_rx = apply_bus_loss(times, rng)

    # Build rows: each frame emits a tx row; surviving frames also emit an rx row.
    n_total = len(all_frames) + int(keep_rx.sum())
    rows = {
        "t_us":           np.empty(n_total, dtype=np.int64),
        "source_dataset": np.empty(n_total, dtype=object),
        "capture_id":     np.empty(n_total, dtype=object),
        "source_file":    np.empty(n_total, dtype=object),
        "arbitration_id": np.empty(n_total, dtype=np.uint32),
        "dlc":            np.empty(n_total, dtype=np.uint8),
        "data":           np.empty(n_total, dtype=object),
        "source_role":    np.empty(n_total, dtype=object),
        "is_injected":    np.empty(n_total, dtype=np.int8),
        "attack_type":    np.empty(n_total, dtype=object),
        "label":          np.empty(n_total, dtype=np.int8),
    }
    idx = 0
    for i, f in enumerate(all_frames):
        # tx row (always)
        rows["t_us"][idx]           = f["t_us"]
        rows["source_dataset"][idx] = "sim"
        rows["capture_id"][idx]     = spec.capture_id
        rows["source_file"][idx]    = spec.capture_id
        rows["arbitration_id"][idx] = np.uint32(f["arbitration_id"])
        rows["dlc"][idx]            = np.uint8(f["dlc"])
        rows["data"][idx]           = f["data"]
        rows["source_role"][idx]    = "tx"
        rows["is_injected"][idx]    = np.int8(f["is_injected"])
        rows["attack_type"][idx]    = f["attack_type"]
        rows["label"][idx]          = np.int8(f["is_injected"])
        idx += 1
        if keep_rx[i]:
            rows["t_us"][idx]           = f["t_us"] + 50    # 50 us bus latency
            rows["source_dataset"][idx] = "sim"
            rows["capture_id"][idx]     = spec.capture_id
            rows["source_file"][idx]    = spec.capture_id
            rows["arbitration_id"][idx] = np.uint32(f["arbitration_id"])
            rows["dlc"][idx]            = np.uint8(f["dlc"])
            rows["data"][idx]           = f["data"]
            rows["source_role"][idx]    = "rx"
            rows["is_injected"][idx]    = np.int8(f["is_injected"])
            rows["attack_type"][idx]    = f["attack_type"]
            rows["label"][idx]          = np.int8(f["is_injected"])
            idx += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("t_us", kind="mergesort").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Corpus spec — the default grid used by 15_generate_sim.py
# --------------------------------------------------------------------------

def default_corpus(seeds: Sequence[int] = (20260512, 20260513)
                   ) -> list[CaptureSpec]:
    """Default grid: 4 attacks x 5 levels x 2 seeds + 5 benign-only captures."""
    specs: list[CaptureSpec] = []
    for seed in seeds:
        for attack in ("dos", "idsweep", "replay", "spoofing"):
            for lvl in range(len(INTENSITY_LADDERS[attack])):
                specs.append(CaptureSpec(
                    capture_id=f"sim_{attack}_lvl{lvl}_seed{seed}",
                    attack_type=attack,
                    intensity_level=lvl,
                    seed=seed,
                ))
    for k in range(5):
        seed = 20260520 + k
        specs.append(CaptureSpec(
            capture_id=f"sim_baseline_seed{seed}",
            attack_type="benign",
            intensity_level=-1,
            seed=seed,
        ))
    return specs
