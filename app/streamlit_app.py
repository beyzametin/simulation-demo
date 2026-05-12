"""Streamlit demo — Honest CAN-IDS Simulator (companion to makale3).

Run locally:   streamlit run app/streamlit_app.py
Deploy:        push to GitHub, create a Streamlit Cloud project pointed at
               app/streamlit_app.py with app/requirements.txt as the deps file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Wire the sim library from the parent repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
from utils.sim import (                                       # noqa: E402
    BENCH_ECUS, INTENSITY_LADDERS, SPOOF_TARGETS,
    CaptureSpec, build_capture, default_corpus,
)

# Reuse the feature extractor.
import importlib.util                                          # noqa: E402
_fspec = importlib.util.spec_from_file_location(
    "_feat", REPO_ROOT / "code" / "07_features.py")
_feat = importlib.util.module_from_spec(_fspec); _fspec.loader.exec_module(_feat)
features_for_capture = _feat._features_for_capture

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Honest CAN-IDS Simulator",
    page_icon="•",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1f4068"
ACCENT  = "#cc6677"
GREY    = "#888888"

# --------------------------------------------------------------------------
# Cockpit decoders — map each tracked ECU to a readable signal.
# Values mirror BENCH_ECUS in code/utils/sim.py; the simulator emits the
# nominal payload defined there, and an attacker that touches one of these
# IDs (spoofing 0x100, replay across the bus, DoS at 0x000, etc.) will
# show up as a glitch, a freeze, or a wrong value in the gauge it drives.
# --------------------------------------------------------------------------

def _u8(d, i=0):  return d[i] if len(d) > i else 0
def _u16(d):      return int.from_bytes(d[:2], "big") if len(d) >= 2 else 0

ECU_GAUGES = {
    0x100: dict(name="Speed",    unit="km/h", decode=lambda d: _u8(d),
                 vmin=0, vmax=200),
    0x110: dict(name="RPM",      unit="",     decode=lambda d: _u16(d),
                 vmin=0, vmax=8000),
    0x140: dict(name="Throttle", unit="%",    decode=lambda d: min(100, _u8(d)),
                 vmin=0, vmax=100),
}

ECU_FLAGS = {
    0x120: dict(name="Brake",     on="PRESSED", off="released",
                decode=lambda d: bool(_u8(d))),
    0x150: dict(name="Door",      on="open",    off="closed",
                decode=lambda d: bool(_u8(d))),
    0x160: dict(name="Headlight", on="on",      off="off",
                decode=lambda d: bool(_u8(d))),
}

ECU_CLIMATE_ID = 0x170


def _decode_state(df: pd.DataFrame, wf: pd.DataFrame, t_us: int) -> dict:
    """Latest defender-side (RX) value of each tracked ECU at or before t_us."""
    rx_up = df[(df["source_role"] == "rx") & (df["t_us"] <= t_us)]
    state: dict = {}
    for ecu_id, spec in {**ECU_GAUGES, **ECU_FLAGS}.items():
        sub = rx_up[rx_up["arbitration_id"] == ecu_id]
        state[ecu_id] = (spec["decode"](sub.iloc[-1]["data"])
                         if not sub.empty else None)
    climate_sub = rx_up[rx_up["arbitration_id"] == ECU_CLIMATE_ID]
    state[ECU_CLIMATE_ID] = (_u8(climate_sub.iloc[-1]["data"])
                              if not climate_sub.empty else None)

    # Window-level detector verdict at this timestamp
    win = wf[(wf["window_start_us"] <= t_us) &
             (wf["window_end_us"] >= t_us)]
    state["pred_prob"]   = float(win["pred_prob"].iloc[0]) if not win.empty else None
    state["true_label"]  = int(win["label"].iloc[0])      if not win.empty else None

    # Is there an injected frame within the last second on the bus?
    recent = df[(df["t_us"] >= max(0, t_us - 1_000_000)) &
                (df["t_us"] <= t_us)]
    state["attack_active"] = bool((recent["label"] == 1).any())

    # Recent bus load (RX frames in past 1 s) for the bus-utilisation gauge.
    rx_recent = rx_up[rx_up["t_us"] >= max(0, t_us - 1_000_000)]
    state["bus_rate_hz"] = float(len(rx_recent))
    return state


def _gauge_trace(value, label, vmin, vmax, color):
    return go.Indicator(
        mode="gauge+number",
        value=value if value is not None else 0,
        title={"text": label, "font": {"size": 12, "color": "#444"}},
        number={"font": {"size": 22, "color": PRIMARY}},
        gauge=dict(
            axis=dict(range=[vmin, vmax], tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.5),
            bgcolor="white", borderwidth=1, bordercolor="#e6e6e6",
            steps=[dict(range=[vmin, vmax], color="#f6f6f8")],
        ),
    )


# --------------------------------------------------------------------------
# Cockpit figure — auto-playing Plotly animation that drives gauges + a
# vector car along the defender's bus timeline. Each animation frame is a
# real-state sample at 0.3 s resolution, so playback is a faithful replay
# of the capture; the bar colour and ATTACK marker turn on when the live
# attack flag is set, without touching any underlying value.
# --------------------------------------------------------------------------

def _car_x_from_speed(speed_kmh: float) -> float:
    return 12.0 + (min(max(speed_kmh, 0.0), 200.0) / 200.0) * 76.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _scene_shapes(t_us: float) -> list[dict]:
    """Compose the bottom-subplot scene: layered sky gradient + a clean
    horizon ridge + a road. Shapes use xref='x'/yref='y' because in this
    figure the scatter subplot is the only cartesian axis pair."""
    sh: list[dict] = []
    # Sky gradient — four pale cool-grey bands, painted-light feel.
    sky_bands = [
        (0.45, 0.55, "#eaf0f7"),
        (0.55, 0.70, "#d8e1ec"),
        (0.70, 0.85, "#c5d2e1"),
        (0.85, 1.00, "#aebccd"),
    ]
    for lo, hi, color in sky_bands:
        sh.append(dict(type="rect", xref="x", yref="y",
                       x0=0, y0=lo, x1=100, y1=hi,
                       line=dict(width=0), fillcolor=color))

    # Distant ridge — a single subtle low silhouette along the horizon.
    ridge_offset = -(t_us / 1e6 * 2.8) % 40.0
    for k in range(-1, 4):
        base = k * 40.0 + ridge_offset
        if base > 100 or base + 40 < 0:
            continue
        sh.append(dict(type="path", xref="x", yref="y",
                       path=(f"M {base:.2f} 0.45 "
                             f"C {base + 8:.2f} 0.43 {base + 16:.2f} 0.40 "
                             f"{base + 22:.2f} 0.43 "
                             f"C {base + 30:.2f} 0.46 {base + 36:.2f} 0.43 "
                             f"{base + 42:.2f} 0.45 "
                             f"L {base + 42:.2f} 0.46 L {base:.2f} 0.46 Z"),
                       line=dict(width=0),
                       fillcolor="#9aa9bc"))

    # Horizon strip — thin haze line between the sky and the ground.
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.44, x1=100, y1=0.46,
                   line=dict(width=0), fillcolor="#b9c3d0"))

    # Ground band — narrow grass strip above the road for separation.
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.32, x1=100, y1=0.45,
                   line=dict(width=0), fillcolor="#cbc9b5"))

    # Asphalt — three subtle horizontal bands for a graduated perspective.
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.10, x1=100, y1=0.32,
                   line=dict(width=0), fillcolor="#34373d"))
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.27, x1=100, y1=0.32,
                   line=dict(width=0), fillcolor="#42464e"))
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.10, x1=100, y1=0.14,
                   line=dict(width=0), fillcolor="#26282c"))

    # Side stripes — solid white at top, solid white at bottom edge of the road.
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.305, x1=100, y1=0.320,
                   line=dict(width=0), fillcolor="#eeece6"))
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=0, y0=0.115, x1=100, y1=0.130,
                   line=dict(width=0), fillcolor="#eeece6"))

    # Centre dashes — strong, fast-scrolling cue.
    dash_offset = -(t_us / 1e6 * 14.0) % 8.0
    for i in range(-1, 16):
        x0 = i * 8 + dash_offset
        if x0 + 4 < 0 or x0 > 100:
            continue
        sh.append(dict(type="rect", xref="x", yref="y",
                       x0=_clamp(x0 + 1, 0, 100),
                       y0=0.207,
                       x1=_clamp(x0 + 4, 0, 100),
                       y1=0.222,
                       line=dict(width=0), fillcolor="#eeece6"))
    return sh


# Car palette tuned for a paper-figure aesthetic: monochrome silhouette,
# attack state is signalled by a stroked outline + soft halo, not a fill flip.
_CAR_BODY_FILL    = "#dde1e8"
_CAR_BODY_LINE    = "#2f343b"
_CAR_WINDOW_FILL  = "#a9b2bd"
_CAR_WINDOW_LINE  = "#5e656e"
_CAR_WHEEL_FILL   = "#15171a"
_CAR_HUB_FILL     = "#6e7480"
_CAR_HEADLIGHT    = "#f5cb78"
_CAR_TAILLIGHT    = "#c95a5a"


def _car_silhouette_path(cx: float) -> str:
    """Modern coupe-sedan side profile centred at cx, spans x in [cx-9.5, cx+9.5].

    Proportions follow a contemporary fastback sedan: long sloping hood,
    steep windshield, low rear-roof line, short rear deck, soft bumper curves.
    Base sits at y=0.225 (the chassis line, partially hidden by wheels)."""
    return (
        f"M {cx-9.5:.3f} 0.225 "
        f"L {cx-9.5:.3f} 0.275 "
        f"C {cx-9.5:.3f} 0.305 {cx-8.6:.3f} 0.318 {cx-7.4:.3f} 0.318 "
        f"L {cx-4.6:.3f} 0.326 "
        f"C {cx-4.0:.3f} 0.326 {cx-3.5:.3f} 0.355 {cx-2.4:.3f} 0.420 "
        f"L {cx+1.7:.3f} 0.430 "
        f"C {cx+2.8:.3f} 0.430 {cx+3.6:.3f} 0.395 {cx+4.5:.3f} 0.336 "
        f"L {cx+7.8:.3f} 0.326 "
        f"C {cx+8.9:.3f} 0.326 {cx+9.5:.3f} 0.305 {cx+9.5:.3f} 0.275 "
        f"L {cx+9.5:.3f} 0.225 "
        f"Z"
    )


def _car_window_path(cx: float) -> str:
    """Greenhouse glass — single trapezoid traced just inside the roof outline."""
    return (
        f"M {cx-3.6:.3f} 0.330 "
        f"L {cx-2.4:.3f} 0.418 "
        f"L {cx+1.6:.3f} 0.422 "
        f"L {cx+4.0:.3f} 0.336 "
        f"Z"
    )


def _wheel_shapes(wx: float, cy: float) -> list[dict]:
    """Wheel built as concentric ellipses (Plotly axes are wide so x-radius
    must be larger than y-radius to look round). Sits centred at (wx, cy)."""
    rx, ry = 1.05, 0.043           # outer tire
    rim_rx, rim_ry = 0.78, 0.032
    hub_rx, hub_ry = 0.28, 0.012
    sh = [
        # Outer tire (matte black)
        dict(type="circle", xref="x", yref="y",
             x0=wx - rx, y0=cy - ry, x1=wx + rx, y1=cy + ry,
             line=dict(color="#0a0b0d", width=1.0), fillcolor=_CAR_WHEEL_FILL),
        # Rim (dark grey)
        dict(type="circle", xref="x", yref="y",
             x0=wx - rim_rx, y0=cy - rim_ry, x1=wx + rim_rx, y1=cy + rim_ry,
             line=dict(color="#161820", width=0.5), fillcolor="#2c3038"),
    ]
    # Five thin spokes
    for ang_deg in (54, 126, 198, 270, 342):
        ang = ang_deg * 3.141592653589793 / 180.0
        dx = rim_rx * 0.94 * np.cos(ang)
        dy = rim_ry * 0.94 * np.sin(ang)
        sh.append(dict(type="line", xref="x", yref="y",
                       x0=wx, y0=cy, x1=wx + dx, y1=cy + dy,
                       line=dict(color="#7e848e", width=0.6)))
    sh.append(dict(type="circle", xref="x", yref="y",
                   x0=wx - hub_rx, y0=cy - hub_ry,
                   x1=wx + hub_rx, y1=cy + hub_ry,
                   line=dict(width=0), fillcolor=_CAR_HUB_FILL))
    return sh


def _car_shapes(state: dict) -> list[dict]:
    """Detailed sedan silhouette sitting on the road. Wheels touch road top
    at y=0.20; body extends up to a roof at y≈0.43."""
    cx = _car_x_from_speed(state[0x100] or 0.0)
    attack = state["attack_active"]
    border_color = ACCENT if attack else _CAR_BODY_LINE
    border_width = 1.8 if attack else 1.0

    wheel_cy = 0.235       # wheel centre; bottom at 0.20 sits on road top
    sh: list[dict] = []

    # Drop shadow on the road (flat ellipse beneath the car)
    sh.append(dict(type="circle", xref="x", yref="y",
                   x0=cx - 9.0, y0=0.198, x1=cx + 9.0, y1=0.215,
                   line=dict(width=0), fillcolor="rgba(0,0,0,0.28)"))

    # Attack halo
    if attack:
        sh.append(dict(type="rect", xref="x", yref="y",
                       x0=cx - 11.0, y0=0.19, x1=cx + 11.0, y1=0.47,
                       line=dict(color=ACCENT, width=0.9, dash="dot"),
                       fillcolor="rgba(204,102,119,0.05)"))

    # Lower body / chassis (between wheels)
    sh.append(dict(type="path", xref="x", yref="y",
                   path=_car_silhouette_path(cx),
                   line=dict(color=border_color, width=border_width),
                   fillcolor=_CAR_BODY_FILL))

    # Body highlight band — long horizontal sheen on the upper surface
    sh.append(dict(type="path", xref="x", yref="y",
                   path=(f"M {cx-7.0:.3f} 0.317 "
                         f"L {cx-4.5:.3f} 0.327 "
                         f"L {cx-2.4:.3f} 0.418 "
                         f"L {cx+1.8:.3f} 0.428 "
                         f"L {cx+4.4:.3f} 0.336 "
                         f"L {cx+7.5:.3f} 0.325 "
                         f"L {cx+7.5:.3f} 0.320 "
                         f"L {cx+4.0:.3f} 0.330 "
                         f"L {cx+1.5:.3f} 0.422 "
                         f"L {cx-2.2:.3f} 0.412 "
                         f"L {cx-4.5:.3f} 0.322 "
                         f"L {cx-7.0:.3f} 0.313 Z"),
                   line=dict(width=0), fillcolor="rgba(255,255,255,0.18)"))

    # Window glass (greenhouse)
    sh.append(dict(type="path", xref="x", yref="y",
                   path=_car_window_path(cx),
                   line=dict(color=_CAR_WINDOW_LINE, width=0.6),
                   fillcolor=_CAR_WINDOW_FILL))
    # Reflection along the top of the glass
    sh.append(dict(type="path", xref="x", yref="y",
                   path=(f"M {cx-3.0:.3f} 0.405 "
                         f"L {cx-2.5:.3f} 0.418 "
                         f"L {cx+1.4:.3f} 0.422 "
                         f"L {cx+2.0:.3f} 0.410 "
                         f"Z"),
                   line=dict(width=0), fillcolor="rgba(255,255,255,0.32)"))
    # B-pillar
    sh.append(dict(type="line", xref="x", yref="y",
                   x0=cx + 0.0, y0=0.330, x1=cx + 0.0, y1=0.420,
                   line=dict(color=_CAR_WINDOW_LINE, width=1.2)))
    # Beltline (under windows, runs full body length)
    sh.append(dict(type="line", xref="x", yref="y",
                   x0=cx - 7.5, y0=0.327, x1=cx + 7.4, y1=0.327,
                   line=dict(color="#7d8590", width=0.7)))
    # Lower character line — gives the body visual weight
    sh.append(dict(type="line", xref="x", yref="y",
                   x0=cx - 7.5, y0=0.265, x1=cx + 7.5, y1=0.265,
                   line=dict(color="#a0a8b2", width=0.5)))

    # Door cut lines
    for dx_split in (-0.2, ):
        sh.append(dict(type="line", xref="x", yref="y",
                       x0=cx + dx_split, y0=0.328, x1=cx + dx_split, y1=0.232,
                       line=dict(color="#8b919b", width=0.5)))

    # Door handles
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=cx - 2.4, y0=0.305, x1=cx - 1.0, y1=0.316,
                   line=dict(width=0), fillcolor="#6b727c"))
    sh.append(dict(type="rect", xref="x", yref="y",
                   x0=cx + 1.0, y0=0.305, x1=cx + 2.4, y1=0.316,
                   line=dict(width=0), fillcolor="#6b727c"))

    # Side mirror
    sh.append(dict(type="path", xref="x", yref="y",
                   path=(f"M {cx-3.8:.3f} 0.343 "
                         f"L {cx-3.1:.3f} 0.343 "
                         f"L {cx-3.0:.3f} 0.360 "
                         f"L {cx-3.75:.3f} 0.360 Z"),
                   line=dict(color=_CAR_BODY_LINE, width=0.5),
                   fillcolor="#8d96a3"))

    # Wheel arches — darker semi-ellipses inside the body, above each wheel
    for wx in (cx - 5.6, cx + 5.6):
        sh.append(dict(type="path", xref="x", yref="y",
                       path=(f"M {wx-1.25:.3f} 0.235 "
                             f"Q {wx:.3f} 0.292 "
                             f"{wx+1.25:.3f} 0.235 Z"),
                       line=dict(width=0), fillcolor="rgba(18,20,24,0.30)"))

    # Wheels
    for wx in (cx - 5.6, cx + 5.6):
        sh.extend(_wheel_shapes(wx, wheel_cy))

    # Headlight cluster (sleek slanted slot at the front of the hood)
    sh.append(dict(type="path", xref="x", yref="y",
                   path=(f"M {cx+7.4:.3f} 0.305 "
                         f"L {cx+9.2:.3f} 0.300 "
                         f"L {cx+9.2:.3f} 0.315 "
                         f"L {cx+7.6:.3f} 0.320 Z"),
                   line=dict(color="#8c7842", width=0.5),
                   fillcolor=_CAR_HEADLIGHT))
    # Taillight (slim, behind the rear quarter)
    sh.append(dict(type="path", xref="x", yref="y",
                   path=(f"M {cx-9.2:.3f} 0.300 "
                         f"L {cx-7.4:.3f} 0.305 "
                         f"L {cx-7.6:.3f} 0.320 "
                         f"L {cx-9.2:.3f} 0.315 Z"),
                   line=dict(color="#7a3038", width=0.5),
                   fillcolor=_CAR_TAILLIGHT))
    # Front grille hint (thin dark line under the headlight)
    sh.append(dict(type="line", xref="x", yref="y",
                   x0=cx + 8.4, y0=0.280, x1=cx + 9.4, y1=0.275,
                   line=dict(color="#3c4148", width=0.8)))
    return sh


def _attack_overlay(state: dict, attack_type: str) -> tuple[list[dict], list[dict]]:
    """Attack-specific visual layer (shapes, annotations). Empty when benign."""
    if not state["attack_active"] or attack_type == "benign":
        return [], []
    cx = _car_x_from_speed(state[0x100] or 0.0)
    shapes: list[dict] = []
    anns: list[dict] = []

    if attack_type == "dos":
        # Cascade of small downward triangles above the car: bus saturation.
        for i in range(7):
            ax = cx - 7 + i * 2.2
            ay_top = 0.62 + ((i % 2) * 0.06)
            shapes.append(dict(type="path", xref="x", yref="y",
                               path=(f"M {ax-0.45:.3f} {ay_top+0.08:.3f} "
                                     f"L {ax+0.45:.3f} {ay_top+0.08:.3f} "
                                     f"L {ax:.3f} {ay_top:.3f} Z"),
                               line=dict(width=0), fillcolor=ACCENT))
        anns.append(dict(text="DoS — flood at 0x000, RX buffer overrun",
                          x=cx, y=0.82, xref="x", yref="y", showarrow=False,
                          xanchor="center",
                          font=dict(color=ACCENT, size=11, family="serif")))

    elif attack_type == "replay":
        # Faded ghost silhouette behind the car: re-injected past frames.
        ghost_cx = max(8.0, cx - 12.0)
        shapes.append(dict(type="path", xref="x", yref="y",
                           path=_car_silhouette_path(ghost_cx),
                           line=dict(color=ACCENT, width=0.8, dash="dot"),
                           fillcolor="rgba(204,102,119,0.10)"))
        # Curved arrow from ghost to live car
        shapes.append(dict(type="path", xref="x", yref="y",
                           path=(f"M {ghost_cx+1.0:.3f} 0.45 "
                                 f"Q {(ghost_cx+cx)/2:.3f} 0.55 "
                                 f"{cx-1.5:.3f} 0.45"),
                           line=dict(color=ACCENT, width=1.2),
                           fillcolor="rgba(0,0,0,0)"))
        anns.append(dict(text="Replay — past frames re-injected onto the bus",
                          x=cx, y=0.82, xref="x", yref="y", showarrow=False,
                          xanchor="center",
                          font=dict(color=ACCENT, size=11, family="serif")))

    elif attack_type == "spoofing":
        # Lightning glyph above the car: false payload on a legitimate ID.
        shapes.append(dict(type="path", xref="x", yref="y",
                           path=(f"M {cx-0.6:.3f} 0.72 "
                                 f"L {cx+0.3:.3f} 0.60 "
                                 f"L {cx-0.1:.3f} 0.58 "
                                 f"L {cx+0.7:.3f} 0.48 "
                                 f"L {cx-0.05:.3f} 0.52 "
                                 f"L {cx+0.35:.3f} 0.54 Z"),
                           line=dict(color=ACCENT, width=0.8),
                           fillcolor=ACCENT))
        anns.append(dict(text="Spoofing — false payload on a legitimate ID",
                          x=cx, y=0.82, xref="x", yref="y", showarrow=False,
                          xanchor="center",
                          font=dict(color=ACCENT, size=11, family="serif")))

    elif attack_type == "idsweep":
        # Scan bar across the top: enumeration of CAN IDs.
        for i in range(12):
            ax = 6 + i * 7.5
            shapes.append(dict(type="rect", xref="x", yref="y",
                               x0=ax, y0=0.66, x1=ax + 4, y1=0.69,
                               line=dict(width=0), fillcolor=ACCENT))
        anns.append(dict(text="ID-sweep — attacker enumerates CAN IDs",
                          x=cx, y=0.82, xref="x", yref="y", showarrow=False,
                          xanchor="center",
                          font=dict(color=ACCENT, size=11, family="serif")))

    return shapes, anns


def _cockpit_annotations(state: dict, t_s: float) -> list[dict]:
    cx = _car_x_from_speed(state[0x100] or 0.0)
    speed = state[0x100] or 0
    anns: list[dict] = []
    # Small speed label tucked above the car silhouette.
    anns.append(dict(text=f"{int(speed)} km/h", x=cx, y=0.46,
                     xref="x", yref="y", showarrow=False,
                     font=dict(color=PRIMARY, size=12, family="serif")))
    # Status strip across the top of the road subplot.
    anns.append(dict(text=f"t = {t_s:.1f} s", x=2, y=0.93,
                     xref="x", yref="y", showarrow=False, xanchor="left",
                     font=dict(color="#555", size=11, family="serif")))
    pred = state.get("pred_prob")
    if pred is not None:
        anns.append(dict(text=f"detector P(attack) = {pred:.3f}",
                         x=50, y=0.93, xref="x", yref="y", showarrow=False,
                         xanchor="center",
                         font=dict(color="#555", size=11, family="serif")))
    if state["attack_active"]:
        anns.append(dict(text="● ATTACK ACTIVE", x=98, y=0.93,
                         xref="x", yref="y", showarrow=False, xanchor="right",
                         font=dict(color=ACCENT, size=11, family="serif")))
    return anns


def _build_cockpit_fig(df: pd.DataFrame, wf: pd.DataFrame,
                        attack_type: str) -> go.Figure:
    """One animated figure: gauges in the top row, a vector car on the bottom road.
    `attack_type` drives the attack-specific overlay (DoS triangles, replay ghost, etc.).
    """
    t_max_us = int(df["t_us"].max())
    step_us = 500_000        # 0.5 s sampling -> ~60 frames for a 30 s capture
    samples_us = list(range(0, t_max_us + 1, step_us))
    if samples_us[-1] != t_max_us:
        samples_us.append(t_max_us)
    states = [_decode_state(df, wf, t) for t in samples_us]
    state0 = states[0]
    ov_shapes0, ov_anns0 = _attack_overlay(state0, attack_type)

    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "indicator"}] * 4,
               [{"type": "scatter", "colspan": 4}, None, None, None]],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
    )
    fig.add_trace(_gauge_trace(state0[0x100], "Speed (km/h)", 0, 200, PRIMARY),
                  row=1, col=1)
    fig.add_trace(_gauge_trace(state0[0x110], "RPM",          0, 8000, PRIMARY),
                  row=1, col=2)
    fig.add_trace(_gauge_trace(state0[0x140], "Throttle (%)", 0, 100, PRIMARY),
                  row=1, col=3)
    fig.add_trace(_gauge_trace(state0["bus_rate_hz"], "Bus rate (Hz)",
                                0, 5000, PRIMARY), row=1, col=4)
    # Anchor scatter for the road subplot (markers invisible)
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 1], mode="markers",
                              marker=dict(opacity=0), showlegend=False,
                              hoverinfo="skip"), row=2, col=1)
    fig.update_xaxes(range=[0, 100], visible=False, fixedrange=True, row=2, col=1)
    fig.update_yaxes(range=[0, 1], visible=False, fixedrange=True, row=2, col=1)

    fig.update_layout(
        shapes=_scene_shapes(samples_us[0]) + ov_shapes0 + _car_shapes(state0),
        annotations=_cockpit_annotations(state0, samples_us[0] / 1e6) + ov_anns0,
    )

    frames = []
    for t_us, state in zip(samples_us, states):
        ov_shapes, ov_anns = _attack_overlay(state, attack_type)
        frames.append(go.Frame(
            data=[
                go.Indicator(value=state[0x100] or 0),
                go.Indicator(value=state[0x110] or 0),
                go.Indicator(value=state[0x140] or 0),
                go.Indicator(value=state["bus_rate_hz"]),
                go.Scatter(x=[0, 100], y=[0, 1]),
            ],
            layout=dict(
                shapes=_scene_shapes(t_us) + ov_shapes + _car_shapes(state),
                annotations=(_cockpit_annotations(state, t_us / 1e6) + ov_anns),
                xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
                yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
            ),
            name=f"{t_us / 1e6:.1f}",
        ))
    fig.frames = frames

    play_args  = [None, dict(frame=dict(duration=220, redraw=True),
                              transition=dict(duration=70),
                              fromcurrent=True, mode="immediate")]
    pause_args = [[None], dict(frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0))]
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=10, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=-0.07, xanchor="left", yanchor="top",
            pad=dict(t=4, r=10),
            buttons=[
                dict(label="Play",  method="animate", args=play_args),
                dict(label="Pause", method="animate", args=pause_args),
            ],
        )],
        sliders=[dict(
            active=0, x=0.12, y=-0.04, len=0.85,
            currentvalue=dict(prefix="t = ", suffix=" s",
                              visible=True, font=dict(size=11, color="#444")),
            transition=dict(duration=0),
            pad=dict(t=4, b=4),
            steps=[dict(method="animate",
                        args=[[f"{t / 1e6:.1f}"],
                              dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))],
                        label=f"{t / 1e6:.1f}")
                   for t in samples_us],
        )],
    )
    return fig

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
    h1 { font-weight: 600; letter-spacing: -0.01em; }
    .stMetric { background: #f7f7f9; border-radius: 8px; padding: 0.6rem 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.2rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; }
    .footer-note { color: #888; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Honest CAN-IDS Simulator")
st.markdown(
    "**Dial attack subtlety from undetectable to extreme. Watch a "
    "calibrated intrusion detector lose confidence in real time.**"
)
st.markdown(
    "<span class='footer-note'>Companion to "
    "<em>Honest Cross-Testbed Evaluation of CAN-Bus Intrusion Detection</em>. "
    "Synthetic traffic generator with 8 simulated ECUs, four attacker profiles, "
    "and a load-dependent observation-loss model. No proprietary data, no GPU; "
    "fits inside the same canonical schema as five public CAN-IDS datasets.</span>",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------
# Background corpus — generate once, train the reference detector once.
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Generating reference corpus and training detector...")
def load_reference_detector() -> tuple[RandomForestClassifier, list[str], pd.DataFrame]:
    """Train a calibrated RF on a fixed multi-intensity sim corpus."""
    specs = default_corpus(seeds=(20260512, 20260513))
    rows: list[dict] = []
    for spec in specs:
        df = build_capture(spec)
        rs = features_for_capture(df, spec.capture_id, 1.0, 0.5, has_tx=True)
        for r in rs:
            r["capture_id"] = spec.capture_id
            r["attack_class"] = (df[df["label"] == 1]["attack_type"].iloc[0]
                                  if (df["label"] == 1).any() else "benign")
        rows.extend(rs)
    feats = pd.DataFrame(rows)
    feat_cols = [
        "n_frames", "n_unique_ids", "iat_mean_us", "iat_std_us",
        "iat_p50_us", "iat_p95_us", "id_entropy_bits", "id_coverage",
        "payload_byte_entropy_mean", "payload_byte_entropy_max",
        "payload_diff_rate", "loss_rate_burst", "id_coverage_shrink",
    ]
    X = feats[feat_cols].fillna(feats[feat_cols].median(numeric_only=True))
    y = feats["label"].astype(int).to_numpy()
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=20260512, n_jobs=-1,
    )
    rf.fit(X, y)
    return rf, feat_cols, feats


rf, FEAT_COLS, ref_feats = load_reference_detector()

# --------------------------------------------------------------------------
# Sidebar — attacker controls
# --------------------------------------------------------------------------

st.sidebar.header("Attacker dials")
st.sidebar.markdown(
    "<span class='footer-note'>Pick an attack class and a subtlety level. "
    "Higher levels reproduce recorded-bench extremity; the lowest level is "
    "engineered to overlap legitimate traffic.</span>",
    unsafe_allow_html=True,
)
st.sidebar.write("")

attack_type = st.sidebar.selectbox(
    "Attack class",
    options=["dos", "idsweep", "replay", "spoofing", "benign"],
    format_func=lambda s: {
        "dos": "DoS — high-rate flood",
        "idsweep": "ID-sweep — fuzz across CAN IDs",
        "replay": "Replay — re-transmit recent frames",
        "spoofing": "Spoofing — frequency-dominance impersonation",
        "benign": "Benign — no attack (control)",
    }[s],
    index=2,
)

if attack_type == "benign":
    intensity_level = -1
    st.sidebar.info("Generating a clean baseline capture for reference.")
else:
    intensity_level = st.sidebar.slider(
        "Subtlety level (0 = undetectable design, 4 = recorded-bench extreme)",
        min_value=0, max_value=4, value=1, step=1,
    )
    params = INTENSITY_LADDERS[attack_type][intensity_level]
    pretty = ", ".join(f"`{k}` = {v}" for k, v in params.items())
    st.sidebar.markdown(f"**Active parameters:** {pretty}")

seed = st.sidebar.number_input("Random seed", value=20260520, step=1)
duration_s = st.sidebar.slider("Capture length (s)", 10, 60, 30, step=5)
attack_s = st.sidebar.slider("Attack window length (s)", 2, 30, 10, step=2)
pre_s = (duration_s - attack_s) / 2

st.sidebar.write("")
generate = st.sidebar.button("Generate trace", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Generate one capture on demand
# --------------------------------------------------------------------------

if not generate and "current_df" not in st.session_state:
    st.info(
        "Use the dials on the left, then press **Generate trace** to "
        "synthesise a CAN-bus capture and feed it through a reference "
        "intrusion detector."
    )
    st.stop()

if generate:
    spec = CaptureSpec(
        capture_id=f"live_{attack_type}_lvl{intensity_level}_seed{int(seed)}",
        attack_type=attack_type,
        intensity_level=intensity_level if attack_type != "benign" else -1,
        seed=int(seed),
        duration_s=float(duration_s),
        pre_s=float(pre_s),
        attack_s=float(attack_s),
    )
    df = build_capture(spec)
    win_rows = features_for_capture(df, spec.capture_id, 1.0, 0.5, has_tx=True)
    wf = pd.DataFrame(win_rows)
    wf["capture_id"] = spec.capture_id
    Xf = wf[FEAT_COLS].fillna(wf[FEAT_COLS].median(numeric_only=True))
    wf["pred_prob"] = rf.predict_proba(Xf)[:, 1]
    st.session_state["current_df"] = df
    st.session_state["current_wf"] = wf
    st.session_state["current_spec"] = spec

df = st.session_state["current_df"]
wf = st.session_state["current_wf"]
spec = st.session_state["current_spec"]

# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------

n_tx = int((df["source_role"] == "tx").sum())
n_rx = int((df["source_role"] == "rx").sum())
loss_pct = (1.0 - n_rx / n_tx) * 100 if n_tx else 0.0
attack_frac = float((df["label"] == 1).mean())

if (wf["label"] > 0).any() and (wf["label"] == 0).any():
    auc = roc_auc_score(wf["label"].astype(int), wf["pred_prob"])
else:
    auc = float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("TX frames", f"{n_tx:,}", help="Frames transmitted on the simulated bus.")
c2.metric("RX frames observed", f"{n_rx:,}", f"-{loss_pct:.1f}% loss")
c3.metric("Attack fraction", f"{attack_frac*100:.1f}%")
c4.metric("Window-level ROC-AUC", f"{auc:.3f}" if auc == auc else "n/a",
          help="Reference detector confidence vs. ground truth.")

st.divider()

# --------------------------------------------------------------------------
# Visual tabs
# --------------------------------------------------------------------------

tab_cockpit, tab1, tab2, tab3, tab4 = st.tabs([
    "Cockpit", "Traffic timeline", "Detector confidence",
    "Calibration", "Asymmetry channel",
])

WINDOW_US = 1_000_000

with tab_cockpit:
    cockpit_fig = _build_cockpit_fig(df, wf, spec.attack_type)
    st.plotly_chart(cockpit_fig, use_container_width=True)

    # Static snapshot of the boolean ECU flags, taken at the attack-window
    # midpoint so the pills show the most informative moment of the capture.
    snap_t_us = int(spec.pre_s * 1e6) + int(spec.attack_s * 1e6 / 2)
    snap_t_us = min(snap_t_us, int(df["t_us"].max()))
    snap = _decode_state(df, wf, snap_t_us)
    pcols = st.columns(4)
    pill_help = f"Snapshot at t = {snap_t_us/1e6:.1f}s (attack-window midpoint)."
    for j, (ecu_id, fspec) in enumerate(ECU_FLAGS.items()):
        val = snap[ecu_id]
        label = "—" if val is None else (fspec["on"] if val else fspec["off"])
        pcols[j].metric(fspec["name"], label, help=pill_help)
    pcols[3].metric(
        "Climate stage",
        str(snap[ECU_CLIMATE_ID]) if snap[ECU_CLIMATE_ID] is not None else "—",
        help=pill_help,
    )

    st.markdown(
        "<span class='footer-note'>Press <b>Play</b> below the dashboard to "
        "watch the capture flow in real time, or drag the slider to scrub. "
        "Gauges decode the defender-side (RX) view of the bus from the "
        "latest payload on each ID. Spoofing 0x100 or 0x110 snaps the "
        "speedometer or tachometer to a false value; heavy DoS or "
        "ID-sweep cause RX loss, which freezes the gauges on stale state; "
        "replay re-injects past frames, which show up as the gauge "
        "jumping back in time. Body colour and the ATTACK ACTIVE marker "
        "turn on whenever the true label is positive in the live window.</span>",
        unsafe_allow_html=True,
    )

with tab1:
    rx = df[df["source_role"] == "rx"].copy()
    rx["t_s"] = rx["t_us"] / 1e6
    bins = np.arange(0, rx["t_s"].max() + 0.1, 0.1)
    rate_total, _   = np.histogram(rx["t_s"], bins=bins)
    rate_attack, _  = np.histogram(rx[rx["label"] == 1]["t_s"], bins=bins)
    rate_benign     = rate_total - rate_attack
    mid = 0.5 * (bins[:-1] + bins[1:])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        vertical_spacing=0.08,
                        subplot_titles=("Bus rate (100 ms buckets, defender RX)",
                                        "Per-window attack probability"))
    fig.add_bar(x=mid, y=rate_benign, name="benign", marker_color=PRIMARY,
                row=1, col=1)
    fig.add_bar(x=mid, y=rate_attack, name="attack", marker_color=ACCENT,
                row=1, col=1)
    fig.add_trace(go.Scatter(x=(wf["window_start_us"] + WINDOW_US/2)/1e6,
                              y=wf["pred_prob"],
                              mode="lines+markers",
                              line=dict(color=PRIMARY, width=2),
                              marker=dict(size=5),
                              name="P(attack)"),
                  row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY,
                  row=2, col=1, annotation_text="threshold")
    fig.update_layout(barmode="stack", height=460,
                      margin=dict(l=30, r=10, t=40, b=20),
                      legend=dict(orientation="h", y=1.12, x=0.5,
                                  xanchor="center"))
    fig.update_yaxes(title_text="frames / 100 ms", row=1, col=1)
    fig.update_yaxes(title_text="P(attack)", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    win_truth = wf["label"].astype(int)
    win_prob  = wf["pred_prob"]
    sorted_idx = np.argsort(-win_prob.to_numpy())
    rank = np.arange(1, len(sorted_idx) + 1)
    cum_pos = np.cumsum(win_truth.to_numpy()[sorted_idx]) / max(1, win_truth.sum())
    cum_neg = np.cumsum((1 - win_truth.to_numpy()[sorted_idx])) / max(1, len(win_truth) - win_truth.sum())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rank, y=cum_pos, mode="lines", name="TPR",
                              line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=rank, y=cum_neg, mode="lines", name="FPR",
                              line=dict(color=PRIMARY, width=2.5)))
    fig.update_layout(height=420,
                      margin=dict(l=30, r=10, t=20, b=30),
                      xaxis_title="windows ranked by P(attack), descending",
                      yaxis_title="cumulative fraction",
                      legend=dict(orientation="h", y=1.08, x=0.5,
                                  xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    bins = np.linspace(0, 1, 6)
    digit = np.digitize(wf["pred_prob"], bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = (digit == b)
        if m.any():
            rows.append({
                "bin_mid": 0.5 * (bins[b] + bins[b + 1]),
                "obs_rate": float(wf.loc[m, "label"].astype(int).mean()),
                "n": int(m.sum()),
            })
    rel = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(color=GREY, dash="dash"),
                              name="perfect calibration"))
    if not rel.empty:
        fig.add_trace(go.Scatter(x=rel["bin_mid"], y=rel["obs_rate"],
                                  mode="lines+markers",
                                  marker=dict(size=8 + rel["n"].clip(0, 80)/8,
                                              color=ACCENT),
                                  line=dict(color=ACCENT, width=2),
                                  name="this trace"))
    fig.update_layout(height=420,
                      margin=dict(l=30, r=10, t=20, b=30),
                      xaxis_title="predicted P(attack), bin centre",
                      yaxis_title="empirical positive rate",
                      xaxis=dict(range=[0, 1]),
                      yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", y=1.08, x=0.5,
                                  xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Per-window loss-rate burst",
                                        "ID-coverage shrinkage"))
    for j, col in enumerate(("loss_rate_burst", "id_coverage_shrink"), start=1):
        for lbl, color in [(0, PRIMARY), (1, ACCENT)]:
            vals = wf.loc[wf["label"] == lbl, col].dropna()
            if vals.empty:
                continue
            fig.add_trace(go.Histogram(x=vals, nbinsx=18,
                                       name=f"label={lbl}",
                                       marker_color=color,
                                       opacity=0.65,
                                       showlegend=(j == 1)),
                          row=1, col=j)
    fig.update_layout(barmode="overlay", height=380,
                      margin=dict(l=30, r=10, t=40, b=30),
                      legend=dict(orientation="h", y=1.18, x=0.5,
                                  xanchor="center"))
    fig.update_xaxes(title_text="loss_rate_burst", row=1, col=1)
    fig.update_xaxes(title_text="id_coverage_shrink", row=1, col=2)
    fig.update_yaxes(title_text="windows", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<span class='footer-note'>The asymmetry channel measures how the "
        "attack distorts the defender's observation of the bus — packet loss "
        "and ID-coverage drift between transmitted and observed frames. In "
        "the recorded bench corpus this channel is null because attacks are "
        "physically extreme; the simulator surfaces a measurable lift in the "
        "subtle-attack regime (replay lvl0, spoofing lvl1).</span>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    "<div class='footer-note'>Source: "
    "<a href='https://github.com/' target='_blank'>makale3 repository</a>. "
    "Headline cross-testbed numbers, Holm-corrected significance and "
    "calibration measurements live in the companion paper.</div>",
    unsafe_allow_html=True,
)
