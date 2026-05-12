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
    t_max_s = float(df["t_us"].max() / 1e6)
    default_t = round(min(t_max_s, max(0.0, spec.pre_s + spec.attack_s / 2)), 1)
    t_sel_s = st.slider(
        "Scrub through the capture (seconds)",
        min_value=0.0, max_value=t_max_s,
        value=default_t, step=0.1,
        help="Drag to see how the gauges, status lights, and the detector's "
             "verdict evolve over the bus timeline.",
    )
    t_sel_us = int(t_sel_s * 1_000_000)
    state = _decode_state(df, wf, t_sel_us)

    # Status banner — defender's view of what is happening right now.
    pred = state["pred_prob"]
    pred_txt = f"P(attack) = {pred:.3f}" if pred is not None else "P(attack) = n/a"
    if state["attack_active"]:
        banner_bg, banner_fg = ACCENT, "white"
        banner_text = (f"Attack window active "
                       f"(true label = {state['true_label']}) — "
                       f"detector reports {pred_txt}")
    else:
        banner_bg, banner_fg = PRIMARY, "white"
        banner_text = f"Benign — detector reports {pred_txt}"
    st.markdown(
        f"<div style='background:{banner_bg};color:{banner_fg};"
        f"padding:0.55rem 1rem;border-radius:8px;font-weight:500;"
        f"margin-bottom:1rem;'>{banner_text}</div>",
        unsafe_allow_html=True,
    )

    # Three primary gauges (Speed, RPM, Throttle) plus a bus-utilisation meter.
    gauge_color = ACCENT if state["attack_active"] else PRIMARY
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        horizontal_spacing=0.04,
    )
    for col, (ecu_id, gspec) in enumerate(ECU_GAUGES.items(), start=1):
        label = gspec["name"] + (f" ({gspec['unit']})" if gspec["unit"] else "")
        fig.add_trace(
            _gauge_trace(state[ecu_id], label, gspec["vmin"], gspec["vmax"],
                         gauge_color),
            row=1, col=col,
        )
    fig.add_trace(
        _gauge_trace(state["bus_rate_hz"], "Bus rate (frames / s)",
                     0, 5000, gauge_color),
        row=1, col=4,
    )
    fig.update_layout(height=230, margin=dict(l=15, r=15, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Boolean ECU status pills.
    pcols = st.columns(4)
    for j, (ecu_id, fspec) in enumerate(ECU_FLAGS.items()):
        val = state[ecu_id]
        if val is None:
            label = "—"
        else:
            label = fspec["on"] if val else fspec["off"]
        pcols[j].metric(fspec["name"], label)
    pcols[3].metric("Climate stage",
                     str(state[ECU_CLIMATE_ID]) if state[ECU_CLIMATE_ID] is not None else "—")

    # Road strip with a car icon whose position tracks the decoded speed.
    speed = state[0x100] or 0
    car_x = 5.0 + (min(speed, 200) / 200.0) * 90.0
    road = go.Figure()
    road.add_shape(type="rect", x0=0, y0=0.2, x1=100, y1=0.8,
                    line=dict(width=0), fillcolor="#3a3a3a")
    for i in range(0, 100, 8):
        road.add_shape(type="rect", x0=i, y0=0.47, x1=i + 4, y1=0.53,
                        line=dict(width=0), fillcolor="#f4f4f4")
    car_emoji = "🚗"
    road.add_trace(go.Scatter(
        x=[car_x], y=[0.5], mode="text",
        text=[car_emoji], textfont=dict(size=44),
        showlegend=False, hoverinfo="skip",
    ))
    road.update_layout(
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[0, 1], visible=False),
        height=110, margin=dict(l=10, r=10, t=4, b=4),
        plot_bgcolor="white",
    )
    if state["attack_active"]:
        road.add_annotation(
            text="ATTACK", x=50, y=0.92, xref="x", yref="y",
            showarrow=False,
            font=dict(color=ACCENT, size=14, family="serif"),
        )
    st.plotly_chart(road, use_container_width=True)

    st.markdown(
        "<span class='footer-note'>The gauges read the defender-side (RX) "
        "view of the bus, decoded from the latest payload on each ID. "
        "Spoofing 0x100 or 0x110 snaps the speedometer or tachometer to a "
        "false value; heavy DoS and ID-sweep cause RX loss, which freezes "
        "the gauges on stale state; replay re-injects past frames, which "
        "show up as the gauge jumping back in time. Move the slider to "
        "scrub through the capture.</span>",
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
