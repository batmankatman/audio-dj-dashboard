"""Streamlit DJ dashboard for test.wav — TRUE single-page layout.
Run with: streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import librosa

AUDIO_PATH = Path(__file__).with_name("test.wav")

st.set_page_config(
    page_title="DJ Dashboard",
    layout="wide",
    page_icon="🎚️",
    initial_sidebar_state="expanded",
)

_COMPACT_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    overflow: hidden !important;
    height: 100vh !important;
}
.block-container {
    padding: 0.4rem 1rem 0.2rem 1rem !important;
    max-width: 100% !important;
    background: radial-gradient(ellipse at top, rgba(90,0,140,0.25), transparent 60%), #030303;
}
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background-color: #0a0a0a;
    padding-top: 1rem;
}
.stMetric { padding: 0 !important; }
.stMetric label { font-size: 0.7rem !important; }
.stMetric [data-testid="stMetricValue"] { font-size: 1rem !important; }
</style>
"""
st.markdown(_COMPACT_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path.as_posix(), sr=None)
    return y, int(sr)


@st.cache_data(show_spinner=False)
def compute_mel(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop = 2048
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop)
    freqs = librosa.mel_frequencies(n_mels=mel_db.shape[0], fmin=0, fmax=sr / 2)
    return mel_db, times, freqs


def extract_segment(audio: np.ndarray, sr: int, start: float, dur: float):
    s = int(max(0, start) * sr)
    e = min(len(audio), s + int(dur * sr))
    seg = audio[s:e]
    t = np.linspace(start, start + dur, len(seg), endpoint=False)
    return t, seg


def quantize(sig: np.ndarray, bits: int) -> np.ndarray:
    lvl = 2 ** bits
    q = np.round((sig + 1) * (lvl / 2)) / (lvl / 2) - 1
    return np.clip(q, -1, 1)


def main() -> None:
    y, sr = load_audio(AUDIO_PATH)
    dur = len(y) / sr
    mel_db, mel_t, mel_f = compute_mel(y, sr)

    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest = peak / rms
    zcr = float(librosa.feature.zero_crossing_rate(y.reshape(1, -1)).mean())
    tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=512)[0])

    # ─── Sidebar controls ───
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        seg_start = st.slider("Segment start (s)", 0.0, max(0.0, dur - 0.02), 1.0, 0.01)
        seg_dur = st.slider("Segment ms", 5, 500, 50, 5) / 1000
        bit_depth = st.slider("Bit depth", 3, 16, 8)
        alias_rate = st.select_slider("Downsample Hz", [500, 1000, 2000, 4000, 8000, 16000], value=4000)

        st.markdown("---")
        st.markdown("### 🔊 Playback")
        seg_samples = y[int(seg_start * sr):int((seg_start + seg_dur) * sr)]
        st.audio(seg_samples, sample_rate=sr)

        st.download_button(
            "CSV export",
            pd.DataFrame({
                "t": np.linspace(seg_start, seg_start + seg_dur, int(seg_dur * sr)),
                "amp": y[int(seg_start * sr):int((seg_start + seg_dur) * sr)][:int(seg_dur * sr)],
            }).to_csv(index=False).encode(),
            "segment.csv",
            "text/csv",
        )

        st.markdown("---")
        st.markdown("### 🎭 Mood Estimator")
        # Compute mood metrics on full track
        first_diff = np.diff(y)
        movement_metric = float(np.mean(np.abs(first_diff)))
        transient_metric = float(np.mean(np.abs(np.diff(y, n=2))))
        spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

        calm_votes = sum([
            movement_metric < 0.001,
            transient_metric < 0.0005,
            zcr < 0.05,
            tempo < 90,
        ])
        if calm_votes >= 2:
            mood_label = "😌 Mellow / Sad"
            mood_color = "#7986cb"
        else:
            mood_label = "🔥 Energetic / Happy"
            mood_color = "#ffca28"

        st.markdown(f"<div style='text-align:center; padding:8px; background:{mood_color}22; border-radius:6px;'>"
                    f"<span style='font-size:1.3rem;'>{mood_label}</span></div>", unsafe_allow_html=True)
        st.caption(f"Movement: {movement_metric:.5f} · Transients: {transient_metric:.6f}")

    seg_t, seg_v = extract_segment(y, sr, seg_start, seg_dur)
    alias_wave = librosa.resample(seg_v, orig_sr=sr, target_sr=alias_rate)
    alias_t = np.linspace(seg_t[0], seg_t[-1], len(alias_wave))
    q_seg = quantize(seg_v, bit_depth)

    # ─── Header row: title + stats ───
    hdr = st.columns([3, 1, 1, 1, 1, 1])
    hdr[0].markdown("###DJ Dashboard")
    hdr[1].metric("Dur", f"{dur:.1f}s")
    hdr[2].metric("SR", f"{sr//1000}k")
    hdr[3].metric("RMS", f"{rms:.3f}")
    hdr[4].metric("Crest", f"{crest:.1f}")
    hdr[5].metric("BPM", f"{tempo:.0f}")

    # ─── Main 2×3 grid of tiny plots ───
    fig = make_subplots(
        rows=2,
        cols=3,
        column_widths=[0.38, 0.31, 0.31],
        row_heights=[0.55, 0.45],
        horizontal_spacing=0.045,
        vertical_spacing=0.12,
        subplot_titles=("Waveform", "Sampling", "Aliasing", "Mel spectrogram", "Quantization", "Dynamic range"),
    )

    # Waveform — zoom to show segment context (±0.5s padding around segment)
    pad = 0.5  # seconds of padding around segment
    wf_start = max(0, seg_start - pad)
    wf_end = min(dur, seg_start + seg_dur + pad)
    wf_s_idx = int(wf_start * sr)
    wf_e_idx = int(wf_end * sr)
    wf_section = y[wf_s_idx:wf_e_idx]
    stride = max(1, len(wf_section) // 2000)
    wf_t = np.linspace(wf_start, wf_end, len(wf_section))[::stride]
    wf_v = wf_section[::stride]
    fig.add_trace(go.Scatter(x=wf_t, y=wf_v, line=dict(color="#29b6f6", width=1), showlegend=False), row=1, col=1)
    # Highlight selected segment region
    fig.add_vrect(x0=seg_start, x1=seg_start + seg_dur, row=1, col=1,
                  fillcolor="#ff2d95", opacity=0.25, line_width=0)

    # Sampling
    pts = min(50, len(seg_v))
    idx = np.linspace(0, len(seg_v) - 1, pts, dtype=int)
    fig.add_trace(go.Scatter(x=seg_t, y=seg_v, line=dict(color="#ffb347", width=1), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=seg_t[idx], y=seg_v[idx], mode="markers", marker=dict(size=5, color="#f92efd"), showlegend=False), row=1, col=2)

    # Aliasing
    fig.add_trace(go.Scatter(x=seg_t, y=seg_v, line=dict(color="#00e5ff", width=1), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=alias_t, y=alias_wave, mode="lines+markers", line=dict(color="#ff1744", dash="dot", width=1), marker=dict(size=4), showlegend=False), row=1, col=3)

    # Mel spectrogram
    fig.add_trace(go.Heatmap(z=mel_db, x=mel_t, y=mel_f, colorscale="Turbo", showscale=False), row=2, col=1)

    # Quantization
    fig.add_trace(go.Scatter(x=seg_t, y=seg_v, line=dict(color="#ffd54f", width=1), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=seg_t, y=q_seg, line=dict(color="#00bfa5", width=1, shape="hv"), showlegend=False), row=2, col=2)

    # Dynamic range bar
    bits_arr = np.array([8, 16, 24])
    dr_arr = 6.02 * bits_arr
    fig.add_trace(go.Bar(y=[f"{b}b" for b in bits_arr], x=dr_arr, orientation="h", marker_color="#7c4dff", showlegend=False), row=2, col=3)

    fig.update_layout(
        height=520,
        margin=dict(l=30, r=10, t=28, b=18),
        paper_bgcolor="#030303",
        plot_bgcolor="#030303",
        font=dict(color="#eee", size=10),
    )
    for i in range(1, 7):
        fig.update_xaxes(showgrid=False, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
        fig.update_yaxes(showgrid=False, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()
