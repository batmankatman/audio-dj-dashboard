"""Streamlit DJ dashboard for test.wav.
Run with: streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import librosa

AUDIO_PATH = Path(__file__).with_name("test.wav")
DEFAULT_SEGMENT_START = 1.0  # seconds
DEFAULT_SEGMENT_DURATION = 0.02  # 20 ms

st.set_page_config(
    page_title="test.wav DJ Dashboard",
    layout="wide",
    page_icon="🎚️",
)

_DJ_CSS = """
<style>
body {
    background-color: #050505;
}
section[data-testid="stSidebar"] > div:first-child {
    background-color: #080808;
    border-right: 1px solid #111;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    background: radial-gradient(circle at top, rgba(60,0,90,0.35), transparent 55%), #050505;
}
.stButton > button,
.stDownloadButton > button,
.stRadio > div > label,
.stSlider {
    color: #f2f2f2;
}
</style>
"""
st.markdown(_DJ_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    y, sr = librosa.load(audio_path.as_posix(), sr=None)
    return y, int(sr)


@st.cache_data(show_spinner=False)
def compute_spectra(audio: np.ndarray, sr: int) -> dict:
    hop = 1024
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=120, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop)
    mel_freqs = librosa.mel_frequencies(n_mels=mel_db.shape[0], fmin=0, fmax=sr / 2)

    n_bins = 72
    fmin = 32.7
    cqt = librosa.cqt(audio, sr=sr, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_times = librosa.frames_to_time(np.arange(cqt_db.shape[1]), sr=sr, hop_length=hop)
    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=12)

    return {
        "mel_db": mel_db,
        "mel_times": mel_times,
        "mel_freqs": mel_freqs,
        "cqt_db": cqt_db,
        "cqt_times": cqt_times,
        "cqt_freqs": cqt_freqs,
    }


def extract_segment(audio: np.ndarray, sr: int, start: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    start = max(0.0, min(start, len(audio) / sr))
    end = min(len(audio) / sr, start + duration)
    start_idx = int(start * sr)
    end_idx = int(end * sr)
    seg = audio[start_idx:end_idx]
    times = np.linspace(start, end, num=len(seg), endpoint=False)
    return times, seg


def quantize(signal: np.ndarray, bit_depth: int) -> np.ndarray:
    levels = 2 ** bit_depth
    quantized = np.round((signal + 1) * (levels / 2)) / (levels / 2) - 1
    return np.clip(quantized, -1, 1)


def build_waveform_fig(times: np.ndarray, values: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            line=dict(color="#29b6f6", width=2),
            hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def build_sampling_fig(segment_times: np.ndarray, segment_values: np.ndarray, sample_points: int) -> go.Figure:
    sample_points = min(sample_points, len(segment_values))
    idx = np.linspace(0, len(segment_values) - 1, sample_points, dtype=int)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=segment_times, y=segment_values, name="Segment", line=dict(color="#ffb347"))
    )
    fig.add_trace(
        go.Scatter(
            x=segment_times[idx],
            y=segment_values[idx],
            mode="markers",
            marker=dict(size=9, color="#f92efd", symbol="diamond"),
            name=f"{sample_points} samples",
        )
    )
    fig.update_layout(
        title="Sampling spotlight",
        template="plotly_dark",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def build_alias_fig(segment_times: np.ndarray, native: np.ndarray, alias_times: np.ndarray, alias_values: np.ndarray, target_sr: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=segment_times, y=native, line=dict(color="#00e5ff", width=2.5), name="Original")
    )
    fig.add_trace(
        go.Scatter(
            x=alias_times,
            y=alias_values,
            mode="lines+markers",
            line=dict(color="#ff1744", width=1.2, dash="dot"),
            marker=dict(size=6, symbol="cross"),
            name=f"Downsampled @{target_sr} Hz",
        )
    )
    fig.update_layout(
        title="Aliasing playground",
        template="plotly_dark",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def build_quant_fig(segment_times: np.ndarray, original: np.ndarray, quantized: np.ndarray, bits: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=segment_times, y=original, line=dict(color="#ffd54f"), name="Analog")
    )
    fig.add_trace(
        go.Scatter(
            x=segment_times,
            y=quantized,
            mode="lines+markers",
            line=dict(color="#00bfa5", shape="hv"),
            marker=dict(size=4),
            name=f"{bits}-bit",
        )
    )
    fig.update_layout(
        title=f"Quantization sculpting ({bits}-bit)",
        template="plotly_dark",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def build_dynamic_range_fig() -> go.Figure:
    bit_depths = np.array([8, 12, 16, 20, 24, 32])
    dynamic_ranges = 6.02 * bit_depths
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=bit_depths,
            x=dynamic_ranges,
            orientation="h",
            marker=dict(color="#7c4dff"),
        )
    )
    fig.update_layout(
        title="Ideal digital dynamic range",
        template="plotly_dark",
        height=300,
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        xaxis_title="Range (dB)",
        yaxis_title="Bit depth",
    )
    return fig


def main() -> None:
    st.title("🎚️ test.wav One-Page DJ Dashboard")
    st.caption("Waveform • Sampling • Aliasing • Quantization • Spectral energy • Stats")

    y, sr = load_audio(AUDIO_PATH)
    duration = len(y) / sr
    spectra = compute_spectra(y, sr)

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / rms)
    zcr = float(librosa.feature.zero_crossing_rate(y.reshape(1, -1)).mean())
    tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=512)[0])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{duration:.2f} s")
    col2.metric("Sample rate", f"{sr:,} Hz")
    col3.metric("RMS", f"{rms:.4f}")
    col4.metric("Tempo (est)", f"{tempo:.0f} BPM")

    base_segment_start = st.sidebar.slider(
        "Segment start (s)", 0.0, max(0.0, duration - DEFAULT_SEGMENT_DURATION), DEFAULT_SEGMENT_START, 0.01
    )
    base_segment_duration = st.sidebar.slider(
        "Segment duration (ms)", 5, 60, int(DEFAULT_SEGMENT_DURATION * 1000), 1
    ) / 1000
    segment_times, segment_values = extract_segment(y, sr, base_segment_start, base_segment_duration)

    waveform_window = st.slider(
        "Waveform window (seconds)", 0.25, min(2.0, duration), 0.8, 0.05, help="Use to zoom the full waveform panel"
    )
    waveform_start = st.slider(
        "Waveform start", 0.0, max(0.0, duration - waveform_window), 0.0, 0.01
    )
    wf_end_idx = min(len(y), int((waveform_start + waveform_window) * sr))
    wf_start_idx = int(waveform_start * sr)
    waveform_times = np.linspace(waveform_start, waveform_start + waveform_window, wf_end_idx - wf_start_idx, endpoint=False)
    waveform_values = y[wf_start_idx:wf_end_idx]
    st.plotly_chart(build_waveform_fig(waveform_times, waveform_values, "Full-track waveform"), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        sample_points = st.slider("Sample markers", 20, 200, 80, 5)
        st.plotly_chart(build_sampling_fig(segment_times, segment_values, sample_points), use_container_width=True)
    with col_b:
        alias_rate = st.select_slider(
            "Downsample to (Hz)", options=[500, 1000, 2000, 4000, 8000, 16000, 22050], value=4000
        )
        alias_wave = librosa.resample(segment_values, orig_sr=sr, target_sr=alias_rate)
        alias_times = np.linspace(segment_times[0], segment_times[-1], len(alias_wave))
        st.plotly_chart(
            build_alias_fig(segment_times, segment_values, alias_times, alias_wave, alias_rate),
            use_container_width=True,
        )

    col_c, col_d = st.columns([0.6, 0.4])
    with col_c:
        bit_depth = st.slider("Quantization bit-depth", 3, 16, 8)
        st.plotly_chart(
            build_quant_fig(segment_times, segment_values, quantize(segment_values, bit_depth), bit_depth),
            use_container_width=True,
        )
    with col_d:
        st.plotly_chart(build_dynamic_range_fig(), use_container_width=True)

    mel_tab, cqt_tab = st.tabs(["Mel energy", "Harmonic CQT"])
    with mel_tab:
        fig = px.imshow(
            spectra["mel_db"],
            x=spectra["mel_times"],
            y=spectra["mel_freqs"],
            color_continuous_scale="Turbo",
            origin="lower",
            aspect="auto",
            labels=dict(x="Time (s)", y="Freq (Hz)", color="dB"),
        )
        fig.update_layout(
            template="plotly_dark",
            title="Mel spectrogram",
            paper_bgcolor="#050505",
            plot_bgcolor="#050505",
        )
        st.plotly_chart(fig, use_container_width=True)
    with cqt_tab:
        fig = px.imshow(
            spectra["cqt_db"],
            x=spectra["cqt_times"],
            y=spectra["cqt_freqs"],
            color_continuous_scale="Viridis",
            origin="lower",
            aspect="auto",
            labels=dict(x="Time (s)", y="Freq (Hz)", color="dB"),
        )
        fig.update_layout(
            template="plotly_dark",
            title="Constant-Q texture",
            paper_bgcolor="#050505",
            plot_bgcolor="#050505",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick stats & exports")
    stats_df = pd.DataFrame(
        {
            "Metric": ["Duration", "Sample rate", "RMS", "Peak", "Crest factor", "Zero-crossing rate"],
            "Value": [f"{duration:.2f} s", f"{sr:,} Hz", f"{rms:.4f}", f"{peak:.4f}", f"{crest:.2f}", f"{zcr:.4f}"],
        }
    )
    st.dataframe(
        stats_df.style.highlight_max(color="#ff4081", axis=0),
        use_container_width=True,
        hide_index=True,
    )

    csv_times, csv_values = extract_segment(y, sr, DEFAULT_SEGMENT_START, DEFAULT_SEGMENT_DURATION)
    csv_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(csv_values)),
            "time_s": csv_times,
            "amplitude": csv_values,
            "int16_level": np.round(csv_values * (2**15 - 1)).astype(int),
        }
    )
    st.download_button(
        "Download 20 ms segment CSV",
        csv_df.to_csv(index=False).encode("utf-8"),
        file_name="test_segment_20ms.csv",
        mime="text/csv",
    )

    st.caption("Built with Streamlit · Plotly · Librosa — optimized for a single neon-dark page.")


if __name__ == "__main__":
    main()
