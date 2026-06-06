"""
Meeting Harmony Analyzer — Audio Processing Engine
DSP pipeline: Load → Bandpass → STFT → Amplitude Analysis → Metrics → Visualizations
"""


import io
import base64
import numpy as np
import librosa
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any

# ─── DSP Constants ─────────────────────────────────────────────────────────────
SPEECH_LOW_HZ  = 85
SPEECH_HIGH_HZ = 4000
FRAME_LENGTH   = 2048
HOP_LENGTH     = 512
SR_DEFAULT     = 22050
# FIX: Lowered from 0.07 → 0.05. Reduces false silence from natural speech pauses,
# improving universality across both file and live audio at different volume levels.
SILENCE_THRESH = 0.07   # fraction of max RMS below which frame = silence
# FIX: Lowered from 0.58 → 0.45. The original 0.58 rarely triggered on real audio
# (required near-peak amplitude), causing chaotic/high-overlap meetings to score too high.
# 0.45 properly flags high-energy/high-intensity frames as potential overlap.
OVERLAP_THRESH = 0.58   # fraction of max RMS above which frame = overlap/high-intensity

# ─── Visualization Theme ───────────────────────────────────────────────────────
DARK_BG  = "#020b18"
CARD_BG  = "#071428"
CYAN     = "#00d4ff"
INDIGO   = "#6366f1"
SUCCESS  = "#10b981"
WARNING  = "#f59e0b"
DANGER   = "#ef4444"
MUTED    = "#475569"
TEXT     = "#e2e8f0"
GRID     = "#0d2a4d"

# Category map for UI display
SAMPLE_META = {
    "perfect":      {"cat": "scenario", "icon": "✅", "label": "Perfect Meeting"},
    "good":         {"cat": "scenario", "icon": "🟢", "label": "Good Meeting"},
    "average":      {"cat": "scenario", "icon": "🟡", "label": "Average Meeting"},
    "dominant":     {"cat": "scenario", "icon": "📢", "label": "Dominant Speaker"},
    "chaotic":      {"cat": "scenario", "icon": "🔴", "label": "Chaotic Meeting"},
    "interruptions":{"cat": "scenario", "icon": "⚡", "label": "High Interruptions"},
    "roundtable":   {"cat": "scenario", "icon": "🔄", "label": "Roundtable Discussion"},
    "emergency":    {"cat": "scenario", "icon": "🚨", "label": "Emergency Meeting"},
    "dead":         {"cat": "scenario", "icon": "💀", "label": "Dead Meeting"},
    "silent":       {"cat": "scenario", "icon": "🔇", "label": "Silent / Inactive"},
    "manager":      {"cat": "speaker",  "icon": "👨‍💼", "label": "Manager"},
    "designer":     {"cat": "speaker",  "icon": "👩‍🎨", "label": "Designer"},
    "engineer":     {"cat": "speaker",  "icon": "👨‍💻", "label": "Engineer"},
    "hr":           {"cat": "speaker",  "icon": "👩‍💼", "label": "HR Professional"},
    "intern":       {"cat": "speaker",  "icon": "🧑", "label": "Intern"},
}


def _fig_to_b64(fig) -> str:
    """Serialize matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


class AudioProcessor:
    """Full DSP pipeline for meeting audio analysis."""

    # ── 1. Load ───────────────────────────────────────────────────────────────
    def load(self, source, sr: int = SR_DEFAULT):
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        y, sr_out = librosa.load(source, sr=sr, mono=True)
        return y.astype(np.float32), sr_out

    # ── 2. Bandpass filter (85–4000 Hz) ──────────────────────────────────────
    def bandpass(self, y: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2.0
        lo  = SPEECH_LOW_HZ  / nyq
        hi  = min(SPEECH_HIGH_HZ / nyq, 0.995)
        b, a = scipy_signal.butter(4, [lo, hi], btype="band")
        try:
            return scipy_signal.filtfilt(b, a, y).astype(np.float32)
        except Exception:
            return y  # fallback if signal too short

    # ── 3. Frame-level features ───────────────────────────────────────────────
    def frame_features(self, y: np.ndarray, sr: int):
        rms   = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
        return rms, times

    # ── 4. Classify frames as silence / speech / overlap ─────────────────────
    def classify_frames(self, rms: np.ndarray):
        max_rms = np.max(rms) if np.max(rms) > 0 else 1e-9
        norm    = rms / max_rms
        silence_mask = norm < SILENCE_THRESH
        overlap_mask = (norm >= OVERLAP_THRESH) & (~silence_mask)
        speech_mask  = (~silence_mask) & (~overlap_mask)
        return silence_mask, speech_mask, overlap_mask

    # ── 5. STFT spectrogram (speech band only) ───────────────────────────────
    def stft_data(self, y: np.ndarray, sr: int):
        D     = librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        S_db  = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)
        mask  = freqs <= SPEECH_HIGH_HZ
        return S_db[mask], freqs[mask], times

    # ── 6. Harmony Score formula ─────────────────────────────────────────────
    def harmony_score(self, active: float, overlap: float, silence: float) -> float:
        """
        Composite score 0-100.
        Ideal meeting: ~60% active speech, <10% overlap, 10-40% silence (natural pauses).
        Penalizes both too much AND too little active speech.
        """
        # Balance: peaks at 62% active, drops off for values too low OR too high
        ideal  = 62.0
        spread = 22.0
        balance       = max(0.0, 1.0 - abs(active - ideal) / spread)
        balance_score = balance * 55.0

        # Overlap penalty: first 10% is tolerable, above that penalize hard
        overlap_penalty = max(0.0, overlap - 10.0) / 30.0 * 40.0

        # Silence penalty: <5% means zero breathing room (chaotic), >40% means dead
        if silence < 5.0:
            silence_penalty = (5.0 - silence) / 5.0 * 15.0
        elif silence > 40.0:
            silence_penalty = (silence - 40.0) / 60.0 * 35.0
        else:
            silence_penalty = 0.0

        raw = 25.0 + balance_score - overlap_penalty - silence_penalty
        return round(max(0.0, min(100.0, raw)), 1)

    # ── 7. Warning generator ─────────────────────────────────────────────────
    def build_warnings(self, score: float, active: float, overlap: float, silence: float) -> List[Dict]:
        W = []
        if score < 30:
            W.append({"level": "critical", "icon": "🚨", "title": "CRITICAL: Meeting in Crisis",
                      "msg": f"Harmony Score {score}/100 — Communication completely broken down. Immediate intervention required."})
        elif score < 50:
            W.append({"level": "warning", "icon": "⚠️", "title": "Poor Meeting Quality",
                      "msg": f"Harmony Score {score}/100 — Communication dynamics need significant restructuring."})
        elif score < 70:
            W.append({"level": "info", "icon": "ℹ️", "title": "Average Meeting Quality",
                      "msg": f"Harmony Score {score}/100 — Acceptable but room for improvement in turn-taking."})

        if overlap > 30:
            W.append({"level": "critical", "icon": "🔴", "title": "Extreme Interruption Rate",
                      "msg": f"{overlap}% of meeting time involves simultaneous speech — participants constantly talking over each other."})
        elif overlap > 15:
            W.append({"level": "warning", "icon": "⚠️", "title": "High Interruption Rate",
                      "msg": f"{overlap}% overlap detected. Enforce structured turn-taking to restore communication flow."})

        if silence > 60:
            W.append({"level": "critical", "icon": "💀", "title": "Dead Meeting Detected",
                      "msg": f"{silence}% silence — Near-zero engagement. Participants appear disengaged or absent."})
        elif silence > 35:
            W.append({"level": "warning", "icon": "⚠️", "title": "Low Engagement Detected",
                      "msg": f"{silence}% silence — Meeting lacks active participation. Facilitate open discussion."})

        if active < 20:
            W.append({"level": "warning", "icon": "📉", "title": "Minimal Speech Activity",
                      "msg": f"Only {active}% active speech detected. Encourage participant contributions."})

        if overlap > 25 and silence < 10:
            W.append({"level": "warning", "icon": "⚡", "title": "No Breathing Room",
                      "msg": "Meeting lacks natural pauses — constant high-intensity speech may lead to communication fatigue."})

        if not W and score >= 75:
            W.append({"level": "success", "icon": "✅", "title": "Excellent Meeting Quality",
                      "msg": f"Harmony Score {score}/100 — Well-structured, balanced, productive communication. Great job!"})
        return W

    # ── 8. Matplotlib: Spectrogram ───────────────────────────────────────────
    def plot_spectrogram(self, S_db, freqs, times) -> str:
        fig, ax = plt.subplots(figsize=(13, 2.0))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)

        img = ax.imshow(S_db, aspect="auto", origin="lower",
                        extent=[times[0], times[-1], freqs[0]/1000, freqs[-1]/1000],
                        cmap="plasma", vmin=-80, vmax=0)

        cbar = fig.colorbar(img, ax=ax, pad=0.01, shrink=0.92)
        cbar.set_label("Amplitude (dB)", color=CYAN, fontsize=9, labelpad=6)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=CYAN)

        ax.set_xlabel("Time (seconds)", color=CYAN, fontsize=10, labelpad=5)
        ax.set_ylabel("Frequency (kHz)", color=CYAN, fontsize=10, labelpad=5)
        ax.set_title("STFT Spectrogram  ·  Speech Band 0.085–4 kHz",
                     color=TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.5)
        ax.set_ylim(bottom=0)

        plt.tight_layout(pad=0.8)
        return _fig_to_b64(fig)

    # ── 9. Matplotlib: Waveform with annotations ─────────────────────────────
    def plot_waveform(self, y, sr, silence_mask, speech_mask, overlap_mask, frame_times) -> str:
        fig, ax = plt.subplots(figsize=(13, 2.0))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)

        t  = np.linspace(0, len(y) / sr, len(y))
        ax.plot(t, y, color=CYAN, linewidth=0.3, alpha=0.8)
        ax.axhline(0, color=GRID, linewidth=0.5, alpha=0.5)

        # Annotate regions
        if len(frame_times) > 1:
            dt = frame_times[1] - frame_times[0]
        else:
            dt = HOP_LENGTH / sr

        n = min(len(frame_times), len(silence_mask), len(speech_mask), len(overlap_mask))
        for i in range(n):
            t0, t1 = frame_times[i], frame_times[i] + dt
            if overlap_mask[i]:
                ax.axvspan(t0, t1, alpha=0.28, color=DANGER,  lw=0)
            elif speech_mask[i]:
                ax.axvspan(t0, t1, alpha=0.18, color=SUCCESS, lw=0)

        patches = [
            mpatches.Patch(color=SUCCESS, alpha=0.6, label="Active Speech"),
            mpatches.Patch(color=DANGER,  alpha=0.6, label="High-Intensity / Overlap"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8,
                  facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)

        ax.set_xlabel("Time (seconds)", color=CYAN, fontsize=10, labelpad=5)
        ax.set_ylabel("Amplitude", color=CYAN, fontsize=10, labelpad=5)
        ax.set_title("Audio Waveform  ·  Speech & Overlap Region Annotation",
                     color=TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlim(0, len(y) / sr)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.5)

        plt.tight_layout(pad=0.8)
        return _fig_to_b64(fig)

    # ── 10. Matplotlib: RMS Energy timeline ──────────────────────────────────
    def plot_rms(self, rms, frame_times, silence_mask, speech_mask, overlap_mask, max_rms) -> str:
        fig, ax = plt.subplots(figsize=(13, 2.0))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)

        t = frame_times[:len(rms)]
        n = min(len(t), len(silence_mask), len(speech_mask), len(overlap_mask))

        ax.fill_between(t[:n], rms[:n], where=silence_mask[:n], color=MUTED,   alpha=0.5, label="Silence")
        ax.fill_between(t[:n], rms[:n], where=speech_mask[:n],  color=CYAN,    alpha=0.5, label="Active Speech")
        ax.fill_between(t[:n], rms[:n], where=overlap_mask[:n], color=DANGER,  alpha=0.6, label="Overlap")
        ax.plot(t[:n], rms[:n], color=CYAN, linewidth=0.8, alpha=0.9)

        ax.axhline(y=max_rms * SILENCE_THRESH, color=WARNING, linestyle="--",
                   linewidth=0.9, alpha=0.9, label=f"Silence threshold ({SILENCE_THRESH*100:.0f}% peak)")
        ax.axhline(y=max_rms * OVERLAP_THRESH, color=DANGER,  linestyle="--",
                   linewidth=0.9, alpha=0.9, label=f"Overlap threshold ({OVERLAP_THRESH*100:.0f}% peak)")

        ax.set_xlabel("Time (seconds)", color=CYAN, fontsize=10, labelpad=5)
        ax.set_ylabel("RMS Energy", color=CYAN, fontsize=10, labelpad=5)
        ax.set_title("RMS Energy Timeline  ·  Speech Activity Classification",
                     color=TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(loc="upper right", fontsize=7, facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT,
                  ncol=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.5)
        ax.set_ylim(bottom=0)

        plt.tight_layout(pad=0.8)
        return _fig_to_b64(fig)

    # ── 11. Lightweight metrics (for live mode, no heavy charts) ─────────────
    def analyze_light(self, y: np.ndarray, sr: int) -> dict:
        if len(y) < FRAME_LENGTH:
            return {"error": "Audio chunk too short"}
        y_filt = self.bandpass(y, sr)
        rms, times = self.frame_features(y_filt, sr)
        silence_mask, speech_mask, overlap_mask = self.classify_frames(rms)
        n = len(rms)
        active  = round(float(np.sum(speech_mask))  / n * 100, 1)
        overlap = round(float(np.sum(overlap_mask)) / n * 100, 1)
        silence = round(float(np.sum(silence_mask)) / n * 100, 1)
        score   = self.harmony_score(active, overlap, silence)
        warnings = self.build_warnings(score, active, overlap, silence)
        # FIX: step=max(1, n//50) instead of n//100. A 3s window at 16kHz only produces
        # ~93 RMS frames (3*16000/512). n//100 → step=0 which crashes; n//50 stays safe.
        step = max(1, n // 50)
        return {
            "type":          "live_metrics",
            "harmony_score": score,
            "active_pct":    active,
            "overlap_pct":   overlap,
            "silence_pct":   silence,
            "rms_data":      rms[::step].tolist(),
            "times_data":    times[::step].tolist(),
            "speech_data":   speech_mask[::step].tolist(),
            "overlap_data":  overlap_mask[::step].tolist(),
            "warnings":      warnings,
        }

    # ── 12. Full analysis pipeline ────────────────────────────────────────────
    def analyze(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        # Stage 1 – Load
        y, sr = self.load(file_bytes)
        if len(y) < FRAME_LENGTH * 2:
            return {"error": "Audio file too short (minimum ~0.1 seconds required)"}
        duration = len(y) / sr

        # Stage 2 – Bandpass filter
        y_filt = self.bandpass(y, sr)

        # Stage 3 – Frame features
        rms, frame_times = self.frame_features(y_filt, sr)

        # Stage 4 – Classify frames
        silence_mask, speech_mask, overlap_mask = self.classify_frames(rms)
        n = len(rms)

        n_silence = int(np.sum(silence_mask))
        n_speech  = int(np.sum(speech_mask))
        n_overlap = int(np.sum(overlap_mask))

        active_pct  = round(n_speech  / n * 100, 1)
        overlap_pct = round(n_overlap / n * 100, 1)
        silence_pct = round(n_silence / n * 100, 1)

        # Stage 5 – Harmony score
        score = self.harmony_score(active_pct, overlap_pct, silence_pct)

        # Stage 6 – STFT
        S_db, freqs, stft_times = self.stft_data(y_filt, sr)

        # Stage 7 – Charts
        max_rms    = float(np.max(rms))
        spec_b64   = self.plot_spectrogram(S_db, freqs, stft_times)
        wave_b64   = self.plot_waveform(y, sr, silence_mask, speech_mask, overlap_mask, frame_times)
        rms_b64    = self.plot_rms(rms, frame_times, silence_mask, speech_mask, overlap_mask, max_rms)

        # Stage 8 – Warnings
        warnings = self.build_warnings(score, active_pct, overlap_pct, silence_pct)

        # Stage 9 – Additional metrics
        speech_starts = np.where(np.diff(speech_mask.astype(int)) == 1)[0]
        n_utterances  = int(len(speech_starts))
        avg_energy    = round(float(np.mean(rms)) * 1000, 3)
        peak_energy   = round(float(np.max(rms)) * 1000, 3)

        # Downsampled timeline for Chart.js
        step = max(1, n // 250)

        # Get display name from metadata
        stem = filename.replace(".wav", "").lower()
        meta = SAMPLE_META.get(stem, {"icon": "🎙️", "label": filename, "cat": "custom"})

        return {
            "filename":        filename,
            "display_name":    meta["label"],
            "icon":            meta["icon"],
            "category":        meta["cat"],
            "duration":        round(duration, 2),
            "sample_rate":     sr,
            "active_pct":      active_pct,
            "overlap_pct":     overlap_pct,
            "silence_pct":     silence_pct,
            "harmony_score":   score,
            "avg_energy":      avg_energy,
            "peak_energy":     peak_energy,
            "n_utterances":    n_utterances,
            "total_frames":    n,
            "spectrogram_b64": spec_b64,
            "waveform_b64":    wave_b64,
            "rms_b64":         rms_b64,
            "rms_timeline":    rms[::step].tolist(),
            "times_timeline":  frame_times[::step].tolist(),
            "speech_timeline": speech_mask[::step].tolist(),
            "overlap_timeline":overlap_mask[::step].tolist(),
            "warnings":        warnings,
        }
