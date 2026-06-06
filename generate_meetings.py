import librosa
import numpy as np
import soundfile as sf
import os

# ── Run from project root: python generate_meetings.py
# ── This overwrites all 10 scenario WAVs in the audio/ folder.

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def path(name):
    return os.path.join(AUDIO_DIR, name)

# ── Load the 5 speaker source files ──────────────────────────────────────────
print("Loading speaker files...")
manager,  sr = librosa.load(path("manager.wav"),  sr=None)
engineer, _  = librosa.load(path("engineer.wav"), sr=None)
hr,       _  = librosa.load(path("hr.wav"),       sr=None)
designer, _  = librosa.load(path("designer.wav"), sr=None)
intern_,  _  = librosa.load(path("intern.wav"),   sr=None)

# Silence helpers
sil_2s  = np.zeros(int(sr * 2.0))   # 2-second gap
sil_05s = np.zeros(int(sr * 0.5))   # 0.5-second gap (natural turn pause)

# ─────────────────────────────────────────────────────────────────────────────
# PERFECT MEETING  →  target score ~78-82
# Uses only the 3 cleanest speakers (lowest internal overlap):
#   hr=8.3%  engineer=10.9%  manager=13.6%
# Short 0.5s gaps between turns = natural pauses, not dead silence.
# Avoids designer (19.1% overlap) and intern (15.1% overlap).
# ─────────────────────────────────────────────────────────────────────────────
perfect = np.concatenate([
    hr,
    sil_05s,
    engineer,
    sil_05s,
    manager,
    sil_05s,
    hr,
    sil_05s,
    engineer,
    sil_05s,
    manager,
])

# ─────────────────────────────────────────────────────────────────────────────
# GOOD MEETING  →  target score ~68-75
# All speakers, short gaps, balanced turns.
# ─────────────────────────────────────────────────────────────────────────────
good = np.concatenate([
    manager,
    sil_05s,
    engineer,
    sil_05s,
    hr,
    sil_05s,
    designer,
    sil_05s,
    intern_,
])

# ─────────────────────────────────────────────────────────────────────────────
# AVERAGE MEETING  →  target score ~55-65
# Longer gaps = more silence, slightly disengaged feel.
# ─────────────────────────────────────────────────────────────────────────────
average = np.concatenate([
    manager,
    sil_2s,
    sil_2s,
    engineer,
    sil_2s,
    designer,
])

# ─────────────────────────────────────────────────────────────────────────────
# DEAD MEETING  →  target score ~25-35
# Mostly silence with one speaker.
# ─────────────────────────────────────────────────────────────────────────────
dead = np.concatenate([
    sil_2s,
    sil_2s,
    hr,
    sil_2s,
    sil_2s,
])

# ─────────────────────────────────────────────────────────────────────────────
# DOMINANT SPEAKER  →  target score ~60-68
# One person talks most of the time.
# ─────────────────────────────────────────────────────────────────────────────
dominant = np.concatenate([
    manager,
    manager,
    manager,
    engineer,
    hr,
])

# ─────────────────────────────────────────────────────────────────────────────
# ROUNDTABLE  →  target score ~68-75
# All speakers, equal turns, no gaps.
# ─────────────────────────────────────────────────────────────────────────────
roundtable = np.concatenate([
    manager,
    engineer,
    hr,
    designer,
    intern_,
])

# ─────────────────────────────────────────────────────────────────────────────
# CHAOTIC MEETING  →  target score ~35-45
# Two speakers mixed (summed) simultaneously = overlap throughout.
# ─────────────────────────────────────────────────────────────────────────────
length_chaotic = min(len(manager), len(engineer))
chaotic = manager[:length_chaotic] + engineer[:length_chaotic]

# ─────────────────────────────────────────────────────────────────────────────
# HIGH INTERRUPTIONS  →  target score ~25-35
# Two speakers mixed with near-equal volumes = constant interruption.
# ─────────────────────────────────────────────────────────────────────────────
length_int = min(len(manager), len(designer))
interruptions = manager[:length_int] + 0.8 * designer[:length_int]

# ─────────────────────────────────────────────────────────────────────────────
# SILENT / INACTIVE  →  target score ~15-25
# One short speaker buried in silence.
# ─────────────────────────────────────────────────────────────────────────────
silent_team = np.concatenate([
    intern_,
    sil_2s,
    sil_2s,
    sil_2s,
])

# ─────────────────────────────────────────────────────────────────────────────
# EMERGENCY MEETING  →  target score ~18-25
# Three speakers simultaneously = very high energy, no breathing room.
# ─────────────────────────────────────────────────────────────────────────────
length_emg = min(len(manager), len(engineer), len(hr))
emergency = manager[:length_emg] + engineer[:length_emg] + hr[:length_emg]

# ── Write all files ───────────────────────────────────────────────────────────
files = {
    "perfect.wav":       perfect,
    "good.wav":          good,
    "average.wav":       average,
    "dead.wav":          dead,
    "dominant.wav":      dominant,
    "roundtable.wav":    roundtable,
    "chaotic.wav":       chaotic,
    "interruptions.wav": interruptions,
    "silent.wav":        silent_team,
    "emergency.wav":     emergency,
}

print("Writing scenario files...")
for name, audio in files.items():
    # Normalise to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    sf.write(path(name), audio, sr)
    print(f"  ✓ {name}")

print("\nAll meeting scenarios regenerated successfully.")
print(f"Files written to: {AUDIO_DIR}")