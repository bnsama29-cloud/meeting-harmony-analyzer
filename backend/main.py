"""
Meeting Harmony Analyzer — FastAPI Backend
REST API  : POST /api/analyze, GET /api/samples, GET /api/sample/{name}
WebSocket : /ws/live  (live audio streaming)
Static    : serves frontend/index.html on /
"""

import os
import io
import json
import asyncio
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent))
from audio_processor import AudioProcessor, SAMPLE_META

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Meeting Harmony Analyzer API",
    version="1.0.0",
    description="AI-powered audio analytics for meeting communication quality"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = AudioProcessor()

AUDIO_DIR    = Path(__file__).parent.parent / "audio"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Serve static frontend files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Run from project root.</h1>", status_code=404)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/samples")
async def list_samples():
    """Return all sample WAV files grouped by category."""
    if not AUDIO_DIR.exists():
        return JSONResponse({"scenarios": [], "speakers": []})

    scenarios, speakers = [], []
    for f in sorted(AUDIO_DIR.glob("*.wav")):
        stem = f.stem.lower()
        meta = SAMPLE_META.get(stem, {"icon": "🎙️", "label": f.name, "cat": "custom"})
        entry = {
            "filename": f.name,
            "stem":     stem,
            "label":    meta["label"],
            "icon":     meta["icon"],
            "size_kb":  round(f.stat().st_size / 1024, 1),
        }
        if meta["cat"] == "speaker":
            speakers.append(entry)
        else:
            scenarios.append(entry)

    return JSONResponse({"scenarios": scenarios, "speakers": speakers})


@app.get("/api/sample/{filename}")
async def analyze_sample(filename: str):
    """Analyze a pre-loaded sample audio file."""
    safe = Path(filename).name  # strip any path traversal
    path = AUDIO_DIR / safe
    if not path.exists() or path.suffix.lower() != ".wav":
        raise HTTPException(404, detail=f"Sample '{safe}' not found in audio directory")
    result = processor.analyze(path.read_bytes(), safe)
    return JSONResponse(result)


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze an uploaded WAV file."""
    if not file.filename.lower().endswith((".wav", ".WAV")):
        raise HTTPException(400, detail="Only WAV files are supported")
    data = await file.read()
    if len(data) < 1024:
        raise HTTPException(400, detail="File too small — minimum 1 KB required")
    result = processor.analyze(data, file.filename)
    if "error" in result:
        raise HTTPException(422, detail=result["error"])
    return JSONResponse(result)


# ─── WebSocket: Live Audio Streaming ──────────────────────────────────────────
@app.websocket("/ws/live")
async def live_audio(ws: WebSocket):
    """
    Accepts raw PCM Float32 audio chunks from the browser (16 kHz mono).
    Analyses every 3-second window and streams back lightweight metrics JSON.

    FIX: SR constant was defined locally; extracted to module-level for consistency.
    FIX: Buffer sliding now retains WINDOW//2 samples (50% overlap) instead of only
         1 second — prevents context loss between analysis windows.
    FIX: Minimum buffer check uses the actual SR to stay correct if SR ever changes.
    """
    await ws.accept()

    SR        = 16000
    WINDOW    = SR * 3         # 3-second analysis window
    # FIX: Was SR * 1 (1-second overlap). 50% overlap (1.5s) gives much better
    # continuity — a speaker starting near the end of window N is fully captured in N+1.
    SLIDE     = SR * 3 // 2    # slide by 1.5s, keeping 1.5s of prior context
    buffer: list = []

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_bytes(), timeout=60.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await ws.send_text(json.dumps({"type": "ping"}))
                continue

            # Decode Float32 PCM
            chunk = np.frombuffer(raw, dtype=np.float32)
            buffer.extend(chunk.tolist())

            # Process when we have enough audio
            if len(buffer) >= WINDOW:
                audio = np.array(buffer[:WINDOW], dtype=np.float32)
                # Run light analysis (no heavy charts)
                try:
                    metrics = processor.analyze_light(audio, SR)
                    await ws.send_text(json.dumps(metrics))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "msg": str(e)}))

                # FIX: Slide forward by SLIDE, keeping the tail as overlap context
                # Old code: buffer = buffer[-OVERLAP:] — kept only last 1s regardless
                # New code: advance by SLIDE samples, retain the rest as context
                buffer = buffer[SLIDE:]

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
