"""
Meeting Harmony Analyzer — Launch Script
Run from the project root: python run.py
"""
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).parent
    backend = root / "backend"

    # Validate structure
    missing = []
    for p in [backend / "main.py", backend / "audio_processor.py", root / "frontend" / "index.html"]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[ERROR] Missing files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    n_wavs = len(list((root / "audio").glob("*.wav"))) if (root / "audio").exists() else 0
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       SMART MEETING HARMONY ANALYZER  v1.0               ║
╠══════════════════════════════════════════════════════════╣
║  Project root : {str(root):<41}║
║  Audio files  : {n_wavs} WAV files loaded                        ║
║  Dashboard    : http://localhost:8000                    ║
║  API docs     : http://localhost:8000/docs               ║
╚══════════════════════════════════════════════════════════╝
""")

    # Launch uvicorn from project root so imports resolve correctly
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-dir", str(backend),
    ]
    subprocess.run(cmd, cwd=str(root))

if __name__ == "__main__":
    main()
