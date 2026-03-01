#!/usr/bin/env python3
"""
Startup helper: trains models if artefacts are missing, then launches Streamlit.
Usage: python startup.py
"""
import os
import subprocess
import sys

REQUIRED = [
    "models/best_model.pkl",
    "models/scaler.pkl",
    "models/feature_names.json",
]

if not all(os.path.exists(p) for p in REQUIRED):
    print("🔧 Model artefacts not found — running training pipeline...")
    result = subprocess.run([sys.executable, "src/train.py"])
    if result.returncode != 0:
        print("❌ Training failed!")
        sys.exit(1)
    print("✅ Training complete!")

port = os.environ.get("PORT", "8501")
print(f"🚀 Launching Streamlit on port {port}...")
os.execvp("streamlit", [
    "streamlit", "run", "app.py",
    "--server.address=0.0.0.0",
    f"--server.port={port}",
    "--server.headless=true",
])
