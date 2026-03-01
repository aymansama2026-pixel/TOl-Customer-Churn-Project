"""
RUN THIS FIRST — Auto Setup Script
Just double-click this file or run: python setup.py
"""
import os
import sys
import subprocess

print("="*50)
print("  AUTO SETUP — House Price Predictor")
print("="*50)

# Step 1 — Create all required folders
folders = ["data", "models", "outputs", "src"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✅ Folder ready: {folder}/")

# Step 2 — Install dependencies
print("\n  📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "numpy", "pandas", "scikit-learn", "matplotlib",
    "seaborn", "joblib", "streamlit", "--quiet"])
print("  ✅ Dependencies installed!")

# Step 3 — Run training
print("\n  🤖 Training models...")
exec(open("src/train.py").read())

print("\n" + "="*50)
print("  ✅ SETUP COMPLETE!")
print("="*50)
print("\n  Now run:")
print("  streamlit run app_pro.py")
print("="*50)