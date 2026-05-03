#!/bin/bash
# =============================================================
# setup.sh — One-command local setup for Movie Recommender
# =============================================================
# Usage:   bash setup.sh
# =============================================================

echo "============================================"
echo "  Movie Recommendation System — Setup"
echo "============================================"

# Step 1: Create virtual environment
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
echo "      Done: venv/"

# Step 2: Activate and install dependencies
echo ""
echo "[2/3] Installing dependencies..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "      Done: all packages installed"

# Step 3: Run the app
echo ""
echo "[3/3] Launching Streamlit app..."
echo ""
echo "  → Open your browser at: http://localhost:8501"
echo ""
streamlit run app.py