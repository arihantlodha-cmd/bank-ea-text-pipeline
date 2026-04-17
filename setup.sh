#!/bin/bash
# setup.sh — One-time setup for the Bank EA Text Pipeline
# Run this once before using the pipeline for the first time.

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Bank EA Pipeline — One-Time Setup                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# --- Python packages ---
echo "▶ Installing Python dependencies..."
pip3 install -r "$(dirname "$0")/requirements.txt" -q
echo "  ✓ Python packages installed"
echo ""

# --- Tesseract (optional, for OCR on scanned PDFs) ---
echo "▶ Checking for Tesseract OCR..."
if command -v tesseract &>/dev/null; then
    echo "  ✓ Tesseract already installed: $(tesseract --version 2>&1 | head -1)"
else
    echo "  Tesseract not found. Attempting to install via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install tesseract
        echo "  ✓ Tesseract installed"
    else
        echo "  ⚠ Homebrew not found. Tesseract OCR will be skipped."
        echo "    (Most documents extract fine without it.)"
        echo "    To install manually: https://brew.sh → then: brew install tesseract"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup complete! To run the pipeline:                        ║"
echo "║                                                              ║"
echo "║    python3 launch.py                                         ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
