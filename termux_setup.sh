#!/bin/bash

echo "🤖 Termux Auto-Setup for Transformative Bot (Venv Edition)"
echo "=========================================================="

echo "📦 Updating repositories..."
pkg update -y && pkg upgrade -y

echo "📦 Installing System Dependencies & Python..."
# Core build tools, python, git, and media libraries
# Note: User strictly requires Python 3.10 for compatibility.
pkg install -y tur-repo 
pkg install -y python3.10 git clang make binutils
# Build tools for compiling python packages (Numpy, Pillow, OpenCV fallback)
# Build tools for compiling python packages (Numpy, Pillow, OpenCV fallback)
pkg install -y cmake ninja rust libffi libjpeg-turbo libpng freetype libxml2 libxslt zlib openjpeg libwebp libtiff ffmpeg

echo "🛠️ Creating Virtual Environment (venv)..."

# 1. Locate Python 3.10
PY_BIN=$(command -v python3.10)
if [ -z "$PY_BIN" ]; then
    echo "⚠️ Python 3.10 not found in PATH. Checking default locations..."
    if [ -f "/data/data/com.termux/files/usr/bin/python3.10" ]; then
        PY_BIN="/data/data/com.termux/files/usr/bin/python3.10"
    else
        echo "❌ ERROR: Python 3.10 binary NOT found. Installing..."
        pkg install -y python3.10
        PY_BIN=$(command -v python3.10)
    fi
fi

if [ -z "$PY_BIN" ]; then
    echo "❌ FATAL: Could not locate 'python3.10' even after install."
    echo "   Try running: 'pkg install python3.10' manually."
    exit 1
fi

echo "   > Found Python at: $PY_BIN"

# 2. Try Standard Venv
if [ ! -d "venv" ]; then
    echo "   > Attempting standard 'venv' creation..."
    "$PY_BIN" -m venv venv
    
    if [ ! -d "venv" ]; then
        echo "   ⚠️ Standard 'venv' failed. Falling back to 'virtualenv'..."
        "$PY_BIN" -m pip install virtualenv
        "$PY_BIN" -m virtualenv venv
    fi
    
    if [ ! -d "venv" ]; then
        echo "❌ FATAL: Failed to create 'venv'. Check permissions?"
        exit 1
    fi
    echo "   └─ Created 'venv' successfully."
else
    echo "   └─ 'venv' already exists. Skipping creation."
fi

echo "🔌 Activating venv..."
source venv/bin/activate || . venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to activate venv."
    exit 1
fi

echo "📦 Upgrading pip (inside venv)..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing Python Dependencies (inside venv)..."
# Flag to force compile if wheels missing
# Flag to force compile if wheels missing
export CFLAGS="-Wno-error=incompatible-function-pointer-types -O3"
export MATHLIB="m"
export LDFLAGS="-L/data/data/com.termux/files/usr/lib/"
pip install -r requirements.txt

echo "=========================================================="
echo "✅ Setup Complete!"
echo ""
echo "❗ IMPORTANT ❗"
echo "To run the bot, you must activate the environment first:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "Or run in one line:"
echo "   ./venv/bin/python main.py"
