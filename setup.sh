#!/bin/bash
# Setup script for Super Mario Video Dataset Generation
# Run this in an environment with pip installed

set -e

echo "=========================================="
echo "Super Mario Video Dataset Generator Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✓ pip is available"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take a few minutes..."
echo ""

pip3 install --user -r requirements.txt

echo ""
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
import gym
import gym_super_mario_bros
import cv2
print('✓ PyTorch version:', torch.__version__)
print('✓ Gym version:', gym.__version__)
print('✓ OpenCV version:', cv2.__version__)
print('✓ All dependencies installed successfully!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To generate video, run:"
echo "  python3 video_generator.py --videos 1 --duration 2"
echo ""
echo "To generate a dataset of 10 videos (5 min each):"
echo "  python3 video_generator.py --videos 10 --duration 5"
echo ""
