#!/usr/bin/env bash
set -euo pipefail

sudo apt-get install -y \
    libxrender1 \
    libxi6 \
    libxkbcommon0 \
    libxxf86vm1 \
    libxfixes3 \
    libsm6 \
    libice6 \
    libgl1 \
    libglu1-mesa \
    libegl1 \
    libxext6 \
    libx11-6 \
    libxcb1
