#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_orbslam3_ubuntu.sh [target_dir]
# Example:
#   bash setup_orbslam3_ubuntu.sh ~/slam

TARGET_DIR="${1:-$HOME/slam}"
REPO_DIR="$TARGET_DIR/ORB_SLAM3"

echo "[1/5] Installing Ubuntu dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  libeigen3-dev libopencv-dev \
  libglew-dev libboost-all-dev \
  libgtk-3-dev libtbb-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libpangolin-dev

echo "[2/5] Preparing target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "[3/5] Cloning ORB_SLAM3 (if needed)..."
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$REPO_DIR"
else
  echo "  ORB_SLAM3 already exists, skipping clone."
fi

echo "[4/5] Building ORB_SLAM3..."
cd "$REPO_DIR"
chmod +x build.sh
./build.sh

echo "[5/5] Done."
echo "ORB_SLAM3 root: $REPO_DIR"
echo "Expected binary: $REPO_DIR/Examples/Monocular/mono_tum (or build/Examples/Monocular/mono_tum)"
echo "Expected vocab : $REPO_DIR/Vocabulary/ORBvoc.txt"
