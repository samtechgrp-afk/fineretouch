#!/usr/bin/env sh
# Downloads the MediaPipe Face Landmarker model for local (non-Docker) development.
set -eu
cd "$(dirname "$0")/.."
mkdir -p models
curl -fsSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "Saved models/face_landmarker.task"
