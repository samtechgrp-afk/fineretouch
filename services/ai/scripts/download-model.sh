#!/usr/bin/env sh
# Downloads the models for local (non-Docker) development.
#   face_landmarker.task      MediaPipe Face Landmarker (478-point mesh), Apache-2.0, Google. Required for face tools.
#   selfie_multiclass.tflite  MediaPipe selfie multiclass segmenter (person/hair), Apache-2.0, Google. Optional:
#                             subject mask for bokeh and hair protection for stray-hair cleanup.
#   midas_small.onnx          MiDaS v2.1 small monocular depth, MIT, Intel ISL. Optional: full depth for bokeh.
# Set SKIP_OPTIONAL_MODELS=1 to fetch only the face model.
set -eu
cd "$(dirname "$0")/.."
mkdir -p models
curl -fsSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "Saved models/face_landmarker.task"
if [ "${SKIP_OPTIONAL_MODELS:-0}" != "1" ]; then
  curl -fsSL -o models/selfie_multiclass.tflite \
    https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
  echo "Saved models/selfie_multiclass.tflite"
  curl -fsSL -o models/midas_small.onnx https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx
  echo "Saved models/midas_small.onnx"
fi
