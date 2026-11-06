FROM python:3.12-slim

# MediaPipe's tasks runtime links against EGL/GLES even on CPU.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MediaPipe Face Landmarker (478-point mesh). Apache-2.0, published by Google.
ARG FACE_LANDMARKER_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
RUN mkdir -p models && curl -fsSL -o models/face_landmarker.task "$FACE_LANDMARKER_URL"

COPY app ./app

ENV PYTHONUNBUFFERED=1 WEB_CONCURRENCY=2
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
