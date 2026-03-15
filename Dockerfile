FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install FFmpeg for voice playback and build deps for PyNaCl
RUN apt-get update && apt-get install -y ffmpeg libffi-dev libsodium-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "llmcord.py"]
