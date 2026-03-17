FROM python:3.11-slim

# System deps (Pillow native libs + optional ffmpeg for webm support)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libjpeg-dev \
        zlib1g-dev \
        libwebp-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: uncomment for native Lottie/TGS rendering
# RUN pip install --no-cache-dir rlottie-python==1.3.4

COPY bot.py .
COPY .env .

# Persistent volumes
VOLUME ["/app/fonts_cache", "/app/bot_data.db"]

CMD ["python", "bot.py"]
