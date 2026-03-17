FROM python:3.11-slim

# System deps for Pillow
RUN apt-get update && apt-get install -y \
    libwebp-dev \
    libjpeg-dev \
    libpng-dev \
    libgif-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY .env .

# Persist fonts and DB
VOLUME ["/app/fonts", "/app/users.db"]

CMD ["python", "main.py"]
