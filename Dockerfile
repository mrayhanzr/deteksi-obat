FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deps.txt .
RUN pip install --no-cache-dir -r deps.txt
COPY . .

# BARIS INI ADALAH KUNCI UTAMA:
# Kita paksa jalankan gunicorn langsung di sini.
# $PORT adalah variabel dari Railway, gunicorn akan otomatis membacanya.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "120"]