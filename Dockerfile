# Gunakan base image python resmi
FROM python:3.11-slim

# Gunakan libgl1 sebagai pengganti libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sisa file proyek
COPY . .

# Jalankan aplikasi (sesuaikan dengan command di Procfile Anda)
CMD ["python", "app.py"]