# Wir nutzen ein leichtes Python-Image als Basis
FROM python:3.9-slim

# Arbeitsordner im Container setzen
WORKDIR /app

# System-Tools installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- TRICK GEGEN TIMEOUTS ---
# 1. Wir installieren Torch separat und zwingen es auf CPU-Modus (nur ~100MB statt 3GB)
# --default-timeout=1000 verhindert, dass er bei schlechtem Netz sofort abbricht
RUN pip install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Jetzt kopieren wir die restlichen Requirements (ohne Torch)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Sprachdaten für TextBlob laden
RUN python -m textblob.download_corpora

# Code kopieren
COPY . .

# Environment Variable
ENV PYTHONUNBUFFERED=1

# Startbefehl (wird von docker-compose überschrieben, aber gut als Fallback)
CMD ["python", "scheduler.py"]