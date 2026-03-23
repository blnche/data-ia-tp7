FROM python:3.11-slim

# Evite les prompts interactifs pendant l'install
ENV DEBIAN_FRONTEND=noninteractive

# Dépendances système minimales pour PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Copier et installer les dépendances Python d'abord (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY app.py .
COPY SANOFI-Integrated-Annual-Report-2022-EN.pdf .

# Port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancement
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
