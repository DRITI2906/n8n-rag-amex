FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    antiword \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (~180MB vs ~800MB for full torch) — no GPU needed in container
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Pre-download NLTK sentence tokeniser data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
