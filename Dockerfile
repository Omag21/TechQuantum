# TechQuantum — crop detection interface (8-qubit quantum kernel, exact simulation).
#   docker build -t techquantum-app .
#   docker run --rm -p 8501:8501 techquantum-app      ->  http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt requirements-app.txt ./
RUN pip install -r requirements-app.txt

# Only what the app needs at runtime. The data loader checks SHA-256 hashes,
# so data/ must keep LF line endings (see .dockerignore and the README note).
COPY challenge/ challenge/
COPY configs/ configs/
COPY data/ data/
COPY app/ app/

# Fail the build early if the data hashes do not match (e.g. CRLF conversion on Windows).
RUN python -c "from challenge.data import load_data; load_data(); print('data checks passed')"

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
