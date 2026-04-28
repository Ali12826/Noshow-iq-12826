FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
RUN useradd -m mluser
WORKDIR /home/mluser/app
COPY --from=builder /install /usr/local
COPY --chown=mluser:mluser noshow_iq/ ./noshow_iq/
COPY --chown=mluser:mluser model.joblib .
ENV PYTHONPATH="/home/mluser/app"
USER mluser
CMD ["python", "-m", "uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "8000"]
