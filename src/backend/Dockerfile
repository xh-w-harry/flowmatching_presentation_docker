FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ROOT=/app
ENV MODEL_PATH=/app/models/best_fm_multimodal.pth
ENV RUN_ROOT=/app/runs

WORKDIR /app/src/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6052
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6052"]
