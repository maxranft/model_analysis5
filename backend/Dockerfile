FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data /app/model

ENV MEDIMG_DB_PATH=/app/data/app.db
ENV MEDIMG_MODEL_PATH=/app/model/model.onnx
ENV MEDIMG_ENABLE_MOCK_MODEL=true

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

