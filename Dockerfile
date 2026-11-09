FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY models/fraud_model.joblib ./models/fraud_model.joblib
COPY models/decision_config.joblib ./models/decision_config.joblib

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]