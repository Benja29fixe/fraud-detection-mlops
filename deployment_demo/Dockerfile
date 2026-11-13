
FROM python:3.11-slim

WORKDIR /app

COPY deployment_demo/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment_demo/app.py ./deployment_demo/app.py

COPY models/synthetic/demo_model.joblib \
     ./models/synthetic/demo_model.joblib

COPY models/synthetic/demo_metrics.json \
     ./models/synthetic/demo_metrics.json

EXPOSE 8000

CMD ["uvicorn", "deployment_demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
