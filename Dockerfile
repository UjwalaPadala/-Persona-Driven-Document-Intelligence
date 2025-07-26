FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

COPY app/ ./app/
WORKDIR /app/app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]
