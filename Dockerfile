FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Ingest data before starting the server
RUN python backend_new.py

EXPOSE 8000

CMD ["uvicorn", "chatbot_new:app", "--host", "0.0.0.0", "--port", "8000"]