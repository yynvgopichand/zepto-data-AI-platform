
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the policy documents.
COPY docs ./docs

# Copy the Python source code.
COPY src ./src

# Build the ChromaDB index.
RUN python -m src.rag

EXPOSE 7860

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]