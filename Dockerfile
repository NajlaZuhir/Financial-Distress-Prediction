# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies first (for caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY main.py .

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]