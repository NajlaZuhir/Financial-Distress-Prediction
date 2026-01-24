# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and install dependencies first (for caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4️⃣ Copy project files (excluding things in .dockerignore)
COPY . .

# 5️⃣ Expose FastAPI port
EXPOSE 8000

# 6️⃣ Start FastAPI
CMD ["uvicorn", "src.predict_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
