# ---- Base Image ----
FROM python:3.11-slim

# ---- Set Workdir ----
WORKDIR /app

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Project Files ----
COPY pyproject.toml requirements.txt* ./ 
RUN pip install --upgrade pip

# Install dependencies
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy source code
COPY . .

# ---- Expose Port ----
EXPOSE 8000

# ---- Start API ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]