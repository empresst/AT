FROM python:3.12-slim

# Install system dependencies for blis and faiss
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libopenblas-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway assigns PORT dynamically)
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]