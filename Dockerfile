# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r fraud && useradd -r -g fraud fraud

# Copy installed packages from builder
COPY --from=builder /root/.local /home/fraud/.local

# Make sure scripts in .local are usable
ENV PATH=/home/fraud/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Change ownership to non-root user
RUN chown -R fraud:fraud /app

# Switch to non-root user
USER fraud

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
