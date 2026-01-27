#!/bin/bash
# Build and run the fraud detection API

set -e

echo "🔨 Building Docker image..."
docker build -t fraud-detection:latest .

echo "🚀 Starting container..."
docker run -d \
    --name fraud-detection-api \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    --health-cmd="python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\"" \
    --health-interval=30s \
    --restart=unless-stopped \
    fraud-detection:latest

echo "⏳ Waiting for container to be healthy..."
sleep 5

# Check health
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ Container is healthy!"
    echo "📊 API available at: http://localhost:8000"
    echo "📖 Docs available at: http://localhost:8000/docs"
else
    echo "❌ Container health check failed"
    docker logs fraud-detection-api
    exit 1
fi
