# Docker Compose for MIMII Anomaly Detection (example)

version: '3.8'
services:
  backend:
    build: .
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    ports:
      - "8080:8080"
    command: uvicorn api.main:app --host 0.0.0.0 --port 8080
    environment:
      - CUDA_VISIBLE_DEVICES=0
    depends_on:
      - redis
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html:ro
    depends_on:
      - backend
