# Dockerfile for Sepsis Early Warning API
FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU version first (separate step)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the sepsis package with API dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]"

# Copy application code and the model bundle
COPY api/ ./api/
COPY models/production/ ./models/production/
ENV MODEL_DIR=/app/models/production

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
