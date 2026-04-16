# Dockerfile for Sepsis Early Warning API
FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU version first (separate step)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    shap \
    numpy \
    pandas \
    scikit-learn \
    python-dotenv \
    pydantic

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]