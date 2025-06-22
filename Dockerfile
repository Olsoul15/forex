# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpq-dev \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib
RUN curl -L -o /tmp/ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && cd /tmp \
    && tar -xvf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn fastapi gunicorn sse-starlette>=1.6.5

# Copy application code
COPY . .

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000
ENV ENVIRONMENT=production
ENV DEBUG=false
ENV LOG_LEVEL=INFO
ENV ENABLE_DOCS=true

# Set default values for Supabase environment variables
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""
ENV SUPABASE_SERVICE_KEY=""

WORKDIR /app

# Copy ta-lib files from builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib

# Install minimal runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn fastapi gunicorn sse-starlette>=1.6.5

# Copy Python packages and application code from builder
COPY --from=builder /app /app

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

# Create logs directory with proper permissions
RUN mkdir -p /app/forex_ai/logs && \
    chown -R appuser:appuser /app/forex_ai/logs && \
    chmod 777 /app/forex_ai/logs

# Switch to non-root user
USER appuser

# Create a healthcheck script
COPY --chown=appuser:appuser healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py

# Expose port
EXPOSE 8000

# Start the application with gunicorn
CMD exec gunicorn forex_ai.api.main:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --graceful-timeout 300 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --preload