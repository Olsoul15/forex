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

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir sse-starlette>=1.6.5 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn fastapi gunicorn

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

# Copy application code and dependencies
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Create non-root user and set permissions
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /usr/local/lib/python3.11/site-packages

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

# Start the application with a command that does not exit
# This allows us to exec into the container for debugging
CMD ["tail", "-f", "/dev/null"]