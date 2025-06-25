# Build stage
FROM python:3.11-slim as builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY README.md ./
COPY LICENSE ./

# Install uv for fast package management
RUN pip install uv

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the application with all production dependencies
RUN uv pip install -e ".[text-processing,monitoring]"

# Production stage
FROM python:3.11-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r v10r && useradd -r -g v10r -d /app -s /bin/bash v10r

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./
COPY --from=builder /app/README.md ./
COPY --from=builder /app/LICENSE ./

# Create directories for configuration and logs
RUN mkdir -p /app/config /app/logs /app/data && \
    chown -R v10r:v10r /app

# Copy example configuration (can be overridden by volume mount)
COPY examples/simple-config.yaml /app/config/v10r-config.yaml

# Health check script
COPY --chmod=755 <<EOF /app/health-check.sh
#!/bin/bash
# Health check for v10r services
set -e

CONFIG_FILE=\${V10R_CONFIG_FILE:-/app/config/v10r-config.yaml}

if [ "\$V10R_WORKER_MODE" = "true" ]; then
    # Worker health check - check if process is running and Redis is accessible
    pgrep -f "v10r worker" > /dev/null || exit 1
    v10r test-connection --config "\$CONFIG_FILE" 2>/dev/null || exit 1
else
    # Listener health check - check if process is running and can connect to DB
    pgrep -f "v10r listen" > /dev/null || exit 1
    v10r test-connection --config "\$CONFIG_FILE" 2>/dev/null || exit 1
fi

echo "Health check passed"
EOF

# Switch to non-root user
USER v10r

# Set default environment variables
ENV V10R_CONFIG_FILE=/app/config/v10r-config.yaml
ENV V10R_LOG_LEVEL=INFO
ENV V10R_ENVIRONMENT=production
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose health check port (if running as web service)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /app/health-check.sh

# Default command (can be overridden by Porter)
CMD ["v10r", "listen", "--config", "/app/config/v10r-config.yaml"]

# Labels for better container management
LABEL org.opencontainers.image.title="v10r - PostgreSQL Vectorizer"
LABEL org.opencontainers.image.description="Generic PostgreSQL NOTIFY-based vectorization service"
LABEL org.opencontainers.image.vendor="v10r Team"
LABEL org.opencontainers.image.source="https://github.com/v10r/v10r"
LABEL org.opencontainers.image.documentation="https://v10r.readthedocs.io" 