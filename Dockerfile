# Multi-stage Dockerfile for 金融智能客服
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Use Chinese PyPI mirror for faster downloads
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
# Increase pip timeout for slow networks
ENV PIP_DEFAULT_TIMEOUT=300

# Use Chinese apt mirror for faster downloads
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true

# Install system dependencies with build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools FIRST (separate layer for caching)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file for better caching
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Use Chinese apt mirror for faster downloads
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true

# Install runtime dependencies (including netcat for health checks)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    libpq5 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/app && \
    chown -R appuser:appuser /home/appuser

# Set working directory
WORKDIR /home/appuser/app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy entrypoint script
COPY --chown=appuser:appuser scripts/entrypoint.sh /home/appuser/app/scripts/entrypoint.sh

# Make entrypoint executable
USER root
RUN chmod +x /home/appuser/app/scripts/entrypoint.sh
USER appuser

# Run application via entrypoint
CMD ["/home/appuser/app/scripts/entrypoint.sh"]
