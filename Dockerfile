# =============================================================================
# Myanmar Spell Checker - Production Dockerfile
# Multi-stage build with security hardening
# =============================================================================

# Build arguments
ARG PYTHON_VERSION=3.11
ARG APP_ENV=production

# =============================================================================
# Stage 1: Builder - Compile Cython extensions and install dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies (including OpenMP for parallel processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip install --upgrade pip setuptools wheel Cython>=3.0.0

# Copy dependency files and source code
COPY pyproject.toml ./
COPY setup.py ./
COPY src/ ./src/

# Install package with Cython extensions
# Note: pyyaml is now a core dependency (no need for .[config])
RUN pip install --prefix=/install .

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="Myanmar Spell Checker" \
      org.opencontainers.image.description="High-performance Myanmar spell checker with syllable-first architecture" \
      org.opencontainers.image.source="https://github.com/thettwe/my-spellchecker" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=appuser \
    APP_GROUP=appgroup

WORKDIR ${APP_HOME}

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_GROUP} && \
    useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

# Install runtime dependencies (including libgomp for OpenMP parallel processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder (includes compiled Cython extensions,
# .py source, YAML rules, schemas, and all package data)
COPY --from=builder /install /usr/local

# Create data directory for database files (to be mounted as volume)
RUN mkdir -p ${APP_HOME}/data && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# Switch to non-root user
USER ${APP_USER}

# Default command (can be overridden)
CMD ["python", "-m", "myspellchecker", "--help"]

# =============================================================================
# Stage 3: Development image (optional, for docker-compose)
# =============================================================================
FROM runtime AS development

USER root

# Install development dependencies (testing + linting, excludes heavy training deps)
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-benchmark>=4.0.0 \
    pytest-xdist>=3.0.0 \
    pytest-timeout>=2.2.0 \
    hypothesis>=6.0.0 \
    ruff>=0.3.0 \
    mypy>=1.0.0

# Copy test files
COPY --chown=${APP_USER}:${APP_GROUP} tests/ ./tests/

# Switch back to non-root user
USER ${APP_USER}

# Override command for development
CMD ["python", "-m", "myspellchecker", "--help"]

# =============================================================================
# Stage 4: CLI-only image (minimal, no web server)
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS cli

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    APP_USER=appuser

WORKDIR ${APP_HOME}

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home ${APP_USER}

# Install runtime dependencies (including libgomp for OpenMP parallel processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create data directory
RUN mkdir -p ${APP_HOME}/data && \
    chown -R ${APP_USER}:appgroup ${APP_HOME}

USER ${APP_USER}

# Default to CLI help
ENTRYPOINT ["python", "-m", "myspellchecker"]
CMD ["--help"]
