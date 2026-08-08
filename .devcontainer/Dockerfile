# Standardized SOTA Dockerfile for house-prices-kaggle
# Pre-packaged with Python 3.12 and Astral uv for ultra-fast, reproducible package installs
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Avoid prompts, disable pyc generation, and ensure output is printed instantly
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Set the path to auto-resolve the virtual environment interpreter
    PATH="/workspaces/.venv/bin:${PATH}" \
    # Silence cross-filesystem warnings between mounted volumes and the container cache
    UV_LINK_MODE=copy \
    # Precompile python files to bytecode for faster startup times on import
    UV_COMPILE_BYTECODE=1 \
    # Put the virtual environment at a stable absolute path VS Code can easily reference
    UV_PROJECT_ENVIRONMENT=/workspaces/.venv

# Install minimal OS-level dependencies (Adding glpk and cbc)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    glpk-utils \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Work from the workspaces mount directory (default for Codespaces and VS Code Dev Containers)
WORKDIR /workspaces
