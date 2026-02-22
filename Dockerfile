FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---- install uv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ---- install cc ----
RUN curl -fsSL https://claude.ai/install.sh | bash

# ---- install cs ----
RUN curl -fsSL https://code-server.dev/install.sh | bash

# ---- git config ----
RUN git config --global init.defaultBranch main

# ---- app directory ----
WORKDIR /workspace