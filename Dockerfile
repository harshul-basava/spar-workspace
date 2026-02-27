FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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
RUN git config --global init.defaultBranch main \
    && git config --global user.name "harshul-basava" \
    && git config --global user.email "harshulbasava7@gmail.com"

# ---- code-server port ----
EXPOSE 8080

# ---- app directory ----
WORKDIR /workspaces