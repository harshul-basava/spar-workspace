FROM python:3.11-slim

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

# ---- app directory ----
WORKDIR /app

# ---- copy dependency files first (for layer caching) ----
COPY pyproject.toml uv.lock README.md ./

# ---- copy rest of repo ----
COPY . .

# ---- install python deps (after source is available) ----
RUN uv pip install --system .

# ---- default ----
CMD ["python3", "experiment001-political_persona/finetune.py"]