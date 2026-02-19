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

# ---- copy dependency files ----
COPY pyproject.toml uv.lock README.md ./

# ---- install python deps ----
RUN uv pip install --system .

# ---- copy rest of repo ----
COPY . .

# ---- default ----
CMD ["bash"]