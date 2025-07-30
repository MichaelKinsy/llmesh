FROM ghcr.io/ggml-org/llama.cpp:full AS llama-cpp
FROM ubuntu:22.04

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake clang git python3 python3-pip wget curl \
    software-properties-common gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM 18 for BitNet.cpp
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && bash llvm.sh 18 && rm llvm.sh

# Copy ALL llama.cpp tools from official FULL image
COPY --from=llama-cpp /app /usr/local/llama-cpp/
ENV PATH="/usr/local/llama-cpp:${PATH}"

# Set working directory
WORKDIR /app

# Clone and build BitNet.cpp
RUN git clone --recursive https://github.com/microsoft/BitNet.git /app/BitNet
WORKDIR /app/BitNet
RUN uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements.txt

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/logs

# Copy application code
COPY . /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
