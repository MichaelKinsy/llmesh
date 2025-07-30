# Local Models Server

A simple FastAPI server to wrap inference engines for BitNet (1-bit), and LLaMA.cpp (GGUF) models with other management functionality via API requests.

## Features

- **Dual Engine Support**: BitNet.cpp for 1-bit models and llama.cpp for standard GGUF models
- **Dynamic Model Management**: Download models on-demand from Hugging Face Hub
- **Streaming Inference**: Real-time token streaming for responsive UX
- **Official Optimizations**: Uses official Docker images and tools for maximum performance
- **Memory Efficient**: Smart model loading/unloading to manage memory usage
- **Container Ready**: Docker and Docker Compose support

## API Endpoints

### Model Management

- `GET /models/available` - List available models from Hugging Face
- `GET /models` - List downloaded models
- `POST /models/download?repo_id={repo_id}` - Download a model
- `POST /models/{model_name}/load` - Load model into memory
- `POST /models/unload` - Unload current model
- `DELETE /models/{model_name}` - Delete downloaded model

### Inference

- `POST /generate` - Generate text (streaming)
- `POST /chat` - Chat completion (streaming)

### System

- `GET /health` - Health check
- `GET /docs` - API documentation

## Supported Models

### BitNet Models (1-bit)

- `microsoft/BitNet-b1.58-2B-4T-gguf` - Official 2B parameter model
- `1bitLLM/bitnet_b1_58-3B` - Community 3B model

### Standard Models (GGUF)

Any GGUF-compatible model from Hugging Face can be used with the llama.cpp engine.

## Configuration

Environment variables (optional):

```bash
# Paths
MODELS_DIR=/app/models          # Model storage directory
LOGS_DIR=/app/logs             # Log directory

# Model defaults
DEFAULT_MAX_TOKENS=256         # Default max tokens
DEFAULT_TEMPERATURE=0.7        # Default temperature
DEFAULT_CONTEXT_SIZE=2048      # Default context window

# Performance
CPU_THREADS=8                  # CPU threads for inference
REQUEST_TIMEOUT=300            # Request timeout in seconds

# Hugging Face
HF_TOKEN=your_token_here       # For private models

# Debug
DEBUG=false                    # Enable debug mode
```
