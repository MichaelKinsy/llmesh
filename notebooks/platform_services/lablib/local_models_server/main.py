from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from model_manager import model_manager
from inference_engines import inference_manager

# Request models
class GenerateRequest(BaseModel):
    name: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    name: str = Field(..., description="Name of the model to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    yield
    await inference_manager.unload_current_model()

app = FastAPI(
    title="BitNet & LLaMA.cpp Inference Server",
    description="High-performance inference server for BitNet and LLaMA.cpp models with dynamic model management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BitNet & LLaMA.cpp Inference Server",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    downloaded_models = model_manager.list_downloaded()
    current_engine = await inference_manager.get_current_engine()
    
    return {
        "status": "healthy",
        "models_downloaded": len(downloaded_models),
        "current_model": inference_manager.current_model,
        "engine_loaded": current_engine is not None
    }

@app.get("/models/available")
async def list_available_models():
    """List all available models from supported repositories."""
    try:
        models = await model_manager.list_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list available models: {str(e)}")

@app.get("/models")
async def list_downloaded_models():
    """List all downloaded models."""
    try:
        models = model_manager.list_downloaded()
        return {
            "models": [
                {
                    "name": model.name,
                    "repo_id": model.repo_id,
                    "type": model.model_type.value,
                    "is_downloaded": model.is_downloaded,
                    "local_path": model.local_path
                }
                for model in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/models/download")
async def download_model(repo_id: str):
    """Download a model from Hugging Face Hub."""
    try:
        model_info = await model_manager.download_model(repo_id)
        return {
            "message": f"Successfully downloaded {model_info.name}",
            "model_name": model_info.name,
            "model_type": model_info.model_type.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model."""
    try:
        # Unload if currently loaded
        if inference_manager.current_model == model_name:
            await inference_manager.unload_current_model()
        
        success = await model_manager.delete_model(model_name)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"message": f"Model {model_name} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a model into memory for inference."""
    try:
        engine = await inference_manager.load_model(model_name)
        return {
            "message": f"Model {model_name} loaded successfully",
            "engine_type": type(engine).__name__
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"DEBUG: Failed to load model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/models/unload")
async def unload_current_model():
    """Unload the currently loaded model."""
    try:
        await inference_manager.unload_current_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using the specified model."""
    try:
        engine = await inference_manager.load_model(request.name)
        
        async def stream():
            async for token in engine.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                yield f"{token}"
        
        return StreamingResponse(
            stream(), 
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """Generate a chat completion using the specified model."""
    try:
        engine = await inference_manager.load_model(request.name)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        async def stream():
            async for token in engine.chat_completion(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                yield f"{token}"
        
        return StreamingResponse(
            stream(), 
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.debug
    )
