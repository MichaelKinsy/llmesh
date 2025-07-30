import asyncio
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from huggingface_hub import snapshot_download, list_repo_files
from config import settings

class ModelType(Enum):
    BITNET = "bitnet"
    GGUF = "gguf"

@dataclass
class ModelInfo:
    name: str
    repo_id: str
    model_type: ModelType
    local_path: Optional[str] = None
    is_downloaded: bool = False

class ModelManager:
    """Manages model downloading and information."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)
        self._scan_existing_models()
    
    def _scan_existing_models(self):
        """Scan models directory for existing downloaded models."""
        models_path = Path(settings.models_dir)
        if not models_path.exists():
            return
            
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                # Check if it contains model files
                model_files = (
                    list(model_dir.glob("*.gguf")) + 
                    list(model_dir.glob("*.safetensors")) + 
                    list(model_dir.glob("pytorch_model.bin"))
                )
                if model_files:
                    model_type = self._detect_model_type(model_dir.name)
                    self.models[model_dir.name] = ModelInfo(
                        name=model_dir.name,
                        repo_id=f"local/{model_dir.name}",
                        model_type=model_type,
                        local_path=str(model_dir),
                        is_downloaded=True
                    )
    
    def _detect_model_type(self, identifier: str) -> ModelType:
        """Detect model type based on repository ID or name."""
        bitnet_indicators = ["bitnet", "1bit", "1.58bit"]
        if any(indicator in identifier.lower() for indicator in bitnet_indicators):
            return ModelType.BITNET
        return ModelType.GGUF
    
    def _get_local_model_name(self, repo_id: str) -> str:
        """Get the correct local directory name for a repo_id."""
        print(f"DEBUG: Getting local model name for repo_id: {repo_id}")
        print(f"DEBUG: Supported models: {settings.supported_bitnet_models}")
        if repo_id in settings.supported_bitnet_models:
            return settings.supported_bitnet_models[repo_id]
        # Fallback to last part of repo_id
        return repo_id.split('/')[-1]
    
    async def list_available_models(self) -> List[dict]:
        """List available models from supported repositories."""
        available = []
        for repo_id in settings.supported_bitnet_models.keys():
            try:
                files = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda r=repo_id: list_repo_files(r, token=settings.hf_token)
                )
                model_files = [f for f in files if f.endswith(('.gguf', '.safetensors'))]
                model_name = self._get_local_model_name(repo_id)
                
                available.append({
                    "repo_id": repo_id,
                    "name": model_name,
                    "type": self._detect_model_type(repo_id).value,
                    "model_files": model_files,
                    "is_downloaded": model_name in self.models and self.models[model_name].is_downloaded
                })
            except Exception:
                continue
        
        return available
    
    async def download_model(self, repo_id: str) -> ModelInfo:
        """Download model from Hugging Face Hub."""
        model_name = self._get_local_model_name(repo_id)
        
        # Check if already downloaded
        if model_name in self.models and self.models[model_name].is_downloaded:
            return self.models[model_name]
        
        model_type = self._detect_model_type(repo_id)
        local_path = Path(settings.models_dir) / model_name
        
        try:
            # Download in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_path),
                    token=settings.hf_token
                )
            )
            
            model_info = ModelInfo(
                name=model_name,
                repo_id=repo_id,
                model_type=model_type,
                local_path=str(local_path),
                is_downloaded=True
            )
            
            self.models[model_name] = model_info
            return model_info
        
        except Exception as e:
            # Cleanup partial download
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
            raise RuntimeError(f"Failed to download {repo_id}: {str(e)}")
    
    async def prepare_bitnet_model(self, model_info: ModelInfo) -> str:
        """Prepare BitNet model for inference."""
        if not model_info.local_path or not Path(model_info.local_path).exists():
            raise ValueError(f"Model not downloaded: {model_info.name}")
        
        model_path = Path(model_info.local_path)
        
        # Check if the GGUF model file exists
        gguf_files = list(model_path.glob("ggml-model-i2_s.gguf"))
        if not gguf_files:
            raise RuntimeError(f"No ggml-model-i2_s.gguf file found in {model_path}")
        
        gguf_file_path = str(gguf_files[0])
        
        # Create a marker file to track if setup has been run for this specific model
        setup_marker = Path(settings.bitnet_path) / f".setup_done_{model_info.name}"
        bitnet_binary = Path(settings.bitnet_path) / "build" / "bin" / "llama-cli"
        
        if setup_marker.exists() and bitnet_binary.exists():
            print(f"DEBUG: BitNet already set up for model {model_info.name}")
            return gguf_file_path
        
        print(f"DEBUG: Setting up BitNet engine for model {model_info.name}...")
        print(f"DEBUG: Model path: {model_path}")
        print(f"DEBUG: GGUF file: {gguf_file_path}")
        
        # Calculate relative path from BitNet directory to model directory
        bitnet_path = Path(settings.bitnet_path)
        try:
            # Get relative path from BitNet directory to model directory
            relative_model_path = model_path.relative_to(bitnet_path)
            relative_model_path_str = str(relative_model_path)
        except ValueError:
            # If we can't get a relative path, use absolute path
            relative_model_path_str = str(model_path)
        
        print(f"DEBUG: Relative model path: {relative_model_path_str}")
        
        # Setup BitNet environment - run from BitNet directory with relative path
        setup_cmd = [
            "python3", "setup_env.py",  # Use relative script name
            "-md", relative_model_path_str,  # Use relative model path
            "-q", "i2_s"
        ]
        
        print(f"DEBUG: Running setup command from {settings.bitnet_path}: {' '.join(setup_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *setup_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=settings.bitnet_path  # This is the key - run from BitNet directory
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"DEBUG: Setup failed. STDOUT: {stdout.decode()}")
            print(f"DEBUG: Setup failed. STDERR: {stderr.decode()}")
            raise RuntimeError(f"BitNet setup failed: {stderr.decode()}")
        
        print(f"DEBUG: Setup completed successfully")
        print(f"DEBUG: Setup STDOUT: {stdout.decode()}")
        
        # Verify the binary was created
        if not bitnet_binary.exists():
            raise RuntimeError(f"BitNet binary not found after setup: {bitnet_binary}")
        
        # Create marker file to avoid re-running setup
        setup_marker.touch()
        print(f"DEBUG: Created setup marker: {setup_marker}")
        
        print(f"DEBUG: BitNet engine ready: {bitnet_binary}")
        return gguf_file_path
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self.models.get(model_name)
    
    def list_downloaded(self) -> List[ModelInfo]:
        """List all downloaded models."""
        return [m for m in self.models.values() if m.is_downloaded]
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model."""
        if model_name not in self.models:
            return False
        
        model_info = self.models[model_name]
        
        # Delete local files
        if model_info.local_path and Path(model_info.local_path).exists():
            shutil.rmtree(model_info.local_path, ignore_errors=True)
        
        # Remove from registry
        del self.models[model_name]
        return True

# Global model manager instance
model_manager = ModelManager()
