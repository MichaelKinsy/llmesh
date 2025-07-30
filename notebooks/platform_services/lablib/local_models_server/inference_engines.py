# inference_engines.py - Fixed BitNetEngine class

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator
from pathlib import Path
import os

from llama_cpp import Llama
from config import settings
from model_manager import ModelInfo, ModelType, model_manager

class InferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    async def load_model(self, model_info: ModelInfo) -> None:
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    async def chat_completion(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        pass

class BitNetEngine(InferenceEngine):
    """BitNet.cpp inference engine using subprocess calls."""
    
    def __init__(self):
        self.model_info: Optional[ModelInfo] = None
        self.model_path: Optional[str] = None
        self.is_loaded: bool = False
    
    async def load_model(self, model_info: ModelInfo) -> None:
        """Load a BitNet model."""
        if self.is_loaded and self.model_info and self.model_info.name == model_info.name:
            return
        
        # Prepare the model
        self.model_path = await model_manager.prepare_bitnet_model(model_info)
        self.model_info = model_info
        self.is_loaded = True
    
    async def unload_model(self) -> None:
        """Unload the BitNet model."""
        self.model_path = None
        self.model_info = None
        self.is_loaded = False
    
    def _check_bitnet_installation(self) -> bool:
        """Check if BitNet is properly installed."""
        bitnet_path = Path(settings.bitnet_path)
        run_inference_script = bitnet_path / "run_inference.py"
        
        if not bitnet_path.exists():
            raise RuntimeError(f"BitNet directory not found: {bitnet_path}")
        
        if not run_inference_script.exists():
            raise RuntimeError(f"BitNet run_inference.py not found: {run_inference_script}")
        
        return True
    
    async def _run_inference(self, prompt: str, conversation_mode: bool = False, **kwargs) -> AsyncGenerator[str, None]:
        """Run BitNet inference as a subprocess using shell command."""
        if not self.is_loaded or not self.model_path:
            raise RuntimeError("No BitNet model loaded")
        
        # Check BitNet installation
        self._check_bitnet_installation()
        
        # Use absolute paths
        model_path_abs = os.path.abspath(self.model_path)
        working_dir = os.path.abspath(settings.bitnet_path)
        
        # Build the exact command that works in shell
        cmd_parts = [
            f"cd {working_dir}",
            "&&",
            "python3",
            f"{working_dir}/run_inference.py",
            "-m", model_path_abs,
            "-p", f"'{prompt}'",  # Quote the prompt
            "-n", str(kwargs.get('max_tokens', settings.default_max_tokens)),
            "-temp", str(kwargs.get('temperature', settings.default_temperature)),
            "-c", str(kwargs.get('context_size', settings.default_context_size))
        ]
        
        if conversation_mode:
            cmd_parts.append("-cnv")
        
        if settings.cpu_threads:
            cmd_parts.extend(["-t", str(settings.cpu_threads)])
        
        shell_cmd = " ".join(cmd_parts)
        
        print(f"DEBUG: Executing shell command: {shell_cmd}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Read output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                text = line.decode().strip()
                if text:
                    yield text
            
            await process.wait()
            
            if process.returncode != 0:
                stderr = await process.stderr.read()
                error_msg = stderr.decode().strip()
                print(f"DEBUG: Process failed with return code {process.returncode}")
                print(f"DEBUG: stderr: {error_msg}")
                raise RuntimeError(f"BitNet inference failed: {error_msg}")
                
        except Exception as e:
            print(f"DEBUG: Shell command error: {e}")
            if 'process' in locals() and process.returncode is None:
                process.terminate()
                await process.wait()
            raise RuntimeError(f"BitNet inference error: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text using BitNet."""
        async for token in self._run_inference(prompt, conversation_mode=False, **kwargs):
            yield token
    
    async def chat_completion(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Generate chat completion using BitNet."""
        prompt = self._format_chat_messages(messages)
        async for token in self._run_inference(prompt, conversation_mode=True, **kwargs):
            yield token
    
    def _format_chat_messages(self, messages: list) -> str:
        """Format chat messages for BitNet."""
        if not messages:
            return ""
        
        # Use the last user message as the prompt
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        
        return messages[-1].get("content", "") if messages else ""

class LlamaCppEngine(InferenceEngine):
    """llama.cpp inference engine using llama-cpp-python with access to official tools."""
    
    def __init__(self):
        self.model: Optional[Llama] = None
        self.model_info: Optional[ModelInfo] = None
        self.is_loaded: bool = False
        
        self.llama_quantize = "/usr/local/llama-cpp/llama-quantize"
        self.convert_hf = "/usr/local/llama-cpp/convert-hf-to-gguf.py"
    
    async def convert_model_if_needed(self, model_path: Path) -> str:
        """Convert HF model to GGUF if needed."""
        gguf_files = list(model_path.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])
        
        # Look for HF files to convert
        hf_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model.bin"))
        if not hf_files:
            raise RuntimeError(f"No model files found in {model_path}")
        
        output_path = str(model_path / "converted_model.gguf")
        
        cmd = ["python3", self.convert_hf, str(model_path), "--outfile", output_path]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Model conversion failed: {stderr.decode()}")
        
        return output_path
    
    async def load_model(self, model_info: ModelInfo) -> None:
        """Load a GGUF model with optional conversion."""
        if self.is_loaded and self.model_info and self.model_info.name == model_info.name:
            return
        
        model_path = Path(model_info.local_path)
        
        # Convert to GGUF if needed
        gguf_file = await self.convert_model_if_needed(model_path)
        
        # Load model in executor to avoid blocking
        self.model = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: Llama(
                model_path=gguf_file,
                n_ctx=settings.default_context_size,
                n_threads=settings.cpu_threads,
                verbose=False
            )
        )
        
        self.model_info = model_info
        self.is_loaded = True
    
    async def unload_model(self) -> None:
        """Unload the llama.cpp model."""
        if self.model:
            self.model = None
        self.model_info = None
        self.is_loaded = False
    
    async def generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text using llama.cpp."""
        if not self.is_loaded or not self.model:
            raise RuntimeError("No llama.cpp model loaded")
        
        # Run generation in executor
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.create_completion(
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', settings.default_max_tokens),
                temperature=kwargs.get('temperature', settings.default_temperature),
                stream=True
            )
        )
        
        for token_data in stream:
            token = token_data['choices'][0]['text']
            if token:
                yield token
    
    async def chat_completion(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Generate chat completion using llama.cpp."""
        if not self.is_loaded or not self.model:
            raise RuntimeError("No llama.cpp model loaded")
        
        # Run chat completion in executor
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.create_chat_completion(
                messages=messages,
                max_tokens=kwargs.get('max_tokens', settings.default_max_tokens),
                temperature=kwargs.get('temperature', settings.default_temperature),
                stream=True
            )
        )
        
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta and delta['content']:
                yield delta['content']

class InferenceManager:
    """Manages inference engines and model loading."""
    
    def __init__(self):
        self.engines: dict[str, InferenceEngine] = {}
        self.current_model: Optional[str] = None
        self._load_lock = asyncio.Lock()
    
    async def load_model(self, model_name: str) -> InferenceEngine:
        """Load a model and return the appropriate engine."""
        async with self._load_lock:
            model_info = model_manager.get_model(model_name)
            if not model_info:
                raise ValueError(f"Model not found: {model_name}")
            
            if not model_info.is_downloaded:
                raise ValueError(f"Model not downloaded: {model_name}")
            
            # Unload current model if different
            if self.current_model and self.current_model != model_name:
                await self.unload_current_model()
            
            # Create engine if not exists
            if model_name not in self.engines:
                if model_info.model_type == ModelType.BITNET:
                    self.engines[model_name] = BitNetEngine()
                else:
                    self.engines[model_name] = LlamaCppEngine()
            
            # Load the model
            engine = self.engines[model_name]
            await engine.load_model(model_info)
            
            self.current_model = model_name
            return engine
    
    async def get_current_engine(self) -> Optional[InferenceEngine]:
        """Get the currently loaded engine."""
        if self.current_model and self.current_model in self.engines:
            return self.engines[self.current_model]
        return None
    
    async def unload_current_model(self) -> None:
        """Unload the currently loaded model."""
        if self.current_model and self.current_model in self.engines:
            await self.engines[self.current_model].unload_model()
            self.current_model = None

# Global inference manager
inference_manager = InferenceManager()
