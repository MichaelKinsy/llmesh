from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List, Union

class Settings(BaseSettings):
    """Application configuration."""
    
    # Core paths
    models_dir: str = Field("/app/models", env="MODELS_DIR")
    bitnet_path: str = Field("/app/BitNet", env="BITNET_PATH")
    logs_dir: str = Field("/app/logs", env="LOGS_DIR")
    
    # Core model settings  
    default_max_tokens: int = Field(256, env="DEFAULT_MAX_TOKENS")
    default_temperature: float = Field(0.7, env="DEFAULT_TEMPERATURE")
    default_context_size: int = Field(2048, env="DEFAULT_CONTEXT_SIZE")
    
    # Performance settings
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")
    
    # HuggingFace
    hf_token: Optional[str] = Field(None, env="HF_TOKEN")
    
    # Hardware
    cpu_threads: Optional[int] = Field(None, env="CPU_THREADS")
    debug: bool = Field(False, env="DEBUG")
    
    # Supported models - key is repo_id, value is local directory name
    supported_bitnet_models: dict = {
        "microsoft/BitNet-b1.58-2B-4T-gguf": "BitNet-b1.58-2B-4T",
        "1bitLLM/bitnet_b1_58-3B": "bitnet_b1_58-3B"
    }
    
    @field_validator('cpu_threads', mode='before')
    @classmethod
    def parse_cpu_threads(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convert empty strings to None for cpu_threads."""
        if v is None or v == '' or v == 'None':
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return v
    
    @field_validator('hf_token', mode='before')
    @classmethod
    def parse_hf_token(cls, v: Union[str, None]) -> Optional[str]:
        """Convert empty strings to None for hf_token."""
        if v is None or v == '' or v == 'None':
            return None
        return v
    
    class Config:
        env_file = ".env"

settings = Settings()
