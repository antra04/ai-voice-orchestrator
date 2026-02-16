from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # External API URLs
    stt_api_url: str
    llm_api_url: str
    tts_api_url: str
    
    # Server config
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
