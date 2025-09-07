import os
from functools import lru_cache
from typing import List


class Settings:
    """Application configuration settings"""
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:19006",  # Expo dev server
        "http://localhost:8081",  # React Native packager
        "*"  # In production, replace with specific origins
    ]
    
    # AI Configuration
    OPENAI_MODEL: str = "gpt-4o-mini"
    CLAUDE_MODEL: str = "claude-3-sonnet-20240229"
    AI_TEMPERATURE: float = 0.8
    MAX_TOKENS: int = 200
    
    # Bible API Configuration
    # We now fetch verses from jsDelivr (wldeh/bible-api) for public-domain versions.
    REQUEST_TIMEOUT: int = 30
    
    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)
    
    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()