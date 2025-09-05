# config.py - Configuration settings
import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_pullup_model.h5")
    SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "10"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.7"))
    MIN_TRACKING_CONFIDENCE: float = float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.5"))
    
    # File upload settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = [".mp4", ".mov", ".avi", ".mkv"]
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/pullup_videos")
    
    # Performance settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    PROCESSING_TIMEOUT: int = int(os.getenv("PROCESSING_TIMEOUT", "300"))  # 5 minutes
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/pullup_api.log")
    
    # TensorFlow optimization
    TF_ENABLE_GPU: bool = os.getenv("TF_ENABLE_GPU", "true").lower() == "true"
    TF_MEMORY_GROWTH: bool = os.getenv("TF_MEMORY_GROWTH", "true").lower() == "true"
    TF_XLA_FLAGS: str = os.getenv("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices")
    
    @classmethod
    def get_tf_config(cls) -> Dict[str, Any]:
        """Get TensorFlow configuration"""
        return {
            "enable_gpu": cls.TF_ENABLE_GPU,
            "memory_growth": cls.TF_MEMORY_GROWTH,
            "inter_op_threads": 0,
            "intra_op_threads": 0,
            "allow_soft_placement": True,
            "enable_xla": True
        }