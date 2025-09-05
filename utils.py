# utils.py - Utility functions
import os
import tempfile
import aiofiles
from typing import Optional
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

async def save_uploaded_file(file_content: bytes, filename: str, temp_dir: str = None) -> str:
    """Save uploaded file to temporary location"""
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(temp_dir, safe_filename)
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"File saved to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

async def cleanup_temp_file(file_path: str, delay: int = 300) -> None:
    """Clean up temporary file after delay"""
    
    await asyncio.sleep(delay)
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

def validate_video_file(filename: str, content_type: str, file_size: int) -> bool:
    """Validate uploaded video file"""
    
    from config import Config
    
    # Check file extension
    file_ext = os.path.splitext(filename.lower())[1]
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        return False
    
    # Check content type
    if not content_type.startswith('video/'):
        return False
    
    # Check file size
    if file_size > Config.MAX_FILE_SIZE:
        return False
    
    return True

class VideoValidator:
    """Video file validation"""
    
    def __init__(self, max_size_mb: int = 100, allowed_formats: list = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_formats = allowed_formats or ['.mp4', '.mov', '.avi', '.mkv']
    
    def validate(self, filename: str, content_type: str, file_size: int) -> tuple[bool, str]:
        """Validate video file"""
        
        # Check file size
        if file_size > self.max_size_bytes:
            return False, f"File too large. Max size: {self.max_size_bytes // (1024*1024)}MB"
        
        # Check file extension
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.allowed_formats:
            return False, f"Invalid format. Allowed: {', '.join(self.allowed_formats)}"
        
        # Check MIME type
        if not content_type.startswith('video/'):
            return False, "File must be a video"
        
        return True, "Valid"