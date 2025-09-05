# monitoring.py - Monitoring and metrics
import time
from functools import wraps
from typing import Callable
import redis.asyncio as redis
import json
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and store API metrics"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def record_request(self, endpoint: str, method: str, 
                           status_code: int, duration: float):
        """Record API request metrics"""
        
        timestamp = int(time.time())
        metrics = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration": duration,
            "timestamp": timestamp
        }
        
        try:
            # Store in Redis with TTL
            await self.redis.lpush("metrics:requests", json.dumps(metrics))
            await self.redis.expire("metrics:requests", 86400)  # 24 hours
            
            # Update counters
            counter_key = f"counter:{endpoint}:{method}"
            await self.redis.incr(counter_key)
            await self.redis.expire(counter_key, 86400)
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    async def record_processing_metrics(self, video_id: str, 
                                      processing_time: float, 
                                      frames_processed: int):
        """Record video processing metrics"""
        
        metrics = {
            "video_id": video_id,
            "processing_time": processing_time,
            "frames_processed": frames_processed,
            "fps": frames_processed / processing_time if processing_time > 0 else 0,
            "timestamp": int(time.time())
        }
        
        try:
            await self.redis.lpush("metrics:processing", json.dumps(metrics))
            await self.redis.expire("metrics:processing", 86400)
        except Exception as e:
            logger.error(f"Error recording processing metrics: {e}")
    
    async def get_metrics_summary(self) -> dict:
        """Get metrics summary"""
        
        try:
            # Get request metrics
            request_metrics = await self.redis.lrange("metrics:requests", 0, 99)
            processing_metrics = await self.redis.lrange("metrics:processing", 0, 99)
            
            # Parse and aggregate
            total_requests = len(request_metrics)
            avg_response_time = 0
            status_codes = {}
            
            if request_metrics:
                durations = []
                for metric_str in request_metrics:
                    metric = json.loads(metric_str)
                    durations.append(metric['duration'])
                    status_code = metric['status_code']
                    status_codes[status_code] = status_codes.get(status_code, 0) + 1
                
                avg_response_time = sum(durations) / len(durations)
            
            # Processing metrics
            total_videos = len(processing_metrics)
            avg_processing_time = 0
            avg_fps = 0
            
            if processing_metrics:
                processing_times = []
                fps_values = []
                for metric_str in processing_metrics:
                    metric = json.loads(metric_str)
                    processing_times.append(metric['processing_time'])
                    fps_values.append(metric['fps'])
                
                avg_processing_time = sum(processing_times) / len(processing_times)
                avg_fps = sum(fps_values) / len(fps_values)
            
            return {
                "total_requests": total_requests,
                "avg_response_time": avg_response_time,
                "status_codes": status_codes,
                "total_videos_processed": total_videos,
                "avg_processing_time": avg_processing_time,
                "avg_fps": avg_fps
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}

def timing_middleware(func: Callable) -> Callable:
    """Middleware to time function execution"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.info(f"{func.__name__} executed in {duration:.3f}s")
    
    return wrapper
