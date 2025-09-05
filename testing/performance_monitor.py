# performance_monitor.py - Performance monitoring utilities
import psutil
import asyncio
from datetime import datetime, timedelta

class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.monitoring = False
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        while self.monitoring:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def collect_system_metrics(self):
        """Collect system performance metrics"""
        
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (if available)
            gpu_metrics = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    gpu_metrics = {
                        "utilization": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
            except ImportError:
                pass
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "gpu": gpu_metrics
            }
            
            # Store in Redis
            await self.redis.lpush("metrics:system", json.dumps(metrics))
            await self.redis.ltrim("metrics:system", 0, 1439)  # Keep last 24 hours
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
    
    async def get_performance_summary(self) -> dict:
        """Get performance summary"""
        
        try:
            # Get recent system metrics
            recent_metrics = await self.redis.lrange("metrics:system", 0, 59)  # Last hour
            
            if not recent_metrics:
                return {"status": "no_data"}
            
            # Parse and average
            cpu_values = []
            memory_values = []
            
            for metric_str in recent_metrics:
                metric = json.loads(metric_str)
                cpu_values.append(metric['cpu_percent'])
                memory_values.append(metric['memory_percent'])
            
            avg_cpu = sum(cpu_values) / len(cpu_values)
            avg_memory = sum(memory_values) / len(memory_values)
            max_cpu = max(cpu_values)
            max_memory = max(memory_values)
            
            return {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "max_cpu_percent": max_cpu,
                "max_memory_percent": max_memory,
                "samples": len(recent_metrics),
                "period": "last_hour"
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}