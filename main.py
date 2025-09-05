import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

import cv2 as cv
import numpy as np
import redis.asyncio as redis
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles

from objdetect_async import AsyncPullUpDetector
from rnn_optimized import OptimizedPullUpRNN

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI initialization
app = FastAPI(
    title="Pull-Up Detection API",
    description="Real-time pull-up detection and form analysis using computer vision and ML",
    version="1.0.0"
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
redis_client = None
detector = None

# API responses
class AnalysisResult(BaseModel):
    video_id: str
    total_reps: int
    phases_detected: List[int]
    form_score: float
    timestamp: datetime
    processing_time: float
    detailed_analysis: Dict[str, Any]

class AnalysisStatus(BaseModel):
    video_id: str
    status: str  # "processing", "completed", "failed"
    progress: float
    message: Optional[str] = None

class LiveStreamConfig(BaseModel):
    stream_id: str
    confidence_threshold: float = 0.7
    phase_smoothing: bool = True

# Redis config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hour cache TTL

# TF settings
TF_OPTIMIZATION_FLAGS = {
    "inter_op_parallelism_threads": 0,
    "intra_op_parallelism_threads": 0,
    "allow_soft_placement": True
}

async def get_redis():
    """Dependency to get Redis client"""
    return redis_client

def optimize_tensorflow():
    """Configure TensorFlow for optimal performance"""
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    
    # Set threading options
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    
    # Enable XLA compilation for better performance
    tf.config.optimizer.set_jit(True)

@app.on_event("startup")
async def startup_event():
    """Initialize Redis, TensorFlow model, and detector on startup"""
    global redis_client, model, detector
    
    try:
        # Initialize Redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established")
        
        # Optimize TensorFlow
        optimize_tensorflow()
        
        # Load optimized model
        model = OptimizedPullUpRNN.load_optimized_model('best_pullup_model.h5')
        logger.info("TensorFlow model loaded and optimized")
        
        # Initialize async detector
        detector = AsyncPullUpDetector()
        logger.info("Pull-up detector initialized")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global redis_client
    if redis_client:
        await redis_client.close()

def generate_video_hash(file_content: bytes) -> str:
    """Generate unique hash for video content"""
    return hashlib.sha256(file_content).hexdigest()

async def cache_result(key: str, data: dict, ttl: int = CACHE_TTL):
    """Cache analysis result in Redis"""
    try:
        await redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.error(f"Cache write error: {e}")

async def get_cached_result(key: str) -> Optional[dict]:
    """Retrieve cached analysis result"""
    try:
        cached = await redis_client.get(key)
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None

async def update_analysis_status(video_id: str, status: str, progress: float = 0.0, message: str = None):
    """Update analysis status in Redis"""
    status_data = {
        "video_id": video_id,
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": datetime.now().isoformat()
    }
    await redis_client.setex(f"status:{video_id}", 1800, json.dumps(status_data))  # 30 min TTL

async def process_video_analysis(video_path: str, video_id: str) -> Dict[str, Any]:
    """Process video analysis asynchronously"""
    try:
        await update_analysis_status(video_id, "processing", 10.0, "Starting video analysis")
        
        start_time = datetime.now()
        
        # Process video with async detector
        sequences, labels = await detector.process_video_async(video_path, sequence_length=10)
        await update_analysis_status(video_id, "processing", 50.0, "Video processing complete")
        
        # Run inference with optimized model
        predictions = await asyncio.to_thread(model.predict_optimized, sequences)
        await update_analysis_status(video_id, "processing", 80.0, "Model inference complete")
        
        # Analyze results
        analysis = await analyze_predictions(predictions, labels)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "video_id": video_id,
            "total_reps": analysis["rep_count"],
            "phases_detected": analysis["phases"],
            "form_score": analysis["form_score"],
            "timestamp": datetime.now(),
            "processing_time": processing_time,
            "detailed_analysis": analysis
        }
        
        # Cache the result
        await cache_result(f"result:{video_id}", result)
        await update_analysis_status(video_id, "completed", 100.0, "Analysis complete")
        
        return result
        
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        await update_analysis_status(video_id, "failed", 0.0, str(e))
        raise

async def analyze_predictions(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """Analyze model predictions and extract metrics"""
    # Convert predictions to phase sequence
    predicted_phases = np.argmax(predictions, axis=-1)
    
    # Count repetitions (transition from phase 3 back to phase 0)
    rep_count = 0
    phase_transitions = []
    
    for i in range(1, len(predicted_phases)):
        prev_phase = predicted_phases[i-1][-1] if len(predicted_phases[i-1]) > 0 else 0
        curr_phase = predicted_phases[i][0] if len(predicted_phases[i]) > 0 else 0
        
        if prev_phase == 3 and curr_phase == 0:
            rep_count += 1
        
        phase_transitions.append((prev_phase, curr_phase))
    
    # Calculate form score (simplified)
    form_score = calculate_form_score(predicted_phases, ground_truth)
    
    return {
        "rep_count": rep_count,
        "phases": predicted_phases.flatten().tolist(),
        "form_score": form_score,
        "phase_transitions": phase_transitions,
        "confidence_scores": np.max(predictions, axis=-1).flatten().tolist()
    }

def calculate_form_score(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate form score based on prediction accuracy"""
    if ground_truth.size == 0:
        return 0.85  # Default score when no ground truth
    
    # Flatten arrays for comparison
    pred_flat = predicted.flatten()
    truth_flat = ground_truth.flatten()
    
    # Ensure same length
    min_len = min(len(pred_flat), len(truth_flat))
    accuracy = np.mean(pred_flat[:min_len] == truth_flat[:min_len])
    
    return float(accuracy)

@app.post("/analyze-video", response_model=Dict[str, str])
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Upload and analyze video for pull-up detection"""
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Read file content
        content = await file.read()
        video_hash = generate_video_hash(content)
        
        # Check cache first
        cached_result = await get_cached_result(f"result:{video_hash}")
        if cached_result:
            return {"video_id": video_hash, "status": "completed", "cached": True}
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        async with aiofiles.open(temp_file.name, 'wb') as f:
            await f.write(content)
        
        # Start background processing
        background_tasks.add_task(process_video_analysis, temp_file.name, video_id)
        
        # Initialize status
        await update_analysis_status(video_id, "processing", 0.0, "Video uploaded successfully")
        
        return {"video_id": video_id, "status": "processing", "message": "Analysis started"}
        
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{video_id}", response_model=AnalysisResult)
async def get_analysis_result(
    video_id: str,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Retrieve analysis results for a video"""
    
    # Check for completed result
    result = await get_cached_result(f"result:{video_id}")
    if result:
        return AnalysisResult(**result)
    
    # Check processing status
    status = await get_cached_result(f"status:{video_id}")
    if status and status["status"] == "processing":
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    elif status and status["status"] == "failed":
        raise HTTPException(status_code=500, detail=status.get("message", "Analysis failed"))
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/status/{video_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    video_id: str,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get current analysis status"""
    
    status = await get_cached_result(f"status:{video_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return AnalysisStatus(**status)

@app.post("/live-stream/start")
async def start_live_stream(config: LiveStreamConfig):
    """Websockets"""
    stream_id = config.stream_id
    
    # Store stream configuration in Redis
    stream_config = {
        "stream_id": stream_id,
        "confidence_threshold": config.confidence_threshold,
        "phase_smoothing": config.phase_smoothing,
        "status": "active",
        "started_at": datetime.now().isoformat()
    }
    
    await redis_client.setex(f"stream:{stream_id}", 3600, json.dumps(stream_config))
    
    return {"stream_id": stream_id, "status": "started", "message": "Live stream analysis started"}

@app.delete("/live-stream/{stream_id}")
async def stop_live_stream(stream_id: str):
    """Stop live stream analysis"""
    
    # Update stream status
    stream_config = await get_cached_result(f"stream:{stream_id}")
    if stream_config:
        stream_config["status"] = "stopped"
        stream_config["stopped_at"] = datetime.now().isoformat()
        await cache_result(f"stream:{stream_id}", stream_config, 300)  # 5 min TTL
    
    return {"stream_id": stream_id, "status": "stopped"}

@app.get("/exercise-stats")
async def get_exercise_stats(
    days: int = 7,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get aggregated exercise statistics"""
    
    stats_key = f"stats:last_{days}_days"
    cached_stats = await get_cached_result(stats_key)
    
    if cached_stats:
        return cached_stats
    
    # Mock stats
    stats = {
        "total_workouts": 15,
        "total_reps": 127,
        "average_form_score": 0.87,
        "best_session_reps": 12,
        "improvement_trend": 0.15,
        "last_updated": datetime.now().isoformat()
    }
    
    await cache_result(stats_key, stats, 1800)  # 30 min cache
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_status,
            "tensorflow": "healthy" if model else "unhealthy",
            "detector": "healthy" if detector else "unhealthy"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # Increase for prod
        loop="asyncio"
    )