# websocket.py - WebSocket support for live streaming
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
import cv2 as cv
import numpy as np
import base64

class ConnectionManager:
    """Manage WebSocket connections for live streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, stream_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[stream_id] = websocket
        logger.info(f"WebSocket connected: {stream_id}")
    
    def disconnect(self, stream_id: str):
        """Remove WebSocket connection"""
        if stream_id in self.active_connections:
            del self.active_connections[stream_id]
        
        # Cancel associated stream task
        if stream_id in self.stream_tasks:
            self.stream_tasks[stream_id].cancel()
            del self.stream_tasks[stream_id]
        
        logger.info(f"WebSocket disconnected: {stream_id}")
    
    async def send_personal_message(self, message: dict, stream_id: str):
        """Send message to specific connection"""
        if stream_id in self.active_connections:
            try:
                websocket = self.active_connections[stream_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {stream_id}: {e}")
                self.disconnect(stream_id)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        if self.active_connections:
            tasks = []
            for stream_id in list(self.active_connections.keys()):
                task = self.send_personal_message(message, stream_id)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

# Add WebSocket endpoints to main.py
websocket_manager = ConnectionManager()

@app.websocket("/ws/live-stream/{stream_id}")
async def websocket_live_stream(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for live video streaming"""
    
    await websocket_manager.connect(websocket, stream_id)
    
    try:
        # Start processing task
        task = asyncio.create_task(
            process_live_stream(websocket, stream_id)
        )
        websocket_manager.stream_tasks[stream_id] = task
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive frame data from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "frame":
                    # Process frame asynchronously
                    await handle_frame_data(message, stream_id)
                elif message.get("type") == "stop":
                    break
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {stream_id}: {e}")
                break
    
    finally:
        websocket_manager.disconnect(stream_id)

async def process_live_stream(websocket: WebSocket, stream_id: str):
    """Process live stream frames"""
    
    try:
        # Initialize detector for this stream
        stream_detector = AsyncPullUpDetector()
        frame_buffer = []
        
        while stream_id in websocket_manager.active_connections:
            await asyncio.sleep(0.1)  # Process at ~10 FPS
            
            # Process buffered frames if available
            if len(frame_buffer) >= 5:  # Process in small batches
                batch = frame_buffer[:5]
                frame_buffer = frame_buffer[5:]
                
                # Process batch
                results = await process_frame_batch(batch, stream_detector)
                
                # Send results back to client
                await websocket_manager.send_personal_message({
                    "type": "results",
                    "data": results
                }, stream_id)
    
    except Exception as e:
        logger.error(f"Live stream processing error: {e}")

async def handle_frame_data(message: dict, stream_id: str):
    """Handle incoming frame data from WebSocket"""
    
    try:
        # Decode base64 frame data
        frame_data = message.get("frame")
        if frame_data:
            # Convert base64 to numpy array
            img_data = base64.b64decode(frame_data)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            
            if frame is not None:
                # Add frame to processing queue
                # This would be handled by the processing task
                pass
    
    except Exception as e:
        logger.error(f"Error handling frame data: {e}")

async def process_frame_batch(frames: List[np.ndarray], detector: AsyncPullUpDetector) -> dict:
    """Process batch of frames"""
    
    results = {
        "timestamp": time.time(),
        "frames_processed": len(frames),
        "detections": []
    }
    
    try:
        for i, frame in enumerate(frames):
            # Process frame
            features, phase = await detector.process_frame_async(frame)
            
            results["detections"].append({
                "frame_index": i,
                "phase": int(phase),
                "features": features.tolist() if isinstance(features, np.ndarray) else features
            })
    
    except Exception as e:
        logger.error(f"Error processing frame batch: {e}")
        results["error"] = str(e)
    
    return results