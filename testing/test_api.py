import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from main import app

class TestPullUpAPI:
    """Test cases for Pull-Up Detection API"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    async def test_upload_video(self, async_client):
        """Test video upload endpoint"""
        
        # Create dummy video file
        video_content = b"dummy_video_content"
        
        files = {"file": ("test_video.mp4", video_content, "video/mp4")}
        response = await async_client.post("/analyze-video", files=files)
        
        assert response.status_code in [200, 202]
        assert "video_id" in response.json()
    
    def test_get_stats(self, client):
        """Test exercise stats endpoint"""
        response = client.get("/exercise-stats")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
    
    async def test_live_stream_config(self, async_client):
        """Test live stream configuration"""
        
        config = {
            "stream_id": "test_stream_123",
            "confidence_threshold": 0.8,
            "phase_smoothing": True
        }
        
        response = await async_client.post("/live-stream/start", json=config)
        assert response.status_code == 200
        assert response.json()["stream_id"] == config["stream_id"]