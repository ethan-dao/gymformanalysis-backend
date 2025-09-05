from locust import HttpUser, task, between
import io
import random

class PullUpAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        pass
    
    @task(3)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")
    
    @task(1)
    def get_stats(self):
        """Test stats endpoint"""
        self.client.get("/exercise-stats")
    
    @task(2)
    def upload_video(self):
        """Test video upload (with dummy data)"""
        
        # Create dummy video content
        video_content = b"dummy_video_content_" + str(random.randint(1000, 9999)).encode()
        
        files = {
            "file": ("test_video.mp4", io.BytesIO(video_content), "video/mp4")
        }
        
        response = self.client.post("/analyze-video", files=files)
        
        if response.status_code in [200, 202]:
            # If successful, try to get results
            video_id = response.json().get("video_id")
            if video_id:
                self.client.get(f"/results/{video_id}")