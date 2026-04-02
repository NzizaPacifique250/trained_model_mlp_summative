"""
Locust load-testing configuration for the ML Pipeline API.

Usage:
    locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089 in your browser to configure
the number of users, spawn rate, and start the test.

Recommended test matrix (for the assignment report):
    Containers | Users | Spawn Rate
    ---------------------------------
         1     |  50   |     5
         1     | 100   |    10
         3     | 100   |    10
         3     | 200   |    20
"""

from locust import HttpUser, task, between
import io
from PIL import Image


class MLPipelineUser(HttpUser):
    """Simulates a user interacting with the ML Pipeline API."""

    wait_time = between(1, 3)

    def on_start(self):
        """Generate a synthetic JPEG image in memory for load testing."""
        img = Image.new("RGB", (150, 150), color=(73, 109, 137))
        self.img_bytes = io.BytesIO()
        img.save(self.img_bytes, format="JPEG")

    @task(3)
    def predict_endpoint(self):
        """POST an image to /predict — the heaviest endpoint."""
        self.img_bytes.seek(0)
        files = {"file": ("test_image.jpg", self.img_bytes.read(), "image/jpeg")}
        self.client.post("/predict", files=files)

    @task(5)
    def health_endpoint(self):
        """GET /health — lightweight liveness check."""
        self.client.get("/health")

    @task(1)
    def stats_endpoint(self):
        """GET /data/stats — dataset statistics."""
        self.client.get("/data/stats")
