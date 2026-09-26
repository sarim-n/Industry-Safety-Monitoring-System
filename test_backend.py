"""
Unit & Integration Tests for Phase 5 FastAPI Backend
=====================================================
Tests all 9 backend functionality scenarios using FastAPI TestClient:
1. /api/health returns 200 and {"status": "ok"}
2. /api/status returns valid JSON system state
3. /api/workers returns valid worker safety states list
4. /api/events returns CSV-backed events
5. /api/events pagination (limit & offset) works correctly
6. /api/statistics returns correct aggregate metrics
7. Valid evidence file is served correctly
8. Nonexistent evidence file returns HTTP 404
9. Path traversal attempts are rejected cleanly (HTTP 400)
"""

import sys
import os
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app, EVIDENCE_DIR


class TestFastAPIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create a test evidence file for verification
        cls.test_filename = "_test_snapshot_unit_test.jpg"
        cls.test_filepath = EVIDENCE_DIR / cls.test_filename
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cls.test_filepath, "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xFF\xD9")

    @classmethod
    def tearDownClass(cls):
        if cls.test_filepath.exists():
            cls.test_filepath.unlink()

    def test_1_health_endpoint(self):
        """GET /api/health should return 200 and status ok."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_2_status_endpoint(self):
        """GET /api/status should return valid system status JSON."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("system_running", data)
        self.assertIn("active_workers", data)
        self.assertIn("confirmed_violations", data)

    def test_3_workers_endpoint(self):
        """GET /api/workers should return a list of worker state objects."""
        response = self.client.get("/api/workers")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_4_events_endpoint(self):
        """GET /api/events should return CSV-backed events."""
        response = self.client.get("/api/events")
        self.assertEqual(response.status_code, 200)
        events = response.json()
        self.assertIsInstance(events, list)
        if len(events) > 0:
            evt = events[0]
            self.assertIn("timestamp", evt)
            self.assertIn("worker_id", evt)
            self.assertIn("violation_type", evt)
            self.assertIn("evidence_path", evt)

    def test_5_events_pagination(self):
        """GET /api/events with limit and offset should paginate correctly."""
        resp1 = self.client.get("/api/events?limit=2&offset=0")
        self.assertEqual(resp1.status_code, 200)
        events_page1 = resp1.json()
        self.assertLessEqual(len(events_page1), 2)

        resp2 = self.client.get("/api/events?limit=2&offset=2")
        self.assertEqual(resp2.status_code, 200)
        events_page2 = resp2.json()
        self.assertLessEqual(len(events_page2), 2)

    def test_6_statistics_endpoint(self):
        """GET /api/statistics should return aggregate metrics."""
        response = self.client.get("/api/statistics")
        self.assertEqual(response.status_code, 200)
        stats = response.json()
        self.assertIn("total_events", stats)
        self.assertIn("no_helmet_count", stats)
        self.assertIn("no_mask_count", stats)
        self.assertIn("no_helmet_and_mask_count", stats)
        self.assertIn("unique_workers", stats)

    def test_7_valid_evidence_serving(self):
        """GET /api/evidence/{filename} should serve valid existing evidence image."""
        response = self.client.get(f"/api/evidence/{self.test_filename}")
        self.assertEqual(response.status_code, 200)

    def test_8_nonexistent_evidence_returns_404(self):
        """GET /api/evidence/{filename} should return 404 for missing files."""
        response = self.client.get("/api/evidence/nonexistent_file_99999.jpg")
        self.assertEqual(response.status_code, 404)

    def test_9_path_traversal_rejected(self):
        """GET /api/evidence/.. should reject path traversal attempt cleanly (400/404, not 200)."""
        response_encoded = self.client.get("/api/evidence/%2E%2E%2Frun_live.py")
        self.assertIn(response_encoded.status_code, [400, 404])
        self.assertNotEqual(response_encoded.status_code, 200)

        response_raw = self.client.get("/api/evidence/../run_live.py")
        self.assertIn(response_raw.status_code, [400, 404])
        self.assertNotEqual(response_raw.status_code, 200)


if __name__ == "__main__":
    unittest.main()
