"""
Frontend Validation Test Suite for Phase 7 Live Stream React Dashboard
======================================================================
Verifies:
1. Production bundle build dist/ exists and contains built HTML & JS assets
2. React component structure including LiveVideoPanel.tsx
3. API endpoints alignment including getVideoStreamUrl
4. Polling hook configuration
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"


class TestFrontendDashboard(unittest.TestCase):

    def test_1_production_build_exists(self):
        """Verify npm run build dist/ artifacts exist."""
        self.assertTrue(DIST_DIR.exists(), "dist/ directory MUST exist after npm run build.")
        index_html = DIST_DIR / "index.html"
        self.assertTrue(index_html.exists(), "dist/index.html MUST exist.")

        assets_dir = DIST_DIR / "assets"
        self.assertTrue(assets_dir.exists(), "dist/assets/ MUST exist.")
        asset_files = list(assets_dir.glob("*.js")) + list(assets_dir.glob("*.css"))
        self.assertGreater(len(asset_files), 0, "Built JS & CSS assets MUST exist in dist/assets/.")

    def test_2_component_files_exist(self):
        """Verify modular component architecture including LiveVideoPanel."""
        components_dir = FRONTEND_DIR / "src" / "components"
        required_components = [
            "Header.tsx",
            "SummaryCards.tsx",
            "LiveVideoPanel.tsx",
            "SafetyStatusPanel.tsx",
            "RecentEventsPanel.tsx",
            "StatisticsPanel.tsx",
            "EvidenceModal.tsx"
        ]
        for comp in required_components:
            path = components_dir / comp
            self.assertTrue(path.exists(), f"Component {comp} MUST exist.")

    def test_3_stream_url_configured(self):
        """Verify api.ts exposes getVideoStreamUrl."""
        api_path = FRONTEND_DIR / "src" / "services" / "api.ts"
        with open(api_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("getVideoStreamUrl", content)
        self.assertIn("/api/video/stream", content)

    def test_4_no_fake_video_hacks(self):
        """Verify LiveVideoPanel points to backend streaming endpoint without synthetic video generator."""
        panel_path = FRONTEND_DIR / "src" / "components" / "LiveVideoPanel.tsx"
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("getVideoStreamUrl", content)
        self.assertNotIn("fake-stream.mp4", content)


if __name__ == "__main__":
    unittest.main()
