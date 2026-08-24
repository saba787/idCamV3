"""Vision pipeline tests."""

import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.sharpness import laplacian_variance, mean_brightness
from idcam.vision.stability_detector import StabilityDetector


@pytest.fixture
def detection_config():
    return DetectionConfig(
        motion_threshold=5.0,
        edge_density_threshold=0.01,
        stable_frames_required=3,
        removal_frames_required=3,
    )


class TestDocumentDetector:
    def test_no_document_on_uniform_frame(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.update_baseline(frame)
        assert not detector.detect(frame)

    def test_document_detected_on_change(self, detection_config):
        detector = DocumentDetector(detection_config)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        # Add high-contrast document region
        frame[100:380, 100:540] = 50
        frame[100:120, 100:540] = 0
        frame[360:380, 100:540] = 0
        frame[100:380, 100:120] = 0
        frame[100:380, 520:540] = 0
        assert detector.detect(frame)

    def test_scene_empty_after_removal(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        for _ in range(20):
            detector.update_baseline(frame)
        assert detector.is_scene_empty(frame)


class TestStabilityDetector:
    def test_not_stable_initially(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert not sd.update(frame)

    def test_stable_after_consecutive_frames(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = False
        for _ in range(10):
            result = sd.update(frame)
        assert result

    def test_reset_on_motion(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(2):
            sd.update(frame)
        moving = frame.copy()
        moving[200:300, 200:300] = 255
        sd.update(moving)
        assert sd.stable_count == 0


class TestSharpness:
    def test_blurry_lower_than_sharp(self):
        blurry = np.full((480, 640, 3), 128, dtype=np.uint8)
        sharp = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        assert laplacian_variance(blurry) < laplacian_variance(sharp)

    def test_brightness(self):
        dark = np.full((100, 100), 30, dtype=np.uint8)
        bright = np.full((100, 100), 200, dtype=np.uint8)
        assert mean_brightness(dark) < mean_brightness(bright)
