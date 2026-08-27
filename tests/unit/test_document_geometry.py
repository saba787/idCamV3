"""Document geometry detection tests."""

import numpy as np

from idcam.vision.document_geometry import detect_document_geometry, order_corners


def _document_frame(width=640, height=480):
    frame = np.full((height, width, 3), 210, dtype=np.uint8)
    x0, y0, x1, y1 = 80, 60, width - 80, height - 60
    frame[y0:y1, x0:x1] = 60
    frame[y0:y0 + 8, x0:x1] = 0
    frame[y1 - 8:y1, x0:x1] = 0
    frame[y0:y1, x0:x0 + 8] = 0
    frame[y0:y1, x1 - 8:x1] = 0
    return frame


class TestDocumentGeometry:
    def test_detects_paper_like_rectangle(self):
        frame = _document_frame()
        geom = detect_document_geometry(frame)
        assert geom is not None
        assert geom.confidence >= 0.4
        assert geom.area_fraction > 0.1
        assert geom.bbox[2] > 100
        assert geom.bbox[3] > 100

    def test_uniform_scene_returns_none(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert detect_document_geometry(frame) is None

    def test_order_corners_tl_tr_br_bl(self):
        pts = np.array([[10, 10], [100, 12], [98, 90], [12, 88]], dtype=np.float32)
        ordered = order_corners(pts)
        assert ordered[0][1] <= ordered[2][1]
        assert ordered[0][0] <= ordered[1][0]
