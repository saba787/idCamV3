#!/usr/bin/env python3
"""
Preview + document geometry diagnostic (Phases 2-4).

Saves:
  /tmp/idcam-preview.jpg
  /tmp/idcam-geometry-debug.jpg

Prints JSON with channel analysis and every geometry candidate.

Usage:
  /opt/idcam/venv/bin/python scripts/geometry_diagnostic.py --config /etc/idcam/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from idcam.camera.diagnostic import (
    analyze_channels,
    camera_hardware_info,
    capture_native_preview,
    save_rgb_jpeg,
)
from idcam.camera.manager import CameraManager
from idcam.config.loader import load_config
from idcam.config.schema import CameraConfig
from idcam.vision.document_geometry import draw_geometry_debug_image, scan_document_candidates


def _candidate_dict(c) -> dict:
    return {
        "accepted": c.accepted,
        "bbox": list(c.bbox),
        "corners": [list(p) for p in c.corners] if c.corners else None,
        "area_fraction": round(c.area_fraction, 4),
        "aspect_ratio": round(c.aspect_ratio, 3),
        "rectangularity": round(c.rectangularity, 3),
        "edge_strength": round(c.edge_strength, 2),
        "final_score": round(c.final_score, 3),
        "rejection_reason": c.rejection_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="idCam preview + geometry diagnostic")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preview-output", type=Path, default=Path("/tmp/idcam-preview.jpg"))
    parser.add_argument("--geometry-debug", type=Path, default=Path("/tmp/idcam-geometry-debug.jpg"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    report: dict = {"ok": False}

    try:
        config = load_config(args.config)
        camera_cfg = config.camera
    except Exception as exc:
        report["config_error"] = str(exc)
        camera_cfg = CameraConfig()

    mgr = CameraManager(camera_cfg, use_mock=args.mock)

    try:
        mgr.initialize()
        report["hardware"] = camera_hardware_info(mgr)

        native = capture_native_preview(mgr)
        rgb = mgr.capture_preview_array()

        report["preview_native"] = analyze_channels(native, "native_picamera2") if native is not None else None
        report["preview_rgb"] = analyze_channels(rgb, "idcam_rgb_normalized")

        save_rgb_jpeg(rgb, args.preview_output)
        report["preview_jpeg_path"] = str(args.preview_output)
        report["preview_jpeg_bytes"] = args.preview_output.stat().st_size

        best, candidates = scan_document_candidates(rgb)
        report["candidates"] = [_candidate_dict(c) for c in candidates]

        if best is None:
            reason = "no_contour_passed_filters" if candidates else "no_contours_found"
            report["document_detected"] = False
            report["reason"] = reason
            draw_geometry_debug_image(rgb, None, args.geometry_debug, reason=reason)
        else:
            report["document_detected"] = True
            report["document"] = {
                "bbox": list(best.bbox),
                "corners": [list(p) for p in best.corners],
                "confidence": round(best.confidence, 3),
                "area_fraction": round(best.area_fraction, 4),
                "area_percent": round(best.area_fraction * 100, 1),
                "aspect_ratio": round(best.aspect_ratio, 3),
                "rectangularity": round(best.rectangularity, 3),
                "rotation_deg": round(best.rotation_deg, 2),
            }
            draw_geometry_debug_image(rgb, best, args.geometry_debug)

        report["geometry_debug_path"] = str(args.geometry_debug)
        report["ok"] = True

    except Exception as exc:
        report["ok"] = False
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2, default=str))
        mgr.shutdown()
        return 1

    mgr.shutdown()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
