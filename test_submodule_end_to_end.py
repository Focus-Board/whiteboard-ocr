#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from whiteboardOCRService.main import app
from whiteboardOCRService.pipeline import prepareImageForOcr


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end whiteboard OCR submodule test")
    parser.add_argument("image_path", type=Path, help="Path to an input image")
    parser.add_argument("--device-id", default="local-test-device", help="Device ID to submit with the job")
    parser.add_argument("--timestamp", default="2026-04-15T00:00:00Z", help="Timestamp to submit with the job")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Poll interval for job completion")
    parser.add_argument("--max-polls", type=int, default=60, help="Maximum number of polling attempts")
    parser.add_argument("--debug-dir", type=Path, default=Path("debug"), help="Directory to save debug artifacts")
    return parser


def _print_json(label: str, payload: object) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2))


def _save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _build_segmentation_boxes(prepared: Image.Image) -> tuple[np.ndarray, list[dict[str, int]]]:
    rgb = np.array(prepared)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    numLabels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    boxes: list[dict[str, int]] = []
    vis = bgr.copy()
    for idx in range(1, numLabels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 150:
            continue
        boxes.append({"x": x, "y": y, "w": w, "h": h, "area": area})
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)

    return vis, boxes


def _run() -> int:
    args = _build_parser().parse_args()
    if not args.image_path.exists():
        print(f"Image not found: {args.image_path}", file=sys.stderr)
        return 2

    runId = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    debugDir = args.debug_dir / runId
    debugDir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.image_path) as image:
        prepared = prepareImageForOcr(image)
        print("\n=== SEGMENTATION / PREPROCESSING ===")
        print(f"Input mode: {image.mode}")
        print(f"Input size: {image.size}")
        print(f"Prepared mode: {prepared.mode}")
        print(f"Prepared size: {prepared.size}")
        image.save(debugDir / "01_input_image.png")
        prepared.save(debugDir / "02_preprocessed_image.png")

        vis, boxes = _build_segmentation_boxes(prepared)
        cv2.imwrite(str(debugDir / "03_segmentation_boxes.png"), vis)
        _save_json(debugDir / "03_segmentation_boxes.json", boxes)

    with TestClient(app) as client:
        with args.image_path.open("rb") as image_file:
            debug_response = client.post(
                "/api/v1/debug/ocr-preview",
                files={"file": (args.image_path.name, image_file, "image/jpeg")},
            )
        print(f"Debug preview status: {debug_response.status_code}")
        debug_response.raise_for_status()
        debug_json = debug_response.json()
        _save_json(debugDir / "04_debug_preview.json", debug_json)
        pipeline = debug_json.get("pipeline", {}) if isinstance(debug_json, dict) else {}
        stage1 = pipeline.get("stage1_ocrText", {}) if isinstance(pipeline, dict) else {}
        stage2 = pipeline.get("stage2_llmPrompt", {}) if isinstance(pipeline, dict) else {}
        stage3 = pipeline.get("stage3_llmRawResponse", {}) if isinstance(pipeline, dict) else {}
        _save_text(debugDir / "05_ocr_text.txt", str(stage1.get("value") or ""))
        _save_text(debugDir / "06_llm_prompt.txt", str(stage2.get("value") or ""))
        _save_text(debugDir / "07_llm_raw_response.txt", str(stage3.get("value") or ""))

        print("\n=== UPLOAD ===")
        with args.image_path.open("rb") as image_file:
            submit_response = client.post(
                "/api/v1/jobs",
                data={
                    "deviceId": args.device_id,
                    "timestamp": args.timestamp,
                },
                files={"file": (args.image_path.name, image_file, "image/jpeg")},
            )
        print(f"Status: {submit_response.status_code}")
        submit_response.raise_for_status()
        submit_json = submit_response.json()
        _print_json("JOB SUBMIT RESPONSE", submit_json)
        _save_json(debugDir / "08_job_submit_response.json", submit_json)

        job_id = submit_json["jobId"]

        print("\n=== OCR / PARSING POLL LOOP ===")
        result_json: dict[str, object] | None = None
        for attempt in range(1, args.max_polls + 1):
            status_response = client.get(f"/api/v1/jobs/{job_id}")
            status_response.raise_for_status()
            status_json = status_response.json()
            print(f"Poll {attempt}: {status_json['status']}")
            _save_json(debugDir / f"09_job_status_poll_{attempt:02d}.json", status_json)
            if status_json["status"] == "failed":
                _print_json("JOB STATUS", status_json)
                return 1
            if status_json["status"] == "done":
                result_response = client.get(f"/api/v1/jobs/{job_id}/result")
                result_response.raise_for_status()
                result_json = result_response.json()
                break
            time.sleep(args.poll_seconds)

        if result_json is None:
            print("Job did not complete before timeout", file=sys.stderr)
            return 1

        _print_json("USER RESPONSE", result_json)
        _save_json(debugDir / "10_user_response.json", result_json)

        events = result_json.get("events", []) if isinstance(result_json, dict) else []
        notes_vjournal = result_json.get("notesVjournal", "") if isinstance(result_json, dict) else ""
        notes = result_json.get("structured", {}).get("notes", []) if isinstance(result_json, dict) else []

        print("\n=== USER APPROVAL ===")
        approval_payload = {
            "events": events,
            "notes": notes,
        }
        _print_json("APPROVAL REQUEST", approval_payload)
        _save_json(debugDir / "11_approval_request.json", approval_payload)

        approve_response = client.post(f"/api/v1/jobs/{job_id}/approve", json=approval_payload)
        print(f"Approval status: {approve_response.status_code}")
        approve_response.raise_for_status()
        approve_json = approve_response.json()
        _print_json("APPROVAL RESPONSE", approve_json)
        _save_json(debugDir / "12_approval_response.json", approve_json)

        if notes_vjournal:
            print("\n=== NOTES VJOURNAL (user response) ===")
            print(notes_vjournal)
            _save_text(debugDir / "13_notes_vjournal.ics", notes_vjournal)

    print(f"\nSaved debug artifacts to: {debugDir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
