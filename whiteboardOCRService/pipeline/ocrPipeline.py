from __future__ import annotations

import io
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..models.trocr import TrOCRManager
from ..utils.jobStore import jobQueue, jobStore


@dataclass(frozen=True)
class OcrPipelineResult:
    text: str
    structured: dict[str, Any]


def _grayWorldWhiteBalance(imgBgr: np.ndarray) -> np.ndarray:
    img = imgBgr.astype(np.float32)
    b, g, r = cv2.split(img)
    avg_b = max(float(np.mean(b)), 1.0)
    avg_g = max(float(np.mean(g)), 1.0)
    avg_r = max(float(np.mean(r)), 1.0)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    b *= avg_gray / avg_b
    g *= avg_gray / avg_g
    r *= avg_gray / avg_r

    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def prepareImageForOcr(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = ImageOps.exif_transpose(image)
    imgBgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    balanced = _grayWorldWhiteBalance(imgBgr)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def parseStructuredText(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "lines": lines,
        "lineCount": len(lines),
        "rawText": text,
    }


def processImageBytes(imageBytes: bytes, ocrManager: TrOCRManager) -> OcrPipelineResult:
    image = Image.open(io.BytesIO(imageBytes))
    prepared = prepareImageForOcr(image)
    text = ocrManager.predict(prepared)
    structured = parseStructuredText(text)
    return OcrPipelineResult(text=text, structured=structured)


async def runJobWorker(ocrManager: TrOCRManager) -> None:
    while True:
        jobId = await jobQueue.get()
        try:
            job = jobStore.getJob(jobId)
            if job is None:
                continue

            jobStore.markProcessing(jobId)

            if job.inputType == "pdf":
                raise NotImplementedError("PDF processing is not implemented yet.")

            result = processImageBytes(job.fileBytes, ocrManager)
            jobStore.markDone(jobId, text=result.text, structured=result.structured)
        except Exception as exc:
            with suppress(Exception):
                jobStore.markFailed(jobId, str(exc))
        finally:
            jobQueue.task_done()
