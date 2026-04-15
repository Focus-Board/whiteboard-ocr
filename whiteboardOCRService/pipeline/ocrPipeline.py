from __future__ import annotations

import io
from contextlib import suppress
from dataclasses import dataclass
import json
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..models.localLlm import localJsonLlmManager
from ..models.trocr import TrOCRManager
from ..parsing import CalendarDraft, buildFallbackCalendarDraft, buildVjournalFromDraft
from ..utils.jobStore import jobQueue, jobStore


@dataclass(frozen=True)
class OcrPipelineResult:
    text: str
    structured: dict[str, Any]
    vjournal: str


@dataclass(frozen=True)
class OcrDebugArtifact:
    text: str
    llmModel: str
    llmPrompt: str
    llmRawResponse: str
    calendarDraft: dict[str, Any]
    structured: dict[str, Any]
    vjournal: str


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


def _buildCalendarDraftFromText(text: str) -> tuple[str, str, CalendarDraft]:
    llmResult = localJsonLlmManager.generateDraft(text)
    return llmResult.prompt, llmResult.rawResponse, llmResult.parsedDraft


def parseStructuredText(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        _, _, calendarDraft = _buildCalendarDraftFromText(text)
    except Exception:
        calendarDraft = buildFallbackCalendarDraft(text)
    return {
        "lines": lines,
        "lineCount": len(lines),
        "rawText": text,
        "calendarDraft": calendarDraft.model_dump(),
    }


def buildOcrDebugArtifact(imageBytes: bytes, ocrManager: TrOCRManager) -> OcrDebugArtifact:
    image = Image.open(io.BytesIO(imageBytes))
    prepared = prepareImageForOcr(image)
    text = ocrManager.predict(prepared)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    try:
        llmPrompt, llmRawResponse, parsedDraft = _buildCalendarDraftFromText(text)
    except Exception as exc:
        fallbackDraft = buildFallbackCalendarDraft(text)
        llmPrompt = f"LLM unavailable, fallback to deterministic parsing. Error: {exc}"
        llmRawResponse = json.dumps({"error": str(exc), "fallback": True}, indent=2)
        parsedDraft = fallbackDraft

    vjournal = buildVjournalFromDraft(parsedDraft)
    structured = {
        "lines": lines,
        "lineCount": len(lines),
        "rawText": text,
        "calendarDraft": parsedDraft.model_dump(),
    }

    return OcrDebugArtifact(
        text=text,
        llmModel=localJsonLlmManager.modelName,
        llmPrompt=llmPrompt,
        llmRawResponse=llmRawResponse,
        calendarDraft=parsedDraft.model_dump(),
        structured=structured,
        vjournal=vjournal,
    )


def processImageBytes(imageBytes: bytes, ocrManager: TrOCRManager) -> OcrPipelineResult:
    image = Image.open(io.BytesIO(imageBytes))
    prepared = prepareImageForOcr(image)
    text = ocrManager.predict(prepared)

    try:
        _, _, parsedDraft = _buildCalendarDraftFromText(text)
    except Exception:
        parsedDraft = buildFallbackCalendarDraft(text)

    structured = {
        "lines": [line.strip() for line in text.splitlines() if line.strip()],
        "lineCount": len([line for line in text.splitlines() if line.strip()]),
        "rawText": text,
        "calendarDraft": parsedDraft.model_dump(),
    }
    return OcrPipelineResult(text=text, structured=structured, vjournal=buildVjournalFromDraft(parsedDraft))


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
