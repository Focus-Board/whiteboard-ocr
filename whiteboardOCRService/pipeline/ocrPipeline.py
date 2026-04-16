from __future__ import annotations

import io
from contextlib import suppress
from dataclasses import dataclass
import json
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..models.localLlm import localJsonLlmManager
from ..models.trocr import TrOCRManager
from ..parsing import CalendarDraft, buildVjournalFromDraft
from ..utils.jobStore import jobQueue, jobStore


@dataclass(frozen=True)
class OcrPipelineResult:
    text: str
    structured: dict[str, Any]
    notesVjournal: str


@dataclass(frozen=True)
class OcrDebugArtifact:
    text: str
    llmModel: str
    llmPrompt: str
    llmRawResponse: str
    events: list[dict[str, Any]]
    structured: dict[str, Any]
    notesVjournal: str


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


def _extractLineCrops(preparedImage: Image.Image) -> list[Image.Image]:
    rgb = np.array(preparedImage)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Binarize text strokes and connect nearby letters into words.
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
    imageHeight, imageWidth = gray.shape

    wordBoxes: list[tuple[int, int, int, int]] = []
    for idx in range(1, numLabels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < 120:
            continue
        if w >= int(imageWidth * 0.9) and h >= int(imageHeight * 0.7):
            # Ignore giant background component.
            continue
        if w < 8 or h < 8:
            continue

        wordBoxes.append((x, y, w, h))

    if not wordBoxes:
        return [preparedImage]

    # Group words into text lines by vertical alignment.
    wordBoxes.sort(key=lambda box: (box[1], box[0]))
    lineGroups: list[dict[str, Any]] = []
    for x, y, w, h in wordBoxes:
        cy = y + (h / 2.0)
        matched = False
        for group in lineGroups:
            if abs(cy - group["centerY"]) <= max(18.0, group["avgHeight"] * 0.8):
                group["boxes"].append((x, y, w, h))
                group["centerY"] = float(np.mean([b[1] + (b[3] / 2.0) for b in group["boxes"]]))
                group["avgHeight"] = float(np.mean([b[3] for b in group["boxes"]]))
                matched = True
                break
        if not matched:
            lineGroups.append(
                {
                    "boxes": [(x, y, w, h)],
                    "centerY": cy,
                    "avgHeight": float(h),
                }
            )

    lineGroups.sort(key=lambda group: min(b[1] for b in group["boxes"]))

    lineCrops: list[Image.Image] = []
    for group in lineGroups:
        boxes = group["boxes"]
        minX = max(min(b[0] for b in boxes) - 8, 0)
        minY = max(min(b[1] for b in boxes) - 8, 0)
        maxX = min(max(b[0] + b[2] for b in boxes) + 8, imageWidth)
        maxY = min(max(b[1] + b[3] for b in boxes) + 8, imageHeight)

        if (maxX - minX) < 24 or (maxY - minY) < 12:
            continue

        lineCrops.append(preparedImage.crop((minX, minY, maxX, maxY)))

    return lineCrops or [preparedImage]


def _predictOcrText(preparedImage: Image.Image, ocrManager: TrOCRManager) -> str:
    lineCrops = _extractLineCrops(preparedImage)
    predictions: list[str] = []
    for crop in lineCrops:
        text = ocrManager.predict(crop).strip()
        if text:
            predictions.append(text)

    if not predictions:
        return ""

    # Remove duplicate lines while preserving order.
    deduped = list(dict.fromkeys(predictions))
    return "\n".join(deduped)


def _cleanOcrTranscript(text: str) -> str:
    rawLines = [line.strip() for line in text.splitlines() if line.strip()]
    cleanedLines: list[str] = []

    def normalizeLine(line: str) -> str:
        normalized = line
        normalized = re.sub(r"(?<=\d),(?=\d{2}\b)", ":", normalized)
        normalized = re.sub(r"\b(\d{1,2}:\d{2})\s*([aApP][mM])\b", lambda match: f"{match.group(1)} {match.group(2).upper()}", normalized)
        normalized = re.sub(r"\b(\d{1,2}),(\d{2})\s*([aApP][mM])\b", lambda match: f"{match.group(1)}:{match.group(2)} {match.group(3).upper()}", normalized)
        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
        normalized = re.sub(r"([,.;:])\s*$", r"\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def looksLikeJunk(line: str) -> bool:
        compact = re.sub(r"[^a-zA-Z0-9]", "", line)
        if not compact:
            return True
        if re.fullmatch(r"\d+", compact):
            return True
        if len(compact) <= 2:
            return True
        alphaCount = len(re.findall(r"[a-zA-Z]", line))
        digitCount = len(re.findall(r"\d", line))
        if alphaCount == 0 and digitCount > 0:
            return True
        if digitCount >= alphaCount * 2 and alphaCount < 6:
            return True
        return False

    for line in rawLines:
        normalized = normalizeLine(line)
        if looksLikeJunk(normalized):
            continue

        if cleanedLines and re.match(r"^(on|at|for|to|and|with|from|in|of|by|before|after)\b", normalized, flags=re.IGNORECASE):
            cleanedLines[-1] = f"{cleanedLines[-1].rstrip('.')} {normalized}"
            continue

        if cleanedLines and re.match(r"^[a-z]", normalized):
            cleanedLines[-1] = f"{cleanedLines[-1].rstrip('.')} {normalized}"
            continue

        cleanedLines.append(normalized)

    # Merge obvious wrapped time fragments with the previous line.
    mergedLines: list[str] = []
    for line in cleanedLines:
        if mergedLines and re.fullmatch(r"(on|at)\b.*", line, flags=re.IGNORECASE):
            mergedLines[-1] = f"{mergedLines[-1].rstrip('.')} {line}"
            continue
        mergedLines.append(line)

    return "\n".join(mergedLines)


def _splitOcrEntries(text: str) -> list[str]:
    entries: list[str] = []
    for line in text.splitlines():
        entry = line.strip()
        entry = entry.lstrip("-•* ").strip()
        if not entry:
            continue
        entries.append(entry)
    return entries


def _buildCalendarDraftFromText(text: str) -> tuple[str, str, CalendarDraft]:
    entries = _splitOcrEntries(_cleanOcrTranscript(text))
    if not entries:
        return "", "", CalendarDraft(timezone="UTC", source_text=text, events=[], notes=[])

    prompts: list[str] = []
    rawResponses: list[dict[str, Any]] = []
    mergedEvents: list[dict[str, Any]] = []
    mergedNotes: list[str] = []
    timezoneValue = "UTC"

    for entry in entries:
        llmResult = localJsonLlmManager.generateDraft(entry)
        prompts.append(llmResult.prompt)
        rawResponses.append({"input": entry, "rawResponse": llmResult.rawResponse})

        draft = llmResult.parsedDraft
        timezoneValue = draft.timezone or timezoneValue
        mergedEvents.extend(event.model_dump() for event in draft.events)
        mergedNotes.extend(draft.notes)

    dedupedNotes = list(dict.fromkeys(note for note in mergedNotes if note.strip()))
    mergedDraft = CalendarDraft(
        timezone=timezoneValue,
        source_text="\n".join(entries),
        events=[event for event in mergedEvents if isinstance(event, dict)],
        notes=dedupedNotes,
    )
    return "\n\n---\n\n".join(prompts), json.dumps(rawResponses, indent=2), mergedDraft


def parseStructuredText(text: str) -> dict[str, Any]:
    cleanedText = _cleanOcrTranscript(text)
    lines = _splitOcrEntries(cleanedText)
    try:
        _, _, calendarDraft = _buildCalendarDraftFromText(cleanedText)
    except Exception:
        calendarDraft = CalendarDraft(timezone="UTC", source_text=cleanedText, events=[], notes=[])
    notesVjournal = buildVjournalFromDraft(calendarDraft)
    return {
        "lines": lines,
        "lineCount": len(lines),
        "rawText": cleanedText,
        "events": [event.model_dump() for event in calendarDraft.events],
        "notes": calendarDraft.notes,
        "notesVjournal": notesVjournal,
    }


def buildOcrDebugArtifact(imageBytes: bytes, ocrManager: TrOCRManager) -> OcrDebugArtifact:
    image = Image.open(io.BytesIO(imageBytes))
    prepared = prepareImageForOcr(image)
    rawText = _predictOcrText(prepared, ocrManager)
    text = _cleanOcrTranscript(rawText)
    lines = _splitOcrEntries(text)

    try:
        llmPrompt, llmRawResponse, parsedDraft = _buildCalendarDraftFromText(text)
    except Exception as exc:
        llmPrompt = f"{text}\n\nLLM request failed: {exc}"
        llmRawResponse = ""
        parsedDraft = CalendarDraft(timezone="UTC", source_text=text, events=[], notes=[])

    notesVjournal = buildVjournalFromDraft(parsedDraft)
    structured = {
        "lines": lines,
        "lineCount": len(lines),
        "rawText": text,
        "events": parsedDraft.events,
        "notes": parsedDraft.notes,
        "notesVjournal": notesVjournal,
    }

    return OcrDebugArtifact(
        text=text,
        llmModel=localJsonLlmManager.modelName,
        llmPrompt=llmPrompt,
        llmRawResponse=llmRawResponse,
        events=[event.model_dump() for event in parsedDraft.events],
        structured=structured,
        notesVjournal=notesVjournal,
    )


def processImageBytes(imageBytes: bytes, ocrManager: TrOCRManager) -> OcrPipelineResult:
    image = Image.open(io.BytesIO(imageBytes))
    prepared = prepareImageForOcr(image)
    text = _cleanOcrTranscript(_predictOcrText(prepared, ocrManager))

    try:
        _, _, parsedDraft = _buildCalendarDraftFromText(text)
    except Exception:
        parsedDraft = CalendarDraft(timezone="UTC", source_text=text, events=[], notes=[])

    structured = {
        "lines": _splitOcrEntries(text),
        "lineCount": len(_splitOcrEntries(text)),
        "rawText": text,
        "events": [event.model_dump() for event in parsedDraft.events],
        "notes": parsedDraft.notes,
        "notesVjournal": buildVjournalFromDraft(parsedDraft),
    }
    return OcrPipelineResult(text=text, structured=structured, notesVjournal=structured["notesVjournal"])


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
