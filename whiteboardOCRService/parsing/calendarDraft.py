from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


CalendarItemType = Literal["event", "task", "note"]


class CalendarItem(BaseModel):
    itemType: CalendarItemType
    title: str = Field(min_length=1)
    description: str = ""
    date: str | None = Field(default=None, description="YYYY-MM-DD when known")
    startTime: str | None = Field(default=None, description="HH:MM 24h when known")
    endTime: str | None = Field(default=None, description="HH:MM 24h when known")
    dueDate: str | None = Field(default=None, description="YYYY-MM-DD for tasks")
    dueTime: str | None = Field(default=None, description="HH:MM 24h for tasks")
    allDay: bool = False
    location: str = ""
    tags: list[str] = Field(default_factory=list)
    sourceText: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CalendarDraft(BaseModel):
    timezone: str = "UTC"
    summary: str = "Whiteboard draft"
    sourceText: str = ""
    items: list[CalendarItem] = Field(default_factory=list)


def getCalendarDraftTemplate() -> dict[str, Any]:
    return {
        "timezone": "UTC",
        "summary": "Whiteboard draft",
        "sourceText": "original OCR text",
        "items": [
            {
                "itemType": "event",
                "title": "Algebra review",
                "description": "Chapter 2",
                "date": "2026-04-20",
                "startTime": "14:00",
                "endTime": "15:00",
                "allDay": False,
                "location": "Classroom",
                "tags": ["school"],
                "sourceText": "Review session Monday 2pm",
                "confidence": 0.91,
            },
            {
                "itemType": "task",
                "title": "Submit worksheet",
                "description": "Solve problems 1-10",
                "dueDate": "2026-04-22",
                "dueTime": "23:59",
                "allDay": False,
                "tags": ["homework"],
                "sourceText": "worksheet due Wednesday",
                "confidence": 0.86,
            },
            {
                "itemType": "note",
                "title": "Lesson objective",
                "description": "Understand what algebra is and its purpose.",
                "tags": ["lesson"],
                "sourceText": "Lesson Objectives: ...",
                "confidence": 0.82,
            },
        ],
    }


def buildLlmExtractionPrompt(ocrText: str, *, timezone: str = "UTC") -> str:
    templateJson = json.dumps(getCalendarDraftTemplate(), indent=2)
    return (
        "Extract tasks, events, and notes from OCR text into STRICT JSON. "
        "Do not include markdown or explanation. Return only one JSON object matching this schema shape.\n\n"
        f"timezone to use: {timezone}\n"
        "Rules:\n"
        "- Use itemType as one of: event, task, note\n"
        "- Keep unknown date/time fields as null\n"
        "- confidence is 0..1\n"
        "- Preserve sourceText snippets\n\n"
        f"Target JSON shape example:\n{templateJson}\n\n"
        f"OCR text:\n{ocrText}\n"
    )


def _extractJsonPayload(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _normalizeItemPayload(item: dict[str, Any], *, fallbackSourceText: str = "") -> dict[str, Any]:
    normalized = dict(item)
    title = str(normalized.get("title") or "").strip()
    description = str(normalized.get("description") or "").strip()
    sourceText = str(normalized.get("sourceText") or "").strip()

    if not title:
        title = description or sourceText or str(normalized.get("itemType") or "Item").title()

    if not sourceText:
        sourceText = description or title or fallbackSourceText.strip()

    normalized["title"] = title
    normalized["description"] = description
    normalized["sourceText"] = sourceText
    normalized["location"] = str(normalized.get("location") or "").strip()

    tags = normalized.get("tags") or []
    if isinstance(tags, list):
        normalized["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
    else:
        normalized["tags"] = []

    if normalized.get("summary") is None:
        normalized.pop("summary", None)

    return normalized


def _normalizeDraftPayload(payload: dict[str, Any], *, fallbackSourceText: str = "") -> dict[str, Any]:
    normalized = dict(payload)
    normalized["timezone"] = str(normalized.get("timezone") or "UTC").strip() or "UTC"
    summary = str(normalized.get("summary") or "").strip()
    sourceText = str(normalized.get("sourceText") or "").strip() or fallbackSourceText.strip()

    if not summary:
        summary = sourceText.splitlines()[0] if sourceText else "Whiteboard draft"

    normalized["summary"] = summary
    normalized["sourceText"] = sourceText

    items = normalized.get("items") or []
    if isinstance(items, list):
        normalized["items"] = [
            _normalizeItemPayload(item, fallbackSourceText=sourceText)
            for item in items
            if isinstance(item, dict)
        ]
    else:
        normalized["items"] = []

    return normalized


def parseCalendarDraftFromUnknown(raw: Any, *, fallbackSourceText: str = "") -> CalendarDraft:
    if isinstance(raw, CalendarDraft):
        return raw

    if isinstance(raw, str):
        payload = _extractJsonPayload(raw)
        try:
            parsed = json.loads(payload)
            normalized = _normalizeDraftPayload(parsed, fallbackSourceText=fallbackSourceText or payload)
            return CalendarDraft.model_validate(normalized)
        except (json.JSONDecodeError, ValidationError):
            return buildFallbackCalendarDraft(fallbackSourceText or raw)

    if isinstance(raw, dict):
        try:
            normalized = _normalizeDraftPayload(raw, fallbackSourceText=fallbackSourceText)
            return CalendarDraft.model_validate(normalized)
        except ValidationError:
            return buildFallbackCalendarDraft(fallbackSourceText)

    return buildFallbackCalendarDraft(fallbackSourceText)


def buildFallbackCalendarDraft(text: str, *, timezone: str = "UTC") -> CalendarDraft:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0] if lines else "Whiteboard draft"
    items = []

    for line in lines:
        items.append(
            CalendarItem(
                itemType="note",
                title=line[:120],
                description=line,
                sourceText=line,
                confidence=0.35,
            )
        )

    return CalendarDraft(
        timezone=timezone,
        summary=summary,
        sourceText=text,
        items=items,
    )