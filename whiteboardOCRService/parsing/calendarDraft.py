from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


EventType = Literal["event"]


class CalendarEvent(BaseModel):
    type: EventType = "event"
    title: str = Field(min_length=1)
    description: str | None = None
    start_time: str = Field(description="ISO-8601 datetime")
    end_time: str = Field(description="ISO-8601 datetime")
    all_day: bool = False
    location: str | None = None


class CalendarDraft(BaseModel):
    timezone: str = "UTC"
    source_text: str = ""
    events: list[CalendarEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def getCalendarDraftTemplate() -> dict[str, Any]:
    return {
        "timezone": "UTC",
        "source_text": "original OCR text",
        "events": [
            {
                "type": "event",
                "title": "<event title>",
                "description": None,
                "start_time": "<ISO-8601 datetime>",
                "end_time": "<ISO-8601 datetime>",
                "all_day": False,
                "location": None,
            },
        ],
        "notes": ["<note text>"],
    }


def buildLlmExtractionPrompt(ocrText: str, *, timezone: str = "UTC") -> str:
    templateJson = json.dumps(getCalendarDraftTemplate(), indent=2)
    return (
        "Extract only events and notes from OCR text into STRICT JSON. "
        "Do not include markdown or explanation. Return only one JSON object matching this schema shape.\n\n"
        f"timezone to use: {timezone}\n"
        "Rules:\n"
        "- DO NOT return tasks\n"
        "- Event shape must be exactly: type, title, description, start_time, end_time, all_day, location\n"
        "- start_time and end_time must be ISO-8601 with timezone offset\n"
        "- if end_time is unknown, set it to start_time plus one hour\n"
        "- Put non-event actionable text into notes\n"
        "- Do NOT copy placeholder/example values from the schema; use OCR-derived values only\n"
        "- Keep unknown nullable fields as null\n\n"
        f"Target JSON shape example:\n{templateJson}\n\n"
        f"OCR text:\n{ocrText}\n"
    )


def _end_time_with_default(startTimeIso: str, endTimeIso: str | None) -> str:
    if endTimeIso:
        return endTimeIso

    parsedStart = datetime.fromisoformat(startTimeIso.replace("Z", "+00:00"))
    return (parsedStart + timedelta(hours=1)).isoformat()


def _extractJsonPayload(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _to_iso8601(value: str | None) -> str | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    parsedRaw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(parsedRaw)
    except ValueError:
        timeMatch = re.match(r"^(?P<hour>\d{1,2}):(\d{2})$", raw)
        if timeMatch:
            hour, minute = raw.split(":", maxsplit=1)
            now = datetime.now(timezone.utc)
            dt = datetime(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=int(hour),
                minute=int(minute),
                second=0,
                tzinfo=timezone.utc,
            )
        else:
            amPmMatch = re.match(r"^(?P<hour>\d{1,2})(:(?P<minute>\d{2}))?\s*(?P<ampm>AM|PM)$", raw, flags=re.IGNORECASE)
            if not amPmMatch:
                return None

            hour = int(amPmMatch.group("hour"))
            minute = int(amPmMatch.group("minute") or "0")
            ampm = amPmMatch.group("ampm").upper()

            if ampm == "PM" and hour != 12:
                hour += 12
            if ampm == "AM" and hour == 12:
                hour = 0

            now = datetime.now(timezone.utc)
            dt = datetime(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=hour,
                minute=minute,
                second=0,
                tzinfo=timezone.utc,
            )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _normalizeEventPayload(item: dict[str, Any], *, fallbackSourceText: str = "") -> dict[str, Any] | None:
    normalized = dict(item)
    sourceText = str(normalized.get("source_text") or normalized.get("sourceText") or fallbackSourceText or "").strip()
    title = str(normalized.get("title") or "").strip()
    if not title:
        title = sourceText
    if not title:
        return None

    startTime = _to_iso8601(normalized.get("start_time"))
    endTime = _to_iso8601(normalized.get("end_time"))
    if not startTime:
        return None

    try:
        normalizedEndTime = _end_time_with_default(startTime, endTime)
    except ValueError:
        return None

    return {
        "type": "event",
        "title": title,
        "description": None if normalized.get("description") in (None, "") else str(normalized.get("description")),
        "start_time": startTime,
        "end_time": normalizedEndTime,
        "all_day": bool(normalized.get("all_day") or False),
        "location": None if normalized.get("location") in (None, "") else str(normalized.get("location")),
    }


def _normalizeLegacyItemPayload(item: dict[str, Any], timezoneName: str) -> tuple[dict[str, Any] | None, str | None]:
    itemType = str(item.get("itemType") or "").strip().lower()

    if itemType != "event":
        noteText = str(item.get("sourceText") or item.get("description") or item.get("title") or "").strip()
        return None, noteText or None

    title = str(item.get("title") or "").strip()
    if not title:
        return None, None

    date = str(item.get("date") or "").strip()
    startTimeRaw = str(item.get("startTime") or "").strip()
    endTimeRaw = str(item.get("endTime") or "").strip()
    allDay = bool(item.get("allDay") or False)

    if allDay and date:
        startIso = _to_iso8601(f"{date}T00:00:00+00:00")
        endIso = _to_iso8601(f"{date}T23:59:59+00:00")
    elif date and startTimeRaw:
        startIso = _to_iso8601(f"{date}T{startTimeRaw}:00+00:00")
        endIso = _to_iso8601(f"{date}T{endTimeRaw}:00+00:00") if endTimeRaw else None
    else:
        startIso = None
        endIso = None

    if not startIso:
        noteText = str(item.get("sourceText") or item.get("description") or title).strip()
        return None, noteText or None

    try:
        normalizedEndIso = _end_time_with_default(startIso, endIso)
    except ValueError:
        return None, None

    return {
        "type": "event",
        "title": title,
        "description": None if item.get("description") in (None, "") else str(item.get("description")),
        "start_time": startIso,
        "end_time": normalizedEndIso,
        "all_day": allDay,
        "location": None if item.get("location") in (None, "") else str(item.get("location")),
    }, None


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9']+", value.lower()) if len(token) >= 3}


def _looks_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return False
    if "<" in lowered and ">" in lowered:
        return True
    placeholderMarkers = {
        "uuid-v4",
        "uuid",
        "event title",
        "note text",
        "iso-8601",
        "iso 8601",
    }
    return any(marker in lowered for marker in placeholderMarkers)


def _is_grounded_text(value: str, sourceTokens: set[str]) -> bool:
    if not sourceTokens:
        return True
    tokens = _tokenize(value)
    return bool(tokens.intersection(sourceTokens))


def _normalizeDraftPayload(payload: dict[str, Any], *, fallbackSourceText: str = "") -> dict[str, Any]:
    timezoneName = str(payload.get("timezone") or "UTC").strip() or "UTC"
    sourceText = str(payload.get("source_text") or payload.get("sourceText") or "").strip() or fallbackSourceText.strip()
    sourceTokens = _tokenize(sourceText)

    events: list[dict[str, Any]] = []
    notes: list[str] = []

    rawEvents = payload.get("events")
    if isinstance(rawEvents, list):
        for event in rawEvents:
            if not isinstance(event, dict):
                continue
            normalizedEvent = _normalizeEventPayload(event, fallbackSourceText=sourceText)
            if not normalizedEvent:
                continue

            title = str(normalizedEvent.get("title") or "")
            description = str(normalizedEvent.get("description") or "")
            if _looks_placeholder(title) or _looks_placeholder(description):
                continue
            if not _is_grounded_text(title + " " + description, sourceTokens):
                continue

            if normalizedEvent:
                events.append(normalizedEvent)

    rawNotes = payload.get("notes")
    if isinstance(rawNotes, list):
        for note in rawNotes:
            if isinstance(note, dict):
                noteText = str(
                    note.get("text")
                    or note.get("note")
                    or note.get("content")
                    or ""
                ).strip()
            else:
                noteText = str(note or "").strip()
            if not noteText:
                continue
            if _looks_placeholder(noteText):
                continue
            if not _is_grounded_text(noteText, sourceTokens):
                continue
            if noteText:
                notes.append(noteText)

    legacyItems = payload.get("items")
    if isinstance(legacyItems, list):
        for item in legacyItems:
            if not isinstance(item, dict):
                continue
            normalizedEvent, noteText = _normalizeLegacyItemPayload(item, timezoneName)
            if normalizedEvent:
                title = str(normalizedEvent.get("title") or "")
                description = str(normalizedEvent.get("description") or "")
                if not _looks_placeholder(title) and _is_grounded_text(title + " " + description, sourceTokens):
                    events.append(normalizedEvent)
            if noteText:
                if not _looks_placeholder(noteText) and _is_grounded_text(noteText, sourceTokens):
                    notes.append(noteText)

    eventTitles = {str(event.get("title") or "").strip() for event in events if isinstance(event, dict)}
    eventTitlesLower = {title.lower() for title in eventTitles if title}
    notes = [note for note in notes if note.strip().lower() not in eventTitlesLower]

    dedupedNotes = list(dict.fromkeys(notes))
    return {
        "timezone": timezoneName,
        "source_text": sourceText,
        "events": events,
        "notes": dedupedNotes,
    }


def parseCalendarDraftFromUnknown(raw: Any, *, fallbackSourceText: str = "") -> CalendarDraft:
    if isinstance(raw, CalendarDraft):
        return raw

    if isinstance(raw, str):
        payload = _extractJsonPayload(raw)
        try:
            parsed = json.loads(payload)
            normalized = _normalizeDraftPayload(parsed, fallbackSourceText=fallbackSourceText)
            return CalendarDraft.model_validate(normalized)
        except (json.JSONDecodeError, ValidationError):
            return CalendarDraft(timezone="UTC", source_text=fallbackSourceText or raw, events=[], notes=[])

    if isinstance(raw, dict):
        try:
            normalized = _normalizeDraftPayload(raw, fallbackSourceText=fallbackSourceText)
            return CalendarDraft.model_validate(normalized)
        except ValidationError:
            return CalendarDraft(timezone="UTC", source_text=fallbackSourceText, events=[], notes=[])

    return CalendarDraft(timezone="UTC", source_text=fallbackSourceText, events=[], notes=[])


