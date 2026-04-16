from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from typing import Iterable

from .calendarDraft import CalendarDraft


def _escapeIcalText(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _foldIcalLine(line: str, limit: int = 75) -> list[str]:
    if len(line) <= limit:
        return [line]

    chunks = [line[:limit]]
    remainder = line[limit:]
    while remainder:
        chunks.append(" " + remainder[: limit - 1])
        remainder = remainder[limit - 1 :]
    return chunks


def _foldIcalLines(lines: Iterable[str]) -> str:
    folded: list[str] = []
    for line in lines:
        folded.extend(_foldIcalLine(line))
    return "\r\n".join(folded) + "\r\n"


def buildVjournalDocument(*, text: str, summary: str | None = None, sourceLabel: str = "whiteboard-ocr") -> str:
    normalizedText = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    summaryText = summary.strip() if summary else (normalizedText.splitlines()[0] if normalizedText else "Whiteboard note")

    nowUtc = datetime.now(timezone.utc)
    dtstamp = nowUtc.strftime("%Y%m%dT%H%M%SZ")
    uidSeed = f"{sourceLabel}|{normalizedText}|{dtstamp}".encode("utf-8")
    uid = sha1(uidSeed).hexdigest()

    bodyLines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Focus Board//Whiteboard OCR//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VJOURNAL",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"SUMMARY:{_escapeIcalText(summaryText)}",
        f"DESCRIPTION:{_escapeIcalText(normalizedText or text.strip())}",
        "END:VJOURNAL",
        "END:VCALENDAR",
    ]
    return _foldIcalLines(bodyLines)


def buildVjournalFromDraft(draft: CalendarDraft, *, sourceLabel: str = "whiteboard-ocr") -> str:
    return buildVjournalFromNotes(draft.notes, sourceLabel=sourceLabel)


def buildVjournalFromNotes(notes: list[str], *, sourceLabel: str = "whiteboard-ocr") -> str:
    cleanedNotes = [note.strip() for note in notes if note and note.strip()]

    if not cleanedNotes:
        return ""

    body = "\n".join(f"{index}. {note}" for index, note in enumerate(cleanedNotes, start=1))
    summary = cleanedNotes[0][:120]

    return buildVjournalDocument(
        text=body,
        summary=summary,
        sourceLabel=sourceLabel,
    )