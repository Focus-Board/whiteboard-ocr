from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..parsing import CalendarDraft, buildLlmExtractionPrompt, parseCalendarDraftFromUnknown


@dataclass(frozen=True)
class LocalLlmResult:
    modelName: str
    prompt: str
    rawResponse: str
    parsedDraft: CalendarDraft


class LocalJsonLlmManager:
    def __init__(
        self,
        *,
        modelName: str | None = None,
        baseUrl: str | None = None,
        timeoutSeconds: int | None = None,
    ) -> None:
        self.modelName = modelName or os.environ.get("WHITEBOARD_LOCAL_LLM_MODEL", "phi3:mini")
        self.baseUrl = (baseUrl or os.environ.get("WHITEBOARD_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeoutSeconds = timeoutSeconds or int(os.environ.get("WHITEBOARD_LOCAL_LLM_TIMEOUT_SECONDS", "120"))

    def generateDraft(self, ocrText: str, *, timezone: str = "UTC") -> LocalLlmResult:
        prompt = buildLlmExtractionPrompt(ocrText, timezone=timezone)
        rawResponse = self._chat(prompt)
        parsedDraft = parseCalendarDraftFromUnknown(rawResponse, fallbackSourceText=ocrText)
        return LocalLlmResult(
            modelName=self.modelName,
            prompt=prompt,
            rawResponse=rawResponse,
            parsedDraft=parsedDraft,
        )

    def _chat(self, prompt: str) -> str:
        url = f"{self.baseUrl}/api/chat"
        payload = {
            "model": self.modelName,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict information extraction engine. "
                        "Return only valid JSON and no markdown. "
                        "Extract only events and notes into the provided JSON shape. "
                        "Do not create tasks. "
                        "Event times must be ISO-8601 with timezone offset."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "options": {
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeoutSeconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Local LLM request failed: {exc}") from exc

        try:
            responseJson = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Local LLM returned invalid JSON transport payload: {body}") from exc

        message = responseJson.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Local LLM response missing assistant content: {responseJson}")

        return content


localJsonLlmManager = LocalJsonLlmManager()
