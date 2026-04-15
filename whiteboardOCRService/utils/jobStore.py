from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..core.config import JOB_QUEUE_MAX_SIZE
from ..schemas.jobs import JobStatus


@dataclass
class JobRecord:
    jobId: str
    status: JobStatus
    deviceId: str
    timestamp: str
    inputType: str
    fileName: str
    fileBytes: bytes = field(repr=False)
    text: str | None = None
    structured: dict[str, Any] | None = None
    error: str | None = None
    createdAt: float = field(default_factory=time.time)
    updatedAt: float = field(default_factory=time.time)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def createJob(
        self,
        *,
        deviceId: str,
        timestamp: str,
        inputType: str,
        fileName: str,
        fileBytes: bytes,
    ) -> JobRecord:
        jobId = str(uuid.uuid4())
        job = JobRecord(
            jobId=jobId,
            status=JobStatus.queued,
            deviceId=deviceId,
            timestamp=timestamp,
            inputType=inputType,
            fileName=fileName,
            fileBytes=fileBytes,
        )
        with self._lock:
            self._jobs[jobId] = job
        return job

    def getJob(self, jobId: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(jobId)

    def markProcessing(self, jobId: str) -> None:
        self._updateJob(jobId, status=JobStatus.processing)

    def markDone(self, jobId: str, *, text: str, structured: dict[str, Any]) -> None:
        self._updateJob(jobId, status=JobStatus.done, text=text, structured=structured)

    def updateStructured(self, jobId: str, *, structured: dict[str, Any]) -> None:
        self._updateJob(jobId, structured=structured)

    def markFailed(self, jobId: str, error: str) -> None:
        self._updateJob(jobId, status=JobStatus.failed, error=error)

    def deleteJob(self, jobId: str) -> None:
        with self._lock:
            self._jobs.pop(jobId, None)

    def _updateJob(self, jobId: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(jobId)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updatedAt = time.time()


jobStore = JobStore()
jobQueue: asyncio.Queue[str] = asyncio.Queue(maxsize=JOB_QUEUE_MAX_SIZE)
