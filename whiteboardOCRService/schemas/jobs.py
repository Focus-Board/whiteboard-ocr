from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job lifecycle statuses."""
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class JobSubmitResponse(BaseModel):
    """Response when a user submits an image for processing."""
    jobId: str = Field(..., description="Unique job identifier (UUID)")
    status: JobStatus = Field(JobStatus.queued, description="Initial status after submission")


class JobStatusResponse(BaseModel):
    """Response when checking job status via polling."""
    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    error: str | None = Field(default=None, description="Error message when status is 'failed'")


class JobResultResponse(BaseModel):
    """Response when the job is complete and result is retrieved."""
    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Final or current job status")
    text: str = Field(default="", description="Raw extracted text from OCR")
    structured: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed structure: tasks, notes, calendarItems, etc.",
    )
    error: str | None = Field(default=None, description="Error message when status is 'failed'")
