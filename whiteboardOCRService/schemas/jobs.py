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
    calendarDraft: dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized JSON draft from OCR extraction for user review",
    )
    structured: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed structure: tasks, notes, calendarItems, etc.",
    )
    error: str | None = Field(default=None, description="Error message when status is 'failed'")


class ApproveDraftRequest(BaseModel):
    """Request body for approving a reviewed/edited calendar draft."""
    calendarDraft: dict[str, Any] = Field(
        default_factory=dict,
        description="Reviewed or edited calendar draft provided by the app",
    )


class ApproveDraftResponse(BaseModel):
    """Response after approving a draft and generating server upload artifact."""
    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    approvedCalendarDraft: dict[str, Any] = Field(
        default_factory=dict,
        description="Validated and normalized draft accepted by the server",
    )
    uploadArtifact: dict[str, str] = Field(
        default_factory=dict,
        description="Upload-ready calendar artifact to pass to main server display endpoint",
    )
