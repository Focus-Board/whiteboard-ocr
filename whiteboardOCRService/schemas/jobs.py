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
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="User-reviewable event JSON list (ISO-8601 fields)",
    )
    notesVjournal: str = Field(
        default="",
        description="Server-generated VJOURNAL document containing extracted notes",
    )
    structured: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional internal structured data",
    )
    error: str | None = Field(default=None, description="Error message when status is 'failed'")


class ApproveDraftRequest(BaseModel):
    """Request body for approving reviewed events/notes payload."""
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Reviewed or edited event list from the app",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Reviewed or edited note list from the app",
    )


class ApproveDraftResponse(BaseModel):
    """Response after approving events/notes and generating upload artifact."""
    jobId: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    approvedEvents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Validated and normalized events accepted by the server",
    )
    notesVjournal: str = Field(
        default="",
        description="Generated VJOURNAL from approved notes",
    )
    uploadArtifact: dict[str, str] = Field(
        default_factory=dict,
        description="Upload-ready calendar artifact to pass to main server display endpoint",
    )
