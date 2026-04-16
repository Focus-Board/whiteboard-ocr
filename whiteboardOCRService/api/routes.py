from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..core.config import API_PREFIX
from ..parsing import buildVjournalFromDraft, parseCalendarDraftFromUnknown
from ..schemas.jobs import (
    ApproveDraftRequest,
    ApproveDraftResponse,
    JobStatus,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitResponse,
)
from ..pipeline import buildOcrDebugArtifact
from ..utils.jobStore import jobStore, jobQueue

router = APIRouter(prefix=API_PREFIX, tags=["ocrJobs"])


@router.post("/debug/ocr-preview")
async def debugOcrPreview(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, object]:
    fileBytes = await file.read()
    if not fileBytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    ocrManager = request.app.state.ocrManager
    artifact = buildOcrDebugArtifact(fileBytes, ocrManager)
    
    # Build side-by-side comparison showing full pipeline visibility
    comparison = {
        "inputFileName": file.filename or "upload",
        "pipeline": {
            "stage1_ocrText": {
                "description": "Raw text extracted from image via TrOCR",
                "value": artifact.text,
            },
            "stage2_llmPrompt": {
                "description": "System + user prompt sent to local LLM for extraction",
                "value": artifact.llmPrompt,
                "model": artifact.llmModel,
            },
            "stage3_llmRawResponse": {
                "description": "Raw JSON response from phi3:mini (may have weak fields)",
                "value": artifact.llmRawResponse,
            },
            "stage4_normalizedDraft": {
                "description": "Normalized extraction output: events JSON + notes list",
                "value": {
                    "events": artifact.events,
                    "notesVjournal": artifact.notesVjournal,
                },
            },
        },
        "userPayload": {
            "description": "Final object returned to mobile app for user review",
            "text": artifact.text,
            "events": artifact.events,
            "notesVjournal": artifact.notesVjournal,
        },
        "workflow": {
            "currentStep": "User reviews events and notes",
            "nextStep": "POST /api/v1/jobs/{jobId}/approve with edited events/notes",
            "finalStep": "Server uploads returned ICS payload to e-ink endpoint",
        },
    }
    return comparison

@router.post("/jobs", response_model=JobSubmitResponse)
async def submitJob(
    deviceId: str = Form(...),
    timestamp: str = Form(...),
    file: UploadFile = File(...),
) -> JobSubmitResponse:
    fileBytes = await file.read()
    if not fileBytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    inputType = _resolveInputType(file)

    job = jobStore.createJob(
        deviceId=deviceId,
        timestamp=timestamp,
        inputType=inputType,
        fileName=file.filename or "upload",
        fileBytes=fileBytes,
    )

    try:
        jobQueue.put_nowait(job.jobId)
    except asyncio.QueueFull as exception:
        jobStore.deleteJob(job.jobId)
        raise HTTPException(status_code=503, detail="Server is busy. Please try again later.") from exception

    return JobSubmitResponse(jobId=job.jobId, status=job.status)

@router.get("/jobs/{jobId}", response_model=JobStatusResponse)
async def getJobStatus(jobId: str) -> JobStatusResponse:
    job = jobStore.getJob(jobId)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(jobId=job.jobId, status=job.status, error=job.error)

@router.get("/jobs/{jobId}/result", response_model=JobResultResponse)
async def getJobResult(jobId: str) -> JobResultResponse:
    job = jobStore.getJob(jobId)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == JobStatus.failed:
        return JobResultResponse(
            jobId=job.jobId,
            status=job.status,
            text="",
            events=[],
            notesVjournal="",
            structured={},
            error=job.error or "Unknown processing error",
        )

    if job.status != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"Job is not completed yet. Current status: {job.status}.")

    return JobResultResponse(
        jobId=job.jobId,
        status=job.status,
        text=job.text or "",
        events=(job.structured or {}).get("events", []),
        notesVjournal=(job.structured or {}).get("notesVjournal", ""),
        structured=job.structured or {},
        error=job.error,
    )


@router.post("/jobs/{jobId}/approve", response_model=ApproveDraftResponse)
async def approveJobDraft(jobId: str, payload: ApproveDraftRequest) -> ApproveDraftResponse:
    job = jobStore.getJob(jobId)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == JobStatus.failed:
        raise HTTPException(status_code=409, detail="Cannot approve a failed job")

    if job.status != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"Job is not completed yet. Current status: {job.status}.")

    parsedDraft = parseCalendarDraftFromUnknown(
        {
            "events": payload.events,
            "notes": payload.notes,
            "source_text": job.text or "",
        },
        fallbackSourceText=job.text or "",
    )
    approvedEvents = [event.model_dump() for event in parsedDraft.events]
    notesVjournal = buildVjournalFromDraft(parsedDraft)

    structured = dict(job.structured or {})
    structured["events"] = approvedEvents
    structured["notes"] = parsedDraft.notes
    structured["notesVjournal"] = notesVjournal
    structured["approval"] = {
        "approvedAt": datetime.now(timezone.utc).isoformat(),
        "status": "approved",
    }
    jobStore.updateStructured(jobId, structured=structured)

    return ApproveDraftResponse(
        jobId=job.jobId,
        status=job.status,
        approvedEvents=approvedEvents,
        notesVjournal=notesVjournal,
        uploadArtifact={
            "contentType": "text/calendar",
            "fileName": f"{job.jobId}.ics",
            "payload": notesVjournal,
        },
    )

def _resolveInputType(file: UploadFile) -> str:
    contentType = (file.content_type or "").lower()
    fileName = (file.filename or "").lower()

    if contentType == "application/pdf" or fileName.endswith(".pdf"):
        return "pdf"

    if contentType.startswith("image/"):
        return "image"

    raise HTTPException(status_code=415, detail="Unsupported file type. Only images and PDFs are accepted.")