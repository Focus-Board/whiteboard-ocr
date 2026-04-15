from __future__ import annotations

import asyncio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..core.config import API_PREFIX
from ..schemas.jobs import JobStatus, JobResultResponse, JobSubmitResponse, JobStatusResponse
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
                "description": "Calendar draft after normalization (repairs weak LLM fields)",
                "value": artifact.calendarDraft,
            },
        },
        "userPayload": {
            "description": "Final object returned to mobile app for user review",
            "text": artifact.text,
            "calendarDraft": artifact.calendarDraft,
        },
        "serverArtifact": {
            "description": "Server-side only (not exposed to app)",
            "vjournal": artifact.vjournal,
        },
        "workflow": {
            "currentStep": "User reviews calendarDraft and edits (if needed)",
            "nextStep": "POST /api/v1/jobs/{jobId}/approve with edited draft",
            "finalStep": "Server appends approved VJOURNAL to e-ink device",
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
            calendarDraft={},
            structured={},
            error=job.error or "Unknown processing error",
        )

    if job.status != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"Job is not completed yet. Current status: {job.status}.")

    return JobResultResponse(
        jobId=job.jobId,
        status=job.status,
        text=job.text or "",
        calendarDraft=(job.structured or {}).get("calendarDraft", {}),
        structured=job.structured or {},
        error=job.error,
    )

def _resolveInputType(file: UploadFile) -> str:
    contentType = (file.content_type or "").lower()
    fileName = (file.filename or "").lower()

    if contentType == "application/pdf" or fileName.endswith(".pdf"):
        return "pdf"

    if contentType.startswith("image/"):
        return "image"

    raise HTTPException(status_code=415, detail="Unsupported file type. Only images and PDFs are accepted.")