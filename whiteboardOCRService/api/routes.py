from __future__ import annotations

import asyncio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..core.config import API_PREFIX
from ..schemas.jobs import JobStatus, JobResultResponse, JobSubmitResponse, JobStatusResponse
from ..utils.jobStore import jobStore, jobQueue

router = APIRouter(prefix=API_PREFIX, tags=["ocrJobs"])

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
    return JobStatusResponse(jobId=job.jobId, status=job.status)

@router.get("/jobs/{jobId}/result", response_model=JobResultResponse)
async def getJobResult(jobId: str) -> JobResultResponse:
    job = jobStore.getJob(jobId)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"Job is not completed yet. Current status: {job.status}.")
    return JobResultResponse(
        jobId=job.jobId,
        status=job.status,
        text=job.text or "",
        structured=job.structured or {},
    )

def _resolveInputType(file: UploadFile) -> str:
    contentType = (file.content_type or "").lower()
    fileName = (file.filename or "").lower()

    if contentType == "application/pdf" or fileName.endswith(".pdf"):
        return "pdf"

    if contentType.startswith("image/"):
        return "image"

    raise HTTPException(status_code=415, detail="Unsupported file type. Only images and PDFs are accepted.")