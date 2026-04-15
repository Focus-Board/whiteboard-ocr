import asyncio
from contextlib import suppress

from fastapi import FastAPI, UploadFile, File

from whiteboardOCRService.api.routes import router as jobsRouter
from whiteboardOCRService.core.config import TROCR_ALLOW_ONLINE_FALLBACK, TROCR_LOCAL_FILES_ONLY
from whiteboardOCRService.models.trocr import TrOCRManager
from whiteboardOCRService.pipeline import processImageBytes, runJobWorker

app = FastAPI(title="WhiteboardUnderstandingService", version="0.1.0")
ocrManager = TrOCRManager()
workerTask = None

@app.on_event("startup")
async def onStartup() -> None:
    ocrManager.load(
        localFilesOnly=TROCR_LOCAL_FILES_ONLY,
        allowOnlineFallback=TROCR_ALLOW_ONLINE_FALLBACK,
    )
    app.state.ocrManager = ocrManager

    global workerTask
    if workerTask is None or workerTask.done():
        workerTask = asyncio.create_task(runJobWorker(ocrManager))


@app.on_event("shutdown")
async def onShutdown() -> None:
    global workerTask
    if workerTask is None:
        return

    workerTask.cancel()
    with suppress(asyncio.CancelledError):
        await workerTask
    workerTask = None
    app.state.ocrManager = ocrManager

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/ocr")
async def performOcr(file: UploadFile = File(...)) -> dict[str, object]:
    imageBytes = await file.read()
    result = processImageBytes(imageBytes, ocrManager)
    return {
        "text": result.text,
        "calendarDraft": result.structured.get("calendarDraft", {}),
    }

app.include_router(jobsRouter)