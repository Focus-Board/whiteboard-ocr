from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from whiteboardOCRService.api.routes import router as jobsRouter
from whiteboardOCRService.models.trocr import TrOCRManager

app = FastAPI(title="WhiteboardUnderstandingService", version="0.1.0")
ocrManager = TrOCRManager()

@app.on_event("startup")
async def onStartup() -> None:
    ocrManager.load()

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/ocr")
async def performOcr(file: UploadFile = File(...)) -> dict[str, str]:
    imageBytes = await file.read()
    image = Image.open(io.BytesIO(imageBytes))
    text = ocrManager.predict(image)
    return {"text": text}

app.include_router(jobsRouter)