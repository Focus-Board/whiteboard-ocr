from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from whiteboardOCRService.models.trocr import TrOCRManager

app = FastAPI()
ocr = TrOCRManager()

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    imageBytes = await file.read()
    image = Image.open(io.BytesIO(imageBytes))

    text = ocr.predict(image)
    return {"text": text}