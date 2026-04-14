import threading

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCRManager:
    def __init__(self, modelName: str = "microsoft/trocr-large-handwritten") -> None:
        self.modelName = modelName
        self.device = torch.device("cpu")
        self._lock = threading.Lock()
        self._isLoaded = False
        self.processor = None
        self.model = None

    def load(self) -> None:
        with self._lock:
            if self._isLoaded:
                return

            print(f"Loading TrOCR model on device: {self.device}")
            self.processor = TrOCRProcessor.from_pretrained(self.modelName)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.modelName).to(self.device)
            self.model.eval()
            self._isLoaded = True

    def predict(self, image: Image.Image) -> str:
        if not self._isLoaded:
            raise RuntimeError("TrOCR model is not loaded. Call load() at startup.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixelValues = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.inference_mode():
            generatedIds = self.model.generate(pixelValues, max_new_tokens=256)

        text = self.processor.batch_decode(generatedIds, skip_special_tokens=True)[0]
        return text.strip()