import threading
from typing import Any

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRManager:
    def __init__(self, modelName: str = "microsoft/trocr-large-handwritten") -> None:
        self.modelName = modelName
        self.device = torch.device("cpu")
        self._lock = threading.Lock()
        self._isLoaded = False
        self.processor: TrOCRProcessor | None = None
        self.model: VisionEncoderDecoderModel | None = None

    def load(self, *, localFilesOnly: bool = False, allowOnlineFallback: bool = True) -> None:
        with self._lock:
            if self._isLoaded:
                return

            print(f"Loading TrOCR model on device: {self.device}")

            try:
                self._loadArtifacts(localFilesOnly=localFilesOnly)
            except Exception as exc:
                if localFilesOnly and allowOnlineFallback:
                    print("Local cache load failed; retrying with online download enabled.")
                    self._loadArtifacts(localFilesOnly=False)
                elif localFilesOnly:
                    raise RuntimeError(
                        "Model files are not fully cached locally and offline mode is enabled. "
                        "Run once with localFilesOnly=False to download artifacts, then retry offline."
                    ) from exc
                else:
                    raise

            assert self.model is not None
            self.model.eval()
            self._isLoaded = True

    def _loadArtifacts(self, *, localFilesOnly: bool) -> None:
        commonKwargs: dict[str, Any] = {"local_files_only": localFilesOnly}
        self.processor = TrOCRProcessor.from_pretrained(
            self.modelName,
            use_fast=False,
            **commonKwargs,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.modelName,
            **commonKwargs,
        ).to(self.device)

    def predict(self, image: Image.Image) -> str:
        if not self._isLoaded:
            raise RuntimeError("TrOCR model is not loaded. Call load() at startup.")

        assert self.processor is not None
        assert self.model is not None

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixelValues = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.inference_mode():
            generatedIds = self.model.generate(pixelValues, max_new_tokens=256)

        text = self.processor.batch_decode(generatedIds, skip_special_tokens=True)[0]
        return text.strip()