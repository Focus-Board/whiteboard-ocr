import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TrOCRManager:
    _instance = None

    def __new__(cls, model_name = "microsoft/trocr-base-handwritten"):
        if cls._instance is None:
            cls._instance = super(TrOCRManager, cls).__new__(cls)
            cls._instance._initialize(model_name)
        return cls._instance
    
    def __init__(self, model_name = "microsoft/trocr-base-handwritten"):
        if self._initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TrOCR model on device: {self.device}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._initialized = True
    
    def predict(self, image: Image.Image) -> str:
        """
        Perform OCR on a Single Line Image
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=256)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text