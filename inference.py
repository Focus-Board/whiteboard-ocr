from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)

image = Image.open('line.jpeg').convert('RGB')

pixelValues = processor(image, return_tensors='pt').pixel_values.to(device)

generatedIDs = model.generate(pixelValues, max_new_tokens=256, num_beams=4, early_stopping=True)
text = processor.batch_decode(generatedIDs, skip_special_tokens=True)[0]
print(text)