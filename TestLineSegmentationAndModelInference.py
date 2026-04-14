import cv2
from pathlib import Path
from PIL import Image

from whiteboardOCRService.pipeline.LineSegmentation import segmentLines
from whiteboardOCRService.models.trocr import TrOCRManager


def main() -> None:
    imagePath = "whiteboardtexttoosmall.png"
    img = cv2.imread(imagePath)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {imagePath}")

    res = segmentLines(img)

    outDir = Path("debug_lines")
    outDir.mkdir(exist_ok=True)

    ocr = TrOCRManager()
    ocr.load()

    lineTexts: list[str] = []
    for i, crop in enumerate(res.lineImages):
        linePath = outDir / f"line_{i:02d}.png"
        cv2.imwrite(str(linePath), crop)

        lineImage = Image.fromarray(crop)
        text = ocr.predict(lineImage)
        lineTexts.append(text)
        print(f"line {i:02d}: {text}")

    combinedText = "\n".join(lineTexts).strip()
    print("\n=== OCR OUTPUT ===")
    print(combinedText)

    (outDir / "ocr_output.txt").write_text(combinedText, encoding="utf-8")

    print("\nlines:", len(res.lineImages))
    print("valleys:", res.valleys[:10])
    print("estimatedLineHeight:", res.estimatedLineHeight)


if __name__ == "__main__":
    main()