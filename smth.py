import cv2
import numpy as np
from PIL import Image
import easyocr

from whiteboardOCRService.models.trocr import TrOCRManager


def _gray_world_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    avg_b = max(float(np.mean(b)), 1.0)
    avg_g = max(float(np.mean(g)), 1.0)
    avg_r = max(float(np.mean(r)), 1.0)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    b *= avg_gray / avg_b
    g *= avg_gray / avg_g
    r *= avg_gray / avg_r

    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def preprocess_for_detection(img_bgr):
    # Global normalization for different board colors and lighting conditions.
    wb = _gray_world_white_balance(img_bgr)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Light sharpening helps detector lock onto marker edges.
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
    return sharpened


def preprocess_for_trocr(line_bgr):
    h, w = line_bgr.shape[:2]
    if h == 0 or w == 0:
        return line_bgr

    # Upscale tiny crops so character shapes are easier for TrOCR to decode.
    target_h = 96
    if h < target_h:
        scale = target_h / float(h)
        new_w = max(int(w * scale), 1)
        line_bgr = cv2.resize(line_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Keep a soft blend of grayscale and adaptive threshold to preserve stroke details.
    bin_img = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    soft = cv2.addWeighted(enhanced, 0.65, bin_img, 0.35, 0)
    rgb = cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB)
    return rgb


def group_boxes_into_lines(boxes):
    if not boxes:
        return []
    
    rects = []
    heights = []
    for b in boxes:
        x1, x2, y1, y2 = map(int, b)
        h = max(1, y2 - y1)
        rects.append((x1, x2, y1, y2, (y1 + y2) / 2.0, h))
        heights.append(h)

    median_h = float(np.median(heights)) if heights else 20.0
    y_thresh = max(12.0, 0.75 * median_h)

    rects.sort(key=lambda r: r[4])  # sort by y-center
    lines = []

    for r in rects:
        placed = False
        for line in lines:
            line_y1 = min(b[2] for b in line["boxes"])
            line_y2 = max(b[3] for b in line["boxes"])
            inter = max(0, min(r[3], line_y2) - max(r[2], line_y1))
            min_h = max(1, min(r[5], line_y2 - line_y1))
            overlap = inter / float(min_h)

            if abs(r[4] - line["y_center"]) <= y_thresh or overlap >= 0.35:
                line["boxes"].append(r)
                ys = [b[4] for b in line["boxes"]]
                line["y_center"] = float(np.mean(ys))
                placed = True
                break
        if not placed:
            lines.append({"y_center": r[4], "boxes": [r]})

    # sort boxes left->right inside each line, then sort lines top->bottom
    for line in lines:
        line["boxes"].sort(key=lambda r: r[0])
    lines.sort(key=lambda l: l["y_center"])

    return lines    

def merge_line_boxes(line_boxes, img_h, img_w, pad=6):
    x1 = min(b[0] for b in line_boxes)
    x2 = max(b[1] for b in line_boxes)
    y1 = min(b[2] for b in line_boxes)
    y2 = max(b[3] for b in line_boxes)

    # Slightly asymmetric padding keeps ascenders/descenders and punctuation.
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - int(1.3 * pad))
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + int(1.4 * pad))

    return x1, y1, x2, y2

def main():
    image_path = "whiteboardtexttoosmall.png"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    detect_img = preprocess_for_detection(img)

    reader = easyocr.Reader(["en"], detector=True, recognizer=False, gpu=False)
    h_list, _ = reader.detect(detect_img)
    boxes = h_list[0] if h_list and len(h_list) > 0 else []

    print("Detected word boxes:", len(boxes))

    lines = group_boxes_into_lines(boxes)
    print("Merged lines:", len(lines))

    ocr = TrOCRManager(modelName="microsoft/trocr-large-handwritten")
    ocr.load(localFilesOnly=False, allowOnlineFallback=True)

    debug = img.copy()
    outputs = []

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = merge_line_boxes(line["boxes"], img.shape[0], img.shape[1], pad=8)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        trocr_ready = preprocess_for_trocr(crop)
        pil_img = Image.fromarray(trocr_ready)
        text = ocr.predict(pil_img)
        cleaned = " ".join(text.split())
        outputs.append(cleaned)

        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        print(f"line {i:02d}: {cleaned}")

    print("\n=== FINAL TEXT ===")
    print("\n".join(outputs))

    cv2.imwrite("debug_craft/detection_input.jpg", detect_img)
    cv2.imwrite("debug_craft/line_boxes_for_trocr.jpg", debug)

if __name__ == "__main__":
    main()