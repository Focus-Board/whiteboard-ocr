from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    lineImages: List[np.ndarray]
    valleys: List[int]
    estimatedLineHeight: int


def segmentLines(image: np.ndarray) -> SegmentationResult:
    """
    Segment a full-page/whiteboard image into line crops ordered top->bottom.
    Returns grayscale line crops.
    """
    gray = _toGray(image)
    binary = _binarize(gray)  # text = 255, background = 0

    estimatedLineHeight = _estimateLineHeight(binary)
    valleys = _findValleys(binary, estimatedLineHeight)

    lineImages = _extractLineRegions(gray, binary, valleys, estimatedLineHeight)

    if not lineImages:
        lineImages = [gray.copy()]

    return SegmentationResult(
        lineImages=lineImages,
        valleys=valleys,
        estimatedLineHeight=estimatedLineHeight,
    )


def _toGray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binarize(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    # Remove tiny noise and slightly connect strokes.
    kernelOpen = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernelClose = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernelOpen, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernelClose, iterations=1)
    return binary


def _estimateLineHeight(binary: np.ndarray) -> int:
    numLabels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights: List[int] = []

    for i in range(1, numLabels):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Ignore tiny noise and huge blobs.
        if area < 20:
            continue
        if h < 5 or h > binary.shape[0] // 2:
            continue
        if w < 2:
            continue
        heights.append(h)

    if not heights:
        return 30

    medianH = int(np.median(np.array(heights, dtype=np.int32)))
    return max(20, min(80, int(medianH * 1.6)))


def _findValleys(binary: np.ndarray, estimatedLineHeight: int) -> List[int]:
    # Horizontal projection of ink pixels.
    histogram = np.sum(binary > 0, axis=1).astype(np.float32)

    # Smooth histogram to reduce local noise.
    smooth = cv2.GaussianBlur(histogram.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=3).ravel()
    h = smooth.shape[0]

    if h < 5:
        return []

    threshold = float(np.percentile(smooth, 25))
    minDistance = max(10, int(estimatedLineHeight * 0.6))

    candidates: List[int] = []
    for i in range(1, h - 1):
        if smooth[i] <= threshold and smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            candidates.append(i)

    # Keep valleys separated by minDistance (pick deeper valley in each group).
    valleys: List[int] = []
    groupStart = 0
    while groupStart < len(candidates):
        groupEnd = groupStart + 1
        while groupEnd < len(candidates) and candidates[groupEnd] - candidates[groupEnd - 1] <= minDistance:
            groupEnd += 1

        group = candidates[groupStart:groupEnd]
        best = min(group, key=lambda idx: smooth[idx])
        valleys.append(int(best))
        groupStart = groupEnd

    return valleys


def _extractLineRegions(
    gray: np.ndarray,
    binary: np.ndarray,
    valleys: List[int],
    estimatedLineHeight: int,
) -> List[np.ndarray]:
    h, _ = gray.shape
    boundaries = [0] + sorted([v for v in valleys if 0 < v < h]) + [h]

    lineImages: List[np.ndarray] = []
    minHeight = max(12, int(estimatedLineHeight * 0.35))
    pad = max(2, int(estimatedLineHeight * 0.1))

    for i in range(len(boundaries) - 1):
        top = max(0, boundaries[i] - pad)
        bottom = min(h, boundaries[i + 1] + pad)
        if bottom - top < minHeight:
            continue

        regionBinary = binary[top:bottom, :]
        inkRatio = float(np.count_nonzero(regionBinary)) / float(regionBinary.size)

        # Skip empty gaps.
        if inkRatio < 0.003:
            continue

        crop = gray[top:bottom, :]
        lineImages.append(crop)

    return lineImages