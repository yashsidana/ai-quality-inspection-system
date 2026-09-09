"""
OpenCV Image Processing and Defect Feature Extraction Pipeline.
Provides CLAHE contrast enhancement, contour analysis, defect heatmaps,
and side-by-side inspection visualizers.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


class OpenCVPipeline:
    """
    Industrial Computer Vision processing pipeline for surface quality inspection.
    """

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB color space
        to highlight faint scratches, cracks, and surface imperfections.
        """
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def generate_defect_heatmap(image: np.ndarray) -> np.ndarray:
        """
        Computes a gradient anomaly heatmap of the manufacturing part
        revealing structural discontinuities, stress fractures, and micro-cracks.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel gradient computation (dx and dy)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize to 0-255
        norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply false-color heatmap (COLORMAP_JET)
        heatmap = cv2.applyColorMap(norm_magnitude, cv2.COLORMAP_JET)

        # Alpha blend with original image for HUD inspection overlay
        overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
        return overlay

    @staticmethod
    def compute_edge_mask(image: np.ndarray, low_thresh: int = 40, high_thresh: int = 140) -> np.ndarray:
        """
        Generates Canny edge contours for structural defect inspection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def extract_defect_crops(image: np.ndarray, boxes: List[Any], padding: int = 8) -> List[Dict[str, Any]]:
        """
        Crops localized defect areas for individual thumbnail inspection.
        """
        crops = []
        h, w = image.shape[:2]

        for idx, box in enumerate(boxes):
            x1 = max(0, box.x1 - padding)
            y1 = max(0, box.y1 - padding)
            x2 = min(w, box.x2 + padding)
            y2 = min(h, box.y2 + padding)

            crop_img = image[y1:y2, x1:x2]
            if crop_img.size > 0:
                crops.append({
                    "index": idx + 1,
                    "defect_type": box.defect_type,
                    "confidence": box.confidence,
                    "severity": box.severity,
                    "crop_image": crop_img,
                    "dimensions": f"{box.x2 - box.x1}x{box.y2 - box.y1} px"
                })
        return crops

    @staticmethod
    def create_side_by_side(raw_image: np.ndarray, annotated_image: np.ndarray, heatmap_image: np.ndarray) -> np.ndarray:
        """
        Stitches Raw, Heatmap, and Annotated images side-by-side into an industrial comparison strip.
        """
        h1, w1 = raw_image.shape[:2]
        target_h = 360
        target_w = int(w1 * (target_h / h1))

        resized_raw = cv2.resize(raw_image, (target_w, target_h))
        resized_heat = cv2.resize(heatmap_image, (target_w, target_h))
        resized_ann = cv2.resize(annotated_image, (target_w, target_h))

        # Add headers to each view
        cv2.putText(resized_raw, "RAW CAMERA FEED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(resized_heat, "CV GRADIENT HEATMAP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(resized_ann, "YOLOv8 DETECTIONS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Concatenate horizontally
        return np.hstack([resized_raw, resized_heat, resized_ann])
