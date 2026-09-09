"""
Core Computer Vision and Defect Detection Package.
AI-Based Quality Inspection System for Manufacturing.
"""

from .detector import DefectDetector, DefectDetectionResult, BoundingBox
from .opencv_pipeline import OpenCVPipeline
from .metrics import InspectionMetricsCalculator

__all__ = [
    "DefectDetector",
    "DefectDetectionResult",
    "BoundingBox",
    "OpenCVPipeline",
    "InspectionMetricsCalculator"
]
