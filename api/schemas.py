"""
Pydantic Validation Schemas for FastAPI Endpoints.
Defines data structures for image inspection requests, defect predictions,
and PostgreSQL analytical queries.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class BoundingBoxCoordinates(BaseModel):
    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")
    width: int
    height: int


class DefectPrediction(BaseModel):
    defect_type: str = Field(..., example="scratch")
    confidence: float = Field(..., example=0.925)
    severity: str = Field(..., example="Minor")
    area_px: int = Field(..., example=420)
    bounding_box: Dict[str, int]


class InspectionResponse(BaseModel):
    part_id: str = Field(..., example="PART-2026-901")
    status: str = Field(..., example="FAIL")
    is_defective: bool = Field(..., example=True)
    defects_found: int = Field(..., example=1)
    highest_severity: str = Field(..., example="Minor")
    inference_time_ms: float = Field(..., example=24.5)
    defect_summary: Dict[str, int] = Field(default_factory=dict)
    defects: List[Dict[str, Any]] = Field(default_factory=list)


class AnalyticsSummaryResponse(BaseModel):
    total_inspected: int
    total_passed: int
    total_failed: int
    yield_rate_percent: float
    avg_latency_ms: float
    defect_counts: Dict[str, int]
    severity_breakdown: Dict[str, int]


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "ai-quality-inspection-service"
    version: str = "1.0.0"
    model_loaded: bool = True
    database_connected: bool = True
