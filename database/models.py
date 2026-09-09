"""
SQLAlchemy ORM Data Models for Quality Inspection Analytics.
Stores inspection logs, defect telemetry, and manufacturing line metrics.
Compatible with both PostgreSQL and SQLite.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class InspectionRecord(Base):
    """
    Primary log entry representing an individual manufactured component inspection.
    """
    __tablename__ = "inspection_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    part_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    batch_number = Column(String(64), index=True, default="BATCH-DEFAULT")
    camera_id = Column(String(32), default="CAM-INSPECT-01")
    shift = Column(String(16), default="Shift-A")  # Shift-A, Shift-B, Shift-C
    status = Column(String(16), nullable=False)    # 'PASS' or 'FAIL'
    is_defective = Column(Boolean, default=False)
    defect_count = Column(Integer, default=0)
    highest_severity = Column(String(16), default="None")
    avg_confidence = Column(Float, default=0.0)
    inference_time_ms = Column(Float, default=0.0)
    image_name = Column(String(128), nullable=True)
    notes = Column(Text, nullable=True)

    # One-to-many relationship with detected defects
    defects = relationship("DefectDetail", back_populates="inspection", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "part_id": self.part_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "batch_number": self.batch_number,
            "camera_id": self.camera_id,
            "shift": self.shift,
            "status": self.status,
            "is_defective": self.is_defective,
            "defect_count": self.defect_count,
            "highest_severity": self.highest_severity,
            "avg_confidence": round(self.avg_confidence, 4),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_name": self.image_name,
            "notes": self.notes,
            "defects": [d.to_dict() for d in self.defects]
        }


class DefectDetail(Base):
    """
    Granular defect localization and severity records linked to an inspection.
    """
    __tablename__ = "defect_details"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(Integer, ForeignKey("inspection_records.id", ondelete="CASCADE"), nullable=False)
    defect_type = Column(String(32), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String(16), default="Minor")  # Minor, Moderate, Critical
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    area_px = Column(Integer, default=0)

    inspection = relationship("InspectionRecord", back_populates="defects")

    def to_dict(self):
        return {
            "id": self.id,
            "defect_type": self.defect_type,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "bounding_box": {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2},
            "area_px": self.area_px
        }
