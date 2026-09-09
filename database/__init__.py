"""
Database Persistence & Analytics Package.
AI-Based Quality Inspection System for Manufacturing.
"""

from .db import engine, SessionLocal, get_db, init_db
from .models import Base, InspectionRecord, DefectDetail
from .repository import InspectionRepository

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "Base",
    "InspectionRecord",
    "DefectDetail",
    "InspectionRepository"
]
