"""
Repository Layer for Quality Inspection Analytics.
Performs CRUD operations, SQL querying, defect Pareto aggregations,
and automated seed generation.
"""

from datetime import datetime, timedelta
import random
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from .models import InspectionRecord, DefectDetail
from .db import get_db, init_db


class InspectionRepository:
    """
    Data access and analytics reporting repository.
    """

    @staticmethod
    def save_inspection(
        part_id: str,
        status: str,
        is_defective: bool,
        defect_count: int,
        highest_severity: str,
        avg_confidence: float,
        inference_time_ms: float,
        defects_data: List[Dict[str, Any]],
        batch_number: str = "BATCH-A1",
        camera_id: str = "CAM-INSPECT-01",
        shift: str = "Shift-A",
        image_name: Optional[str] = None
    ) -> InspectionRecord:
        """Saves a new inspection log with associated defect items."""
        init_db()
        with get_db() as db:
            record = InspectionRecord(
                part_id=part_id,
                timestamp=datetime.utcnow(),
                batch_number=batch_number,
                camera_id=camera_id,
                shift=shift,
                status=status,
                is_defective=is_defective,
                defect_count=defect_count,
                highest_severity=highest_severity,
                avg_confidence=avg_confidence,
                inference_time_ms=inference_time_ms,
                image_name=image_name
            )
            db.add(record)
            db.flush()  # Populates record.id

            for item in defects_data:
                bbox = item.get("bounding_box", {})
                defect = DefectDetail(
                    inspection_id=record.id,
                    defect_type=item.get("defect_type", "scratch"),
                    confidence=float(item.get("confidence", 0.0)),
                    severity=item.get("severity", "Minor"),
                    x1=int(bbox.get("x1", item.get("x1", 0))),
                    y1=int(bbox.get("y1", item.get("y1", 0))),
                    x2=int(bbox.get("x2", item.get("x2", 0))),
                    y2=int(bbox.get("y2", item.get("y2", 0))),
                    area_px=int(item.get("area_px", 0))
                )
                db.add(defect)

            db.commit()
            db.refresh(record)
            return record

    @staticmethod
    def get_recent_inspections(
        limit: int = 50,
        status_filter: Optional[str] = None,
        part_id_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves inspection logs ordered by newest first."""
        init_db()
        with get_db() as db:
            query = db.query(InspectionRecord)
            if status_filter and status_filter.upper() in ["PASS", "FAIL"]:
                query = query.filter(InspectionRecord.status == status_filter.upper())
            if part_id_query:
                query = query.filter(InspectionRecord.part_id.ilike(f"%{part_id_query}%"))

            records = query.order_by(desc(InspectionRecord.timestamp)).limit(limit).all()
            return [r.to_dict() for r in records]

    @staticmethod
    def get_analytics_summary() -> Dict[str, Any]:
        """
        Computes production KPIs:
        - Total parts scanned
        - Pass / Fail counts and Yield Rate %
        - Top defect modes (Pareto breakdown)
        - Average inference latency
        """
        init_db()
        with get_db() as db:
            total_inspections = db.query(func.count(InspectionRecord.id)).scalar() or 0

            if total_inspections == 0:
                return {
                    "total_inspected": 0,
                    "total_passed": 0,
                    "total_failed": 0,
                    "yield_rate_percent": 100.0,
                    "avg_latency_ms": 0.0,
                    "defect_counts": {},
                    "severity_breakdown": {}
                }

            passed_count = db.query(func.count(InspectionRecord.id))\
                .filter(InspectionRecord.status == "PASS").scalar() or 0
            failed_count = total_inspections - passed_count
            yield_rate = (passed_count / total_inspections) * 100.0

            avg_latency = db.query(func.avg(InspectionRecord.inference_time_ms)).scalar() or 0.0

            # Defect breakdown by type
            defect_type_counts = db.query(
                DefectDetail.defect_type, func.count(DefectDetail.id)
            ).group_by(DefectDetail.defect_type).all()
            defect_counts = {dtype: count for dtype, count in defect_type_counts}

            # Severity breakdown
            severity_counts = db.query(
                DefectDetail.severity, func.count(DefectDetail.id)
            ).group_by(DefectDetail.severity).all()
            severity_breakdown = {sev: count for sev, count in severity_counts}

            return {
                "total_inspected": total_inspections,
                "total_passed": passed_count,
                "total_failed": failed_count,
                "yield_rate_percent": round(yield_rate, 2),
                "avg_latency_ms": round(float(avg_latency), 2),
                "defect_counts": defect_counts,
                "severity_breakdown": severity_breakdown
            }

    @staticmethod
    def seed_demo_data(count: int = 40):
        """
        Populates sample historical manufacturing logs if database is empty.
        Ensures a vibrant dashboard upon initial launch.
        """
        init_db()
        with get_db() as db:
            existing = db.query(func.count(InspectionRecord.id)).scalar() or 0
            if existing >= 15:
                return  # Data already exists

            now = datetime.utcnow()
            part_prefixes = ["PCB-MAIN", "STEEL-PANEL", "GEAR-DRIVE", "ALUM-CHASSIS"]
            defect_pool = ["scratch", "crack", "dent", "burr", "hole", "surface_stain"]
            shifts = ["Shift-A (Morning)", "Shift-B (Evening)", "Shift-C (Night)"]

            for i in range(count):
                delta_mins = (count - i) * 18 + random.randint(1, 10)
                rec_time = now - timedelta(minutes=delta_mins)
                part_type = random.choice(part_prefixes)
                part_id = f"{part_type}-{1000 + i}"

                # Realistic manufacturing line has ~92-96% pass yield
                is_fail = random.random() < 0.08
                status = "FAIL" if is_fail else "PASS"
                latency = round(random.uniform(18.5, 32.4), 2)
                shift = random.choice(shifts)

                num_defects = random.randint(1, 3) if is_fail else (1 if random.random() < 0.1 else 0)
                highest_sev = "None"
                if num_defects > 0:
                    highest_sev = "Critical" if is_fail and random.random() < 0.6 else "Minor"

                rec = InspectionRecord(
                    part_id=part_id,
                    timestamp=rec_time,
                    batch_number=f"BATCH-2026-M{random.randint(1, 4)}",
                    camera_id=f"CAM-STATION-0{random.randint(1, 3)}",
                    shift=shift,
                    status=status,
                    is_defective=(num_defects > 0),
                    defect_count=num_defects,
                    highest_severity=highest_sev,
                    avg_confidence=round(random.uniform(0.82, 0.96), 3) if num_defects > 0 else 0.99,
                    inference_time_ms=latency,
                    image_name=f"inspection_{part_id.lower()}.jpg"
                )
                db.add(rec)
                db.flush()

                for _ in range(num_defects):
                    dtype = random.choice(defect_pool)
                    sev = "Critical" if dtype in ["crack", "hole"] else ("Moderate" if random.random() < 0.4 else "Minor")
                    db.add(DefectDetail(
                        inspection_id=rec.id,
                        defect_type=dtype,
                        confidence=round(random.uniform(0.75, 0.95), 3),
                        severity=sev,
                        x1=random.randint(40, 250),
                        y1=random.randint(40, 250),
                        x2=random.randint(280, 500),
                        y2=random.randint(280, 480),
                        area_px=random.randint(120, 1800)
                    ))

            db.commit()
            print(f"[Database] Seeded {count} historical manufacturing inspection logs successfully.")
