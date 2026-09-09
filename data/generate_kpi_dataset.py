"""
Generates realistic manufacturing quality inspection KPI dataset.
Simulates 800+ inspection cycles across 4 assembly lines, 3 shifts,
and diverse component categories (PCBs, Machined Gears, Steel Plates, Welded Joints).
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manufacturing_kpi_dataset.csv")

def generate_kpi_dataset(num_records: int = 800):
    lines = ["Line-01 (PCB SMT)", "Line-02 (Metal Stamping)", "Line-03 (Precision Machining)", "Line-04 (Welding & Joinery)"]
    products = {
        "Line-01 (PCB SMT)": ("Circuit_Board", ["Scratch", "Burr", "Misalignment"]),
        "Line-02 (Metal Stamping)": ("Chassis_Panel", ["Dent", "Scratch", "Surface_Stain"]),
        "Line-03 (Precision Machining)": ("Machined_Gear", ["Dent", "Burr", "Crack"]),
        "Line-04 (Welding & Joinery)": ("Pipe_Weld", ["Void", "Crack", "Surface_Stain"])
    }
    shifts = ["Shift-A (06:00-14:00)", "Shift-B (14:00-22:00)", "Shift-C (22:00-06:00)"]
    operators = ["OP-104 (Elena R.)", "OP-109 (Marcus V.)", "OP-212 (Devin K.)", "OP-305 (Priya S.)", "OP-418 (Chen W.)"]

    start_date = datetime.now() - timedelta(days=14)
    data = []

    for i in range(num_records):
        rec_time = start_date + timedelta(minutes=i * 24 + random.randint(1, 15))
        line = random.choice(lines)
        product_cat, possible_defects = products[line]
        shift = random.choice(shifts)
        operator = random.choice(operators)

        # Baseline factory yield is ~94-98%
        is_defect = random.random() < 0.055
        status = "REJECT" if is_defect else "PASS"

        if is_defect:
            defect_type = random.choice(possible_defects)
            severity = "Critical" if defect_type in ["Crack", "Void"] else ("Moderate" if random.random() < 0.5 else "Minor")
            scrap_cost = round(random.uniform(45.0, 320.0), 2) if severity == "Critical" else round(random.uniform(12.0, 65.0), 2)
        else:
            defect_type = "None"
            severity = "None"
            scrap_cost = 0.0

        latency_ms = round(random.uniform(18.2, 34.5), 2)
        confidence = round(random.uniform(0.88, 0.98), 3) if is_defect else round(random.uniform(0.96, 0.999), 3)

        data.append({
            "Inspection_ID": f"INSP-{10000 + i}",
            "Timestamp": rec_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Date": rec_time.strftime("%Y-%m-%d"),
            "Hour": rec_time.hour,
            "Assembly_Line": line,
            "Product_Category": product_cat,
            "Shift": shift,
            "Operator": operator,
            "Inspection_Result": status,
            "Defect_Type": defect_type,
            "Severity": severity,
            "Confidence": confidence,
            "Inference_Latency_ms": latency_ms,
            "Scrap_Cost_USD": scrap_cost
        })

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[KPIDataset] Generated {num_records} manufacturing quality records at: {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    generate_kpi_dataset()
