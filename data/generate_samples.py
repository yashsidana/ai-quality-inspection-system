"""
Synthetic Manufacturing Part Generator.
Generates realistic industrial inspection parts (PCBs, machined plates, welds, gears)
with authentic simulated defect topologies (scratches, cracks, dents, voids, burrs)
for zero-download testing and demonstration.
"""

import os
import numpy as np
import cv2

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")


def create_pcb_sample(has_scratch: bool = True) -> np.ndarray:
    """Generates a high-resolution printed circuit board (PCB) image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Deep emerald green solder mask
    img[:] = (18, 72, 35)

    # Grid texture
    for x in range(0, 640, 40):
        cv2.line(img, (x, 0), (x, 480), (22, 85, 42), 1)
    for y in range(0, 480, 40):
        cv2.line(img, (0, y), (640, y), (22, 85, 42), 1)

    # Copper traces
    trace_color = (40, 160, 210)  # Gold/Copper in BGR
    traces = [
        [(60, 100), (200, 100), (260, 160), (450, 160)],
        [(60, 180), (140, 180), (190, 230), (320, 230), (380, 290), (550, 290)],
        [(100, 380), (240, 380), (300, 320), (520, 320)],
        [(400, 80), (400, 220), (460, 280), (580, 280)]
    ]
    for tr in traces:
        pts = np.array(tr, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, trace_color, 4, cv2.LINE_AA)

    # IC chip packages
    cv2.rectangle(img, (220, 170), (340, 270), (28, 28, 30), -1)
    cv2.rectangle(img, (220, 170), (340, 270), (80, 80, 85), 2)
    cv2.putText(img, "ARM-CORTEX", (230, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # IC chip pins
    for py in range(180, 265, 12):
        cv2.rectangle(img, (210, py), (220, py + 6), (190, 200, 210), -1)
        cv2.rectangle(img, (340, py), (350, py + 6), (190, 200, 210), -1)

    # Solder pads / vias
    vias = [(100, 100), (200, 100), (450, 160), (320, 230), (550, 290), (100, 380), (520, 320)]
    for vx, vy in vias:
        cv2.circle(img, (vx, vy), 8, (190, 200, 210), -1)
        cv2.circle(img, (vx, vy), 3, (15, 45, 25), -1)

    # Defect: Sharp diagonal metallic surface scratch
    if has_scratch:
        scratch_pts = [(160, 110), (195, 155), (230, 190), (280, 215)]
        for i in range(len(scratch_pts) - 1):
            p1, p2 = scratch_pts[i], scratch_pts[i+1]
            cv2.line(img, p1, p2, (230, 240, 250), 3, cv2.LINE_AA)
            cv2.line(img, (p1[0]+1, p1[1]+1), (p2[0]+1, p2[1]+1), (10, 30, 15), 1, cv2.LINE_AA)

    return img


def create_steel_sample(has_crack: bool = True) -> np.ndarray:
    """Generates brushed steel plate with stress fracture crack."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Brushed metal base
    base_val = 145
    noise = np.random.normal(0, 12, (480, 640)).astype(np.int16)
    metal = np.clip(base_val + noise, 0, 255).astype(np.uint8)
    # Brushed horizontal streaks
    kernel = np.ones((1, 15), np.float32) / 15
    metal = cv2.filter2D(metal, -1, kernel)
    img[:, :, 0] = metal
    img[:, :, 1] = np.clip(metal - 2, 0, 255)
    img[:, :, 2] = np.clip(metal + 4, 0, 255)

    # Chamfered mounting bolt holes
    bolts = [(80, 80), (560, 80), (80, 400), (560, 400)]
    for bx, by in bolts:
        cv2.circle(img, (bx, by), 24, (100, 100, 105), -1)
        cv2.circle(img, (bx, by), 16, (40, 40, 45), -1)
        cv2.circle(img, (bx, by), 25, (210, 210, 215), 2)

    # Defect: Jagged stress crack
    if has_crack:
        crack_nodes = [
            (290, 140), (305, 175), (298, 205), (320, 250),
            (315, 290), (335, 330), (328, 365)
        ]
        # Main crack line
        for i in range(len(crack_nodes) - 1):
            cv2.line(img, crack_nodes[i], crack_nodes[i+1], (20, 20, 25), 3, cv2.LINE_AA)
            # Crack shadow / specular edge
            cv2.line(img, (crack_nodes[i][0] + 2, crack_nodes[i][1]), (crack_nodes[i+1][0] + 2, crack_nodes[i+1][1]), (220, 220, 230), 1, cv2.LINE_AA)

        # Micro-branch fissure
        cv2.line(img, (320, 250), (355, 275), (25, 25, 30), 2, cv2.LINE_AA)

    return img


def create_machined_gear_sample(has_dent: bool = True) -> np.ndarray:
    """Generates a precision machined gear with an impact dent."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (35, 35, 40)

    center = (320, 240)
    outer_r = 160
    inner_r = 60

    # Draw gear teeth
    num_teeth = 24
    for i in range(num_teeth):
        angle = i * (2 * np.pi / num_teeth)
        tx = int(center[0] + (outer_r + 25) * np.cos(angle))
        ty = int(center[1] + (outer_r + 25) * np.sin(angle))
        cv2.circle(img, (tx, ty), 16, (160, 165, 175), -1)

    # Gear body
    cv2.circle(img, center, outer_r, (170, 175, 185), -1)
    cv2.circle(img, center, outer_r - 20, (130, 135, 145), 3)

    # Center bore & keyway
    cv2.circle(img, center, inner_r, (30, 30, 35), -1)
    cv2.rectangle(img, (center[0] - 12, center[1] - inner_r - 15), (center[0] + 12, center[1] - inner_r + 5), (30, 30, 35), -1)

    # Weight reduction circles
    for i in range(6):
        ang = i * (2 * np.pi / 6)
        wx = int(center[0] + 105 * np.cos(ang))
        wy = int(center[1] + 105 * np.sin(ang))
        cv2.circle(img, (wx, wy), 20, (35, 35, 40), -1)

    # Defect: Deep impact dent deformation on gear rim
    if has_dent:
        dent_center = (410, 175)
        cv2.ellipse(img, dent_center, (22, 14), 45, 0, 360, (50, 50, 55), -1)
        cv2.ellipse(img, dent_center, (24, 16), 45, 0, 360, (230, 235, 240), 2)
        cv2.circle(img, dent_center, 6, (20, 20, 25), -1)

    return img


def create_welded_seam_sample(has_void: bool = True) -> np.ndarray:
    """Generates an industrial pipeline welded joint with porosity void defects."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (85, 90, 95)

    # Heat affected zone
    cv2.rectangle(img, (180, 0), (460, 480), (65, 70, 80), -1)
    cv2.rectangle(img, (250, 0), (390, 480), (120, 125, 135), -1)

    # Weld ripple bead texture
    for y in range(10, 470, 16):
        cv2.ellipse(img, (320, y), (60, 12), 0, 0, 180, (155, 160, 170), 2)
        cv2.ellipse(img, (320, y + 2), (55, 10), 0, 0, 180, (70, 75, 80), 2)

    # Defect: Porosity pinholes / gas voids
    if has_void:
        voids = [(310, 180, 9), (325, 205, 12), (305, 225, 7), (338, 192, 6)]
        for vx, vy, vr in voids:
            cv2.circle(img, (vx, vy), vr, (15, 15, 20), -1)
            cv2.circle(img, (vx + 1, vy + 1), vr + 2, (200, 205, 215), 1)

    return img


def create_flawless_sample() -> np.ndarray:
    """Generates a pristine, zero-defect precision machined component (PASS sample)."""
    return create_pcb_sample(has_scratch=False)


def generate_all_samples():
    """Builds and stores all sample manufacturing parts into data/sample_images."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    samples = {
        "pcb_defect_scratch.jpg": create_pcb_sample(has_scratch=True),
        "steel_plate_crack.jpg": create_steel_sample(has_crack=True),
        "machined_gear_dent.jpg": create_machined_gear_sample(has_dent=True),
        "welded_joint_void.jpg": create_welded_seam_sample(has_void=True),
        "flawless_part_pass.jpg": create_flawless_sample()
    }

    for fname, img in samples.items():
        out_path = os.path.join(SAMPLE_DIR, fname)
        cv2.imwrite(out_path, img)
        print(f"[SampleGenerator] Generated: {out_path}")

    print("[SampleGenerator] All 5 industrial manufacturing samples generated.")


if __name__ == "__main__":
    generate_all_samples()
