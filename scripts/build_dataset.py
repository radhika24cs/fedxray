#!/usr/bin/env python3
"""
build_dataset.py
Generates the synthetic clinical Chest X-ray dataset modeled after standard 
NIH ChestX-ray14 and CheXpert radiograph protocols for Federated Learning.
"""

import os
import csv
import json
import math
import random
import struct
import zlib

CLASSES = [
    "Normal",
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Pleural Effusion",
    "Infiltration"
]

WIDTH = 224
HEIGHT = 224

def make_png(width, height, pixels):
    """Write 8-bit grayscale PNG using standard python zlib and struct."""
    raw = b''.join(b'\x00' + bytes(pixels[y*width:(y+1)*width]) for y in range(height))
    compressor = zlib.compressobj(level=6)
    idat = compressor.compress(raw) + compressor.flush()
    
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
    png += chunk(b'IDAT', idat)
    png += chunk(b'IEND', b'')
    return png

def render_xray(condition: str, seed_val: int) -> bytes:
    """
    Renders a 224x224 chest radiograph simulation with anatomical structures:
    - Rib cage & clavicles
    - Trachea & mediastinum
    - Bilateral lung fields (darker)
    - Cardiac silhouette (heart shadow)
    - Costophrenic angles & diaphragm
    - Condition-specific radiopacity / infiltrates / cardiomegaly / effusion
    """
    rng = random.Random(seed_val)
    pixels = [0] * (WIDTH * HEIGHT)
    
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    
    for y in range(HEIGHT):
        ny = y / HEIGHT
        for x in range(WIDTH):
            nx = x / WIDTH
            
            # Baseline background (chest soft tissue & body wall): medium dark (30-50)
            val = 40.0 + rng.uniform(-3, 3)
            
            # Body outline (elliptical thoracic perimeter)
            dx = (x - cx) / (WIDTH * 0.46)
            dy = (y - cy) / (HEIGHT * 0.48)
            dist_sq = dx*dx + dy*dy
            if dist_sq > 1.0:
                # Outside patient body / air
                pixels[y * WIDTH + x] = int(max(5, min(25, 15 + rng.uniform(-3, 3))))
                continue
                
            # Soft tissue attenuation increases toward margins
            val += dist_sq * 35.0
            
            # Lung fields: two dark lobes (low density)
            # Left hemithorax (patient's right, display left): x in [0.18, 0.46], y in [0.20, 0.75]
            # Right hemithorax (patient's left, display right): x in [0.54, 0.82], y in [0.20, 0.75]
            in_left_lung = False
            in_right_lung = False
            
            # Elliptical bounds for lungs
            lx_center = 0.32
            rx_center = 0.68
            ly_center = 0.48
            
            # Lung distance metrics
            ldx = (nx - lx_center) / 0.14
            ldy = (ny - ly_center) / 0.26
            if (ldx*ldx + ldy*ldy) < 1.0 and ny < 0.74:
                in_left_lung = True
                
            rdx = (nx - rx_center) / 0.14
            rdy = (ny - ly_center) / 0.26
            if (rdx*rdx + rdy*rdy) < 1.0 and ny < 0.74:
                in_right_lung = True
                
            # Lung attenuation (radiolucent = darker, 40-75)
            if in_left_lung:
                lung_depth = 1.0 - (ldx*ldx + ldy*ldy)
                val = 45.0 + (1.0 - lung_depth) * 30.0 + rng.uniform(-4, 4)
            elif in_right_lung:
                lung_depth = 1.0 - (rdx*rdx + rdy*rdy)
                val = 45.0 + (1.0 - lung_depth) * 30.0 + rng.uniform(-4, 4)
                
            # Central Mediastinum / Spine (bright radiopaque vertical column)
            med_dist = abs(nx - 0.50)
            if med_dist < 0.07:
                val += (1.0 - (med_dist / 0.07)) * 120.0
            elif med_dist < 0.14:
                val += (1.0 - (med_dist / 0.14)) * 50.0
                
            # Diaphragm & Abdominal base (radiopaque shelf at y > 0.70)
            if ny > 0.68:
                diaph_rise = math.sin((nx - 0.18) * math.pi / 0.64)
                if diaph_rise > 0:
                    edge = 0.78 - 0.08 * diaph_rise
                    if ny > edge:
                        val = max(val, 160.0 + (ny - edge) * 150.0 + rng.uniform(-5, 5))
                        
            # Rib shadows (curved horizontal bands)
            for rib_idx in range(4, 10):
                rib_y = 0.12 * rib_idx + 0.03 * math.sin((nx - 0.5) * 3.14)
                if abs(ny - rib_y) < 0.015:
                    val += 22.0
                    
            # Clavicles (upper horizontal dense bars at ny ~ 0.20)
            if 0.18 < ny < 0.23:
                clav_dist = abs(nx - 0.50)
                if 0.08 < clav_dist < 0.42:
                    val += 45.0
                    
            # Normal Cardiac Silhouette (Heart shadow)
            # Extends from center toward patient left (display right: nx ~ 0.45 to 0.64, ny ~ 0.48 to 0.72)
            cardiac_radius = 0.14 if condition != "Cardiomegaly" else 0.23
            cdx = (nx - 0.53) / cardiac_radius
            cdy = (ny - 0.60) / 0.13
            if (cdx*cdx + cdy*cdy) < 1.0:
                c_dense = (1.0 - (cdx*cdx + cdy*cdy))
                val = max(val, 120.0 + c_dense * 75.0)
                
            # Condition-specific radiopacity features:
            if condition == "Pneumonia":
                # Alveolar consolidation: dense, fluffy opacity in right lower/middle lobe (display right or left)
                pdx = (nx - 0.68) / 0.11
                pdy = (ny - 0.56) / 0.10
                if (pdx*pdx + pdy*pdy) < 1.0:
                    consolidation = (1.0 - (pdx*pdx + pdy*pdy))
                    val += consolidation * 95.0 + rng.uniform(-10, 10)
                    
            elif condition == "Cardiomegaly":
                # Marked cardiac enlargement across midline
                cdx_large = (nx - 0.52) / 0.25
                cdy_large = (ny - 0.62) / 0.15
                if (cdx_large*cdx_large + cdy_large*cdy_large) < 1.0:
                    val = max(val, 130.0 + (1.0 - (cdx_large*cdx_large + cdy_large*cdy_large)) * 85.0)
                    
            elif condition == "Pleural Effusion":
                # Blunting of costophrenic sulcus & fluid meniscus at lateral base (nx > 0.62, ny > 0.64)
                if nx > 0.62 and ny > 0.64:
                    meniscus_slope = (nx - 0.62) * 0.6
                    if (ny - 0.64) > (0.10 - meniscus_slope):
                        val = max(val, 150.0 + rng.uniform(-6, 6))
                        
            elif condition == "Atelectasis":
                # Linear / platelike band of volume collapse
                if abs((ny - 0.58) - 0.15 * (nx - 0.32)) < 0.02 and (0.22 < nx < 0.44):
                    val += 85.0 + rng.uniform(-8, 8)
                    
            elif condition == "Infiltration":
                # Diffuse reticular / perihilar haziness
                if (in_left_lung or in_right_lung) and 0.35 < ny < 0.65:
                    if rng.random() > 0.35:
                        val += rng.uniform(25, 55)
                        
            pixels[y * WIDTH + x] = int(max(0, min(255, val)))
            
    return make_png(WIDTH, HEIGHT, pixels)

def generate_full_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    os.makedirs(base_dir, exist_ok=True)
    
    # Hospital non-IID distributions
    # Hospital A: Specialized in Pulmonology & Acute Respiratory Care (High Pneumonia, Infiltration)
    # Hospital B: Specialized in Cardiovascular & Heart Failure Center (High Cardiomegaly, Pleural Effusion)
    # Hospital C: Metropolitan General & Thoracic Surgery (Atelectasis, Mixed, Balanced)
    
    plan = [
        # (Split/Hospital folder, count_per_class dictionary, source_name)
        (
            "hospital_a",
            {
                "Normal": 30,
                "Pneumonia": 45,
                "Infiltration": 35,
                "Atelectasis": 15,
                "Cardiomegaly": 8,
                "Pleural Effusion": 12,
            },
            "Hospital A - Pulmonology & Acute Respiratory Institute"
        ),
        (
            "hospital_b",
            {
                "Normal": 30,
                "Cardiomegaly": 45,
                "Pleural Effusion": 40,
                "Pneumonia": 10,
                "Atelectasis": 12,
                "Infiltration": 8,
            },
            "Hospital B - Cardiovascular & Thoracic Care Center"
        ),
        (
            "hospital_c",
            {
                "Normal": 35,
                "Atelectasis": 40,
                "Pneumonia": 25,
                "Pleural Effusion": 25,
                "Infiltration": 20,
                "Cardiomegaly": 15,
            },
            "Hospital C - Metro General Teaching Hospital"
        ),
        (
            "validation",
            {
                "Normal": 15,
                "Pneumonia": 15,
                "Atelectasis": 15,
                "Cardiomegaly": 15,
                "Pleural Effusion": 15,
                "Infiltration": 15,
            },
            "Multi-Center Cross-Validation Benchmark"
        ),
        (
            "test",
            {
                "Normal": 20,
                "Pneumonia": 20,
                "Atelectasis": 20,
                "Cardiomegaly": 20,
                "Pleural Effusion": 20,
                "Infiltration": 20,
            },
            "Federated Global Held-Out Evaluation Test Set"
        )
    ]
    
    csv_rows = []
    global_manifest = {
        "dataset_name": "FedChestXray-Bench-6",
        "standard": "NIH ChestX-ray14 / CheXpert Radiographic Benchmark Protocol",
        "image_dimensions": f"{WIDTH}x{HEIGHT}",
        "color_channels": "Grayscale (1-channel, 8-bit depth PNG)",
        "classes": CLASSES,
        "splits": {},
        "total_images": 0
    }
    
    global_seed = 42
    img_counter = 1
    
    for folder, counts, hospital_desc in plan:
        target_dir = os.path.join(base_dir, folder)
        os.makedirs(target_dir, exist_ok=True)
        split_count = 0
        
        for c_idx, condition in enumerate(CLASSES):
            class_subfolder = os.path.join(target_dir, condition.lower().replace(" ", "_"))
            os.makedirs(class_subfolder, exist_ok=True)
            num_samples = counts.get(condition, 0)
            
            for i in range(num_samples):
                global_seed += 1
                img_id = f"CXR_{img_counter:05d}.png"
                img_counter += 1
                split_count += 1
                
                # Render PNG
                png_bytes = render_xray(condition, global_seed)
                file_path = os.path.join(class_subfolder, img_id)
                with open(file_path, "wb") as f:
                    f.write(png_bytes)
                    
                # Patient demographic metadata
                patient_num = (global_seed % 450) + 1001
                age = 22 + (global_seed * 17) % 65
                gender = "M" if (global_seed % 2 == 0) else "F"
                view = "PA" if (global_seed % 3 != 0) else "AP"
                
                csv_rows.append({
                    "image_id": img_id,
                    "file_path": os.path.relpath(file_path, base_dir),
                    "finding_label": condition,
                    "category": "Normal" if condition == "Normal" else "Abnormal",
                    "patient_id": f"PAT_{patient_num}",
                    "patient_age": age,
                    "patient_gender": gender,
                    "view_position": view,
                    "hospital_split": folder,
                    "institution_name": hospital_desc,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "format": "PNG"
                })
                
        global_manifest["splits"][folder] = {
            "institution": hospital_desc,
            "total_images": split_count,
            "class_distribution": {k: counts.get(k, 0) for k in CLASSES}
        }
        global_manifest["total_images"] += split_count

    # Write CSV
    csv_file = os.path.join(base_dir, "metadata.csv")
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "image_id", "file_path", "finding_label", "category",
            "patient_id", "patient_age", "patient_gender",
            "view_position", "hospital_split", "institution_name",
            "width", "height", "format"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    # Write classes.json
    classes_file = os.path.join(base_dir, "classes.json")
    with open(classes_file, "w") as f:
        json.dump({
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "normal_class": "Normal",
            "pathology_classes": [c for c in CLASSES if c != "Normal"],
            "class_indices": {c: idx for idx, c in enumerate(CLASSES)}
        }, f, indent=2)
        
    # Write dataset_manifest.json
    manifest_file = os.path.join(base_dir, "dataset_manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(global_manifest, f, indent=2)
        
    print(f"Dataset generation complete! Total images: {global_manifest['total_images']}")
    print(f"Metadata CSV: {csv_file}")
    print(f"Manifest JSON: {manifest_file}")

if __name__ == "__main__":
    generate_full_dataset()
